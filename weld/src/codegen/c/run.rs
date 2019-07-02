//!
//! This module manages compiling with optimization the generated C module
//! to the machine code.

use libc;
use time;

use std::ptr;
use std::sync::{Once, ONCE_INIT};

use libc::{uint64_t, c_void};

use self::time::PreciseTime;

use crate::WeldError;
use crate::ast::Type;
use crate::ast::Type::*;
use crate::ast::ScalarKind;
use crate::util::stats::RunStats;
use crate::util::veoffload::*;

use crate::codegen::Runnable;

static ONCE: Once = ONCE_INIT;
static mut INITIALIZE_FAILED: bool = false;

// Several configuration of this run function
// Whether to check and dump data or not.
static CHECK_DATA: bool = false;
static DUMP_DATA: bool = false;
static USE_CONVERT_TOP_PARAMS: bool = false;
static SERIALIZE_THRESHOLD: usize = 1024000;

/// The callable function type.
type I64Func = extern "C" fn(i64) -> i64;

// Use CompiledModule defined in compile.rs
use crate::codegen::c::compile::*;

// The codegen interface requires that modules implement this trait.
// This allows supporting multiple backends via dynamic dispatch.
impl Runnable for CompiledModule {
    fn run(&self, arg: i64, stats: &mut RunStats) -> Result<i64, WeldError> {
        use crate::runtime::WeldRuntimeErrno;
        unsafe {
            let veo_ptr = get_global_veo_ptr();
            let start = PreciseTime::now();
            while !(*veo_ptr).ready {}
            let end = PreciseTime::now();
            stats.run_times.push((
                "wait initialize veo".to_string(), start.to(end)));

            // Load generated shared-object by Weld on VE.
            let start = PreciseTime::now();
            let libname = format!("./{}", self.filename);
            let libhdl_run = (*veo_ptr).load_library(libname)?;
            let end = PreciseTime::now();
            stats.run_times.push(("veo_load_library".to_string(), start.to(end)));

            // Prepare arguments for entry function.
            let start = PreciseTime::now();
            //      WeldInputArgs
            //   0: i64      input (refer a struct of parameters)
            //   8: i32      nworkers
            //  16: i64      memlimit
            //  24: i64      run
            use crate::codegen::WeldInputArgs;
            let input_ptr = arg as *const WeldInputArgs;
            // let data_ptr = (*input_ptr).input as u64;
            let data_ptr = &(*input_ptr).input as *const i64 as u64;
            let nworkers = (*input_ptr).nworkers;
            let mem_limit = (*input_ptr).mem_limit;
            let mut run = (*input_ptr).run;
            use crate::runtime::WeldRuntimeContext;
            let context: *mut WeldRuntimeContext;
            if run == 0 {
                // Call weld_runst_init to instanciate WeldRuntimeContext.
                use crate::runtime::ffi::weld_runst_init;
                context = weld_runst_init(nworkers, mem_limit);
                run = context as i64;
            } else {
                // Use existing WeldRuntimeContext.
                // (Usually, it is created by caller of this function)
                context = run as *mut WeldRuntimeContext;
            }
            let end = PreciseTime::now();
            stats.run_times.push(("from_raw".to_string(), start.to(end)));

            // Check parameters.
            if CHECK_DATA {
                println!("parameters {:?}", self.params);
                self.check_data(&self.params, data_ptr)?;
            }

            let mut input_generated = WeldInputArgs {
                input: 0,
                nworkers,
                mem_limit,
                run: 0, // FIXME: need to recover VE's run if it is avilable.
            };

            let (_buffer, addrs_ve, _buffer_size) = if USE_CONVERT_TOP_PARAMS {
                self.send_data_using_convert_top_params(
                    &self.params,
                    data_ptr,
                    &mut input_generated,
                    stats,
                )?
            } else {
                self.send_data_using_new_mechanism(
                    &self.params,
                    data_ptr,
                    &mut input_generated,
                    stats,
                )?
            };

            let start = PreciseTime::now();
            let args = VEOffload::args_alloc();
            VEOffload::args_set(args, 0, addrs_ve[0]);
            let end = PreciseTime::now();
            stats.run_times.push(("prepare arguments".to_string(), start.to(end)));

            let start = PreciseTime::now();
            let retval_ve_ptr: uint64_t = (*veo_ptr).call_and_wait(libhdl_run, "run", args)?;
            let errno = WeldRuntimeErrno::Success;
            let end = PreciseTime::now();
            stats.run_times.push(("call run".to_string(), start.to(end)));

            let start = PreciseTime::now();
            // Allocate and read WeldOutputArgs from VE memory.
            //      WeldOutputArgs
            //   0: intptr_t output
            //   8: i64      run
            //  16: i64      errno
            use crate::codegen::WeldOutputArgs;
            use crate::runtime::ffi::weld_runst_malloc;
            use std::mem::size_of;
            let output_size = size_of::<WeldOutputArgs>();
            let ret = weld_runst_malloc(context, output_size as i64)
                as *mut WeldOutputArgs;
            if errno != WeldRuntimeErrno::Unknown {
                (*veo_ptr).read_mem(retval_ve_ptr, ret as *mut c_void, output_size)?;
                // FIXME: need to handle VE's run correctly for latter calls.
                // Overwrite VE's `run` by HOST's run.
                (*ret).run = run;
            } else {
                // NOTE: never pass below
                (*ret).output = 0;
                (*ret).run = run;
                (*ret).errno = errno;
            }
            // Copy VE's output to VH if calculation was succeeded
            if (*ret).errno == WeldRuntimeErrno::Success {
                let proc_handle = (*veo_ptr).proc;
                self.convert_results(&self.ret_ty, (*ret).output as u64,
                                     &mut (*ret).output as *mut i64 as u64,
                                     proc_handle)?;
            }
            let end = PreciseTime::now();
            stats.run_times.push(("convert_results".to_string(), start.to(end)));

            let start = PreciseTime::now();
            VEOffload::args_free(args);
            for addr_ve in addrs_ve {
                (*veo_ptr).free_mem(addr_ve)?;
            }
            let end = PreciseTime::now();
            stats.run_times.push(("free and destroy".to_string(), start.to(end)));

            Ok(ret as i64)
        }
    }
}

impl CompiledModule {
    fn check_data(&self, ty: &Type, addr: u64) -> Result<u64, WeldError> {
        // Dump information of weld's type
        let view = addr as *const u8;
        println!("type {:?} addr {:?}", *ty, view);
        // Dump memory data
        for i in 0..16 {
            print!("{:02x} ", unsafe {*view.offset(i)});
        }
        println!("");

        match *ty {
            Scalar(kind) => {
                println!("  is Scalar({})", kind);
                Ok(0)
            }
            Simd(kind) => {
                println!("  is Simd({})", kind);
                Ok(0)
            }
            Struct(ref fields) => {
                println!("  is Struct({:?})", fields);
                let data = unsafe {*(addr as *const u64)};
                let mut faddr = data;
                for f in fields.iter() {
                    let size = self.calc_data_size(f)?;
                    self.check_data(f, faddr)?;
                    faddr += size as u64;
                }
                Ok(0)
            }
            Vector(ref elem) => {
                // Vector(elem) is following 16 bytes data structure.
                //  0:  intptr_t  data
                //  8:  u64       length
                let data = unsafe {*(addr as *const u64)};
                let len = unsafe {*(addr as *const i64).offset(1)};
                println!("  is Vector(elem={}, len={})", elem, len);
                /*
                // Dump 16 eleements.
                let size = self.calc_data_size(elem)?;
                let mut eaddr = data;
                for i in 0..16 {
                    self.check_data(elem, eaddr)?;
                    eaddr += size as u64;
                }
                */
                Ok(0)
            }
            Dict(_, _) => {
                weld_err!("Unsupported dict type {}", ty)
            }
            Builder(_, _) => {
                weld_err!("Unsupported builder type {}", ty)
            }
            _ => {
                weld_err!("Unsupported type {}", ty)
            }
        }
    }

    fn send_data_using_convert_top_params(
        &self,
        ty: &Type,
        data_ptr: u64,
        input_generated: &mut crate::codegen::WeldInputArgs,
        stats: &mut RunStats,
    ) -> Result<(Vec<u8>, Vec<u64>, usize), WeldError> {
        use crate::codegen::WeldInputArgs;
        unsafe {
            let veo_ptr = get_global_veo_ptr();
            // First, calculate the size of memory needed to be allocated by
            // the size of WeldInputArgs and the given data.
            let start = PreciseTime::now();
            let data_size = self.calc_size(ty, data_ptr)?;
            let end = PreciseTime::now();
            stats.run_times.push(("calc_size".to_string(), start.to(end)));

            // The calc_size returns the size of whole data including a pointer
            // the the struct of parameters, so reduce 8.
            use std::mem::size_of;
            let input_size = size_of::<WeldInputArgs>();
            let sz = input_size + data_size - 8;

            let start = PreciseTime::now();
            let mut addrs_ve = Vec::new();
            addrs_ve.push((*veo_ptr).alloc_mem(sz)?);
            let end = PreciseTime::now();
            stats.run_times.push(("veo_alloc_mem".to_string(), start.to(end)));

            // prepare arguments and whole input data
            let start = PreciseTime::now();
            input_generated.input = (addrs_ve[0] + input_size as u64) as i64;

            let mut buffer = Vec::<u8>::with_capacity(sz);
            buffer.set_len(sz);
            copy_nonoverlapping(input_generated as *const WeldInputArgs
                                as *const u8, &mut buffer[0],
                                input_size as usize);
            let addr_vh = &buffer[0] as *const u8 as u64;
            // let input_p = addr_vh as *const WeldInputArgs;
            // println!("nworkers {}", (*input_p).nworkers);
            // println!("mem_limit {}", (*input_p).mem_limit);
            // println!("run {}", (*input_p).run);

            self.convert_top_params(
                ty,
                data_ptr,
                addr_vh + input_size as u64,
                addrs_ve[0] + input_size as u64,
            )?;
            let end = PreciseTime::now();
            stats.run_times.push(("convert_top_params".to_string(), start.to(end)));

            let start = PreciseTime::now();
            (*veo_ptr).write_mem(&buffer[0] as *const u8 as *const c_void, addrs_ve[0], sz)?;
            let end = PreciseTime::now();
            stats.run_times.push(("veo_write_mem".to_string(), start.to(end)));

            Ok((buffer, addrs_ve, sz))
        }
    }

    fn send_data_using_new_mechanism(
        &self,
        ty: &Type,
        data_ptr: u64,
        input_generated: &mut crate::codegen::WeldInputArgs,
        stats: &mut RunStats,
    ) -> Result<(Vec<u8>, Vec<u64>, usize), WeldError> {
        use crate::codegen::WeldInputArgs;
        unsafe {
            let veo_ptr = get_global_veo_ptr();
            // Retrieve files and table data of structure first.
            let field_tys = { match *ty {
                Struct(ref fields) => fields,
                _ => unreachable!(),
            }};
            let fields_ptr = *(data_ptr as *const u64);

            // Calculate the size of WeldInputArgs.
            use std::mem::size_of;
            let input_size = size_of::<WeldInputArgs>();
            let num_params = field_tys.len();
            let flag_size = (num_params + 7) / 8 * 8;

            // Convert parameters.
            //
            //   Return converted data like below.
            //    0:                    input_data_area (input_size bytes)
            //    input_size:           serialized flag (#params x 1 bytes)
            //    input_size+#param:    serialized data (data_size)
            //
            //   bufer:         points the beginning of converted data
            //   size:    equals to input_size+#params+data_size
            let start = PreciseTime::now();
            let (mut buffer, buffer_size) = self.convert_data(
                &field_tys,
                fields_ptr,
            )?;
            let end = PreciseTime::now();
            stats.run_times.push(("convert_data".to_string(), start.to(end)));

            let mut addrs = Vec::<*const c_void>::new();
            let mut sizes = Vec::new();
            let mut serialized_flags = Vec::new();
            // Prepare for the first buffer.
            addrs.push(&buffer[0] as *const u8 as *const c_void);
            sizes.push(buffer_size);
            serialized_flags.push(false);
            // Prepare for the rests.
            let mut field_offset = 0;
            for (i, f) in field_tys.iter().enumerate() {
                let faddr = fields_ptr + field_offset as u64;
                let addr = *(faddr as *const u64);
                addrs.push(addr as *const c_void);
                sizes.push(self.calc_contents_size(f, faddr)?);
                if buffer[input_size + i] == 1 {
                    // No need to alloc_mem/write_mem for serialized data.
                    // Data is serialized into the first buffer.
                    serialized_flags.push(true);
                } else {
                    // Need to alloc_mem/write_mem for this non-serialized data.
                    serialized_flags.push(false)
                }
                field_offset += self.calc_data_size(f)?;
            }

            // prepare arguments and whole input data
            input_generated.input = (input_size + flag_size) as i64;
            copy_nonoverlapping(input_generated as *const WeldInputArgs
                                as *const u8, &mut buffer[0],
                                input_size);

            // dump serialized data
            if DUMP_DATA {
                dump_data(&buffer[0] as *const u8, 160);
            }

            // Allocate VE memory.
            let start = PreciseTime::now();
            let mut addrs_ve = Vec::new();
            for i in 0..addrs.len() {
                if !serialized_flags[i] {
                    addrs_ve.push((*veo_ptr).alloc_mem(sizes[i])?);
                } else {
                    addrs_ve.push(0 as u64);
                }
            }
            let end = PreciseTime::now();
            stats.run_times.push(("veo_alloc_mem".to_string(), start.to(end)));

            // Deserialize data on host.
            let start = PreciseTime::now();
            self.deserialize_on_host(
                field_tys,
                &mut buffer,
                &serialized_flags,
                &addrs_ve,
            )?;
            let end = PreciseTime::now();
            stats.run_times.push(("deserialize_on_host".to_string(), start.to(end)));

            // dump deserialized data
            if DUMP_DATA {
                dump_data(&buffer[0] as *const u8, 160);
            }

            let start = PreciseTime::now();
            for i in 0..addrs.len() {
                if !serialized_flags[i] {
                    (*veo_ptr).write_mem(addrs[i], addrs_ve[i], sizes[i])?;
                }
            }
            let end = PreciseTime::now();
            stats.run_times.push(("veo_write_mem".to_string(), start.to(end)));

            Ok((buffer, addrs_ve, buffer_size))
        }
    }
    fn deserialize_on_host(
        &self,
        field_tys: &[Type],             // 0..N
        buffer: &mut Vec<u8>,
        serialized_flags: &[bool],      // 0..N+1
        addrs_ve: &Vec<u64>,            // 0..N+1
    ) -> Result<(), WeldError> {
        // First deserialize the pointer to struct.
        let ptr = &mut buffer[0] as *mut u8 as u64;
        let offset = unsafe { *(ptr as *mut u64) } as usize;
        unsafe {
            *(ptr as *mut u64) = addrs_ve[0] + offset as u64;
        }
        // Work on top params.
        let mut field_offset = 0;
        for (i, f) in field_tys.iter().enumerate() {
            // Work on each param.
            if serialized_flags[i+1] {
                // Deserialized this param recursively.
                self._deserialize_on_host(
                    f,
                    buffer,
                    addrs_ve[0],
                    offset + field_offset,
                )?;
            } else {
                // This param is transfered directly to VE,
                // so deserialize a pointer to it.
                let ptr = &mut buffer[offset + field_offset] as *mut u8 as u64;
                unsafe {
                    *(ptr as *mut u64) = addrs_ve[i+1];
                }
            }
            field_offset += self.calc_data_size(f)?;
        }
        Ok(())
    }
    fn _deserialize_on_host(
        &self,
        ty: &Type,
        buffer: &mut Vec<u8>,
        addr_ve: u64,
        offset: usize,
    ) -> Result<(), WeldError> {
        let ptr = &mut buffer[offset] as *mut u8 as u64;
        match *ty {
            Scalar(_) | Simd(_) => Ok(()),
            Struct(ref fields) => {
                let offset = unsafe { *(ptr as *mut u64) } as usize;
                unsafe {
                    *(ptr as *mut u64) = addr_ve + offset as u64;
                }
                let mut field_offset = 0;
                for f in fields.iter() {
                    self._deserialize_on_host(
                        f,
                        buffer,
                        addr_ve,
                        offset + field_offset,
                    )?;
                    field_offset += self.calc_data_size(f)?;
                }
                Ok(())
            }
            Vector(ref elem) => {
                unsafe {
                    let offset = *(ptr as *mut u64);
                    *(ptr as *mut u64) = addr_ve + offset;
                }
                /*
                match **elem {
                    Struct(ref fields) => {}
                    Vector(ref elem) => {}
                    Dict(_, _) => {}
                }
                */
                Ok(())
            }
            Dict(_, _) => {
                weld_err!("Unsupported dict type {}", ty)
            }
            Builder(_, _) => {
                weld_err!("Unsupported builder type {}", ty)
            }
            _ => {
                weld_err!("Unsupported type {}", ty)
            }
        }
    }

    fn should_serialize_data(&self, ty: &Type, data: u64)
        -> Result<bool, WeldError> {
        match *ty {
            // Should not serialize if the data is a vector of scalar or simd
            // and its size is more than or equal to SERIALIZE_THRESHOLD.
            Vector(ref elem) => {
                match **elem {
                    Scalar(_) | Simd(_) =>
                        Ok(self.calc_contents_size(ty, data)? <
                           SERIALIZE_THRESHOLD),
                    _ => Ok(true),
                }
            }
            _ => Ok(true),
        }
    }

    fn convert_data(&self, field_tys: &[Type], field_data: u64)
        -> Result<(Vec<u8>, usize), WeldError> {
        // We have a list of parameters of top level funtion and
        // a pointer of them.
        //
        // For example,
        //   field_tys:
        //       [Vec[f64], Vec[i64]]
        //   field_data:
        //       0x00007F5B1B90C010 (addr of f64 array)
        //       0x0000000000989680 (length of f64 array (=10000000))
        //       0x00007F5B12074010 (addr of i64 array)
        //       0x0000000000989680 (length of i64 array (=10000000))
        //
        // We serialized and packed data.  Data may be serialized or be not
        // serialized depend on the purpose.  For example, we may want to
        // translate data without serialization if the data is too big and
        // doesn't contain any pointers.
        //
        // Example of serialized data
        //   result:
        //       0x0000000000000028 (serialized addr to real data)
        //       0x0000000000000000 (area for WeldInputArgs' nworkers)
        //       0x0000000000000000 (area for WeldInputArgs' mem_limit)
        //       0x0000000000000000 (area for WeldInputArgs' run)
        //       0x0000000000000101 (serialized flag for each parameter)
        //       0x0000000000000048 (serialized addr to f64 array)
        //       0x0000000000989680 (length of f64 array (=10000000))
        //       0x0000000004C4B448 (serialized addr of i64 array)
        //       0x0000000000989680 (length of i64 array (=10000000))
        //       ....               (seralized f64 array and i64 array)
        //
        // Example of not serialized data
        //   result:
        //       0x0000000000000028 (serialized addr to real data)
        //       0x0000000000000000 (area for WeldInputArgs' nworkers)
        //       0x0000000000000000 (area for WeldInputArgs' mem_limit)
        //       0x0000000000000000 (area for WeldInputArgs' run)
        //       0x0000000000000000 (serialized flag for each parameter)
        //       0x00007F5B1B90C010 (addr of f64 array on host)
        //       0x0000000000989680 (length of f64 array (=10000000))
        //       0x00007F5B12074010 (addr of i64 array on host)
        //       0x0000000000989680 (length of i64 array (=10000000))

        use std::mem::size_of;
        use crate::codegen::WeldInputArgs;
        let input_size = size_of::<WeldInputArgs>() as usize;
        let num_params = field_tys.len();
        let flag_size = (num_params + 7) / 8 * 8;

        let mut serialized_data_size = input_size + flag_size;
        let mut field_offset = 0;
        let mut flags = Vec::<i8>::new();

        for f in field_tys.iter() {
            let faddr = field_data + field_offset as u64;

            // Serialize small data or data containing pointers in it.
            if self.should_serialize_data(f, faddr)? {
                serialized_data_size += self.calc_size(f, faddr)?;
                flags.push(1);
            } else {
                serialized_data_size += self.calc_data_size(f)?;
                flags.push(0);
            }
            field_offset += self.calc_data_size(f)?;
        }

        // Allocate buffer
        let mut buffer = Vec::<u8>::with_capacity(serialized_data_size);
        unsafe { buffer.set_len(serialized_data_size) };
        let offset = input_size + flag_size;
        let mut contents_offset = input_size + flag_size + field_offset;
        let mut flag_offset = input_size;
        let mut field_offset = 0;

        for (f, flag) in field_tys.iter().zip(flags.iter()) {

            let faddr = field_data + field_offset as u64;
            let mut contents_size = 0;

            // Convert contents first
            if *flag == 1 {
                contents_size = self._copy_contents(
                    f,
                    &mut buffer[contents_offset] as *mut u8 as u64,
                    faddr,
                )?;
            } else {
            }

            // Convert data
            if *flag == 1 {
                self._copy_serialized_data(
                    f,
                    &mut buffer[offset + field_offset] as *mut u8 as u64,
                    contents_offset,
                    faddr,
                )?;
                buffer[flag_offset] = 1;
            } else {
                self._copy_data(
                    f,
                    &mut buffer[offset + field_offset] as *mut u8 as u64,
                    faddr,
                )?;
                buffer[flag_offset] = 0;
            }

            contents_offset += contents_size;
            flag_offset += 1;
            field_offset += self.calc_data_size(f)?;
        }
        Ok((buffer, serialized_data_size))
    }

    fn _copy_contents(&self, ty: &Type, dest: u64, src: u64)
        -> Result<usize, WeldError> {
        match *ty {
            // Value of Sclar and Simd will be copied by _copy_data function.
            Scalar(_) | Simd(_) => Ok(0),
            Struct(_) => {
                weld_err!("Unsupported struct type {}", ty)
            }
            Vector(ref elem) => {
                self._copy_contents_of_vector(&*elem, dest, src)
            }
            Dict(_, _) => {
                weld_err!("Unsupported dict type {}", ty)
            }
            Builder(_, _) => {
                weld_err!("Unsupported builder type {}", ty)
            }
            _ => {
                weld_err!("Unsupported type {}", ty)
            }
        }
    }
    fn _copy_contents_of_vector(&self, elem_ty: &Type, dest: u64, src: u64)
        -> Result<usize, WeldError> {
        //      WeldVec
        //  0:  intptr_t  data
        //  8:  u64       length

        let elem_data = unsafe { *(src as *const u64) };
        let length = unsafe { *((src + 8) as *const u64) } as usize;
        let elem_size = self.calc_data_size(elem_ty)?;
        let size = elem_size * length;

        match *elem_ty {
            Scalar(_) | Simd(_) => {
                unsafe {
                    copy_nonoverlapping(
                        elem_data as *const u8,
                        dest as *mut u8,
                        size as usize,
                    );
                }
                Ok(size)
            }
            // FIXME: doesn't support Vec[Vec[i8]] or Vec[Struct[...]].
            _ => {
                weld_err!("Unsupported vector element type {}", *elem_ty)
            }
        }
    }

    fn _copy_data(&self, ty: &Type, dest: u64, src: u64)
        -> Result<(), WeldError> {
        unsafe {
            copy_nonoverlapping(
                src as *const u8,
                dest as *mut u8,
                self.calc_data_size(ty)?,
            );
        }
        Ok(())
    }

    fn _copy_serialized_data(
        &self,
        ty: &Type,
        dest: u64,
        contents_offset: usize,
        src: u64,
    ) -> Result<(), WeldError> {
        match *ty {
            Scalar(_) | Simd(_) => {
                self._copy_data(ty, dest, src)
            }
            Struct(_) => {
                weld_err!("Unsupported struct type {}", ty)
            }
            Vector(_) => {
                unsafe {
                    let length = *((src + 8) as *const u64);
                    *(dest as *mut u64) = contents_offset as u64;
                    *((dest + 8) as *mut u64) = length;
                }
                Ok(())
            }
            Dict(_, _) => {
                weld_err!("Unsupported dict type {}", ty)
            }
            Builder(_, _) => {
                weld_err!("Unsupported builder type {}", ty)
            }
            _ => {
                weld_err!("Unsupported type {}", ty)
            }
        }
    }
    fn calc_size(&self, ty: &Type, data: u64) -> Result<usize, WeldError> {
        Ok(self.calc_data_size(ty)? +
           self.calc_contents_size(ty, data)?)
    }
    // Calculate the size of data contents.
    //
    // For example, a vector has its data area and contents area like below.
    // We calculate the size of each of them by calc_data_size and
    // calc_contents_size functions.
    //
    //   Vector Data:
    //     pointer to contents
    //     length of contents
    //
    //   Vector Contents:
    //     arrray of elemnts
    fn calc_contents_size(&self, ty: &Type, data: u64)
        -> Result<usize, WeldError> {
        match *ty {
            Struct(ref fields) => {
                let mut sz = 0;
                let mut ptr = unsafe {
                    *(data as *const u64)
                };
                for f in fields.iter() {
                    let data_size = self.calc_data_size(f)?;
                    let contents_size = self.calc_contents_size(f, ptr)?;
                    // println!("field {:?} data size = {}, contents size = {}", f, data_size, contents_size);
                    sz += data_size + contents_size;
                    ptr += data_size as u64;
                }
                Ok(sz)
            }
            Vector(ref elem) => {
                //      WeldVec
                //  0:  intptr_t  data
                //  8:  u64       length
                let elem_data = unsafe {
                    *(data as *const u64)
                };
                let elem_size = self.calc_data_size(elem)?;
                let length = unsafe {
                    *((data + 8) as *const u64)
                } as usize;
                // println!("Vector({:?}) elem size = {}, length = {}", elem, elem_size, length);
                let mut sz = elem_size * length;
                match **elem {
                    Scalar(_) | Simd(_) => Ok(sz),
                    Struct(_) | Vector(_) => {
                        for i in 0..length {
                            let ptr = elem_data + (i * elem_size) as u64;
                            let elem_size = self.calc_contents_size(elem, ptr)?;
                            let pad_elem_size = (elem_size + 7)/8*8;
                            sz += pad_elem_size;
                            // println!("elem[{}] size = {}", i, elem_size);
                        }
                        // println!("total size = {}", sz);
                        Ok(sz)
                    }
                    /*
                    Dict(ref key, ref value) => 0,
                    Builder(ref bk, _) => 0,
                    */
                    _ => Ok(0),
                }
            }
            Dict(_, _) => {
                weld_err!("Unsupported dict type {}", ty)
            }
            Builder(_, _) => {
                weld_err!("Unsupported builder type {}", ty)
            }
            _ => Ok(0),
        }
    }
    // Calculate the size of outermost data structure.
    // In order to calculate the size of whole data, use both
    // this function and calc_contents_size().
    fn calc_data_size(&self, ty: &Type) -> Result<usize, WeldError> {
        match *ty {
            Scalar(kind) => Ok(scalar_kind_size(kind)),
            Simd(kind) => Ok(simd_size(&Scalar(kind)) * scalar_kind_size(kind)),

            // Struct(fields) is a pointer to an array of each field
            Struct(_) => Ok(8),
            // Vector(elem) is following 16 bytes data structure.
            //  0:  intptr_t  data
            //  8:  u64       length
            Vector(_) => Ok(16),
            Dict(_, _) => {
                weld_err!("Unsupported dict type {}", ty)
            }
            Builder(_, _) => {
                weld_err!("Unsupported builder type {}", ty)
            }
            _ => {
                weld_err!("Unsupported type {}", ty)
            }
        }
    }
    // Calculate of the size of Struct's field data size.
    fn calc_struct_field_size(&self, field_tys: &[Type])
        -> Result<usize, WeldError> {
        let mut sz = 0;
        for f in field_tys.iter() {
            let data_size = self.calc_data_size(f)?;
            sz += data_size;
        }
        Ok(sz)
    }
    fn convert_top_params(&self, ty: &Type, data: u64,
                          vh_addr: u64, ve_addr: u64) -> Result<(), WeldError> {
        match *ty {
            Struct(_) => {
                let mut dummy: u64 = 0;
                self.convert_param(ty, data, &mut dummy as *mut u64 as u64,
                                  vh_addr, ve_addr)
            }
            _ => {
                weld_err!("Invalid type {} for the type of top params", ty)
            }
        }
    }
    fn convert_param(&self, ty: &Type, data: u64, vh_addr: u64,
                     vh_next_addr: u64, ve_next_addr: u64) ->
                     Result<(), WeldError> {
        match *ty {
//            Scalar(kind) => scalar_kind_size(kind),
//            Simd(kind) => simd_size(&Scalar(kind)) * scalar_kind_size(kind),
            Struct(ref fields) => {
                let deref_size = self.calc_struct_field_size(&*fields)?;
                let mut data_ptr = unsafe {
                    *(data as *const u64)
                };
                unsafe {
                    *(vh_addr as *mut u64) = ve_next_addr;
                }
                let mut vh_elem_ptr = vh_next_addr;
                let mut vh_next_ptr = vh_next_addr + deref_size as u64;
                let mut ve_next_ptr = ve_next_addr + deref_size as u64;
                for f in fields.iter() {
                    let field_data_size = self.calc_data_size(f)?;
                    let field_deref_size = self.calc_contents_size(f, data_ptr)?;
                    // println!("field {:?} data size = {}, data deref size = {}", f, field_data_size, field_deref_size);
                    self.convert_param(f, data_ptr, vh_elem_ptr,
                                       vh_next_ptr, ve_next_ptr)?;
                    data_ptr += field_data_size as u64;
                    vh_elem_ptr += field_data_size as u64;
                    vh_next_ptr += field_deref_size as u64;
                    ve_next_ptr += field_deref_size as u64;
                }
                Ok(())
            }
            Vector(ref elem) => {
                //      WeldVec
                //  0:  intptr_t  data
                //  8:  u64       length
                let elem_data = unsafe {
                    *(data as *const u64)
                };
                let elem_size = self.calc_data_size(elem)?;
                let length = unsafe {
                    *((data + 8) as *const u64)
                } as usize;
                // println!("Vector({:?}) elem size = {}, length = {}", elem, elem_size, length);
                let sz = elem_size * length;
                unsafe {
                    *(vh_addr as *mut u64) = ve_next_addr;
                    *((vh_addr + 8) as *mut u64) = length as u64;
                }
                let vh_elem_ptr = vh_next_addr;
                let mut vh_next_ptr = vh_next_addr + sz as u64;
                let mut ve_next_ptr = ve_next_addr + sz as u64;
                match **elem {
                    Scalar(_) | Simd(_) => {
                        unsafe {
                            copy_nonoverlapping(elem_data as *const u8,
                                                vh_elem_ptr as *mut u8,
                                                sz as usize);
                        }
                        Ok(())
                    }
                    Struct(_) | Vector(_) => {
                        for i in 0..length {
                            let ptr = elem_data + (i * elem_size) as u64;
                            let elem_ptr = vh_elem_ptr + (i * elem_size) as u64;
                            self.convert_param(elem, ptr, elem_ptr,
                                               vh_next_ptr, ve_next_ptr)?;
                            let elem_size = self.calc_contents_size(elem, ptr)?;
                            let pad_elem_size = (elem_size + 7)/8*8;
                            vh_next_ptr += pad_elem_size as u64;
                            ve_next_ptr += pad_elem_size as u64;
                            // println!("elem[{}] size = {}", i, elem_size);
                        }
                        // println!("total size = {}", sz);
                        Ok(())
                    }
                    /*
                    Dict(ref key, ref value) => 0,
                    Builder(ref bk, _) => 0,
                    */
                    _ => {
                        weld_err!("Unsupported vector element type {} in convert_param", **elem)
                    }
                }
            }
            Dict(_, _) => {
                weld_err!("Unsupported dict type {} in convert_param", ty)
            }
            Builder(_, _) => {
                weld_err!("Unsupported builder type {} in convert_param", ty)
            }
            _ => {
                weld_err!("Unsupported type {} in convert_param", ty)
            }
        }
    }
    fn convert_results(&self, ty: &Type, ve_addr: u64, vh_addr: u64,
                       proc_handle: VeoProcHandleRef) ->
                       Result<(), WeldError> {
        match *ty {
            Function(_, ref ret_ty) => {
                self.convert_result(&ret_ty, ve_addr, vh_addr, proc_handle)?;
                Ok(())
            }
            _ => {
                self.convert_result(&*ty, ve_addr, vh_addr, proc_handle)?;
                Ok(())
                // weld_err!("Invalid type {} for the type of return type", ty)
            }
        }
    }
    fn convert_result_elements(&self, ty: &Type, ve_addr: u64, vh_addr: u64,
                               proc_handle: VeoProcHandleRef) ->
                               Result<(), WeldError> {
        match *ty {
            Struct(ref fields) => {
                let mut vh_ptr = vh_addr;
                for f in fields.iter() {
                    let field_data_size = self.calc_data_size(f)?;
                    match f {
                        Scalar(_) | Simd(_) => {
                            // Data is already copied, so nothing to do
                        }
                        Struct(_) => {
                            let ve_deref_addr = unsafe {
                                *(vh_ptr as *const u64)
                            };
                            self.convert_result(f, ve_deref_addr, vh_ptr,
                                                proc_handle)?;
                        }
                        Vector(_) => {
                            let ve_deref_addr = unsafe {
                                *(vh_ptr as *const u64)
                            };
                            self.convert_result_elements(
                                f, ve_deref_addr, vh_ptr, proc_handle)?;
                        }
                        Dict(_, _) => {
                            return weld_err!("Unsupported dict field type {} in convert_result_elements", ty);
                        }
                        Builder(_, _) => {
                            return weld_err!("Unsupported builder field type {} in convert_result_elements", ty);
                        }
                        _ => {
                            return weld_err!("Unsupported struct field type {} in convert_result_elements", f);
                        }
                    }
                    vh_ptr += field_data_size as u64;
                }
                Ok(())
            }
            Vector(ref elem) => {
                //      WeldVec
                //  0:  intptr_t  data
                //  8:  u64       length
                let elem_size = self.calc_data_size(elem)?;
                let length = unsafe {
                    *((vh_addr + 8) as *const u64)
                } as usize;
                let sz = elem_size * length;
                let ptr = unsafe { alloc(sz as usize) };
                unsafe {
                    *(vh_addr as *mut u64) = ptr as u64;
                }
                unsafe {
                    if veo_read_mem(proc_handle, ptr as *mut c_void,
                                    ve_addr, sz) != 0 {
                        return weld_err!("cannot read veo memory");
                    }
                };
                let elem_data = ptr as u64;
                match **elem {
                    Scalar(_) | Simd(_) => {
                        // Data is already copied, so nothing to do
                        Ok(())
                    }
                    Struct(_) => {
                        for i in 0..length {
                            let ptr = elem_data + (i * elem_size) as u64;
                            let ve_deref_addr = unsafe {
                                *(ptr as *const u64)
                            };
                            self.convert_result(
                                &**elem, ve_deref_addr, ptr, proc_handle)?;
                        }
                        Ok(())
                    }
                    Vector(_) => {
                        for i in 0..length {
                            let ptr = elem_data + (i * elem_size) as u64;
                            let ve_deref_addr = unsafe {
                                *(ptr as *const u64)
                            };
                            self.convert_result_elements(
                                &**elem, ve_deref_addr, ptr, proc_handle)?;
                        }
                        Ok(())
                    }
                    Dict(_, _) => {
                        weld_err!("Unsupported dict element type {} in convert_result_elements", ty)
                    }
                    Builder(_, _) => {
                        weld_err!("Unsupported builder element type {} in convert_result_elements", ty)
                    }
                    _ => {
                        weld_err!("Unsupported vector element type {} in convert_result_elements", **elem)
                    }
                }
            }
            Dict(_, _) => {
                weld_err!("Unsupported dict type {} in convert_result_elements", ty)
            }
            Builder(_, _) => {
                weld_err!("Unsupported builder type {} in convert_result_elements", ty)
            }
            _ => {
                weld_err!("Unsupported type {} in convert_result", ty)
            }
        }
    }
    fn convert_result(&self, ty: &Type, ve_addr: u64, vh_addr: u64,
                      proc_handle: VeoProcHandleRef) ->
                      Result<(), WeldError> {
        match *ty {
            Scalar(kind) => {
                let ptr = unsafe { alloc(scalar_kind_size(kind) as usize) };
                unsafe {
                    *(vh_addr as *mut u64) = ptr as u64;
                }
                unsafe {
                    if veo_read_mem(proc_handle, ptr as *mut c_void,
                                    ve_addr, scalar_kind_size(kind)) != 0 {
                        return weld_err!("cannot read veo memory");
                    }
                };
                Ok(())
            }
//            Simd(kind) => simd_size(&Scalar(kind)) * scalar_kind_size(kind),
            Struct(ref fields) => {
                let deref_size = self.calc_struct_field_size(&*fields)?;
                let ptr = unsafe { alloc(deref_size as usize) };
                unsafe {
                    *(vh_addr as *mut u64) = ptr as u64;
                }
                unsafe {
                    if veo_read_mem(proc_handle, ptr as *mut c_void,
                                    ve_addr, deref_size) != 0 {
                        return weld_err!("cannot read veo memory");
                    }
                };
                self.convert_result_elements(
                    ty, ve_addr, ptr as u64, proc_handle)
            }
            Vector(_) => {
                let data_size = self.calc_data_size(ty)?;
                let ptr = unsafe { alloc(data_size as usize) };
                unsafe {
                    *(vh_addr as *mut u64) = ptr as u64;
                }
                unsafe {
                    if veo_read_mem(proc_handle, ptr as *mut c_void,
                                    ve_addr, data_size) != 0 {
                        return weld_err!("cannot read veo memory");
                    }
                };
                let new_ve_addr = unsafe { *(ptr as *const u64) };
                self.convert_result_elements(
                    ty, new_ve_addr, ptr as u64, proc_handle)
            }
            Dict(_, _) => {
                weld_err!("Unsupported dict type {} in convert_result", ty)
            }
            Builder(_, _) => {
                weld_err!("Unsupported builder type {} in convert_result", ty)
            }
            _ => {
                weld_err!("Unsupported type {} in convert_result", ty)
            }
        }
    }
}

fn scalar_kind_size(k: ScalarKind) -> usize {
    use crate::ast::ScalarKind::*;
    match k {
        Bool => 1,
        I8 => 1,
        I16 => 2,
        I32 => 4,
        I64 => 8,
        U8 => 1,
        U16 => 2,
        U32 => 4,
        U64 => 8,
        F32 => 4,
        F64 => 8,
    }
}

fn simd_size(_: &Type) -> usize {
    use super::LLVM_VECTOR_WIDTH;
    LLVM_VECTOR_WIDTH as usize
}

unsafe fn alloc(len: usize) -> *mut u8 {
    let mut vec = Vec::<u8>::with_capacity(len);
    vec.set_len(len);
    Box::into_raw(vec.into_boxed_slice()) as *mut u8
}

unsafe fn free(raw: *mut u8, len : usize) {
    use std::slice;
    let s = slice::from_raw_parts_mut(raw, len);
    let _ = Box::from_raw(s);
}

// should defined elsewhere
unsafe fn copy_nonoverlapping<T: Copy>(
    src: *const T,
    dest: *mut T,
    count: usize,
) {
    // currently not to use parallel copy because it is very slow.
    // The reason may be host memory is not pin-down
    let num_thread = 1;
    if num_thread > 1 && count >= 1024 {  // sz is rough threshold
        copy_nonoverlapping_parallel(src, dest, count, num_thread);
    } else {
        ptr::copy_nonoverlapping(src, dest, count);
    }
}

unsafe fn copy_nonoverlapping_parallel<T: Copy>(src: *const T, dst: *mut T, count: usize, nthread: usize) {
    use std::thread;

    assert!(nthread > 0);

    let base = count / nthread;
    let xth = count % nthread;

    let mut childs = Vec::new();
    for th in 0..nthread {
        let sz = if th < xth { base + 1 } else { base };
        let off = if th < xth { base * th + th} else { base * th + xth};
        let _src = src as u64;
        let _dst = dst as u64;

        let ch = thread::spawn(move || {
            for i in 0..sz {
                *(_dst as *mut T).offset((off + i) as isize) = *(_src as *const T).offset((off + i) as isize);
            }
        });
        childs.push(ch);
    }
    for ch in childs.into_iter() {
        let _res = ch.join();
    }
}

unsafe fn dump_data(addr: *const u8, len: usize) {
    // Dump memory data
    for i in 0..len as isize {
        if i % 16 == 0 {
            println!("");
        }
        print!("{:02x} ", *addr.offset(i));
    }
    println!("");
}
