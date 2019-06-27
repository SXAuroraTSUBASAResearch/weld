//!
//! This module manages compiling with optimization the generated C module
//! to the machine code.

use libc;
use time;

use std::ffi::CString;
use std::ptr;
use std::sync::{Once, ONCE_INIT};

use libc::{uint64_t, c_char, c_void};

use self::time::PreciseTime;

use crate::WeldError;
use crate::ast::Type;
use crate::ast::Type::*;
use crate::ast::ScalarKind;
use crate::util::stats::RunStats;
use crate::util::veoffload::*;
use crate::util::offload_ve::*;

use crate::codegen::Runnable;

static ONCE: Once = ONCE_INIT;
static mut INITIALIZE_FAILED: bool = false;

/// The callable function type.
type I64Func = extern "C" fn(i64) -> i64;

// Use CompiledModule defined in compile.rs
use crate::codegen::c::compile::*;
use crate::util::veoffload::VeoCommandState::VeoCommandOk;

// The codegen interface requires that modules implement this trait. This allows supporting
// multiple backends via dynamic dispatch.
impl Runnable for CompiledModule {
    fn run(&self, arg: i64, stats: &mut RunStats) -> Result<i64, WeldError> {
        use crate::runtime::WeldRuntimeErrno;
        unsafe {
            let veo_ptr = get_global_veo_ptr();
            let start = PreciseTime::now();
            while !(*veo_ptr).ready {}
            let end = PreciseTime::now();
            stats.run_times.push(("wait initialize veo".to_string(), start.to(end)));

            /*
            let start = PreciseTime::now();
            let node = get_ve_node_number();
            // Instanciate VE proc.
            // FIXME: call this only once.
            let proc_handle = veo_proc_create(node);
            let end = PreciseTime::now();
            stats.run_times.push(("veo_proc_create".to_string(), start.to(end)));
            if proc_handle.is_null() {
                return weld_err!("cannot create proc");
            }
            let start = PreciseTime::now();
            let ctx = veo_context_open(proc_handle);
            let end = PreciseTime::now();
            stats.run_times.push(("veo_context_open".to_string(), start.to(end)));
            if ctx.is_null() {
                return weld_err!("cannot create veo context");
            }
            */

            /*
            // Load generated shared-object by Weld on VE.
            let start = PreciseTime::now();
            let libname = format!("./{}", self.filename);
            let libcname = CString::new(libname.clone()).unwrap();
            let libcname_raw = libcname.into_raw() as *const c_char;
            let handle = veo_load_library(proc_handle, libcname_raw);
            let end = PreciseTime::now();
            stats.run_times.push(("veo_load_library".to_string(), start.to(end)));
            if handle == 0 {
                return weld_err!("cannot load library {}", libname);
            }
            */
            let start = PreciseTime::now();
            let libname = format!("./{}", self.filename);
            let libhdl_run = (*veo_ptr).load_library(libname)?;
            let end = PreciseTime::now();
            stats.run_times.push(("veo_load_library".to_string(), start.to(end)));

            /*
            // Retreieve entry function address on VE.
            let args = veo_args_alloc();
            let start = PreciseTime::now();
            let funname = "run";
            let funcname = CString::new(funname).unwrap();
            let funcname_raw = funcname.into_raw() as *const c_char;
            let fun = veo_get_sym(proc_handle, handle, funcname_raw);
            let end = PreciseTime::now();
            stats.run_times.push(("veo_get_sym".to_string(), start.to(end)));
            if fun == 0 {
                return weld_err!("cannot find function {}", funname);
            }
            // println!("params is {:?}", self.params);
            // println!("ret_ty is {:?}", self.ret_ty);
            // println!("encoded_params is {}", self.encoded_params);
            */

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
                // (Usually, it is created by caller)
                context = run as *mut WeldRuntimeContext;
            }
            let end = PreciseTime::now();
            stats.run_times.push(("from_raw".to_string(), start.to(end)));

            // Check parameters
            // println!("parameters {:?}", self.params);
            // self.check_data(&self.params, data_ptr);

            // First, calculate the size of memory needed to be allocated by
            // the size of WeldInputArgs and the given data.
            let start = PreciseTime::now();
            use std::mem::size_of;
            let input_size = size_of::<WeldInputArgs>() as u64;
            let data_size = self.calc_size(&self.params, data_ptr)?;
            let end = PreciseTime::now();
            stats.run_times.push(("calc_size".to_string(), start.to(end)));

            // The calc_size returns the size of whole data including a pointer
            // the the struct of parameters, so reduce 8.
            let sz = input_size + data_size - 8;

            /*
            // Allocate VE memory.
            let mut addr_ve: uint64_t = 0;
            let start = PreciseTime::now();
            let err = veo_alloc_mem(proc_handle, &mut addr_ve, sz);
            let end = PreciseTime::now();
            stats.run_times.push(("veo_alloc_mem".to_string(), start.to(end)));
            if err != 0 {
                return weld_err!("cannot allocate veo memory");
            }
            // println!("allocated size = {}", sz);
            */
            let start = PreciseTime::now();
            let mut addr_ve: uint64_t = (*veo_ptr).alloc_mem(sz)?;
            let end = PreciseTime::now();
            stats.run_times.push(("veo_alloc_mem".to_string(), start.to(end)));

            // prepare arguments and whole input data
            let start = PreciseTime::now();
            let input_generated = WeldInputArgs {
                input: (addr_ve + input_size) as i64,
                nworkers,
                mem_limit,
                run: 0,
            };
            let mut buffer = Vec::<u8>::with_capacity(sz as usize);
            buffer.set_len(sz as usize);
            ptr::copy_nonoverlapping(&input_generated as *const WeldInputArgs
                                     as *const u8, &mut buffer[0],
                                     input_size as usize);
            let addr_vh = &buffer[0] as *const u8 as u64;
            // let input_p = addr_vh as *const WeldInputArgs;
            // println!("nworkers {}", (*input_p).nworkers);
            // println!("mem_limit {}", (*input_p).mem_limit);
            // println!("run {}", (*input_p).run);

            self.convert_top_params(
                &self.params,
                data_ptr,
                addr_vh + input_size,
                addr_ve + input_size,
            )?;
            let end = PreciseTime::now();
            stats.run_times.push(("convert_top_params".to_string(), start.to(end)));

            /*
            let start = PreciseTime::now();
            let err = veo_write_mem(proc_handle, addr_ve + 0,
                                    &buffer[0] as *const u8 as *const c_void,
                                    sz);
            let end = PreciseTime::now();
            stats.run_times.push(("veo_write_mem".to_string(), start.to(end)));
            if err != 0 {
                return weld_err!("cannot write veo memory");
            }
            */
            let start = PreciseTime::now();
            (*veo_ptr).write_mem(&buffer[0] as *const u8 as *const c_void, addr_ve + 0, sz);
            let end = PreciseTime::now();
            stats.run_times.push(("veo_write_mem".to_string(), start.to(end)));

            /*
            let start = PreciseTime::now();
            veo_args_clear(args);
            veo_args_set_u64(args, 0, addr_ve);
            let end = PreciseTime::now();
            stats.run_times.push(("prepare arguments".to_string(), start.to(end)));
            */
            let start = PreciseTime::now();
            let args = VEOffload::args_alloc();
            VEOffload::args_set(args, 0, addr_ve);
            let end = PreciseTime::now();
            stats.run_times.push(("prepare arguments".to_string(), start.to(end)));

            /*
            let start = PreciseTime::now();
            let id = veo_call_async(ctx, fun, args);
            // println!("running id {}", id);
            let mut retval_ve_ptr: uint64_t = 0;
            let wait = veo_call_wait_result(ctx, id, &mut retval_ve_ptr);
            let end = PreciseTime::now();
            stats.run_times.push(("call run".to_string(), start.to(end)));
            match wait {
                VeoCommandState::VeoCommandOk => {}
                _ => { return weld_err!("VE causes some errors"); }
            };
            let errno = match wait {
                VeoCommandState::VeoCommandOk => WeldRuntimeErrno::Success,
                _ => WeldRuntimeErrno::Unknown,
            };
            */
            let start = PreciseTime::now();
            let mut retval_ve_ptr: uint64_t = (*veo_ptr).call_and_wait(libhdl_run, "run", args)?;
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
            let output_size = size_of::<WeldOutputArgs>() as u64;
            let ret = weld_runst_malloc(context, output_size as i64)
                as *mut WeldOutputArgs;
            if errno != WeldRuntimeErrno::Unknown {
                /*
                let err = veo_read_mem(proc_handle,
                                       ret as *mut c_void,
                                       retval_ve_ptr,
                                       output_size,
                );
                if err != 0 {
                    return weld_err!("cannot read veo memory");
                }
                */
                (*veo_ptr).read_mem(retval_ve_ptr, ret as *mut c_void, output_size)?;
                // FIXME: need to handle VE's run correctly for latter calls.
                // Overwrite `run` using HOST's run.
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

            /*
            let start = PreciseTime::now();
            veo_args_free(args);
            let err = veo_free_mem(proc_handle, addr_ve);
            if err != 0 {
                return weld_err!("cannot free veo memory");
            }
            let err = veo_context_close(ctx);
            if err != 0 {
                return weld_err!("cannot close veo context");
            }
            let err = veo_proc_destroy(proc_handle);
            if err != 0 {
                return weld_err!("cannot destrocy veo proc");
            }
            let end = PreciseTime::now();
            stats.run_times.push(("free and destroy".to_string(), start.to(end)));
            */
            let start = PreciseTime::now();
            VEOffload::args_free(args);
            (*veo_ptr).free_mem(addr_ve)?;
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
                    self.check_data(f, faddr);
                    faddr += size;
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
                let size = self.calc_data_size(elem)?;
                let mut eaddr = data;
                for i in 0..16 {
                    self.check_data(elem, eaddr);
                    eaddr += size;
                }
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

    fn calc_size(&self, ty: &Type, data: u64) -> Result<u64, WeldError> {
        Ok(self.calc_data_size(ty)? +
           self.calc_deref_size(ty, data)? +
           self.calc_deref_deref_size(ty, data)?)
    }
    // Calculate of the size of twice dereferenced data.
    // For example, a structure has 3 layered data structure.
    // So, use this to calculate the size of 3rd layer data.
    fn calc_deref_deref_size(&self, ty: &Type, data: u64) ->
        Result<u64, WeldError> {
        match *ty {
            Struct(ref fields) => {
                let mut sz = 0;
                let mut ptr = unsafe {
                    *(data as *const u64)
                };
                for f in fields.iter() {
                    let field_data_size = self.calc_data_size(f)?;
                    let field_deref_size = self.calc_deref_size(f, ptr)? +
                        self.calc_deref_deref_size(f, ptr)?;
                    // println!("field {:?} data size = {}, data deref size = {}", f, field_data_size, field_deref_size);
                    sz += field_deref_size;
                    ptr += field_data_size;
                }
                Ok(sz)
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
    // Calculate of the size of once dereferenced data.
    // For example, a vector has 2 layered data structure.
    // So, use this to calculate the size of 2nd layer data.
    fn calc_deref_size(&self, ty: &Type, data: u64) -> Result<u64, WeldError> {
        match *ty {
            Struct(ref fields) => {
                let mut sz = 0;
                for f in fields.iter() {
                    let field_data_size = self.calc_data_size(f)?;
                    sz += field_data_size;
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
                };
                // println!("Vector({:?}) elem size = {}, length = {}", elem, elem_size, length);
                let mut sz = elem_size * length;
                match **elem {
                    Scalar(_) | Simd(_) => Ok(sz),
                    Struct(_) | Vector(_) => {
                        for i in 0..length {
                            let ptr = elem_data + i * elem_size;
                            let elem_size = self.calc_deref_size(elem, ptr)? +
                                self.calc_deref_deref_size(elem, ptr)?;
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
    // this function, calc_deref_size(), and calc_deref_deref_size().
    fn calc_data_size(&self, ty: &Type) -> Result<u64, WeldError> {
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
                let deref_size = self.calc_deref_size(ty, data)?;
                let mut data_ptr = unsafe {
                    *(data as *const u64)
                };
                unsafe {
                    *(vh_addr as *mut u64) = ve_next_addr;
                }
                let mut vh_elem_ptr = vh_next_addr;
                let mut vh_next_ptr = vh_next_addr + deref_size;
                let mut ve_next_ptr = ve_next_addr + deref_size;
                for f in fields.iter() {
                    let field_data_size = self.calc_data_size(f)?;
                    let field_deref_size = self.calc_deref_size(f, data_ptr)? +
                        self.calc_deref_deref_size(f, data_ptr)?;
                    // println!("field {:?} data size = {}, data deref size = {}", f, field_data_size, field_deref_size);
                    self.convert_param(f, data_ptr, vh_elem_ptr,
                                       vh_next_ptr, ve_next_ptr)?;
                    data_ptr += field_data_size;
                    vh_elem_ptr += field_data_size;
                    vh_next_ptr += field_deref_size;
                    ve_next_ptr += field_deref_size;
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
                };
                // println!("Vector({:?}) elem size = {}, length = {}", elem, elem_size, length);
                let sz = elem_size * length;
                unsafe {
                    *(vh_addr as *mut u64) = ve_next_addr;
                    *((vh_addr + 8) as *mut u64) = length;
                }
                let vh_elem_ptr = vh_next_addr;
                let mut vh_next_ptr = vh_next_addr + sz;
                let mut ve_next_ptr = ve_next_addr + sz;
                match **elem {
                    Scalar(_) | Simd(_) => {
                        unsafe {
                            // currently not to use parallel copy because it is very slow. The reason may be host memory is not pin-down
                            let num_thread = 1;
                            if num_thread > 1 && sz >= 1024 {  // sz is rough threshold
                                copy_nonoverlapping_parallel(elem_data as *const u8,
                                                             vh_elem_ptr as *mut u8,
                                                             sz as usize,
                                                             num_thread);
                            } else {
                                ptr::copy_nonoverlapping(elem_data as *const u8,
                                                         vh_elem_ptr as *mut u8,
                                                         sz as usize);
                            }
                        }
                        Ok(())
                    }
                    Struct(_) | Vector(_) => {
                        for i in 0..length {
                            let ptr = elem_data + i * elem_size;
                            let elem_ptr = vh_elem_ptr + i * elem_size;
                            self.convert_param(elem, ptr, elem_ptr,
                                               vh_next_ptr, ve_next_ptr)?;
                            let elem_size = self.calc_deref_size(elem, ptr)? +
                                self.calc_deref_deref_size(elem, ptr)?;
                            let pad_elem_size = (elem_size + 7)/8*8;
                            vh_next_ptr += pad_elem_size;
                            ve_next_ptr += pad_elem_size;
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
                    vh_ptr += field_data_size;
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
                };
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
                            let ptr = elem_data + i * elem_size;
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
                            let ptr = elem_data + i * elem_size;
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
            Struct(_) => {
                let deref_size = self.calc_deref_size(ty, 0)?;
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

fn scalar_kind_size(k: ScalarKind) -> u64 {
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

fn simd_size(_: &Type) -> u64 {
    use super::LLVM_VECTOR_WIDTH;
    LLVM_VECTOR_WIDTH.into()
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
        let res = ch.join();
    }
}
