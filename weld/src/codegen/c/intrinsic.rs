//! Defines intrinsics in the LLVM IR.
//!
//! An intrinsic is any function that is defined but not generated in the current module. This
//! module provides utilities for calling some default intrinsics, as well as a utility for adding
//! an calling new ones.
//!
//! For documentation on the default intrinsics, see `weld::runtime::strt`.

use fnv;
use libc;
use llvm_sys;
use code_builder::CodeBuilder;

use fnv::FnvHashMap;

use libc::c_char;

use crate::ast::ScalarKind;
use crate::error::*;

use std::ffi::CString;

use super::llvm_exts::*;

use super::CodeGenExt;
use super::LLVM_VECTOR_WIDTH;

use self::llvm_sys::core::*;
use self::llvm_sys::prelude::*;

use crate::runtime::ffi;
use libc::c_void;

use crate::codegen::c::CContextRef;

/// A single intrinsic.
#[derive(Debug, Clone)]
pub enum Intrinsic {
    Builtin(LLVMValueRef),
    FunctionPointer(LLVMValueRef, *mut c_void),
}

impl Intrinsic {
    /// Returns the LLVM value for the intrinsic.
    fn value(&self) -> LLVMValueRef {
        match *self {
            Intrinsic::Builtin(val) | Intrinsic::FunctionPointer(val, _) => val,
        }
    }
}

/// A mapping from a function name to its function pointer.
pub type Mapping = (CString, *mut c_void);

/// Intrinsics defined in the code generator.
///
/// An intrinsic is any function that appears without a definition in the generated module. Code
/// generators must ensure that instrinics are properly linked upon compilation.
pub struct Intrinsics {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    ccontext: CContextRef,
    intrinsics: FnvHashMap<String, Intrinsic>,
}

impl CodeGenExt for Intrinsics {
    fn context(&self) -> LLVMContextRef {
        self.context
    }

    fn module(&self) -> LLVMModuleRef {
        self.module
    }

    fn ccontext(&self) -> CContextRef {
        self.ccontext
    }
}

impl Intrinsics {
    /// Returns value to function pointer mappings for non-builtin intrinsics.
    ///
    /// Builtins are filtered out of this list.
    pub fn mappings(&self) -> Vec<Mapping> {
        let mut mappings = vec![];
        for (name, entry) in self.intrinsics.iter() {
            if let Intrinsic::FunctionPointer(_, ptr) = *entry {
                mappings.push((CString::new(name.as_str()).unwrap(), ptr))
            }
        }
        mappings
    }

    pub unsafe fn defaults(context: LLVMContextRef, module: LLVMModuleRef,
                           ccontext: CContextRef) -> Intrinsics {
        let mut intrinsics = Intrinsics {
            context,
            module,
            ccontext,
            intrinsics: FnvHashMap::default(),
        };

        intrinsics.populate_defaults();
        intrinsics
    }

    /// Returns a string name for a numeric type's LLVM intrinsic.
    pub fn llvm_numeric<T: AsRef<str>>(name: T, kind: ScalarKind, simd: bool) -> String {
        use crate::ast::ScalarKind::*;
        let mut result = format!("llvm.{}.", name.as_ref());
        if simd {
            result.push_str(&format!("v{}", LLVM_VECTOR_WIDTH));
        }

        result.push_str(match kind {
            Bool => "i1",
            I8 => "i32",
            I16 => "i16",
            I32 => "i32",
            I64 => "i64",
            U8 => "i32",
            U16 => "i16",
            U32 => "i32",
            U64 => "i64",
            F32 => "f32",
            F64 => "f64",
        });
        result
    }

    /// Get the intrinsic function value with the given name.
    pub fn get<T: AsRef<str>>(&self, key: T) -> Option<LLVMValueRef> {
        self.intrinsics.get(key.as_ref()).map(|r| r.value())
    }

    /// Add a new intrinsic function with the given name, return type, and argument types.
    ///
    /// Returns true if the function was added or false if it was already registered. The intrinsic
    /// must be builtin or linked.
    pub unsafe fn add<T: AsRef<str>>(
        &mut self,
        name: T,
        c_name: &str,
        ret_ty: LLVMTypeRef,
        c_ret_ty: &str,
        arg_tys: &mut [LLVMTypeRef],
        c_arg_tys: &[&str],
    ) -> bool {
        if !self.intrinsics.contains_key(name.as_ref()) {
            // for C
            let args_line = self.c_args(c_arg_tys);
            (*self.ccontext()).prelude_code.add(format!(
                "extern {} {}({});",
                c_ret_ty,
                c_name,
                args_line,
            ));
            // for LLVM
            let name = CString::new(name.as_ref()).unwrap();
            let fn_type = LLVMFunctionType(ret_ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
            let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
            self.intrinsics
                .insert(name.into_string().unwrap(), Intrinsic::Builtin(function));
            true
        } else {
            false
        }
    }

    /// Generate code to call an intrinsic function with the given name and arguments.
    ///
    /// If the intrinsic is not defined, this function throws an error.
    pub unsafe fn call<T: AsRef<str>>(
        &mut self,
        builder: LLVMBuilderRef,
        name: T,
        args: &mut [LLVMValueRef],
    ) -> WeldResult<LLVMValueRef> {
        if let Some(func) = self.intrinsics.get(name.as_ref()) {
            let func = func.value();
            if args.len() != LLVMCountParams(func) as usize {
                panic!("Argument length didn't match number of parameters")
            }
            Ok(LLVMBuildCall(
                builder,
                func,
                args.as_mut_ptr(),
                args.len() as u32,
                c_str!(""),
            ))
        } else {
            unreachable!()
        }
    }

    // helper function
    pub fn c_args(&mut self, args: &[&str]) -> String {
        let mut args_line = String::new();
        let mut last_arg: &str = "";
        for arg in args {
            if !last_arg.is_empty() {
                args_line = format!("{}{}, ", args_line, last_arg);
            }
            last_arg = arg;
        }
        format!("{}{}", args_line, last_arg)
    }

    /// Convinience wrapper for calling any functions.
    pub unsafe fn c_call(
        &mut self,
        builder: &mut CodeBuilder,
        fun: &str,
        args: &[&str],
        ret_ty: &str,
        result: Option<String>,
    ) -> String {
        let args_line = self.c_args(args);
        if let Some(res) = result {
            builder.add(format!(
                "{} = {}({});", res, fun, args_line));
            res
        } else {
            let res = (*self.ccontext()).var_ids.next();
            builder.add(format!(
                "{} {} = {}({});", ret_ty, res, fun, args_line));
            res
        }
    }

    pub unsafe fn c_call_void(
        &mut self,
        builder: &mut CodeBuilder,
        fun: &str,
        args: &[&str],
    ) {
        let args_line = self.c_args(args);
        builder.add(format!(
            "(void){}({});", fun, args_line));
    }

    /// Convinience wrapper for calling the `weld_run_init` intrinsic.
    pub unsafe fn call_weld_run_init(
        &mut self,
        builder: LLVMBuilderRef,
        nworkers: LLVMValueRef,
        memlimit: LLVMValueRef,
        name: Option<*const c_char>,
    ) -> LLVMValueRef {
        let mut args = [nworkers, memlimit];
        LLVMBuildCall(
            builder,
            self.get("weld_runst_init").unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            name.unwrap_or(c_str!("")),
        )
    }
    pub unsafe fn c_call_weld_run_init(
        &mut self,
        builder: &mut CodeBuilder,
        nworkers: &str,
        memlimit: &str,
        name: Option<String>,
    ) -> String {
        let args = [nworkers, memlimit];
        let ret_type = &self.c_run_handle_type();
        self.c_call(builder, "weld_runst_init", &args, ret_type, name)
    }

    /// Convinience wrapper for calling the `weld_run_get_result` intrinsic.
    pub unsafe fn call_weld_run_get_result(
        &mut self,
        builder: LLVMBuilderRef,
        run: LLVMValueRef,
        name: Option<*const c_char>,
    ) -> LLVMValueRef {
        let mut args = [run];
        LLVMBuildCall(
            builder,
            self.get("weld_runst_get_result").unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            name.unwrap_or(c_str!("")),
        )
    }
    pub unsafe fn c_call_weld_run_get_result(
        &mut self,
        builder: &mut CodeBuilder,
        run: &str,
        name: Option<String>,
    ) -> String {
        let args = [run];
        let ret_type = &self.c_void_pointer_type();
        self.c_call(builder, "weld_runst_get_result", &args, ret_type, name)
    }

    /// Convinience wrapper for calling the `weld_run_set_result` intrinsic.
    pub unsafe fn call_weld_run_set_result(
        &mut self,
        builder: LLVMBuilderRef,
        run: LLVMValueRef,
        pointer: LLVMValueRef,
        name: Option<*const c_char>,
    ) -> LLVMValueRef {
        let mut args = [run, pointer];
        LLVMBuildCall(
            builder,
            self.get("weld_runst_set_result").unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            name.unwrap_or(c_str!("")),
        )
    }
    pub unsafe fn c_call_weld_run_set_result(
        &mut self,
        builder: &mut CodeBuilder,
        run: &str,
        pointer: &str,
        name: Option<String>,
    ) -> String {
        let args = [run, pointer];
        let ret_type = &self.c_void_pointer_type();
        self.c_call(builder, "weld_runst_set_result", &args, ret_type, name)
    }

    /// Convinience wrapper for calling the `weld_run_malloc` intrinsic.
    pub unsafe fn call_weld_run_malloc(
        &mut self,
        builder: LLVMBuilderRef,
        run: LLVMValueRef,
        size: LLVMValueRef,
        name: Option<*const c_char>,
    ) -> LLVMValueRef {
        let mut args = [run, size];
        LLVMBuildCall(
            builder,
            self.get("weld_runst_malloc").unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            name.unwrap_or(c_str!("")),
        )
    }
    pub unsafe fn c_call_weld_run_malloc(
        &mut self,
        builder: &mut CodeBuilder,
        run: &str,
        size: &str,
        name: Option<String>,
    ) -> String {
        let args = [run, size];
        let ret_type = &self.c_void_pointer_type();
        self.c_call(builder, "weld_runst_malloc", &args, ret_type, name)
    }


    /// Convinience wrapper for calling the `weld_run_remalloc` intrinsic.
    pub unsafe fn call_weld_run_realloc(
        &mut self,
        builder: LLVMBuilderRef,
        run: LLVMValueRef,
        pointer: LLVMValueRef,
        size: LLVMValueRef,
        name: Option<*const c_char>,
    ) -> LLVMValueRef {
        let mut args = [run, pointer, size];
        LLVMBuildCall(
            builder,
            self.get("weld_runst_realloc").unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            name.unwrap_or(c_str!("")),
        )
    }

    /// Convinience wrapper for calling the `weld_run_free` intrinsic.
    pub unsafe fn call_weld_run_free(
        &mut self,
        builder: LLVMBuilderRef,
        run: LLVMValueRef,
        pointer: LLVMValueRef,
    ) -> LLVMValueRef {
        let mut args = [run, pointer];
        LLVMBuildCall(
            builder,
            self.get("weld_runst_free").unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        )
    }

    /// Convinience wrapper for calling the `weld_run_get_errno` intrinsic.
    pub unsafe fn call_weld_run_get_errno(
        &mut self,
        builder: LLVMBuilderRef,
        run: LLVMValueRef,
        name: Option<*const c_char>,
    ) -> LLVMValueRef {
        let mut args = [run];
        LLVMBuildCall(
            builder,
            self.get("weld_runst_get_errno").unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            name.unwrap_or(c_str!("")),
        )
    }
    pub unsafe fn c_call_weld_run_get_errno(
        &mut self,
        builder: &mut CodeBuilder,
        run: &str,
        name: Option<String>,
    ) -> String {
        let args = [run];
        self.c_call(builder, "weld_runst_get_errno", &args, "i64", name)
    }

    /// Convinience wrapper for calling the `weld_runst_set_errno` intrinsic.
    pub unsafe fn call_weld_run_set_errno(
        &mut self,
        builder: LLVMBuilderRef,
        run: LLVMValueRef,
        errno: LLVMValueRef,
        name: Option<*const c_char>,
    ) -> LLVMValueRef {
        let mut args = [run, errno];
        LLVMBuildCall(
            builder,
            self.get("weld_runst_set_errno").unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            name.unwrap_or(c_str!("")),
        )
    }
    pub unsafe fn c_call_weld_run_set_errno(
        &mut self,
        builder: &mut CodeBuilder,
        run: &str,
        errno: &str,
        _name: Option<String>,
    ) {
        let args = [run, errno];
        self.c_call_void(builder, "weld_runst_get_errno", &args)
    }

    /// Convinience wrapper for calling the `weld_runst_assert` intrinsic.
    pub unsafe fn call_weld_run_assert(
        &mut self,
        builder: LLVMBuilderRef,
        run: LLVMValueRef,
        cond: LLVMValueRef,
        name: Option<*const c_char>,
    ) -> LLVMValueRef {
        let mut args = [run, cond];
        LLVMBuildCall(
            builder,
            self.get("weld_runst_assert").unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            name.unwrap_or(c_str!("")),
        )
    }

    /// Convinience wrapper for calling the `weld_run_print` intrinsic.
    pub unsafe fn call_weld_run_print(
        &mut self,
        builder: LLVMBuilderRef,
        run: LLVMValueRef,
        string: LLVMValueRef,
    ) -> LLVMValueRef {
        let mut args = [run, string];
        LLVMBuildCall(
            builder,
            self.get("weld_runst_print").unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        )
    }

    /// Convinience wrapper for calling `memcpy`.
    ///
    /// This assumes the `memcpy` is non-volatile and uses an default alignment value of 8.
    pub unsafe fn call_memcpy(
        &mut self,
        builder: LLVMBuilderRef,
        dst: LLVMValueRef,
        src: LLVMValueRef,
        size: LLVMValueRef,
    ) -> LLVMValueRef {
        let mut args = [dst, src, size, self.i32(8), self.i1(false)];
        LLVMBuildCall(
            builder,
            self.get("llvm.memcpy.p0i8.p0i8.i64").unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        )
    }

    /// Convinience wrapper for calling `memset` with 0 bytes value.
    ///
    /// This assumes the `memset` is non-volatile.
    pub unsafe fn call_memset_zero(
        &mut self,
        builder: LLVMBuilderRef,
        dst: LLVMValueRef,
        size: LLVMValueRef,
    ) -> LLVMValueRef {
        let mut args = [dst, self.i8(0), size, self.i32(8), self.i1(false)];
        LLVMBuildCall(
            builder,
            self.get("llvm.memset.p0i8.i64").unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        )
    }
}

/// Private methods.
impl Intrinsics {
    /// Populate the default intrinsics.
    ///
    /// By default, the code generator adds the Weld Run API (functions prefixed with `weld_run`)
    /// and a few other utility functions, such as `memcpy`.
    unsafe fn populate_defaults(&mut self) {
        use super::llvm_exts::LLVMExtAttribute::*;

        // Generate WeldRuntimeContext
        (*self.ccontext()).prelude_code.add(format!("\
typedef {i64} WeldRuntimeErrno;

/// WeldRuntimeErrno need to be synced with weld/src/runtime/mod.rs
/// Indicates success.
///
/// This will always be 0.
#define Success                 0
/// Invalid configuration.
#define ConfigurationError      1
/// Dynamic library load error.
#define LoadLibraryError        2
/// Weld compilation error.
#define CompileError            3
/// Array out-of-bounds error.
#define ArrayOutOfBounds        4
/// A Weld iterator was invalid.
#define BadIteratorLength       5
/// Mismatched Zip error.
///
/// This error is thrown if the vectors in a Zip have different lengths.
#define MismatchedZipSize       6
/// Out of memory error.
///
/// This error is thrown if the amount of memory allocated by the runtime exceeds the limit set
/// by the configuration.
#define OutOfMemory             7
#define RunNotFound             8
/// An unknown error.
#define Unknown                 9
/// A deserialization error.
///
/// This error occurs if a buffer being deserialized has an invalid length.
#define DeserializationError    10
/// A key was not found in a dictionary.
#define KeyNotFoundError        11
/// An assertion evaluated to `false`.
#define AssertionError          12
/// Maximum errno value.
///
/// All errors will have a value less than this value and greater than 0.
#define ErrnoMax                13

typedef struct {{
    /// Maps pointers to allocation size in bytes.
    // allocations: FnvHashMap<Ptr, Layout>,
    // ...void* to layout map...
    /// An error code set for the context.
    WeldRuntimeErrno errno;
    /// A result pointer set by the runtime.
    void* result;
    /// The number of worker threads.
    {i32} nworkers;
    /// A memory limit.
    {u64} memlimit;
    /// Number of allocated bytes so far.
    ///
    /// This will always be equal to `allocations.values().sum()`.
    {u64} allocated;
}} WeldRuntimeContext;
typedef WeldRuntimeContext* WeldRuntimeContextRef;",
            i32=self.c_i32_type(),
            i64=self.c_i64_type(),
            u64=self.c_u64_type(),
        ));

        let int8p = LLVMPointerType(self.i8_type(), 0);

        // Defines the default intrinsics used by the Weld runtime.
        let mut params = vec![self.i32_type(), self.i64_type()];
        let name = CString::new("weld_runst_init").unwrap();
        let fn_type = LLVMFunctionType(
            self.run_handle_type(),
            params.as_mut_ptr(),
            params.len() as u32,
            0,
        );
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        self.intrinsics.insert(
            name.into_string().unwrap(),
            Intrinsic::FunctionPointer(function, ffi::weld_runst_init as *mut c_void),
        );
        (*self.ccontext()).prelude_code.add(format!("\
extern void* malloc({u64});
{run_handle} weld_runst_init({i32} nworkers, {i64} memlimit)
{{
    WeldRuntimeContextRef run =
        (WeldRuntimeContextRef)malloc(sizeof(WeldRuntimeContext));
    assert(run != 0);
    // run->allocations = FnvHashMap::default();
    run->errno = Success;
    run->result = 0;
    run->nworkers = nworkers;
    run->memlimit = memlimit;
    run->allocated = 0;
    return ({run_handle})run;
}}",
            run_handle=self.c_run_handle_type(),
            i32=self.c_i32_type(),
            i64=self.c_i64_type(),
            u64=self.c_u64_type(),
        ));

        let mut params = vec![self.run_handle_type()];
        let name = CString::new("weld_runst_get_result").unwrap();
        let fn_type = LLVMFunctionType(int8p, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnFunction(self.context, function, &[NoUnwind]);
        LLVMExtAddAttrsOnParameter(
            self.context,
            function,
            &[NoCapture, NoAlias, NonNull, ReadOnly],
            0,
        );
        self.intrinsics.insert(
            name.into_string().unwrap(),
            Intrinsic::FunctionPointer(function, ffi::weld_runst_get_result as *mut c_void),
        );
        (*self.ccontext()).prelude_code.add(format!("\
void* weld_runst_get_result({run_handle} run)
{{
    return ((WeldRuntimeContextRef)run)->result;
}}",
            run_handle=self.c_run_handle_type(),
        ));

        let mut params = vec![self.run_handle_type(), int8p];
        let name = CString::new("weld_runst_set_result").unwrap();
        let fn_type = LLVMFunctionType(
            self.void_type(),
            params.as_mut_ptr(),
            params.len() as u32,
            0,
        );
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnFunction(self.context, function, &[NoUnwind]);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 0);
        self.intrinsics.insert(
            name.into_string().unwrap(),
            Intrinsic::FunctionPointer(function, ffi::weld_runst_set_result as *mut c_void),
        );
        (*self.ccontext()).prelude_code.add(format!("\
void* weld_runst_set_result({run_handle} run, {voidp} p)
{{
    return ((WeldRuntimeContextRef)run)->result = p;
}}",
            run_handle=self.c_run_handle_type(),
            voidp=self.c_void_pointer_type(),
        ));


        let mut params = vec![self.run_handle_type(), self.i64_type()];
        let name = CString::new("weld_runst_malloc").unwrap();
        let fn_type = LLVMFunctionType(int8p, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 0);
        LLVMExtAddAttrsOnReturn(self.context, function, &[NoAlias]);
        self.intrinsics.insert(
            name.into_string().unwrap(),
            Intrinsic::FunctionPointer(function, ffi::weld_runst_malloc as *mut c_void),
        );
        (*self.ccontext()).prelude_code.add(format!("\
void* weld_runst_malloc({run_handle} run, {u64} size)
{{
    // return ((WeldRuntimeContextRef)run)->result;
    return malloc(size);
}}",
            run_handle=self.c_run_handle_type(),
            u64=self.c_u64_type(),
        ));
/*
    unsafe fn malloc(&mut self, size: i64) -> Ptr {
        if size == 0 {
            trace!("Alloc'd 0-size pointer (null)");
            return ptr::null_mut();
        }

        let size = size as usize;

        if self.allocated + size > self.memlimit {
            self.errno = WeldRuntimeErrno::OutOfMemory;
            panic!(
                "Weld run ran out of memory (limit={}, attempted to allocate {}",
                self.memlimit,
                self.allocated + size
            );
        }
        let layout = Layout::from_size_align_unchecked(size as usize, DEFAULT_ALIGN);
        let mem = Allocator.alloc(layout);

        self.allocated += layout.size();
        trace!("Alloc'd pointer {:?} ({} bytes)", mem, layout.size());

        self.allocations.insert(mem, layout);
        mem
    }

#[no_mangle]
/// Allocate memory within the provided context.
pub unsafe extern "C" fn weld_runst_malloc(run: WeldRuntimeContextRef, size: int64_t) -> Ptr {
    let run = &mut *run;
    run.malloc(size)
}
*/

        let mut params = vec![self.run_handle_type(), int8p, self.i64_type()];
        let name = CString::new("weld_runst_realloc").unwrap();
        let fn_type = LLVMFunctionType(int8p, params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 0);
        LLVMExtAddAttrsOnReturn(self.context, function, &[NoAlias]);
        self.intrinsics.insert(
            name.into_string().unwrap(),
            Intrinsic::FunctionPointer(function, ffi::weld_runst_realloc as *mut c_void),
        );

        let mut params = vec![self.run_handle_type(), int8p];
        let name = CString::new("weld_runst_free").unwrap();
        let fn_type = LLVMFunctionType(
            self.void_type(),
            params.as_mut_ptr(),
            params.len() as u32,
            0,
        );
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 0);
        self.intrinsics.insert(
            name.into_string().unwrap(),
            Intrinsic::FunctionPointer(function, ffi::weld_runst_free as *mut c_void),
        );

        let mut params = vec![self.run_handle_type()];
        let name = CString::new("weld_runst_get_errno").unwrap();
        let fn_type =
            LLVMFunctionType(self.i64_type(), params.as_mut_ptr(), params.len() as u32, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnFunction(self.context, function, &[NoUnwind]);
        LLVMExtAddAttrsOnParameter(
            self.context,
            function,
            &[NoCapture, NoAlias, NonNull, ReadOnly],
            0,
        );
        self.intrinsics.insert(
            name.into_string().unwrap(),
            Intrinsic::FunctionPointer(function, ffi::weld_runst_get_errno as *mut c_void),
        );
        (*self.ccontext()).prelude_code.add(format!("\
{i64} weld_runst_get_errno({run_handle} run)
{{
    return ((WeldRuntimeContextRef)run)->errno;
}}",
            run_handle=self.c_run_handle_type(),
            i64=self.c_i64_type(),
        ));

        let mut params = vec![self.run_handle_type(), self.i64_type()];
        let name = CString::new("weld_runst_set_errno").unwrap();
        let fn_type = LLVMFunctionType(
            self.void_type(),
            params.as_mut_ptr(),
            params.len() as u32,
            0,
        );
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnFunction(self.context, function, &[NoReturn]);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 0);
        self.intrinsics.insert(
            name.into_string().unwrap(),
            Intrinsic::FunctionPointer(function, ffi::weld_runst_set_errno as *mut c_void),
        );
        (*self.ccontext()).prelude_code.add(format!("\
void weld_runst_set_errno({run_handle} run, {i64} errno)
{{
    ((WeldRuntimeContextRef)run)->errno = errno;
    abort();
    // FIXME: need to throw termination signal to caller here.
    //        weld for llvm call panic function in rust like below.
    // panic!(\"Weld runtime threw error: {{}}\", self.errno)
}}",
            run_handle=self.c_run_handle_type(),
            i64=self.c_i64_type(),
        ));

        let mut params = vec![self.run_handle_type(), self.bool_type()];
        let name = CString::new("weld_runst_assert").unwrap();
        let fn_type = LLVMFunctionType(
            self.bool_type(),
            params.as_mut_ptr(),
            params.len() as u32,
            0,
        );
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnParameter(self.context, function, &[NoCapture, NoAlias, NonNull], 0);
        self.intrinsics.insert(
            name.into_string().unwrap(),
            Intrinsic::FunctionPointer(function, ffi::weld_runst_assert as *mut c_void),
        );

        let mut params = vec![self.run_handle_type(), int8p];
        let name = CString::new("weld_runst_print").unwrap();
        let fn_type = LLVMFunctionType(
            self.void_type(),
            params.as_mut_ptr(),
            params.len() as u32,
            0,
        );
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        LLVMExtAddAttrsOnParameter(
            self.context,
            function,
            &[NoCapture, NoAlias, NonNull, ReadOnly],
            0,
        );
        LLVMExtAddAttrsOnParameter(
            self.context,
            function,
            &[NoCapture, NoAlias, NonNull, ReadOnly],
            1,
        );
        self.intrinsics.insert(
            name.into_string().unwrap(),
            Intrinsic::FunctionPointer(function, ffi::weld_runst_print as *mut c_void),
        );

        let mut params = vec![
            int8p,
            int8p,
            self.i64_type(),
            self.i32_type(),
            self.i1_type(),
        ];
        let name = CString::new("llvm.memcpy.p0i8.p0i8.i64").unwrap();
        let fn_type = LLVMFunctionType(
            self.void_type(),
            params.as_mut_ptr(),
            params.len() as u32,
            0,
        );
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        // LLVM sets attributes on `memcpy` automatically.
        self.intrinsics
            .insert(name.into_string().unwrap(), Intrinsic::Builtin(function));

        let mut params = vec![
            int8p,
            self.i8_type(),
            self.i64_type(),
            self.i32_type(),
            self.i1_type(),
        ];
        let name = CString::new("llvm.memset.p0i8.i64").unwrap();
        let fn_type = LLVMFunctionType(
            self.void_type(),
            params.as_mut_ptr(),
            params.len() as u32,
            0,
        );
        let function = LLVMAddFunction(self.module, name.as_ptr(), fn_type);
        // LLVM sets attributes on `memset` automatically.
        self.intrinsics
            .insert(name.into_string().unwrap(), Intrinsic::Builtin(function));
    }
}
