//! An C backend currently optimized for single-threaded execution.
//!
//! The `CGenerator` struct is responsible for converting an SIR program into an LLVM module.
//! The LLVM module is then JIT'd and returned as a runnable executable.
//!
//! # Overview
//!
//! This code generator is divided into a number of submodules, most of which implement extension
//! traits on top of `CGenerator`. For example, the `hash` module implements the `GenHash`
//! trait, whose sole implemenator is `CGenerator`. The `gen_hash` function in the `GenHash`
//! trait thus adds using state maintained in the `CGenerator` to hash Weld types.
//! `CGenerator` tracks code that has been generated already: for most extension traits, this
//! usually involves some state to ensure that the same code is not generated twice.
//!
//! # The `CodeGenExt` trait
//!
//! The `CodeGenExt` trait contains a number of helper functions for generating LLVM code,
//! retrieving types, etc. Implementors should implement the `module` and `context` functions: all
//! other methods in the trait have standard default implementations that should not be overridden.
//!
//! ## Submodules
//!
//! * The `builder` module provides code generation for the builder types.  `builder` also contains
//! extension traits for generating builder-related expressions (Result, Merge, and For).
//!
//! * The `dict` and `vector` modules define the layout of dictionaries and vectors, and also
//! provide methods over them.
//!
//! * The `eq` module defines equality-check code generation.
//!
//! * The `hash` module implements hashing.
//!
//! * The `intrinsics` module manages intrinsics, or functions that are declared but not generated.
//! This module adds a number of "default" intrinsics, such as the Weld runtime functions (prefixed
//! with `weld_strt_`), `memcpy`, and so forth.
//!
//! * The `jit` module manages compiling a constructed LLVM module into a runnable executable.
//! Among other things, it manages LLVM optimization passes and module verification.
//!
//! The `llvm_exts` modules uses `libllvmext` to provide LLVM functionality that `llvm_sys` (and by
//! extension, the `llvm-c` API) does not provide. It is effectively a wrapper around a few
//! required C++ library calls.
//!
//! * The `numeric` module generates code for numeric expressions such as binary and unary
//! operators, comparisons, etc.
//!
//! * The `serde` module generates code for serializing and deserializing types.
//!
//! * The `target` module provides parsed target specific feature information.

extern crate code_builder;

use fnv;

use libc;
use llvm_sys;
use code_builder::CodeBuilder;

use std::ffi::{CStr, CString};
use std::fmt;
use std::mem;

use fnv::FnvHashMap;
use libc::{c_char, c_uint, c_ulonglong};

use crate::conf::ParsedConf;
use crate::error::*;
use crate::sir::*;
use crate::util::stats::CompilationStats;
use crate::util::id::IdGenerator;

use self::llvm_sys::core::*;
use self::llvm_sys::prelude::*;
use self::llvm_sys::LLVMLinkage;

use super::*;

lazy_static! {
    /// Name of the run handle struct in generated code.
    static ref RUN_HANDLE_NAME: CString = CString::new("RunHandle").unwrap();
}

/// Width of a SIMD vector.
// TODO This should be based on the type!
pub const LLVM_VECTOR_WIDTH: u32 = 4;

/// Calling convention for SIR function.
pub const SIR_FUNC_CALL_CONV: u32 = llvm_sys::LLVMCallConv::LLVMFastCallConv as u32;

/// Convert a string literal into a C string.
macro_rules! c_str {
    ($s:expr) => {
        concat!($s, "\0").as_ptr() as *const i8
    };
}

mod builder;
mod cmp;
mod dict;
mod eq;
mod hash;
mod intrinsic;
mod compile;
//mod jit;
mod llvm_exts;
mod numeric;
mod serde;
mod target;
mod vector;

use self::builder::appender;
use self::builder::merger;

/// Loads a dynamic library from a file using LLVMLoadLibraryPermanently.
///
/// It is safe to call this function multiple times for the same library.
pub fn load_library(libname: &str) -> WeldResult<()> {
    let c_string = CString::new(libname).unwrap();
    let c_string_raw = c_string.into_raw() as *const c_char;
    if unsafe { llvm_sys::support::LLVMLoadLibraryPermanently(c_string_raw) } == 0 {
        Ok(())
    } else {
        compile_err!("Couldn't load library {}", libname)
    }
}

/// Returns the size of a type in bytes.
pub fn size_of(ty: &Type) -> usize {
    unsafe {
        let mut gen = CGenerator::new(ParsedConf::default()).unwrap();
        gen.size_of_ty(ty)
    }
}

/// Compile Weld SIR into a runnable module.
///
/// The runnable module is wrapped as a trait object which the `CompiledModule` struct in `codegen`
/// calls.
pub fn compile(
    program: &SirProgram,
    conf: &ParsedConf,
    stats: &mut CompilationStats,
) -> WeldResult<Box<dyn Runnable + Send + Sync>> {
    use crate::runtime;
    use crate::util::dump::{write_code, DumpCodeFormat};

    info!("Compiling using single thread runtime");
    info!("Target architecture is VE");

    let codegen = unsafe { CGenerator::generate(conf.clone(), &program)? };

    nonfatal!(write_code(
        codegen.to_string(),
        DumpCodeFormat::LLVM,
        &conf.dump_code
    ));

    nonfatal!(write_code(
        codegen.gen_c_code(),
        DumpCodeFormat::C,
        &conf.dump_code
    ));

    unsafe {
        runtime::ffi::weld_init();
    }

    let mappings = &codegen.intrinsics.mappings();
    let module = unsafe { compile::compile(codegen.context, codegen.module, mappings, conf, stats)? };

    nonfatal!(write_code(
        module.asm()?,
        DumpCodeFormat::Assembly,
        &conf.dump_code
    ));
    nonfatal!(write_code(
        module.llvm()?,
        DumpCodeFormat::LLVMOpt,
        &conf.dump_code
    ));

    Ok(Box::new(module))
}

/// A helper trait that defines the LLVM type and structure of an input to the runtime.
trait LlvmInputArg {
    /// LLVM type of the input struct.
    unsafe fn llvm_type(context: LLVMContextRef) -> LLVMTypeRef;
    /// C type of the input struct.
    unsafe fn c_type(ccontext: CContextRef) -> &'static str;
    /// Index of the data pointer in the struct.
    fn input_index() -> u32;
    /// Index of the number of workers value in the struct.
    fn nworkers_index() -> u32;
    /// Index of the memory limit value in the struct.
    fn memlimit_index() -> u32;
    /// Index of run handle pointer in the struct.
    fn run_index() -> u32;
}

impl LlvmInputArg for WeldInputArgs {
    unsafe fn c_type(ccontext: CContextRef) -> &'static str {
        if !(*ccontext).input_arg_defined {
            (*ccontext).prelude_code.add(format!("\
typedef struct {{
    {i64} input;
    {i32} nworkers;
    {i64} memlimit;
    {i64} run;
}} input_args_t;

", i64=c_i64_type(ccontext), i32=c_i32_type(ccontext)));
            (*ccontext).input_arg_defined = true;
        }
        "input_args_t"
    }
    unsafe fn llvm_type(context: LLVMContextRef) -> LLVMTypeRef {
        let mut types = [
            LLVMInt64TypeInContext(context),
            LLVMInt32TypeInContext(context),
            LLVMInt64TypeInContext(context),
            LLVMInt64TypeInContext(context),
        ];
        let args = LLVMStructCreateNamed(context, c_str!("input_args_t"));
        LLVMStructSetBody(args, types.as_mut_ptr(), types.len() as u32, 0);
        args
    }

    fn input_index() -> u32 {
        0
    }

    fn nworkers_index() -> u32 {
        1
    }

    fn memlimit_index() -> u32 {
        2
    }

    fn run_index() -> u32 {
        3
    }
}

/// A helper trait that defines the LLVM type and structure of an output from the runtime.
trait LlvmOutputArg {
    /// LLVM type of the output struct.
    unsafe fn llvm_type(context: LLVMContextRef) -> LLVMTypeRef;
    /// C type of the input struct.
    unsafe fn c_type(ccontext: CContextRef) -> &'static str;
    /// Index of the output data pointer in the struct.
    fn output_index() -> u32;
    /// Index of the run ID/data pointer in the struct.
    fn run_index() -> u32;
    /// Index of the errno pointer in the struct.
    fn errno_index() -> u32;
}

impl LlvmOutputArg for WeldOutputArgs {
    unsafe fn c_type(ccontext: CContextRef) -> &'static str {
        if !(*ccontext).output_arg_defined {
            (*ccontext).prelude_code.add(format!("\
typedef struct {{
    {i64} output;
    {i64} run;
    {i64} errno;
}} output_args_t;

", i64=c_i64_type(ccontext)));
            (*ccontext).output_arg_defined = true;
        }
        "output_args_t"
    }
    unsafe fn llvm_type(context: LLVMContextRef) -> LLVMTypeRef {
        let mut types = [
            LLVMInt64TypeInContext(context),
            LLVMInt64TypeInContext(context),
            LLVMInt64TypeInContext(context),
        ];
        let args = LLVMStructCreateNamed(context, c_str!("output_args_t"));
        LLVMStructSetBody(args, types.as_mut_ptr(), types.len() as u32, 0);
        args
    }

    fn output_index() -> u32 {
        0
    }

    fn run_index() -> u32 {
        1
    }

    fn errno_index() -> u32 {
        2
    }
}

/// Booleans are represented as `i8`.
///
/// For instructions that require `i1` (e.g, conditional branching or select), the caller
/// should truncate this type to `i1_type` manually. The distinction between booleans and `i1`
/// is that boolean types are "externally visible", whereas `i1`s only appear in internal code.
unsafe fn c_bool_type(ccontext: CContextRef) -> String {
    use crate::ast::ScalarKind::*;
    if !(*ccontext).basic_types.contains_key(&Bool) {
        (*ccontext).prelude_code.add("typedef char bool;");
        (*ccontext).basic_types.insert(Bool, "bool".to_string());
    }
    (*ccontext).basic_types.get(&Bool).unwrap().to_string()
}

unsafe fn c_i1_type(ccontext: CContextRef) -> String {
    if !(*ccontext).i1_defined {
        (*ccontext).prelude_code.add("typedef char i1;");
        (*ccontext).i1_defined = true;
    }
    "i1".to_string()
}

unsafe fn c_i8_type(ccontext: CContextRef) -> String {
    use crate::ast::ScalarKind::*;
    if !(*ccontext).basic_types.contains_key(&I8) {
        (*ccontext).prelude_code.add("typedef char i8;");
        (*ccontext).basic_types.insert(I8, "i8".to_string());
    }
    (*ccontext).basic_types.get(&I8).unwrap().to_string()
}

unsafe fn c_u8_type(ccontext: CContextRef) -> String {
    use crate::ast::ScalarKind::*;
    if !(*ccontext).basic_types.contains_key(&U8) {
        (*ccontext).prelude_code.add("typedef unsigned char u8;");
        (*ccontext).basic_types.insert(U8, "u8".to_string());
    }
    (*ccontext).basic_types.get(&U8).unwrap().to_string()
}

unsafe fn c_i16_type(ccontext: CContextRef) -> String {
    use crate::ast::ScalarKind::*;
    if !(*ccontext).basic_types.contains_key(&I16) {
        (*ccontext).prelude_code.add("typedef short i16;");
        (*ccontext).basic_types.insert(I16, "i16".to_string());
    }
    (*ccontext).basic_types.get(&I16).unwrap().to_string()
}

unsafe fn c_u16_type(ccontext: CContextRef) -> String {
    use crate::ast::ScalarKind::*;
    if !(*ccontext).basic_types.contains_key(&U16) {
        (*ccontext).prelude_code.add("typedef unsigned short u16;");
        (*ccontext).basic_types.insert(U16, "u16".to_string());
    }
    (*ccontext).basic_types.get(&U16).unwrap().to_string()
}

unsafe fn c_i32_type(ccontext: CContextRef) -> String {
    use crate::ast::ScalarKind::*;
    if !(*ccontext).basic_types.contains_key(&I32) {
        (*ccontext).prelude_code.add("typedef int i32;");
        (*ccontext).basic_types.insert(I32, "i32".to_string());
    }
    (*ccontext).basic_types.get(&I32).unwrap().to_string()
}

unsafe fn c_u32_type(ccontext: CContextRef) -> String {
    use crate::ast::ScalarKind::*;
    if !(*ccontext).basic_types.contains_key(&U32) {
        (*ccontext).prelude_code.add("typedef unsigned int u32;");
        (*ccontext).basic_types.insert(U32, "u32".to_string());
    }
    (*ccontext).basic_types.get(&U32).unwrap().to_string()
}

unsafe fn c_i64_type(ccontext: CContextRef) -> String {
    use crate::ast::ScalarKind::*;
    if !(*ccontext).basic_types.contains_key(&I64) {
        (*ccontext).prelude_code.add("typedef long i64;");
        (*ccontext).basic_types.insert(I64, "i64".to_string());
    }
    (*ccontext).basic_types.get(&I64).unwrap().to_string()
}

unsafe fn c_u64_type(ccontext: CContextRef) -> String {
    use crate::ast::ScalarKind::*;
    if !(*ccontext).basic_types.contains_key(&U64) {
        (*ccontext).prelude_code.add("typedef unsigned long u64;");
        (*ccontext).basic_types.insert(U64, "u64".to_string());
    }
    (*ccontext).basic_types.get(&U64).unwrap().to_string()
}

unsafe fn c_f32_type(ccontext: CContextRef) -> String {
    use crate::ast::ScalarKind::*;
    if !(*ccontext).basic_types.contains_key(&F32) {
        (*ccontext).prelude_code.add("typedef float f32;");
        (*ccontext).basic_types.insert(F32, "f32".to_string());
    }
    (*ccontext).basic_types.get(&F32).unwrap().to_string()
}

unsafe fn c_f64_type(ccontext: CContextRef) -> String {
    use crate::ast::ScalarKind::*;
    if !(*ccontext).basic_types.contains_key(&F64) {
        (*ccontext).prelude_code.add("typedef double f64;");
        (*ccontext).basic_types.insert(F64, "f64".to_string());
    }
    (*ccontext).basic_types.get(&F64).unwrap().to_string()
}

unsafe fn c_void_type(_ccontext: CContextRef) -> String {
    "void".to_string()
}

unsafe fn c_void_pointer_type(_ccontext: CContextRef) -> String {
    "void*".to_string()
}

unsafe fn c_pointer_type(_ccontext: CContextRef, ty: &str) -> String {
    format!("{}*", ty)
}

unsafe fn c_simd_type(_ccontext: CContextRef, ty: &str, size: u32) -> String {
    format!("simd_{}_{}", ty, size)
}

unsafe fn c_type_of(_ccontext: CContextRef, ty: &str) -> String {
    format!("typeof({})", ty)
}

unsafe fn c_run_handle_type(ccontext: CContextRef) -> String {
    if !(*ccontext).run_handle_defined {
        (*ccontext).prelude_code.add(
            "typedef struct { char f; } RunHandle;");
        (*ccontext).run_handle_defined = true;
    }
    "RunHandle*".to_string()
}

/// Specifies whether a type contains a pointer in generated code.
pub trait HasPointer {
    fn has_pointer(&self) -> bool;
}

impl HasPointer for Type {
    fn has_pointer(&self) -> bool {
        use crate::ast::Type::*;
        match *self {
            Scalar(_) => false,
            Simd(_) => false,
            Vector(_) => true,
            Dict(_, _) => true,
            Builder(_, _) => true,
            Struct(ref tys) => tys.iter().any(|ref t| t.has_pointer()),
            Function(_, _) | Unknown | Alias(_, _) => unreachable!(),
        }
    }
}

/// A struct holding the C codegen state.  This is used instead of LLVMContext.
pub struct CContext {
    /// Names of basic types.  This map returns a name of a given type once
    /// the name is declared.  For example, i16_type() outputs declaration of
    /// i16 and set the name to this map.
    basic_types: FnvHashMap<ScalarKind, String>,
    simd_types: FnvHashMap<ScalarKind, String>,

    i1_defined: bool,
    input_arg_defined: bool,
    output_arg_defined: bool,
    run_handle_defined: bool,

    /// A ID generator for prelude functions and an entry function.
    var_ids: IdGenerator,

    /// A CodeBuilder for prelude functions such as type and struct definitions.
    prelude_code: CodeBuilder,

    /// A CodeBuilder for body functions in the module.
    body_code: CodeBuilder,
}

type CContextRef = *mut CContext;

/// A struct holding the global codegen state for an SIR program.
pub struct CGenerator {
    /// A configuration for generating code.
    conf: ParsedConf,
    /// Target-specific information used during code generation.
    target: target::Target,
    /// An LLVM Context for isolating code generation.
    context: LLVMContextRef,
    /// The main LLVM module to which code is added.
    module: LLVMModuleRef,
    /// An C Context for isolating code generation.
    ccontext: CContextRef,
    /// A map that tracks references to an SIR function's LLVM function.
    functions: FnvHashMap<FunctionId, LLVMValueRef>,
    /// A map that tracks references to an SIR function's LLVM function.
    c_functions: FnvHashMap<FunctionId, String>,
    /// A map tracking generated vectors.
    ///
    /// The key maps the *element type* to the vector's type reference and methods on it.
    vectors: FnvHashMap<Type, vector::Vector>,
    /// Counter for unique vector names.
    vector_index: u32,
    /// A map tracking generated mergers.
    ///
    /// The key maps the merger type to the merger's type reference and methods on it.
    mergers: FnvHashMap<BuilderKind, merger::Merger>,
    /// A map tracking generated appenders.
    ///
    /// The key maps the appender type to the appender's type reference and methods on it.
    appenders: FnvHashMap<BuilderKind, appender::Appender>,
    /// Counter for unique appender names.
    appender_index: u32,
    /// A map tracking generated dictionaries.
    ///
    /// The key maps the dictionary's `Dict` type to the type reference and methods on it.
    dictionaries: FnvHashMap<Type, dict::Dict>,
    /// Common intrinsics defined in the module.
    ///
    /// An intrinsic is any function defined outside of module (i.e., is not code generated).
    intrinsics: intrinsic::Intrinsics,
    /// Generated string literal values.
    strings: FnvHashMap<CString, LLVMValueRef>,
    /// Equality functions on various types.
    eq_fns: FnvHashMap<Type, LLVMValueRef>,
    /// Opaque, externally visible wrappers for equality functions.
    ///
    /// These are used by the dicitonary.
    opaque_eq_fns: FnvHashMap<Type, LLVMValueRef>,
    /// Comparison functions on various types.
    cmp_fns: FnvHashMap<Type, LLVMValueRef>,
    /// Opaque comparison functions that contain a key function, indexed by the ID of the key function.
    hash_fns: FnvHashMap<Type, LLVMValueRef>,
    /// Serialization functions on various types.
    serialize_fns: FnvHashMap<Type, LLVMValueRef>,
    /// Deserialization functions on various types.
    deserialize_fns: FnvHashMap<Type, LLVMValueRef>,
    /// Names of structs for readability.
    struct_names: FnvHashMap<Type, CString>,
    /// Names of structs for readability.
    c_struct_names: FnvHashMap<Type, String>,
    /// Counter for unique struct names.
    struct_index: u32,
}

impl Drop for CGenerator {
    fn drop(&mut self) {
        unsafe {
            drop(Box::from_raw(self.ccontext));
        }
    }
}

/// Defines helper methods for LLVM code generation.
///
/// The main methods here have default implementations: implemenators only need to implement the
/// `module` and `context` methods.
pub trait CodeGenExt {
    /// Returns the module used by this code generator.
    fn module(&self) -> LLVMModuleRef;

    /// Returns the context used by this code generator.
    fn context(&self) -> LLVMContextRef;

    /// Returns the context used by this code generator.
    fn ccontext(&self) -> CContextRef;

    /// Loads a value.
    ///
    /// This method includes a check to ensure that `pointer` is actually a pointer: otherwise, the
    /// LLVM API throws a segmentation fault.
    unsafe fn load(
        &mut self,
        builder: LLVMBuilderRef,
        pointer: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        use self::llvm_sys::LLVMTypeKind;
        if LLVMGetTypeKind(LLVMTypeOf(pointer)) != LLVMTypeKind::LLVMPointerTypeKind {
            unreachable!()
        } else {
            let loaded = LLVMBuildLoad(builder, pointer, c_str!(""));
            if LLVMGetTypeKind(LLVMTypeOf(loaded)) == LLVMTypeKind::LLVMVectorTypeKind {
                LLVMSetAlignment(loaded, 1);
            }
            Ok(loaded)
        }
    }

    /// Get a constant zero-value of the given type.
    unsafe fn zero(&self, ty: LLVMTypeRef) -> LLVMValueRef {
        use self::llvm_sys::LLVMTypeKind::*;
        use std::ptr;
        match LLVMGetTypeKind(ty) {
            LLVMFloatTypeKind => self.f32(0.0),
            LLVMDoubleTypeKind => self.f64(0.0),
            LLVMIntegerTypeKind => LLVMConstInt(ty, 0, 0),
            LLVMStructTypeKind => {
                let num_fields = LLVMCountStructElementTypes(ty) as usize;
                let mut fields = vec![ptr::null_mut(); num_fields];
                LLVMGetStructElementTypes(ty, fields.as_mut_ptr());

                let mut value = LLVMGetUndef(ty);
                for (i, field) in fields.into_iter().enumerate() {
                    value =
                        LLVMConstInsertValue(value, self.zero(field), [i as u32].as_mut_ptr(), 1);
                }
                value
            }
            LLVMPointerTypeKind => self.null_ptr(ty),
            LLVMVectorTypeKind => {
                let size = LLVMGetVectorSize(ty);
                let zero = self.zero(LLVMGetElementType(ty));
                let mut constants = vec![zero; size as usize];
                LLVMConstVector(constants.as_mut_ptr(), size)
            }
            // Other types are not used in the backend.
            other => panic!("Unsupported type kind {:?} in CodeGenExt::zero()", other),
        }
    }

    /// Returns the type of a hash code.
    unsafe fn hash_type(&self) -> LLVMTypeRef {
        self.i32_type()
    }
    unsafe fn c_hash_type(&self) -> String {
        self.c_i32_type()
    }

    /// Returns the type of the key comparator over opaque pointers.
    unsafe fn opaque_cmp_type(&self) -> LLVMTypeRef {
        let mut arg_tys = [self.void_pointer_type(), self.void_pointer_type()];
        let fn_type = LLVMFunctionType(
            self.i32_type(),
            arg_tys.as_mut_ptr(),
            arg_tys.len() as u32,
            0,
        );
        LLVMPointerType(fn_type, 0)
    }

    /// Generates code to define a function with the given return type and argument type.
    ///
    /// Returns a reference to the function, a builder used to build the function body, and the
    /// entry basic block. This method uses the default private linkage type, meaning functions
    /// generated using this method cannot be passed or called outside of the module.
    unsafe fn define_function<T: AsRef<str>>(
        &mut self,
        ret_ty: LLVMTypeRef,
        c_ret_ty: &str,
        arg_tys: &mut [LLVMTypeRef],
        c_arg_tys: &[String],
        name: T,
    ) -> (LLVMValueRef, LLVMBuilderRef, LLVMBasicBlockRef, CodeBuilder) {
        self.define_function_with_visibility(
            ret_ty,
            c_ret_ty,
            arg_tys,
            c_arg_tys,
            LLVMLinkage::LLVMPrivateLinkage,
            name,
        )
    }

    /// Generates code to define a function with the given return type and argument type.
    ///
    /// Returns a reference to the function, a builder used to build the function body, and the
    /// entry basic block.
    unsafe fn define_function_with_visibility<T: AsRef<str>>(
        &mut self,
        ret_ty: LLVMTypeRef,
        c_ret_ty: &str,
        arg_tys: &mut [LLVMTypeRef],
        c_arg_tys: &[String],
        visibility: LLVMLinkage,
        name: T,
    ) -> (LLVMValueRef, LLVMBuilderRef, LLVMBasicBlockRef, CodeBuilder) {
        let func_ty = LLVMFunctionType(ret_ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        // for C
        let mut code = CodeBuilder::new();
        let args_line = self.c_define_args(&c_arg_tys);
        code.add(format!(
            "{ret_ty} {fun}({args})",
            ret_ty=c_ret_ty,
            fun=name.as_ref(),
            args=args_line,
        ));

        // for LLVM
        let name = CString::new(name.as_ref()).unwrap();
        let function = LLVMAddFunction(self.module(), name.as_ptr(), func_ty);
        // Add the default attributes to all functions.
        llvm_exts::LLVMExtAddDefaultAttrs(self.context(), function);

        let builder = LLVMCreateBuilderInContext(self.context());
        let block = LLVMAppendBasicBlockInContext(self.context(), function, c_str!(""));
        LLVMPositionBuilderAtEnd(builder, block);
        LLVMSetLinkage(function, visibility);
        (function, builder, block, code)
    }

    /// Converts a `LiteralKind` into a constant LLVM scalar literal value.
    ///
    /// This method does not generate any code.
    unsafe fn scalar_literal(&self, kind: &LiteralKind) -> LLVMValueRef {
        use crate::ast::LiteralKind::*;
        match *kind {
            BoolLiteral(val) => self.bool(val),
            I8Literal(val) => self.i8(val),
            I16Literal(val) => self.i16(val),
            I32Literal(val) => self.i32(val),
            I64Literal(val) => self.i64(val),
            U8Literal(val) => self.u8(val),
            U16Literal(val) => self.u16(val),
            U32Literal(val) => self.u32(val),
            U64Literal(val) => self.u64(val),
            F32Literal(val) => self.f32(f32::from_bits(val)),
            F64Literal(val) => self.f64(f64::from_bits(val)),
            // Handled by the `gen_numeric`.
            StringLiteral(_) => unreachable!(),
        }
    }
    unsafe fn c_scalar_literal(&self, kind: &LiteralKind) -> String {
        use crate::ast::LiteralKind::*;
        match *kind {
            BoolLiteral(val) => self.c_bool(val),
            I8Literal(val) => self.c_i8(val),
            I16Literal(val) => self.c_i16(val),
            I32Literal(val) => self.c_i32(val),
            I64Literal(val) => self.c_i64(val),
            U8Literal(val) => self.c_u8(val),
            U16Literal(val) => self.c_u16(val),
            U32Literal(val) => self.c_u32(val),
            U64Literal(val) => self.c_u64(val),
            F32Literal(val) => self.c_f32(f32::from_bits(val)),
            F64Literal(val) => self.c_f64(f64::from_bits(val)),
            // Handled by the `gen_numeric`.
            StringLiteral(_) => unreachable!(),
        }
    }

    /// Returns the identity for a given scalar kind and binary operator.
    unsafe fn binop_identity(&self, op: BinOpKind, kind: ScalarKind) -> WeldResult<LLVMValueRef> {
        use crate::ast::BinOpKind::*;
        use crate::ast::ScalarKind::*;
        match kind {
            _ if kind.is_integer() => {
                let ty = LLVMIntTypeInContext(self.context(), kind.bits());
                let signed = kind.is_signed() as i32;
                match op {
                    Add => Ok(LLVMConstInt(ty, 0, signed)),
                    Multiply => Ok(LLVMConstInt(ty, 1, signed)),
                    Max => Ok(LLVMConstInt(ty, ::std::u64::MIN, signed)),
                    Min => Ok(LLVMConstInt(ty, ::std::u64::MAX, signed)),
                    _ => unreachable!(),
                }
            }
            F32 => {
                let ty = self.f32_type();
                match op {
                    Add => Ok(LLVMConstReal(ty, 0.0)),
                    Multiply => Ok(LLVMConstReal(ty, 1.0)),
                    Max => Ok(LLVMConstReal(ty, f64::from(::std::f32::MIN))),
                    Min => Ok(LLVMConstReal(ty, f64::from(::std::f32::MAX))),
                    _ => unreachable!(),
                }
            }
            F64 => {
                let ty = self.f64_type();
                match op {
                    Add => Ok(LLVMConstReal(ty, 0.0)),
                    Multiply => Ok(LLVMConstReal(ty, 1.0)),
                    Max => Ok(LLVMConstReal(ty, ::std::f64::MIN)),
                    Min => Ok(LLVMConstReal(ty, ::std::f64::MAX)),
                    _ => unreachable!(),
                }
            }
            _ => unreachable!(),
        }
    }

    /// Returns the constant size of a type.
    unsafe fn size_of(&self, ty: LLVMTypeRef) -> LLVMValueRef {
        LLVMSizeOf(ty)
    }
    unsafe fn c_size_of(&self, ty: &str) -> String {
        format!("sizeof({})", ty)
    }

    /// Returns the constant size of a type in bits.
    ///
    /// Unlike `size_of`, this returns the size of the value at compile time.
    unsafe fn size_of_bits(&self, ty: LLVMTypeRef) -> u64 {
        let layout = llvm_sys::target::LLVMGetModuleDataLayout(self.module());
        llvm_sys::target::LLVMSizeOfTypeInBits(layout, ty) as u64
    }

    /// Returns the LLVM type corresponding to size_t on this architecture.
    unsafe fn size_t_type(&self) -> WeldResult<LLVMTypeRef> {
        Ok(LLVMIntTypeInContext(
            self.context(),
            mem::size_of::<libc::size_t>() as c_uint,
        ))
    }

    /// Computes the next power of two for the given value.
    ///
    /// `value` must be either an `i32` or `i64` type.
    /// Uses the algorithm from https://graphics.stanford.edu/~seander/bithacks.html.
    unsafe fn next_pow2(&self, builder: LLVMBuilderRef, value: LLVMValueRef) -> LLVMValueRef {
        use self::llvm_sys::LLVMTypeKind;
        let ty = LLVMTypeOf(value);
        assert!(LLVMGetTypeKind(ty) == LLVMTypeKind::LLVMIntegerTypeKind);
        let bits = LLVMGetIntTypeWidth(ty);
        let one = LLVMConstInt(ty, 1 as c_ulonglong, 0);
        let mut result = LLVMBuildSub(builder, value, one, c_str!(""));
        let mut shift_amount = 1;
        while shift_amount < bits {
            let amount = LLVMConstInt(ty, u64::from(shift_amount), 0);
            let shift = LLVMBuildAShr(builder, result, amount, c_str!(""));
            result = LLVMBuildOr(builder, result, shift, c_str!(""));
            shift_amount *= 2;
        }
        LLVMBuildAdd(builder, result, one, c_str!(""))
    }

    /// Convert a boolean to an `i1`.
    ///
    /// If the boolean is a vector, a vector of `i1` is produced.
    unsafe fn bool_to_i1(&self, builder: LLVMBuilderRef, v: LLVMValueRef) -> LLVMValueRef {
        let type_kind = LLVMGetTypeKind(LLVMTypeOf(v));
        let mut zero = self.bool(false);
        if type_kind == llvm_sys::LLVMTypeKind::LLVMVectorTypeKind {
            let mut zeroes = [zero; LLVM_VECTOR_WIDTH as usize];
            zero = LLVMConstVector(zeroes.as_mut_ptr(), zeroes.len() as u32);
        }
        LLVMBuildICmp(
            builder,
            llvm_sys::LLVMIntPredicate::LLVMIntNE,
            v,
            zero,
            c_str!(""),
        )
    }

    /// Convert an `i1` to a boolean.
    ///
    /// If the input is a vector, a vector of `boolean` is produced.
    unsafe fn i1_to_bool(&self, builder: LLVMBuilderRef, v: LLVMValueRef) -> LLVMValueRef {
        let type_kind = LLVMGetTypeKind(LLVMTypeOf(v));
        if type_kind == llvm_sys::LLVMTypeKind::LLVMVectorTypeKind {
            LLVMBuildZExt(
                builder,
                v,
                LLVMVectorType(self.bool_type(), LLVM_VECTOR_WIDTH),
                c_str!(""),
            )
        } else {
            LLVMBuildZExt(builder, v, self.bool_type(), c_str!(""))
        }
    }

    /// Booleans are represented as `i8`.
    ///
    /// For instructions that require `i1` (e.g, conditional branching or select), the caller
    /// should truncate this type to `i1_type` manually. The distinction between booleans and `i1`
    /// is that boolean types are "externally visible", whereas `i1`s only appear in internal code.
    unsafe fn c_bool_type(&self) -> String {
        c_bool_type(self.ccontext())
    }

    unsafe fn c_i1_type(&self) -> String {
        c_i1_type(self.ccontext())
    }

    unsafe fn c_i8_type(&self) -> String {
        c_i8_type(self.ccontext())
    }

    unsafe fn c_u8_type(&self) -> String {
        c_u8_type(self.ccontext())
    }

    unsafe fn c_i16_type(&self) -> String {
        c_i16_type(self.ccontext())
    }

    unsafe fn c_u16_type(&self) -> String {
        c_u16_type(self.ccontext())
    }

    unsafe fn c_i32_type(&self) -> String {
        c_i32_type(self.ccontext())
    }

    unsafe fn c_u32_type(&self) -> String {
        c_u32_type(self.ccontext())
    }

    unsafe fn c_i64_type(&self) -> String {
        c_i64_type(self.ccontext())
    }

    unsafe fn c_u64_type(&self) -> String {
        c_u64_type(self.ccontext())
    }

    unsafe fn c_f32_type(&self) -> String {
        c_f32_type(self.ccontext())
    }

    unsafe fn c_f64_type(&self) -> String {
        c_f64_type(self.ccontext())
    }

    unsafe fn c_void_type(&self) -> String {
        c_void_type(self.ccontext())
    }

    unsafe fn c_void_pointer_type(&self) -> String {
        c_void_pointer_type(self.ccontext())
    }

    unsafe fn c_pointer_type(&self, ty: &str) -> String {
        c_pointer_type(self.ccontext(), ty)
    }

    unsafe fn c_type_of(&self, ty: &str) -> String {
        c_type_of(self.ccontext(), ty)
    }

    unsafe fn c_simd_type(&self, ty: &str, size: u32) -> String {
        c_simd_type(self.ccontext(), ty, size)
    }

    unsafe fn c_run_handle_type(&self) -> String {
        c_run_handle_type(self.ccontext())
    }

    unsafe fn c_bool<T: Into<bool>>(&self, v: T) -> String {
        { if v.into() { "1" } else { "0" } }.to_string()
    }

    unsafe fn c_i1<T: Into<bool>>(&self, v: T) -> String {
        { if v.into() { "1" } else { "0" } }.to_string()
    }

    unsafe fn c_i8<T: Into<i8>>(&self, v: T) -> String {
        v.into().to_string()
    }

    unsafe fn c_u8(&self, v: u8) -> String {
        v.to_string()
    }

    unsafe fn c_i16(&self, v: i16) -> String {
        v.to_string()
    }

    unsafe fn c_u16(&self, v: u16) -> String {
        v.to_string()
    }

    unsafe fn c_i32(&self, v: i32) -> String {
        v.to_string()
    }

    unsafe fn c_u32(&self, v: u32) -> String {
        v.to_string()
    }

    unsafe fn c_i64(&self, v: i64) -> String {
        v.to_string()
    }

    unsafe fn c_u64(&self, v: u64) -> String {
        v.to_string()
    }

    unsafe fn c_f32(&self, v: f32) -> String {
        v.to_string()
    }

    unsafe fn c_f64(&self, v: f64) -> String {
        v.to_string()
    }

    unsafe fn c_null_ptr(&self, ty: &str) -> String {
        format!("(({})0)", ty)
    }

    fn c_get_param(&self, index: usize) -> String {
        format!("p{}", index)
    }

    fn c_get_run(&self) -> &'static str {
        "run"
    }

    /// Helper functions to treate arguments.
    fn c_call_args(&mut self, args: &[String]) -> String {
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

    unsafe fn c_define_args(&mut self, arg_tys: &[String]) -> String {
        let mut args_line = String::new();
        let mut last_arg: &str = "";
        let mut last_i = 0;
        for (i, arg) in arg_tys.iter().enumerate() {
            if i != 0 {
                args_line = format!("{}{} {}, ", args_line, last_arg, self.c_get_param(last_i));
            }
            last_arg = arg;
            last_i = i;
        }
        if self.c_run_handle_type() == last_arg {
            // Write "run" as parameter if the type of last arg is "RunHandle*".
            format!("{}{} run", args_line, last_arg)
        } else {
            format!("{}{} {}", args_line, last_arg, self.c_get_param(last_i))
        }
    }

    /// Booleans are represented as `i8`.
    ///
    /// For instructions that require `i1` (e.g, conditional branching or select), the caller
    /// should truncate this type to `i1_type` manually. The distinction between booleans and `i1`
    /// is that boolean types are "externally visible", whereas `i1`s only appear in internal code.
    unsafe fn bool_type(&self) -> LLVMTypeRef {
        LLVMInt8TypeInContext(self.context())
    }

    unsafe fn i1_type(&self) -> LLVMTypeRef {
        LLVMInt1TypeInContext(self.context())
    }

    unsafe fn i8_type(&self) -> LLVMTypeRef {
        LLVMInt8TypeInContext(self.context())
    }

    unsafe fn u8_type(&self) -> LLVMTypeRef {
        LLVMInt8TypeInContext(self.context())
    }

    unsafe fn i16_type(&self) -> LLVMTypeRef {
        LLVMInt16TypeInContext(self.context())
    }

    unsafe fn u16_type(&self) -> LLVMTypeRef {
        LLVMInt16TypeInContext(self.context())
    }

    unsafe fn i32_type(&self) -> LLVMTypeRef {
        LLVMInt32TypeInContext(self.context())
    }

    unsafe fn u32_type(&self) -> LLVMTypeRef {
        LLVMInt32TypeInContext(self.context())
    }

    unsafe fn i64_type(&self) -> LLVMTypeRef {
        LLVMInt64TypeInContext(self.context())
    }

    unsafe fn u64_type(&self) -> LLVMTypeRef {
        LLVMInt64TypeInContext(self.context())
    }

    unsafe fn f32_type(&self) -> LLVMTypeRef {
        LLVMFloatTypeInContext(self.context())
    }

    unsafe fn f64_type(&self) -> LLVMTypeRef {
        LLVMDoubleTypeInContext(self.context())
    }

    unsafe fn void_type(&self) -> LLVMTypeRef {
        LLVMVoidTypeInContext(self.context())
    }

    unsafe fn void_pointer_type(&self) -> LLVMTypeRef {
        LLVMPointerType(self.i8_type(), 0)
    }

    unsafe fn run_handle_type(&self) -> LLVMTypeRef {
        let mut ty = LLVMGetTypeByName(self.module(), RUN_HANDLE_NAME.as_ptr());
        if ty.is_null() {
            let mut layout = [self.i8_type()];
            ty = LLVMStructCreateNamed(self.context(), RUN_HANDLE_NAME.as_ptr());
            LLVMStructSetBody(ty, layout.as_mut_ptr(), layout.len() as u32, 0);
        }
        LLVMPointerType(ty, 0)
    }

    unsafe fn bool<T: Into<bool>>(&self, v: T) -> LLVMValueRef {
        LLVMConstInt(self.bool_type(), if v.into() { 1 } else { 0 }, 0)
    }

    unsafe fn i1<T: Into<bool>>(&self, v: T) -> LLVMValueRef {
        LLVMConstInt(self.i1_type(), if v.into() { 1 } else { 0 }, 0)
    }

    unsafe fn i8<T: Into<i8>>(&self, v: T) -> LLVMValueRef {
        LLVMConstInt(self.i8_type(), v.into() as c_ulonglong, 1)
    }

    unsafe fn u8(&self, v: u8) -> LLVMValueRef {
        LLVMConstInt(self.u8_type(), u64::from(v), 0)
    }

    unsafe fn i16(&self, v: i16) -> LLVMValueRef {
        LLVMConstInt(self.i16_type(), v as u64, 1)
    }

    unsafe fn u16(&self, v: u16) -> LLVMValueRef {
        LLVMConstInt(self.u16_type(), u64::from(v), 0)
    }

    unsafe fn i32(&self, v: i32) -> LLVMValueRef {
        LLVMConstInt(self.i32_type(), v as u64, 1)
    }

    unsafe fn u32(&self, v: u32) -> LLVMValueRef {
        LLVMConstInt(self.u32_type(), u64::from(v), 0)
    }

    unsafe fn i64(&self, v: i64) -> LLVMValueRef {
        LLVMConstInt(self.i64_type(), v as u64, 1)
    }

    unsafe fn u64(&self, v: u64) -> LLVMValueRef {
        LLVMConstInt(self.u64_type(), v, 0)
    }

    unsafe fn f32(&self, v: f32) -> LLVMValueRef {
        LLVMConstReal(self.f32_type(), f64::from(v))
    }

    unsafe fn f64(&self, v: f64) -> LLVMValueRef {
        LLVMConstReal(self.f64_type(), v)
    }

    unsafe fn null_ptr(&self, ty: LLVMTypeRef) -> LLVMValueRef {
        LLVMConstPointerNull(LLVMPointerType(ty, 0))
    }
}

impl CodeGenExt for CGenerator {
    fn module(&self) -> LLVMModuleRef {
        self.module
    }

    fn context(&self) -> LLVMContextRef {
        self.context
    }

    fn ccontext(&self) -> CContextRef {
        self.ccontext
    }
}

impl CGenerator {
    /// Initialize a new CGenerator.
    unsafe fn new(conf: ParsedConf) -> WeldResult<CGenerator> {
        let context = LLVMContextCreate();
        let module = LLVMModuleCreateWithNameInContext(c_str!("main"), context);
        let ccontext_data = Box::new(CContext {
            basic_types: FnvHashMap::default(),
            simd_types: FnvHashMap::default(),
            i1_defined: false,
            input_arg_defined: false,
            output_arg_defined: false,
            run_handle_defined: false,
            var_ids: IdGenerator::new("t"),
            prelude_code: CodeBuilder::new(),
            body_code: CodeBuilder::new(),
        });
        let ccontext: *mut CContext = Box::into_raw(ccontext_data);

        // These methods *must* be called before using any of the `CodeGenExt` extension methods.
        compile::init();
        compile::set_triple_and_layout(module)?;

        // Adds the default intrinsic definitions.
        let intrinsics = intrinsic::Intrinsics::defaults(context, module, ccontext);

        let target = target::Target::from_llvm_strings(
            llvm_exts::PROCESS_TRIPLE.to_str().unwrap(),
            llvm_exts::HOST_CPU_NAME.to_str().unwrap(),
            llvm_exts::HOST_CPU_FEATURES.to_str().unwrap(),
        )?;

        debug!("CGenerator features: {}", target.features);

        Ok(CGenerator {
            conf,
            context,
            module,
            ccontext,
            target,
            functions: FnvHashMap::default(),
            c_functions: FnvHashMap::default(),
            vectors: FnvHashMap::default(),
            vector_index: 0,
            mergers: FnvHashMap::default(),
            appenders: FnvHashMap::default(),
            appender_index: 0,
            dictionaries: FnvHashMap::default(),
            strings: FnvHashMap::default(),
            eq_fns: FnvHashMap::default(),
            opaque_eq_fns: FnvHashMap::default(),
            cmp_fns: FnvHashMap::default(),
            hash_fns: FnvHashMap::default(),
            serialize_fns: FnvHashMap::default(),
            deserialize_fns: FnvHashMap::default(),
            struct_names: FnvHashMap::default(),
            c_struct_names: FnvHashMap::default(),
            struct_index: 0,
            intrinsics,
        })
    }

    /// Generate code for an SIR program.
    unsafe fn generate(conf: ParsedConf, program: &SirProgram) -> WeldResult<CGenerator> {
        let mut gen = CGenerator::new(conf)?;

        // Declare each function first to create a reference to it. Loop body functions are only
        // called by their ParallelForData terminators, so those are generated on-the-fly during
        // loop code generation.
        for func in program.funcs.iter().filter(|f| !f.loop_body) {
            gen.declare_sir_function(func)?;
        }

        // Generate each non-loop body function in turn. Loop body functions are constructed when
        // the For loop terminator is generated, with the loop control flow injected into the function.
        for func in program.funcs.iter().filter(|f| !f.loop_body) {
            gen.gen_sir_function(program, func)?;
        }

        // Generates a callable entry function in the module.
        gen.gen_entry(program)?;
        Ok(gen)
    }

    /// Generates a global string literal and returns a `i8*` to it.
    unsafe fn gen_global_string(
        &mut self,
        builder: LLVMBuilderRef,
        string: CString,
    ) -> LLVMValueRef {
        let ptr = string.as_ptr();
        *self
            .strings
            .entry(string)
            .or_insert_with(|| LLVMBuildGlobalStringPtr(builder, ptr, c_str!("")))
    }

    /// Generates a print call with the given string.
    unsafe fn gen_print(
        &mut self,
        builder: LLVMBuilderRef,
        run: LLVMValueRef,
        string: CString,
    ) -> WeldResult<()> {
        let string = self.gen_global_string(builder, string);
        let pointer = LLVMConstBitCast(string, LLVMPointerType(self.i8_type(), 0));
        let _ = self.intrinsics.call_weld_run_print(builder, run, pointer);
        Ok(())
    }

    /// Generates the entry point to the Weld program.
    ///
    /// The entry function takes an `i64` and returns an `i64`. Both represent pointers that
    /// point to a `WeldInputArgs` and `WeldOutputArgs` respectively.
    unsafe fn gen_entry(&mut self, program: &SirProgram) -> WeldResult<()> {
        use crate::ast::Type::Struct;

        // Declare types.
        // for C
        let c_input_type = WeldInputArgs::c_type(self.ccontext);
        let c_output_type = WeldOutputArgs::c_type(self.ccontext);
        // for LLVM
        let input_type = WeldInputArgs::llvm_type(self.context);
        let output_type = WeldOutputArgs::llvm_type(self.context);

        // Declare run function.
        // for C
        (*self.ccontext()).body_code.add(format!("i64 {}(i64 args)\n{{", self.conf.llvm.run_func_name));
        // for LLVM
        let name = CString::new(self.conf.llvm.run_func_name.as_bytes()).unwrap();
        let func_ty = LLVMFunctionType(self.i64_type(), [self.i64_type()].as_mut_ptr(), 1, 0);
        let function = LLVMAddFunction(self.module, name.as_ptr(), func_ty);

        // Add the default attributes to all functions.
        llvm_exts::LLVMExtAddDefaultAttrs(self.context(), function);

        // This function is the global entry point into the program, so we must give it externally
        // visible linkage.
        LLVMSetLinkage(function, LLVMLinkage::LLVMExternalLinkage);

        let builder = LLVMCreateBuilderInContext(self.context);
        let entry_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));
        let init_run_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));
        let get_arg_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!(""));

        // Generate codes for entry block
        // for C
        (*self.ccontext()).body_code.add(format!("{input}* input = ({input}*)args;", input=c_input_type));
        // for LLVM
        LLVMPositionBuilderAtEnd(builder, entry_block);
        let argument = LLVMGetParam(function, 0);
        let pointer = LLVMBuildIntToPtr(
            builder,
            argument,
            LLVMPointerType(input_type, 0),
            c_str!(""),
        );

        // Check whether we already have an existing run.
        // for C
        (*self.ccontext()).body_code.add(format!("{handle} run = ({handle})input->run;", handle=self.c_run_handle_type()));
        (*self.ccontext()).body_code.add("if (run == 0) {");
        
        // for LLVM
        let run_pointer =
            LLVMBuildStructGEP(builder, pointer, WeldInputArgs::run_index(), c_str!(""));
        let run_pointer = self.load(builder, run_pointer)?;
        let run_arg = LLVMBuildIntToPtr(builder, run_pointer, self.run_handle_type(), c_str!(""));
        let null = LLVMConstNull(self.run_handle_type());
        let null_check = LLVMBuildICmp(
            builder,
            llvm_sys::LLVMIntPredicate::LLVMIntEQ,
            run_arg,
            null,
            c_str!(""),
        );
        LLVMBuildCondBr(builder, null_check, init_run_block, get_arg_block);

        // Generate codes for init_run block.
        // for C
        self.intrinsics.c_call_weld_run_init(
            &mut (*self.ccontext()).body_code,
            "input->nworkers",
            "input->memlimit",
            Some("run".to_string()),
        );
        (*self.ccontext()).body_code.add("}");
        // for LLVM
        LLVMPositionBuilderAtEnd(builder, init_run_block);
        let nworkers_pointer = LLVMBuildStructGEP(
            builder,
            pointer,
            WeldInputArgs::nworkers_index(),
            c_str!("nworkers"),
        );
        let nworkers = self.load(builder, nworkers_pointer)?;
        let memlimit_pointer = LLVMBuildStructGEP(
            builder,
            pointer,
            WeldInputArgs::memlimit_index(),
            c_str!("memlimit"),
        );
        let memlimit = self.load(builder, memlimit_pointer)?;
        let run_new = self
            .intrinsics
            .call_weld_run_init(builder, nworkers, memlimit, None);
        LLVMBuildBr(builder, get_arg_block);

        // Generate codes for get_arg block.
        // for C
        let arg_ty = &Struct(program.top_params.iter().map(|p| p.ty.clone()).collect());
        (*self.ccontext()).body_code.add(format!("{ty}* arg = ({ty}*)(input->input);", ty=self.c_type(arg_ty)?));

        // for LLVM
        LLVMPositionBuilderAtEnd(builder, get_arg_block);
        let run = LLVMBuildPhi(builder, self.run_handle_type(), c_str!(""));
        let mut blocks = [entry_block, init_run_block];
        let mut values = [run_arg, run_new];
        LLVMAddIncoming(
            run,
            values.as_mut_ptr(),
            blocks.as_mut_ptr(),
            blocks.len() as u32,
        );

        let arg_pointer = LLVMBuildStructGEP(
            builder,
            pointer,
            WeldInputArgs::input_index(),
            c_str!("argptr"),
        );
        // Still a pointer, but now as an integer.
        let arg_pointer = self.load(builder, arg_pointer)?;
        // The first SIR function is the entry point.
        let arg_ty = &Struct(program.top_params.iter().map(|p| p.ty.clone()).collect());
        let llvm_arg_ty = self.llvm_type(arg_ty)?;
        let arg_struct_pointer = LLVMBuildIntToPtr(
            builder,
            arg_pointer,
            LLVMPointerType(llvm_arg_ty, 0),
            c_str!("arg"),
        );

        // Function arguments are sorted by symbol name - arrange the inputs in the proper order.
        let mut params: Vec<(&Symbol, u32)> = program
            .top_params
            .iter()
            .enumerate()
            .map(|(i, p)| (&p.name, i as u32))
            .collect();

        params.sort();

        // Prepare entry_function's arguments list.
        // for C
        let mut c_func_args = vec![];
        for (_, i) in params.iter() {
            c_func_args.push(format!("arg->f{}", i));
        }
        // for LLVM
        let mut func_args = vec![];
        for (_, i) in params.iter() {
            let pointer = LLVMBuildStructGEP(builder, arg_struct_pointer, *i, c_str!("param"));
            let value = self.load(builder, pointer)?;
            func_args.push(value);
        }
        // Push the run handle.
        // for C
        c_func_args.push("run".to_string());
        // for LLVM
        func_args.push(run);

        // Run the Weld program.
        // for C
        let args_line = self.c_call_args(&c_func_args);
        self.c_call_sir_function(
            &mut (*self.ccontext()).body_code,
            &program.funcs[0],
            &args_line,
            None,
        )?;
        // for LLVM
        let entry_function = self.functions[&program.funcs[0].id];
        let inst = LLVMBuildCall(
            builder,
            entry_function,
            func_args.as_mut_ptr(),
            func_args.len() as u32,
            c_str!(""),
        );
        LLVMSetInstructionCallConv(inst, SIR_FUNC_CALL_CONV);

        // Get the results, errno, and run.
        // for C
        let res = self.intrinsics.c_call_weld_run_get_result(
            &mut (*self.ccontext()).body_code,
            "run",
            None,
        );
        let c_result = (*self.ccontext()).var_ids.next();
        (*self.ccontext()).body_code.add(format!(
            "{i64} {result} = ({i64}){res};",
            i64=self.c_i64_type(), result=c_result, res=res));
        let c_errno = self.intrinsics.c_call_weld_run_get_errno(
            &mut (*self.ccontext()).body_code,
            "run",
            None,
        );
        let c_run_int = (*self.ccontext()).var_ids.next();
        (*self.ccontext()).body_code.add(format!(
            "{i64} {run_int} = ({i64})run;",
            i64=self.c_i64_type(), run_int=c_run_int));
        // for LLVM
        let result = self.intrinsics.call_weld_run_get_result(builder, run, None);
        let result = LLVMBuildPtrToInt(builder, result, self.i64_type(), c_str!("result"));
        let errno = self
            .intrinsics
            .call_weld_run_get_errno(builder, run, Some(c_str!("errno")));
        let run_int = LLVMBuildPtrToInt(builder, run, self.i64_type(), c_str!("run"));

        // Generate Output
        // for C
        let return_size = self.c_size_of(c_output_type);
        let return_pointer = self
            .intrinsics
            .c_call_weld_run_malloc(
                &mut (*self.ccontext()).body_code,
                "run",
                &return_size,
                None,
            );
        (*self.ccontext()).body_code.add(format!("{output}* output = ({output}*){ptr};", output=c_output_type, ptr=return_pointer));
        (*self.ccontext()).body_code.add(format!(
            "output->output = {};", c_result));
        (*self.ccontext()).body_code.add(format!(
            "output->run = {};", c_run_int));
        (*self.ccontext()).body_code.add(format!(
            "output->errno = {};", c_errno));
        // for LLVM
        let mut output = LLVMGetUndef(output_type);
        output = LLVMBuildInsertValue(
            builder,
            output,
            result,
            WeldOutputArgs::output_index(),
            c_str!(""),
        );
        output = LLVMBuildInsertValue(
            builder,
            output,
            run_int,
            WeldOutputArgs::run_index(),
            c_str!(""),
        );
        output = LLVMBuildInsertValue(
            builder,
            output,
            errno,
            WeldOutputArgs::errno_index(),
            c_str!(""),
        );

        let return_size = self.size_of(output_type);
        let return_pointer = self
            .intrinsics
            .call_weld_run_malloc(builder, run, return_size, None);
        let return_pointer = LLVMBuildBitCast(
            builder,
            return_pointer,
            LLVMPointerType(output_type, 0),
            c_str!(""),
        );

        // Generate Return instruction.
        // for C
        (*self.ccontext()).body_code.add(format!(
            "return ({})output;\n}}", self.c_i64_type()));
        // for LLVM
        LLVMBuildStore(builder, output, return_pointer);
        let return_value = LLVMBuildPtrToInt(builder, return_pointer, self.i64_type(), c_str!(""));
        LLVMBuildRet(builder, return_value);

        LLVMDisposeBuilder(builder);
        Ok(())
    }

    /// Build the list of argument types for an SIR function.
    unsafe fn argument_types(&mut self, func: &SirFunction) -> WeldResult<Vec<LLVMTypeRef>> {
        let mut types = vec![];
        for (_, ty) in func.params.iter() {
            types.push(self.llvm_type(ty)?);
        }
        Ok(types)
    }
    unsafe fn c_argument_types(&mut self, func: &SirFunction) -> WeldResult<Vec<String>> {
        let mut types = vec![];
        for (_, ty) in func.params.iter() {
            types.push(self.c_type(ty)?);
        }
        Ok(types)
    }

    /// Declare a function in the SIR module and track its reference.
    ///
    /// In addition to the SIR-defined parameters, the runtime adds an `i8*` pointer as the last
    /// argument to each function, representing the handle to the run data. This handle is always
    /// guaranteed to be the last argument.
    ///
    /// This method only defines functions and does not generate code for the function.
    unsafe fn declare_sir_function(&mut self, func: &SirFunction) -> WeldResult<()> {
        // for C
        let function = format!("f{}", func.id);
        self.c_functions.insert(func.id, function);

        // for LLVM
        let mut arg_tys = self.argument_types(func)?;
        arg_tys.push(self.run_handle_type());
        let ret_ty = self.llvm_type(&func.return_type)?;
        let func_ty = LLVMFunctionType(ret_ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        let name = CString::new(format!("f{}", func.id)).unwrap();
        let function = LLVMAddFunction(self.module, name.as_ptr(), func_ty);

        // Add attributes, set linkage, etc.
        llvm_exts::LLVMExtAddDefaultAttrs(self.context(), function);
        LLVMSetLinkage(function, LLVMLinkage::LLVMPrivateLinkage);
        LLVMSetFunctionCallConv(function, SIR_FUNC_CALL_CONV);

        self.functions.insert(func.id, function);
        Ok(())
    }

    pub unsafe fn c_call_sir_function(
        &mut self,
        builder: &mut CodeBuilder,
        func: &SirFunction,
        args_line: &str,
        result: Option<String>,
    ) -> WeldResult<String> {
        if let Some(res) = result {
            let fun = &self.c_functions[&func.id];
            builder.add(format!(
                "{} = {}({});", res, fun, args_line));
            Ok(res)
        } else {
            let res = (*self.ccontext()).var_ids.next();
            let ret_ty = self.c_type(&func.return_type)?.to_string();
            let fun = &self.c_functions[&func.id];
            builder.add(format!(
                "{} {} = {}({});", ret_ty, res, fun, args_line));
            Ok(res)
        }
    }

    /// Generates the Allocas for a function.
    ///
    /// The allocas should generally be generated in the entry block of the function. The caller
    /// should ensure that the context builder is appropriately positioned.
    unsafe fn gen_allocas(&mut self, context: &mut FunctionContext<'_>) -> WeldResult<()> {
        // Add the function parameters, which are stored in alloca'd variables. The
        // function parameters are always enumerated alphabetically sorted by symbol name.
        for (symbol, ty) in context.sir_function.params.iter() {
            // for C
            context.body.add(format!("{ty} {name};",
                             ty=self.c_type(ty)?,
                             name=symbol));
            // for LLVM
            let name = CString::new(symbol.to_string()).unwrap();
            let value = LLVMBuildAlloca(context.builder, self.llvm_type(ty)?, name.as_ptr());
            context.symbols.insert(symbol.clone(), value);
        }

        // alloca the local variables.
        for (symbol, ty) in context.sir_function.locals.iter() {
            // for C
            context.body.add(format!("{ty} {name};",
                             ty=self.c_type(ty)?,
                             name=symbol));
            // for LLVM
            let name = CString::new(symbol.to_string()).unwrap();
            let value = LLVMBuildAlloca(context.builder, self.llvm_type(ty)?, name.as_ptr());
            context.symbols.insert(symbol.clone(), value);
        }
        Ok(())
    }

    /// Generates code to store function parameters in alloca'd variables.
    unsafe fn gen_store_parameters(&mut self, context: &mut FunctionContext<'_>) -> WeldResult<()> {
        // Store the parameter values in the alloca'd symbols.
        for (i, (symbol, _)) in context.sir_function.params.iter().enumerate() {
            // for C
            context.body.add(format!("{} = p{};", symbol, i));
            // for LLVM
            let pointer = context.get_value(symbol)?;
            let value = LLVMGetParam(context.llvm_function, i as u32);
            LLVMBuildStore(context.builder, value, pointer);
        }
        Ok(())
    }

    /// Generates code to define each basic block in the function.
    ///
    /// This function does not actually generate the basic block code: it only adds the basic
    /// blocks to the context so they can be forward referenced if necessary.
    unsafe fn gen_basic_block_defs(&mut self, context: &mut FunctionContext<'_>) -> WeldResult<()> {
        for bb in context.sir_function.blocks.iter() {
            let name = CString::new(format!("b{}", bb.id)).unwrap();
            let block =
                LLVMAppendBasicBlockInContext(self.context, context.llvm_function, name.as_ptr());
            context.blocks.insert(bb.id, block);
        }
        Ok(())
    }

    /// Generate code for a defined SIR `function` from `program`.
    ///
    /// This function specifically generates code for non-loop body functions.
    unsafe fn gen_sir_function(
        &mut self,
        program: &SirProgram,
        func: &SirFunction,
    ) -> WeldResult<()> {
        let function = self.functions[&func.id];
        // + 1 to account for the run handle.
        if LLVMCountParams(function) != (1 + func.params.len()) as u32 {
            unreachable!()
        }

        // Create a context for the function.
        let context = &mut FunctionContext::new(self.context, program, func, function);

        // Generates function definition.
        // for C
        let mut arg_tys = self.c_argument_types(func)?;
        arg_tys.push(self.c_run_handle_type());
        let args_line = self.c_define_args(&arg_tys);
        let ret_ty = self.c_type(&func.return_type)?.to_string();
        let function = &self.c_functions[&func.id];
        context.body.add(format!("{ret_ty} {fun}({args})",
            ret_ty=ret_ty,
            fun=function,
            args=args_line,
        ));
        context.body.add("{");

        // Generates function body.
        // for LLVM
        // Create the entry basic block, where we define alloca'd variables.
        let entry_bb =
            LLVMAppendBasicBlockInContext(self.context, context.llvm_function, c_str!(""));
        LLVMPositionBuilderAtEnd(context.builder, entry_bb);

        // for C and LLVM
        self.gen_allocas(context)?;
        self.gen_store_parameters(context)?;
        self.gen_basic_block_defs(context)?;

        // Jump from locals to the first basic block.
        // for C
        context.body.add(format!(
            "goto {};",
            context.c_get_block(func.blocks[0].id),
        ));
        // for LLVM
        LLVMPositionBuilderAtEnd(context.builder, entry_bb);
        LLVMBuildBr(context.builder, context.get_block(func.blocks[0].id)?);

        // Generate code for the basic blocks in order.
        for bb in func.blocks.iter() {
            // for C
            context.body.add(format!(
                "{}:",
                context.c_get_block(func.blocks[0].id),
            ));
            // for LLVM
            LLVMPositionBuilderAtEnd(context.builder, context.get_block(bb.id)?);
            for statement in bb.statements.iter() {
                self.gen_statement(context, statement)?;
            }
            self.gen_terminator(context, &bb, None)?;
        }
        // Write the function defined in C into the program.
        context.body.add("}");
        (*self.ccontext()).body_code.add(context.body.result());
        Ok(())
    }

    /// Generate code for a single SIR statement.
    ///
    /// The code is generated at the position specified by the function context.
    unsafe fn gen_statement(
        &mut self,
        context: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()> {
        use crate::ast::Type::*;
        use crate::sir::StatementKind::*;
        let output = &statement
            .output
            .clone()
            .unwrap_or_else(|| Symbol::new("unused", 0));

        if self.conf.trace_run {
            self.gen_print(
                context.builder,
                context.get_run(),
                CString::new(format!("{}", statement)).unwrap(),
            )?;
        }

        match statement.kind {
            Assign(ref value) => {
                // for C
                context.body.add(format!(
                    "{} = {};",
                    output,
                    value,
                ));

                // for LLVM
                let loaded = self.load(context.builder, context.get_value(value)?)?;
                LLVMBuildStore(context.builder, loaded, context.get_value(output)?);
                Ok(())
            }
            AssignLiteral(_) => {
                // for C and LLVM
                use self::numeric::NumericExpressionGen;
                self.gen_assign_literal(context, statement)
            }
            BinOp { .. } => {
                // for C and LLVM
                use self::numeric::NumericExpressionGen;
                self.gen_binop(context, statement)
            }
            Broadcast(ref child) => {
                // for C
                context.body.add("#error Broadcast is not implemented yet");

                // for LLVM
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let ty = self.llvm_type(context.sir_function.symbol_type(output)?)?;
                let mut result = LLVMGetUndef(ty);
                for i in 0..LLVM_VECTOR_WIDTH {
                    result = LLVMBuildInsertElement(
                        context.builder,
                        result,
                        child_value,
                        self.i32(i as i32),
                        c_str!(""),
                    );
                }
                LLVMBuildStore(context.builder, result, output_pointer);
                Ok(())
            }
            Cast(_, _) => {
                // for C
                context.body.add("#error Cast is not implemented yet");

                // for LLVM
                use self::numeric::NumericExpressionGen;
                self.gen_cast(context, statement)
            }
            CUDF {
                ref symbol_name,
                ref args,
            } => {
                // for C
                context.body.add("#error CUDF is not implemented yet");

                // for LLVM
                let output_pointer = context.get_value(output)?;
                let return_ty = self.llvm_type(context.sir_function.symbol_type(output)?)?;
                let c_return_ty = self.c_type(context.sir_function.symbol_type(output)?)?;
                let mut arg_tys = vec![];
                let mut c_arg_tys = vec![];

                // A CUDF with declaration Name[R](T1, T2, T3) has a signature `void Name(T1, T2, T3, R)`.
                for arg in args.iter() {
                    arg_tys.push(LLVMPointerType(
                        self.llvm_type(context.sir_function.symbol_type(arg)?)?,
                        0,
                    ));
                    let c_ty = self.c_type(context.sir_function.symbol_type(arg)?)?;
                    c_arg_tys.push(self.c_pointer_type(&c_ty));
                }
                arg_tys.push(LLVMPointerType(return_ty, 0));
                c_arg_tys.push(self.c_pointer_type(&c_return_ty));

                let fn_ret_ty = self.void_type();
                let c_fn_ret_ty = self.c_void_type();
                let v8 : Vec<&str> = c_arg_tys.iter().map(AsRef::as_ref).collect();
                self.intrinsics.add(symbol_name, symbol_name, fn_ret_ty, &c_fn_ret_ty, &mut arg_tys, &v8);

                let mut arg_values = vec![];
                for arg in args.iter() {
                    arg_values.push(context.get_value(arg)?);
                }

                arg_values.push(output_pointer);
                let _ = self
                    .intrinsics
                    .call(context.builder, symbol_name, &mut arg_values)?;

                Ok(())
            }
            Deserialize(_) => {
                // for C
                context.body.add("#error Desirialize is not implemented yet");

                // for LLVM
                use self::serde::SerDeGen;
                self.gen_deserialize(context, statement)
            }
            GetField { ref value, index } => {
                // for C
                context.body.add(format!(
                    "{} = {}->f{};",
                    context.c_get_value(output)?,
                    context.c_get_value(value)?,
                    index,
                ));

                // for LLVM
                let output_pointer = context.get_value(output)?;
                let value_pointer = context.get_value(value)?;
                let elem_pointer =
                    LLVMBuildStructGEP(context.builder, value_pointer, index, c_str!(""));
                let elem = self.load(context.builder, elem_pointer)?;
                LLVMBuildStore(context.builder, elem, output_pointer);
                Ok(())
            }
            KeyExists { ref child, ref key } => {
                // for C
                context.body.add("#error KeyExists is not implemented yet");

                // for LLVM
                use self::hash::GenHash;
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let key_pointer = context.get_value(key)?;
                let child_type = context.sir_function.symbol_type(child)?;
                let hash = if let Dict(ref key, _) = *child_type {
                    self.gen_hash(key, context.builder, key_pointer, None)?
                } else {
                    unreachable!()
                };

                let result = {
                    let methods = self.dictionaries.get_mut(child_type).unwrap();
                    methods.gen_key_exists(
                        context.builder,
                        child_value,
                        context.get_value(key)?,
                        hash,
                    )?
                };
                LLVMBuildStore(context.builder, result, output_pointer);
                Ok(())
            }
            Length(ref child) => {
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let child_type = context.sir_function.symbol_type(child)?;
                if let Vector(ref elem_type) = *child_type {
                    let methods = self.vectors.get_mut(elem_type).unwrap();
                    // for C
                    context.body.add(format!(
                        "{} = {};",
                        output,
                        methods.c_gen_size(context.builder,
                                           &context.c_get_value(child)?)?,
                    ));

                    // for LLVM
                    let result = methods.gen_size(context.builder, child_value)?;
                    LLVMBuildStore(context.builder, result, output_pointer);
                    Ok(())
                } else if let Dict(_, _) = *child_type {
                    // for C
                    context.body.add("#error Length for dict is not implemented yet");

                    // for LLVM
                    let pointer = {
                        let methods = self.dictionaries.get_mut(child_type).unwrap();
                        methods.gen_size(context.builder, child_value)?
                    };
                    let result = self.load(context.builder, pointer)?;
                    LLVMBuildStore(context.builder, result, output_pointer);
                    Ok(())
                } else {
                    unreachable!()
                }
            }
            Lookup {
                ref child,
                ref index,
            } => {
                // for C
                context.body.add("#error Lookup is not implemented yet");

                // for LLVM
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let child_type = context.sir_function.symbol_type(child)?;
                if let Vector(_) = *child_type {
                    use self::vector::VectorExt;
                    let index_value = self.load(context.builder, context.get_value(index)?)?;
                    let pointer =
                        self.gen_at(context.builder, child_type, child_value, index_value)?;
                    let result = self.load(context.builder, pointer)?;
                    LLVMBuildStore(context.builder, result, output_pointer);
                    Ok(())
                } else if let Dict(ref key, _) = *child_type {
                    use self::hash::GenHash;
                    let hash =
                        self.gen_hash(key, context.builder, context.get_value(index)?, None)?;
                    let result = {
                        let methods = self.dictionaries.get_mut(child_type).unwrap();
                        let slot = methods.gen_lookup(
                            context.builder,
                            &mut self.intrinsics,
                            child_value,
                            context.get_value(index)?,
                            hash,
                            context.get_run(),
                        )?;
                        let value_pointer = methods.slot_ty.value(context.builder, slot);
                        LLVMBuildLoad(context.builder, value_pointer, c_str!(""))
                    };
                    LLVMBuildStore(context.builder, result, output_pointer);
                    Ok(())
                } else {
                    unreachable!()
                }
            }
            OptLookup {
                ref child,
                ref index,
            } => {
                // for C
                context.body.add("#error OptLookup is not implemented yet");

                // for LLVM
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let child_type = context.sir_function.symbol_type(child)?;
                if let Dict(ref key, _) = *child_type {
                    use self::hash::GenHash;
                    let hash =
                        self.gen_hash(key, context.builder, context.get_value(index)?, None)?;
                    let (filled, value) = {
                        let methods = self.dictionaries.get_mut(child_type).unwrap();
                        let slot = methods.gen_opt_lookup(
                            context.builder,
                            child_value,
                            context.get_value(index)?,
                            hash,
                        )?;
                        let filled = methods.slot_ty.filled(context.builder, slot);
                        let value_pointer = methods.slot_ty.value(context.builder, slot);
                        // NOTE: This could be an invalid (zeroed value) -- code should check the
                        // boolean.
                        let loaded_value =
                            LLVMBuildLoad(context.builder, value_pointer, c_str!(""));

                        (filled, loaded_value)
                    };

                    let filled = self.i1_to_bool(context.builder, filled);

                    let filled_output_pointer =
                        LLVMBuildStructGEP(context.builder, output_pointer, 0, c_str!(""));
                    LLVMBuildStore(context.builder, filled, filled_output_pointer);
                    let value_output_pointer =
                        LLVMBuildStructGEP(context.builder, output_pointer, 1, c_str!(""));
                    LLVMBuildStore(context.builder, value, value_output_pointer);
                    Ok(())
                } else {
                    unreachable!()
                }
            }
            MakeStruct(ref elems) => {
                let output_pointer = context.get_value(output)?;
                let c_output_pointer = context.c_get_value(output)?;
                for (i, elem) in elems.iter().enumerate() {
                    // for C
                    context.body.add(format!(
                        "{}->f{} = {};",
                        c_output_pointer,
                        i,
                        context.c_get_value(elem)?,
                    ));
                    // for LLVM
                    let elem_pointer =
                        LLVMBuildStructGEP(context.builder, output_pointer, i as u32, c_str!(""));
                    let value = self.load(context.builder, context.get_value(elem)?)?;
                    LLVMBuildStore(context.builder, value, elem_pointer);
                }
                Ok(())
            }
            MakeVector(ref elems) => {
                // for C
                context.body.add("#error MakeVector is not implemented yet");

                // for LLVM
                use self::vector::VectorExt;
                let output_pointer = context.get_value(output)?;
                let output_type = context.sir_function.symbol_type(output)?;
                let size = self.i64(elems.len() as i64);
                let vector = self.gen_new(context.builder, output_type, size, context.get_run())?;
                for (i, elem) in elems.iter().enumerate() {
                    let index = self.i64(i as i64);
                    let vec_pointer = self.gen_at(context.builder, output_type, vector, index)?;
                    let loaded = self.load(context.builder, context.get_value(elem)?)?;
                    LLVMBuildStore(context.builder, loaded, vec_pointer);
                }
                LLVMBuildStore(context.builder, vector, output_pointer);
                Ok(())
            }
            Merge { .. } => {
                // for C and LLVM
                use self::builder::BuilderExpressionGen;
                self.gen_merge(context, statement)
            }
            Negate(_) => {
                // for C
                context.body.add("#error Negate is not implemented yet");

                // for LLVM
                use self::numeric::NumericExpressionGen;
                self.gen_negate(context, statement)
            }
            Not(_) => {
                // for C
                context.body.add("#error Not is not implemented yet");

                // for LLVM
                use self::numeric::NumericExpressionGen;
                self.gen_not(context, statement)
            }
            Assert(ref cond) => {
                // for C
                context.body.add("#error Assert is not implemented yet");

                // for LLVM
                let output_pointer = context.get_value(output)?;
                let cond = self.load(context.builder, context.get_value(cond)?)?;
                let result = self.intrinsics.call_weld_run_assert(
                    context.builder,
                    context.get_run(),
                    cond,
                    None,
                );
                // If assert returns, this expression returns true.
                LLVMBuildStore(context.builder, result, output_pointer);
                Ok(())
            }
            NewBuilder { .. } => {
                // for C and LLVM
                use self::builder::BuilderExpressionGen;
                self.gen_new_builder(context, statement)
            }
            ParallelFor(_) => {
                // for C and LLVM
                use self::builder::BuilderExpressionGen;
                self.gen_for(context, statement)
            }
            Res(_) => {
                // for C and LLVM
                use self::builder::BuilderExpressionGen;
                self.gen_result(context, statement)
            }
            Select {
                ref cond,
                ref on_true,
                ref on_false,
            } => {
                // for C
                context.body.add("#error Select is not implemented yet");

                // for LLVM
                let output_pointer = context.get_value(output)?;
                let cond = self.load(context.builder, context.get_value(cond)?)?;
                let cond = self.bool_to_i1(context.builder, cond);
                let on_true = self.load(context.builder, context.get_value(on_true)?)?;
                let on_false = self.load(context.builder, context.get_value(on_false)?)?;
                let result = LLVMBuildSelect(context.builder, cond, on_true, on_false, c_str!(""));
                LLVMBuildStore(context.builder, result, output_pointer);
                Ok(())
            }
            Serialize(_) => {
                // for C
                context.body.add("#error Serialize is not implemented yet");

                // for LLVM
                use self::serde::SerDeGen;
                self.gen_serialize(context, statement)
            }
            Slice {
                ref child,
                ref index,
                ref size,
            } => {
                // for C
                context.body.add("#error Slice is not implemented yet");

                // for LLVM
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let index_value = self.load(context.builder, context.get_value(index)?)?;
                let size_value = self.load(context.builder, context.get_value(size)?)?;
                let child_type = context.sir_function.symbol_type(child)?;
                if let Vector(ref elem_type) = *child_type {
                    let result = {
                        let methods = self.vectors.get_mut(elem_type).unwrap();
                        methods.gen_slice(context.builder, child_value, index_value, size_value)?
                    };
                    LLVMBuildStore(context.builder, result, output_pointer);
                    Ok(())
                } else {
                    unreachable!()
                }
            }
            Sort {
                ref child,
                ref cmpfunc,
            } => {
                // for C
                context.body.add("#error Sort is not implemented yet");

                // for LLVM
                let output_pointer = context.get_value(output)?;
                let output_type = context
                    .sir_function
                    .symbol_type(statement.output.as_ref().unwrap())?;

                if let Vector(ref elem_ty) = *output_type {
                    use self::vector::VectorExt;

                    let child_value = self.load(context.builder, context.get_value(child)?)?;

                    // Sort clones the vector at the moment.
                    let output_value = self.gen_clone(
                        context.builder,
                        output_type,
                        child_value,
                        context.get_run(),
                    )?;

                    let zero = self.zero(self.i64_type());
                    let elems = self.gen_at(context.builder, output_type, output_value, zero)?;
                    let elems_ptr = LLVMBuildBitCast(
                        context.builder,
                        elems,
                        self.void_pointer_type(),
                        c_str!(""),
                    );
                    let size = self.gen_size(context.builder, output_type, output_value)?;
                    let elem_ll_ty = self.llvm_type(elem_ty)?;
                    let ty_size = self.size_of(elem_ll_ty);

                    use self::cmp::GenCmp;
                    let cmpfunc_ll_fn = self.functions[cmpfunc];

                    let run = context.get_run();

                    // Generate the comparator from the provided custom code.
                    let comparator = self.gen_custom_cmp(elem_ll_ty, *cmpfunc, cmpfunc_ll_fn)?;

                    // args to qsort_r are: base array pointer, num elements,
                    // element size, comparator function, run handle.
                    //
                    // MacOS and Linux pass arguments to qsort_r in different order.
                    let (mut args, mut arg_tys, c_arg_tys) = if cfg!(target_os = "macos") {
                        let args = vec![elems_ptr, size, ty_size, run, comparator];
                        let arg_tys = vec![
                            LLVMTypeOf(elems_ptr),
                            LLVMTypeOf(size),
                            LLVMTypeOf(ty_size),
                            LLVMTypeOf(run),
                            LLVMTypeOf(comparator),
                        ];
                        let c_arg_tys = [
                            "dummy",
                            "dummy",
                            "dummy",
                            "dummy",
                            "dummy",
                            /*
                            self.c_type_of(c_elems_ptr),
                            self.c_type_of(c_size),
                            self.c_type_of(c_ty_size),
                            self.c_type_of(c_run),
                            self.c_type_of(c_comparator),
                            */
                        ];
                        (args, arg_tys, c_arg_tys)
                    } else if cfg!(target_os = "linux") {
                        let args = vec![elems_ptr, size, ty_size, comparator, run];
                        let arg_tys = vec![
                            LLVMTypeOf(elems_ptr),
                            LLVMTypeOf(size),
                            LLVMTypeOf(ty_size),
                            LLVMTypeOf(comparator),
                            LLVMTypeOf(run),
                        ];
                        let c_arg_tys = [
                            "dummy",
                            "dummy",
                            "dummy",
                            "dummy",
                            "dummy",
                            /*
                            self.c_type_of(c_elems_ptr),
                            self.c_type_of(c_size),
                            self.c_type_of(c_ty_size),
                            self.c_type_of(c_comparator),
                            self.c_type_of(c_run),
                            */
                        ];
                        (args, arg_tys, c_arg_tys)
                    } else {
                        unimplemented!("Sort not available on this platform.");
                    };

                    // Generate the call to qsort.
                    let void_type = self.void_type();
                    let c_void_type = self.c_void_type();
                    self.intrinsics.add("qsort_r", "qsort_r", void_type, &c_void_type, &mut arg_tys, &c_arg_tys);
                    self.intrinsics
                        .call(context.builder, "qsort_r", &mut args)?;

                    LLVMBuildStore(context.builder, output_value, output_pointer);

                    Ok(())
                } else {
                    unreachable!()
                }
            }
            ToVec(ref child) => {
                // for C
                context.body.add("#error ToVec is not implemented yet");

                // for LLVM
                let output_pointer = context.get_value(output)?;
                let child_value = self.load(context.builder, context.get_value(child)?)?;
                let child_type = context.sir_function.symbol_type(child)?;
                // This is the type of the resulting key/value vector (vec[{K,V}])
                let output_type = context
                    .sir_function
                    .symbol_type(statement.output.as_ref().unwrap())?;
                let elem = if let Vector(ref elem) = *output_type {
                    elem
                } else {
                    unreachable!()
                };

                let _ = self.llvm_type(output_type)?;
                let result = {
                    let vector_methods = self.vectors.get_mut(elem).unwrap();
                    let methods = self.dictionaries.get_mut(child_type).unwrap();
                    methods.gen_to_vec(
                        context.builder,
                        &mut self.intrinsics,
                        vector_methods,
                        child_value,
                        context.get_run(),
                    )?
                };
                LLVMBuildStore(context.builder, result, output_pointer);
                Ok(())
            }
            UnaryOp { .. } => {
                // for C and LLVM
                use self::numeric::NumericExpressionGen;
                self.gen_unaryop(context, statement)
            }
        }
    }

    /// Generate code for a terminator within an SIR basic block.
    ///
    /// `loop_terminator` is an optional tuple that is present only when generating loop body
    /// functions. The first argument is a basic block to jump to to continue looping instead of
    /// returning from a function. The second argument is a value representing the loop's builder
    /// argument pointer. In cases where the loop body function returns, this function stores the
    /// resulting builder into this pointer. The function is guaranteed to return a builder (since
    /// For loop functions must return builders derived from their input builder).
    ///
    /// This function does not make any assumptions about which *LLVM basic block* the
    /// builder is positioned in, as long as the builder is logically within the passed SIR basic
    /// block.
    unsafe fn gen_terminator(
        &mut self,
        context: &mut FunctionContext<'_>,
        bb: &BasicBlock,
        loop_terminator: Option<(LLVMBasicBlockRef, LLVMValueRef, String)>,
    ) -> WeldResult<()> {
        if self.conf.trace_run {
            self.gen_print(
                context.builder,
                context.get_run(),
                CString::new(format!("{}", bb.terminator)).unwrap(),
            )?;
        }

        use crate::sir::Terminator::*;
        match bb.terminator {
            ProgramReturn(ref sym) => {
                // for C
                let value = context.c_get_value(sym)?;
                let run = context.c_get_run();
                let ty = self.c_type_of(&value);
                let size = self.c_size_of(&ty);
                let bytes = self
                    .intrinsics
                    .c_call_weld_run_malloc(
                        &mut context.body,
                        run,
                        &size,
                        None,
                    );
                context.body.add(format!(
                    "*({}){} = {};",
                    self.c_pointer_type(&ty),
                    bytes,
                    value,
                ));
                let _ = self
                    .intrinsics
                    .c_call_weld_run_set_result(
                        &mut context.body,
                        run,
                        &bytes,
                        None,
                    );
                context.body.add(format!(
                    "return {};",
                    value,
                ));

                // for LLVM
                let value = self.load(context.builder, context.get_value(sym)?)?;
                let run = context.get_run();
                let ty = LLVMTypeOf(value);
                let size = self.size_of(ty);
                let bytes = self
                    .intrinsics
                    .call_weld_run_malloc(context.builder, run, size, None);
                let pointer =
                    LLVMBuildBitCast(context.builder, bytes, LLVMPointerType(ty, 0), c_str!(""));
                LLVMBuildStore(context.builder, value, pointer);
                let _ = self
                    .intrinsics
                    .call_weld_run_set_result(context.builder, run, bytes, None);
                LLVMBuildRet(context.builder, value);
            }
            Branch {
                ref cond,
                ref on_true,
                ref on_false,
            } => {
                // for C
                context.body.add("#error Branch is not implemented yet");

                // for LLVM
                let cond = self.load(context.builder, context.get_value(cond)?)?;
                let cond = self.bool_to_i1(context.builder, cond);
                let _ = LLVMBuildCondBr(
                    context.builder,
                    cond,
                    context.get_block(*on_true)?,
                    context.get_block(*on_false)?,
                );
            }
            JumpBlock(ref id) => {
                // for C
                context.body.add("#error JumpBlock is not implemented yet");

                // for LLVM
                LLVMBuildBr(context.builder, context.get_block(*id)?);
            }
            EndFunction(ref sym) => {
                if let Some((jumpto, loop_builder, c_loop_builder)) = loop_terminator {
                    // for C
                    context.body.add("// EndFunction in loop");
                    context.body.add(format!(
                        "{} = {};",
                        c_loop_builder,
                        context.c_get_value(sym)?,
                    ));
                    context.body.add("continue;");
                    // for LLVM
                    let pointer = context.get_value(sym)?;
                    let updated_builder = self.load(context.builder, pointer)?;
                    LLVMBuildStore(context.builder, updated_builder, loop_builder);
                    LLVMBuildBr(context.builder, jumpto);
                } else {
                    // for C
                    context.body.add("// EndFunction");
                    context.body.add(format!(
                        "return {};",
                        context.c_get_value(sym)?,
                    ));
                    // for LLVM
                    let pointer = context.get_value(sym)?;
                    let return_value = self.load(context.builder, pointer)?;
                    LLVMBuildRet(context.builder, return_value);
                }
            }
            Crash => {
                // for C
                context.body.add("#error Crash is not implemented yet");

                // for LLVM
                use crate::runtime::WeldRuntimeErrno;
                let errno = self.i64(WeldRuntimeErrno::Unknown as i64);
                self.intrinsics.call_weld_run_set_errno(
                    context.builder,
                    context.get_run(),
                    errno,
                    None,
                );
                LLVMBuildUnreachable(context.builder);
            }
        };
        Ok(())
    }

    /// Returns the LLVM type for a Weld Type.
    ///
    /// This method may generate auxillary code before returning the type. For example, for complex
    /// data structures, this function may generate a definition for the data structure first.
    unsafe fn llvm_type(&mut self, ty: &Type) -> WeldResult<LLVMTypeRef> {
        use crate::ast::ScalarKind::*;
        use crate::ast::Type::*;
        let result = match *ty {
            Builder(_, _) => {
                use self::builder::BuilderExpressionGen;
                self.builder_type(ty)?
            }
            Dict(ref key, ref value) => {
                use self::eq::GenEq;
                if !self.dictionaries.contains_key(ty) {
                    let key_ty = self.llvm_type(key)?;
                    let c_key_ty = &self.c_type(key)?.to_string();
                    let value_ty = self.llvm_type(value)?;
                    let c_value_ty = &self.c_type(value)?.to_string();
                    let key_comparator = self.gen_eq_fn(key)?;
                    let dict = dict::Dict::define(
                        "dict",
                        key_ty,
                        c_key_ty,
                        key_comparator,
                        value_ty,
                        c_value_ty,
                        self.context,
                        self.module,
                        self.ccontext,
                    );
                    self.dictionaries.insert(ty.clone(), dict);
                }
                self.dictionaries[ty].dict_ty
            }
            Scalar(kind) => match kind {
                Bool => self.bool_type(),
                I8 => self.i8_type(),
                U8 => self.u8_type(),
                I16 => self.i16_type(),
                U16 => self.u16_type(),
                I32 => self.i32_type(),
                U32 => self.u32_type(),
                I64 => self.i64_type(),
                U64 => self.u64_type(),
                F32 => self.f32_type(),
                F64 => self.f64_type(),
            },
            Simd(kind) => {
                let base = self.llvm_type(&Scalar(kind))?;
                LLVMVectorType(base, LLVM_VECTOR_WIDTH)
            }
            Struct(ref elems) => {
                if !self.struct_names.contains_key(ty) {
                    let name = CString::new(format!("s{}", self.struct_index)).unwrap();
                    self.struct_index += 1;
                    let mut llvm_types: Vec<_> = elems
                        .iter()
                        .map(&mut |t| self.llvm_type(t))
                        .collect::<WeldResult<_>>()?;
                    let struct_ty = LLVMStructCreateNamed(self.context, name.as_ptr());
                    LLVMStructSetBody(
                        struct_ty,
                        llvm_types.as_mut_ptr(),
                        llvm_types.len() as u32,
                        0,
                    );
                    self.struct_names.insert(ty.clone(), name);
                }
                LLVMGetTypeByName(
                    self.module,
                    self.struct_names.get(ty).cloned().unwrap().as_ptr(),
                )
            }
            Vector(ref elem_type) => {
                // Vectors are a named type, so only generate the name once.
                if !self.vectors.contains_key(elem_type) {
                    let name = format!("vec{}", self.vector_index);
                    self.vector_index += 1;
                    let c_elem_type = self.c_type(elem_type)?.to_string();
                    let llvm_elem_type = self.llvm_type(elem_type)?;
                    let vector =
                        vector::Vector::define(name, llvm_elem_type, c_elem_type, self.context, self.module, self.ccontext());
                    self.vectors.insert(elem_type.as_ref().clone(), vector);
                }
                self.vectors[elem_type].vector_ty
            }
            Function(_, _) | Unknown | Alias(_, _) => unreachable!(),
        };
        Ok(result)
    }

    /// Returns the C type for a Weld Type.
    ///
    /// This method may generate auxillary code before returning the type. For example, for complex
    /// data structures, this function may generate a definition for the data structure first.
    unsafe fn c_type(&mut self, ty: &Type) -> WeldResult<String> {
        use crate::ast::ScalarKind::*;
        use crate::ast::Type::*;
        let result = match *ty {
            Builder(_, _) => {
                use self::builder::BuilderExpressionGen;
                self.c_builder_type(ty)?
            }
            Dict(ref key, ref value) => {
                use self::eq::GenEq;
                if !self.dictionaries.contains_key(ty) {
                    let key_ty = self.llvm_type(key)?;
                    let c_key_ty = &self.c_type(key)?.to_string();
                    let value_ty = self.llvm_type(value)?;
                    let c_value_ty = &self.c_type(value)?.to_string();
                    let key_comparator = self.gen_eq_fn(key)?;
                    let dict = dict::Dict::define(
                        "dict",
                        key_ty,
                        c_key_ty,
                        key_comparator,
                        value_ty,
                        c_value_ty,
                        self.context,
                        self.module,
                        self.ccontext,
                    );
                    self.dictionaries.insert(ty.clone(), dict);
                }
                self.dictionaries[ty].name.clone()
            }
            Scalar(kind) => match kind {
                Bool => self.c_bool_type(),
                I8 => self.c_i8_type(),
                U8 => self.c_u8_type(),
                I16 => self.c_i16_type(),
                U16 => self.c_u16_type(),
                I32 => self.c_i32_type(),
                U32 => self.c_u32_type(),
                I64 => self.c_i64_type(),
                U64 => self.c_u64_type(),
                F32 => self.c_f32_type(),
                F64 => self.c_f64_type(),
            },
            Simd(kind) => {
                if !(*self.ccontext()).simd_types.contains_key(&kind) {
                    (*self.ccontext()).prelude_code.add(
                        format!("// typedef ... simd_{};",
                        self.c_type(&Scalar(kind))?));
                    (*self.ccontext()).simd_types.insert(
                        kind, format!("simd_{}", self.c_type(&Scalar(kind))?));
                }
                (*self.ccontext()).simd_types.get(&kind).unwrap().to_string()
            }
            Struct(ref elems) => {
                if !self.c_struct_names.contains_key(ty) {
                    let name = format!("s{}", self.struct_index);
                    self.struct_index += 1;
                    let mut def = CodeBuilder::new();
                    def.add("typedef struct {");
                    for (i, e) in elems.iter().enumerate() {
                        def.add(format!("{} f{};", self.c_type(e)?, i));
                    }
                    def.add(format!("}} {};", name));
                    (*self.ccontext()).prelude_code.add(def.result());
                    self.c_struct_names.insert(ty.clone(), name);
                }
                self.c_struct_names.get(ty).unwrap().to_string()
            }
            Vector(ref elem_type) => {
                // Vectors are a named type, so only generate the name once.
                if !self.vectors.contains_key(elem_type) {
                    let name = format!("vec{}", self.vector_index);
                    self.vector_index += 1;
                    let c_elem_type = self.c_type(elem_type)?.to_string();
                    let llvm_elem_type = self.llvm_type(elem_type)?;
                    let vector =
                        vector::Vector::define(name, llvm_elem_type, c_elem_type, self.context, self.module, self.ccontext());
                    self.vectors.insert(elem_type.as_ref().clone(), vector);
                }
                self.vectors[elem_type].name.clone()
            }
            Function(_, _) | Unknown | Alias(_, _) => unreachable!(),
        };
        Ok(result)
    }

    unsafe fn size_of_ty(&mut self, ty: &Type) -> usize {
        let ll_ty = self.llvm_type(ty).unwrap();
        (self.size_of_bits(ll_ty) / 8) as usize
    }

    fn gen_c_code(&self) -> String {
        unsafe {
        format!("// PRELUDE:\n{}\n\n// BODY:\n{}",
                (*self.ccontext()).prelude_code.result(),
                (*self.ccontext()).body_code.result())
        }
    }
}

impl fmt::Display for CGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = unsafe {
            let c_str = LLVMPrintModuleToString(self.module) as *mut c_char;
            let msg = CStr::from_ptr(c_str).to_owned();
            LLVMDisposeMessage(c_str);
            msg
        };
        write!(f, "{}", s.to_str().unwrap())
    }
}

/// A context for generating code in an SIR function.
///
/// The function context holds the position at which new code should be generated.
pub struct FunctionContext<'a> {
    /// A reference to the SIR program.
    sir_program: &'a SirProgram,
    /// The SIR function to which this context belongs.
    ///
    /// Equivalently, this context represents code generation state for this function.
    sir_function: &'a SirFunction,
    /// An LLVM reference to this function.
    llvm_function: LLVMValueRef,
    /// The LLVM values for symbols defined in this function.
    ///
    /// These symbols are the ones defined in the SIR (i.e., locals and parameters). The symbols
    /// values are thus all alloca'd pointers.
    symbols: FnvHashMap<Symbol, LLVMValueRef>,
    /// A mapping from SIR basic blocks to LLVM basic blocks.
    blocks: FnvHashMap<BasicBlockId, LLVMBasicBlockRef>,
    /// The LLVM builder, which marks where to insert new code.
    builder: LLVMBuilderRef,

    /// A ID generator for each function.
    var_ids: IdGenerator,
    /// A CodeBuilder for each function.
    body: CodeBuilder,
    /// Counter for unique basic block names.
    bb_index: u32,
}

impl<'a> FunctionContext<'a> {
    pub fn new(
        llvm_context: LLVMContextRef,
        sir_program: &'a SirProgram,
        sir_function: &'a SirFunction,
        llvm_function: LLVMValueRef,
    ) -> FunctionContext<'a> {
        FunctionContext {
            sir_program,
            sir_function,
            llvm_function,
            builder: unsafe { LLVMCreateBuilderInContext(llvm_context) },
            symbols: FnvHashMap::default(),
            blocks: FnvHashMap::default(),
            var_ids: IdGenerator::new("t"),
            body: CodeBuilder::new(),
            bb_index: 0,
        }
    }

    /// Returns the LLVM value for a symbol in this function.
    pub fn get_value(&self, sym: &Symbol) -> WeldResult<LLVMValueRef> {
        self.symbols.get(sym).cloned().ok_or_else(|| {
            WeldCompileError::new(format!("Undefined symbol {} in function codegen", sym))
        })
    }

    pub fn c_get_value(&self, sym: &Symbol) -> WeldResult<String> {
        if let Some(_) = self.symbols.get(sym) {
            Ok(sym.to_string())
        } else {
            Err(WeldCompileError::new(format!("Undefined symbol {} in function codegen", sym)))
        }
    }

    /// Returns the LLVM basic block for a basic block ID in this function.
    pub fn get_block(&self, id: BasicBlockId) -> WeldResult<LLVMBasicBlockRef> {
        self.blocks
            .get(&id)
            .cloned()
            .ok_or_else(|| WeldCompileError::new("Undefined basic block in function codegen"))
    }
    pub fn c_get_block(&self, id: BasicBlockId) -> String {
        format!("b{}", id)
    }

    /// Get the handle to the run.
    ///
    /// The run handle is always the last argument of an SIR function.
    pub fn get_run(&self) -> LLVMValueRef {
        unsafe { LLVMGetLastParam(self.llvm_function) }
    }
    pub fn c_get_run(&self) -> &'static str {
        "run"
    }
}

impl<'a> Drop for FunctionContext<'a> {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeBuilder(self.builder);
        }
    }
}
