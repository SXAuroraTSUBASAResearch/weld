//!
//! This module manages compiling with optimization the generated C module
//! to the machine code.

use libc;
use llvm_sys;
use time;

use std::ffi::CStr;
use std::mem;
use std::ptr;
use std::sync::{Once, ONCE_INIT};
use std::process::Command;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;

use libc::c_char;

use self::time::PreciseTime;

use crate::conf::ParsedConf;
use crate::error::*;
use crate::ast::Type;
use crate::util::stats::CompilationStats;
use crate::util::dump::unique_filename;

use self::llvm_sys::core::*;
use self::llvm_sys::execution_engine::*;
use self::llvm_sys::prelude::*;
use self::llvm_sys::target::*;
use self::llvm_sys::target_machine::*;

use crate::codegen::c::intrinsic;
use crate::codegen::c::llvm_exts::*;

static ONCE: Once = ONCE_INIT;
static ONCE_VEO: Once = ONCE_INIT;
static mut INITIALIZE_FAILED: bool = false;

/// The callable function type.
type I64Func = extern "C" fn(i64) -> i64;

/// A compiled, runnable LLVM module.
pub struct CompiledModule {
    context: LLVMContextRef,
    module: LLVMModuleRef,
    // engine: LLVMExecutionEngineRef,
    // for C
    pub filename: String,
    pub encoded_params: String,
    pub params: Type,
    pub ret_ty: Type,
}

// Runnable implementation is moved out to run.rs

// LLVM modules are thread-safe.
unsafe impl Send for CompiledModule {}
unsafe impl Sync for CompiledModule {}

impl Drop for CompiledModule {
    fn drop(&mut self) {
        unsafe {
            // Engine owns the module, so do not drop it explicitly.
            // LLVMDisposeExecutionEngine(self.engine);
            LLVMContextDispose(self.context);
        }
    }
}

pub unsafe fn init() {
    ONCE.call_once(|| initialize());
    if INITIALIZE_FAILED {
        unreachable!()
    }
}

unsafe fn init_veo() {
    use crate::util::veoffload::*;
    use libc::atexit;

    ONCE_VEO.call_once(|| {
        // without veorun and dynamic lib
        initialize_veo_global(None, None);

        let ret = atexit(finalize_veo_global);
        if ret as isize != 0 {
            panic!("fail to add veo finalization at exit");
        }
    });
}

pub fn write_code(
    code: String,
) -> WeldResult<String> {
    let path = &mut PathBuf::new();
    path.push(".");
    path.push(&format!("code-{}-{}", time::now().to_timespec().sec, "gen"));
    path.set_extension("c");

    let filename = path.to_str().unwrap().to_string();
    info!("Writing code to {}", filename);

    let mut options = OpenOptions::new();
    let mut file = options.write(true).create_new(true).open(path)?;

    file.write_all(code.as_bytes())?;

    Ok(filename)
}

/// Compile a constructed module in the given LLVM context.
pub unsafe fn compile(
    code: String,
    params: Type,
    ret_ty: Type,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    _mappings: &[intrinsic::Mapping],
    _conf: &ParsedConf,
    stats: &mut CompilationStats,
) -> WeldResult<CompiledModule> {

    let start = PreciseTime::now();
    init_veo();
    let end = PreciseTime::now();
    stats
        .llvm_times
        .push(("VE Offload initialization".to_string(), start.to(end)));

    // Write code to a file
    let filename = write_code(code)?;

    // Compile it using CC
    use crate::util::env::{get_cc,get_cflags,get_home};
    let compiler = get_cc();
    let cflags = get_cflags();
    let home = get_home();
    if home.is_empty() {
        error!("WELD_HOME is not defined");
    }
    let shared_object = format!("libverun-{}.so", unique_filename());

    // Execute C compiler
    let output = Command::new("sh")
        .arg("-c")
        .arg(&format!(concat!("{compiler} {cflags} {shared} -o {out} {file} ",
                              "-L{home}/weld_rt/cpp ",
                              "-Wl,-rpath,{home}/weld_rt/cpp ",
                              "{lib}"),
                      compiler=compiler,
                      cflags=cflags,
                      shared="-shared -fpic",
                      out=shared_object,
                      file=filename,
                      home=home,
                      lib="-lpthread -ldl"))
        .output()
        .expect("failed to execute process");
    println!("status: \n{}", output.status);
    println!("stdout: \n{}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: \n{}", String::from_utf8_lossy(&output.stderr));

    init();

    let start = PreciseTime::now();
    // verify_module(module)?;
    let end = PreciseTime::now();
    stats
        .llvm_times
        .push(("Module Verification".to_string(), start.to(end)));

    let start = PreciseTime::now();
    let end = PreciseTime::now();
    stats
        .llvm_times
        .push(("Find Run Func Address".to_string(), start.to(end)));

    let result = CompiledModule {
        context,
        module,
        // engine,
        filename: shared_object,
        encoded_params: "".to_string(),
        params,
        ret_ty,
    };
    Ok(result)
}

/// Initialize LLVM.
///
/// This function should only be called once.
unsafe fn initialize() {
    use self::llvm_sys::target::*;
    if LLVM_InitializeNativeTarget() != 0 {
        INITIALIZE_FAILED = true;
        return;
    }
    if LLVM_InitializeNativeAsmPrinter() != 0 {
        INITIALIZE_FAILED = true;
        return;
    }
    if LLVM_InitializeNativeAsmParser() != 0 {
        INITIALIZE_FAILED = true;
        return;
    }

    // No version that just initializes the current one?
    LLVM_InitializeAllTargetInfos();
    LLVMLinkInMCJIT();

    use self::llvm_sys::initialization::*;

    let registry = LLVMGetGlobalPassRegistry();
    LLVMInitializeCore(registry);
    LLVMInitializeAnalysis(registry);
    LLVMInitializeCodeGen(registry);
    LLVMInitializeIPA(registry);
    LLVMInitializeIPO(registry);
    LLVMInitializeInstrumentation(registry);
    LLVMInitializeObjCARCOpts(registry);
    LLVMInitializeScalarOpts(registry);
    LLVMInitializeTarget(registry);
    LLVMInitializeTransformUtils(registry);
    LLVMInitializeVectorization(registry);
}

unsafe fn target_machine() -> WeldResult<LLVMTargetMachineRef> {
    let mut target = mem::uninitialized();
    let mut err = ptr::null_mut();
    let ret = LLVMGetTargetFromTriple(PROCESS_TRIPLE.as_ptr(), &mut target, &mut err);
    if ret == 1 {
        let err_msg = CStr::from_ptr(err as *mut c_char)
            .to_string_lossy()
            .into_owned();
        LLVMDisposeMessage(err); // err is only allocated on res == 1
        compile_err!("Target initialization failed with error {}", err_msg)
    } else {
        Ok(LLVMCreateTargetMachine(
            target,
            PROCESS_TRIPLE.as_ptr(),
            HOST_CPU_NAME.as_ptr(),
            HOST_CPU_FEATURES.as_ptr(),
            LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
            LLVMRelocMode::LLVMRelocDefault,
            LLVMCodeModel::LLVMCodeModelDefault,
        ))
    }
}

pub unsafe fn set_triple_and_layout(module: LLVMModuleRef) -> WeldResult<()> {
    LLVMSetTarget(module, PROCESS_TRIPLE.as_ptr() as *const _);
    debug!("Set module target {:?}", PROCESS_TRIPLE.to_str().unwrap());
    let target_machine = target_machine()?;
    let layout = LLVMCreateTargetDataLayout(target_machine);
    LLVMSetModuleDataLayout(module, layout);
    LLVMDisposeTargetMachine(target_machine);
    LLVMDisposeTargetData(layout);
    Ok(())
}

/// Verify a module using LLVM's verifier.
unsafe fn verify_module(module: LLVMModuleRef) -> WeldResult<()> {
    use self::llvm_sys::analysis::LLVMVerifierFailureAction::*;
    use self::llvm_sys::analysis::LLVMVerifyModule;
    let mut error_str = ptr::null_mut();
    let result_code = LLVMVerifyModule(module, LLVMReturnStatusAction, &mut error_str);
    let result = {
        if result_code != 0 {
            let err = CStr::from_ptr(error_str).to_string_lossy().into_owned();
            compile_err!("{}", format!("Module verification failed: {}", err))
        } else {
            Ok(())
        }
    };
    libc::free(error_str as *mut libc::c_void);
    result
}
