//! Utility functions for veoffload

use std::ptr;
use std::os::raw::c_char;
use std::ffi::CString;
use libc::{uint64_t, int64_t, uint32_t, int32_t, c_int, c_void};

use crate::WeldResult;
use crate::util::offload_ve::*;

pub enum VeoProcHandle {}
pub enum VeoThrContext {}
pub enum VeoArgs {}
pub type VeoProcHandleRef = *mut VeoProcHandle;
pub type VeoHandle = uint64_t;
pub type VeoThrContextRef = *mut VeoThrContext;
pub type VeoArgsRef = *mut VeoArgs;
pub type SymHandle = uint64_t;
pub type CallHandle = uint64_t;

#[derive(Debug)]
#[repr(C)]
pub enum VeoArgsIntent {
  VeoIntentIn = 0,
  VeoIntentInOut = 1,
  VeoIntentOut = 2,
}

#[derive(Debug)]
#[repr(C)]
pub enum VeoCommandState {
  VeoCommandOk = 0,
  VeoCommandException = 1,
  VeoCommandError = 2,
  VeoCommandUnfinished = 3,
  VeoCommandInternalError = -1,
}

#[link(name = "veo")]
extern "C" {
    pub fn veo_proc_create(Val: int64_t) -> VeoProcHandleRef;
    pub fn veo_proc_create_static(Val: int64_t, VeoBin: *const c_char) -> VeoProcHandleRef;
    pub fn veo_proc_destroy(PH: VeoProcHandleRef) -> c_int;
    pub fn veo_load_library(PH: VeoProcHandleRef, LibName: *const c_char)
                            -> VeoHandle;
    pub fn veo_get_sym(PH: VeoProcHandleRef,
                       H: VeoHandle,
                       FunName: *const c_char)
                       -> SymHandle;
    pub fn veo_context_open(PH: VeoProcHandleRef) -> VeoThrContextRef;
    pub fn veo_context_close(Ctx: VeoThrContextRef) -> c_int;
    pub fn veo_args_alloc() -> VeoArgsRef;
    pub fn veo_args_clear(Args: VeoArgsRef);
    pub fn veo_args_free(Args: VeoArgsRef);
    pub fn veo_args_set_i64(Args: VeoArgsRef, Index: c_int, Val: int64_t)
                            -> c_int;
    pub fn veo_args_set_u64(Args: VeoArgsRef, Index: c_int, Val: uint64_t)
                            -> c_int;
    pub fn veo_args_set_i32(Args: VeoArgsRef, Index: c_int, Val: int32_t)
                            -> c_int;
    pub fn veo_args_set_u32(Args: VeoArgsRef, Index: c_int, Val: uint32_t)
                            -> c_int;
    pub fn veo_args_set_stack(Args: VeoArgsRef, Intent: VeoArgsIntent,
                              Index: c_int, Data: *mut c_char, Len: uint64_t)
                              -> c_int;
    pub fn veo_call_async(Ctx: VeoThrContextRef,
                          Sym: SymHandle,
                          Args: VeoArgsRef)
                          -> CallHandle;
    pub fn veo_call_async_by_name(Ctx: VeoThrContextRef,
                                  H: VeoHandle,
                                  FunName: *const c_char,
                                  Args: VeoArgsRef)
                                  -> CallHandle;
    pub fn veo_call_wait_result(Ctx: VeoThrContextRef,
                                Id: CallHandle,
                                RetVal: *mut uint64_t)
                                -> VeoCommandState;
    pub fn veo_alloc_mem(PH: VeoProcHandleRef, Addr: *mut uint64_t,
                         Size: uint64_t) -> c_int;
    pub fn veo_free_mem(PH: VeoProcHandleRef, Addr: uint64_t) -> c_int;
    pub fn veo_read_mem(PH: VeoProcHandleRef, Data: *mut c_void,
                        Addr: uint64_t, Size: uint64_t) -> c_int;
    pub fn veo_write_mem(PH: VeoProcHandleRef, Addr: uint64_t,
                         Data: *const c_void, Size: uint64_t) -> c_int;
    pub fn veo_async_read_mem(Ctx: VeoThrContextRef, Data: *mut c_void,
                              Addr: uint64_t, Size: uint64_t) -> uint64_t;
    pub fn veo_async_write_mem(Ctx: VeoThrContextRef, Addr: uint64_t,
                               Data: *const c_void, Size: uint64_t) -> uint64_t;
}


lazy_static! {
    /// global context of VE offload
    pub static ref VEO_GLOBAL: VEOffload = VEOffload::default();
    /// pointer to global context
    pub static ref VEO_GLOBAL_PTR: u64 = &(*VEO_GLOBAL) as *const VEOffload as u64 ;
}
use std::sync::{Once, ONCE_INIT};
static ONCE: Once = ONCE_INIT;

pub unsafe fn get_global_veo_ptr() -> *mut VEOffload {
    return *VEO_GLOBAL_PTR as *mut VEOffload;
}

/// Initialize VEO
pub unsafe fn initialize_veo(veo_ptr: u64, veorun: Option<String>,
                             libs: Option<Vec<String>>) {
    let veo = veo_ptr as *mut VEOffload;
    if !(*veo).ready {
        (*veo).initialize(veorun).unwrap();
        if let Some(libnames) = libs {
            for libname in libnames {
                (*veo).load_library(libname.as_str()).unwrap();
            }
        }
    }
}

/*
/// Initialize VEO and load library in ahead of module run, with background thread.
pub unsafe fn initialize_veo_bg(veo_ptr: u64, veorun: Option<String>,
                                libs: Option<Vec<String>>)
                                -> Option<thread::JoinHandle<()>> {
    let veo = veo_ptr as *mut VEOffload;
    if (*veo).ready { return None }

    let handle = thread::spawn(move || {
        let veo = veo_ptr as *mut VEOffload;
        (*veo).initialize(veorun).unwrap();
        if let Some(libnames) = libs {
            for libname in libnames {
                (*veo).load_library(libname.as_str()).unwrap();
            }
        }
    });
    Some(handle)
}
*/

pub extern "C" fn initialize_veo_global(
    veorun: Option<String>, libs: Option<Vec<String>>) {
    unsafe {
        ONCE.call_once(
            || initialize_veo(*VEO_GLOBAL_PTR, veorun, libs)
        );
    }
}

pub extern "C" fn finalize_veo_global() {
    unsafe {
        let veo = *VEO_GLOBAL_PTR as *mut VEOffload;
        (*veo).finalize();
    }
}


pub struct VEOffload {
    pub proc: VeoProcHandleRef,
    pub ctx: VeoThrContextRef,
    pub libs: fnv::FnvHashMap<String, VeoHandle>,
    pub ready: bool,
//    pub udma_peer: Option<c_int>
}

impl VEOffload {

    pub unsafe fn load_library<T: AsRef<str>>(
        &mut self, libname: T
    ) -> WeldResult<VeoHandle> {
        let c_libname = CString::new(libname.as_ref()).unwrap();
        let handle = veo_load_library(self.proc, c_libname.as_ptr());
        if handle == 0 {
            return weld_err!("cannot load library (name:{})", libname.as_ref());
        }

        use std::path::Path;
        let filename = Path::new(libname.as_ref()).file_name().unwrap().to_str().unwrap();
        self.libs.insert(filename.to_string(), handle);
        Ok(handle)
    }

    pub unsafe fn get_sym<T: AsRef<str>>(
        &mut self, libhdl: VeoHandle, symname: T
    ) -> WeldResult<SymHandle> {
        let c_symname = CString::new(symname.as_ref()).unwrap();
        let sym = veo_get_sym(self.proc, libhdl, c_symname.as_ptr());
        if sym == 0 {
            return weld_err!("cannot find function (name:{})", symname.as_ref());
        }
        Ok(sym)
    }

    pub unsafe fn call_async(
        &mut self, sym: SymHandle, args: VeoArgsRef
    ) -> WeldResult<CallHandle> {
        let call = veo_call_async(self.ctx, sym, args);
        let VeoRequestIdInvalid = !(0_u64) as CallHandle;
        Ok(call)
    }

    pub unsafe fn call_async_by_name<T: AsRef<str>>(
        &mut self, libhdl: VeoHandle, symname: T, args: VeoArgsRef
    ) -> WeldResult<CallHandle> {
        let c_symname = CString::new(symname.as_ref()).unwrap();
        let call = veo_call_async_by_name(self.ctx, libhdl, c_symname.as_ptr(), args);
        let VeoRequestIdInvalid = !(0_u64) as CallHandle;
        Ok(call)
    }

    pub unsafe fn call_wait_result(&mut self, call: CallHandle) -> WeldResult<uint64_t> {
        let mut retp: uint64_t = 0;
        let state = veo_call_wait_result(self.ctx, call, &mut retp);
        match state {
            VeoCommandState::VeoCommandOk => {}
            _ => { return weld_err!("wait result (call handle:{}) failed", call); }
        };
        Ok(retp)
    }

    pub unsafe fn alloc_mem(&mut self, size: u64) -> WeldResult<u64> {
        let mut addr: uint64_t = 0;
        let err = veo_alloc_mem(self.proc, &mut addr as *mut uint64_t, size as uint64_t);
        if err != 0 {
            return weld_err!("alloc memory");
        } else {
            Ok(addr)
        }
    }

    pub unsafe fn free_mem(&mut self, addr: u64) -> WeldResult<()> {
        let err = veo_free_mem(self.proc, addr);
        if err != 0 {
            return weld_err!("free memory");
        } else {
            Ok(())
        }
    }

    pub unsafe fn read_mem(&self, src: u64, dst: *mut c_void, len: u64) -> WeldResult<()> {
        let err = veo_read_mem(self.proc, dst, src as uint64_t, len as uint64_t);
        if err != 0 {
            return weld_err!("veo read mem: failed");
        }
        Ok(())
    }

    pub unsafe fn write_mem(&self, src: *const c_void, dst: u64, len: u64) -> WeldResult<()> {
        let err = veo_write_mem(self.proc, dst as uint64_t, src, len as uint64_t);
        if err != 0 {
            return weld_err!("veo write mem: failed");
        }
        Ok(())
    }

    pub unsafe fn args_alloc() -> VeoArgsRef {
        veo_args_alloc()
    }

    pub unsafe fn args_clear(args: VeoArgsRef) {
        veo_args_clear(args);
    }

    pub unsafe fn args_free(args: VeoArgsRef) {
        veo_args_free(args);
    }
}


impl Default for VEOffload {
    fn default() -> Self {
        Self {
            proc: ptr::null::<VeoProcHandle>() as VeoProcHandleRef,
            ctx: ptr::null::<VeoThrContext>() as VeoThrContextRef,
            libs: fnv::FnvHashMap::default(),
            ready: false,
//            udma_peer: None,
        }
    }
}

unsafe impl Sync for VEOffload {}  // needed to have VEO static


pub trait VEOffloadHelper {
    unsafe fn initialize<T: AsRef<str>>(&mut self, veorun_path: Option<T>) -> WeldResult<()>;
    unsafe fn finalize(&mut self);
    unsafe fn call_and_wait<T: AsRef<str>>(&mut self, libhdl: VeoHandle, symname: T, args: VeoArgsRef) -> WeldResult<uint64_t>;
    fn get_libhdl(&self, filename: &str) -> Option<VeoHandle>;
    fn unwrap_ready(&self) -> WeldResult<()>;
}

impl VEOffloadHelper for VEOffload {
    unsafe fn initialize<T: AsRef<str>>(&mut self, veorun_path: Option<T>) -> WeldResult<()> {
        let node = get_ve_node_number();

        if let Some(vp) = veorun_path {
            let c_vp = CString::new(vp.as_ref()).unwrap();
            self.proc = veo_proc_create_static(node, c_vp.as_ptr());
        } else {
            self.proc = veo_proc_create(node);
        }
        if self.proc.is_null() {
            return weld_err!("cannot create proc");
        }

        self.ctx = veo_context_open(self.proc);
        if self.ctx.is_null() {
            let err = veo_proc_destroy(self.proc);
            return weld_err!("cannot create veo context");
        }

        self.ready = true;
        Ok(())
    }

    unsafe fn finalize(&mut self) {
        if !self.ready { return }

        let err = veo_context_close(self.ctx);
        if err != 0 {
            panic!("cannot close veo context");
        }

        let err = veo_proc_destroy(self.proc);
        if err != 0 {
            panic!("cannot destrocy veo proc");
        }
    }

    unsafe fn call_and_wait<T: AsRef<str>>(
        &mut self, libhdl: VeoHandle, symname: T, args: VeoArgsRef
    ) -> WeldResult<uint64_t> {
        let call = self.call_async_by_name(libhdl, symname, args)?;
        self.call_wait_result(call)
    }

    fn get_libhdl(&self, filename: &str) -> Option<VeoHandle> {
        let hdlref = self.libs.get(&filename.to_string());
        match hdlref {
            Some(tmp) => Some(*tmp),
            None => None,
        }
    }

    fn unwrap_ready(&self) -> WeldResult<()> {
        if self.ready {
            Ok(())
        } else {
            return weld_err!("veo is not initialized");
        }
    }
}

/// argument for VEO call
pub trait VEOffloadArgument<V> {
    unsafe fn args_set(args: VeoArgsRef, argnum: isize, val: V);
}

impl VEOffloadArgument<i64> for VEOffload {
    unsafe fn args_set(args: VeoArgsRef, argnum: isize, val: i64) {
        let ret = veo_args_set_i64(args, argnum as c_int, val as int64_t);
        if ret < -1 {
            panic!("cannot set veo args");
        }
    }
}

impl VEOffloadArgument<u64> for VEOffload {
    unsafe fn args_set(args: VeoArgsRef, argnum: isize, val: u64) {
        let ret = veo_args_set_u64(args, argnum as c_int, val as uint64_t);
        if ret < -1 {
            panic!("cannot set veo args");
        }
    }
}

impl VEOffloadArgument<i32> for VEOffload {
    unsafe fn args_set(args: VeoArgsRef, argnum: isize, val: i32) {
        let ret = veo_args_set_i32(args, argnum as c_int, val as int32_t);
        if ret < -1 {
            panic!("cannot set veo args");
        }
    }
}

impl VEOffloadArgument<u32> for VEOffload {
    unsafe fn args_set(args: VeoArgsRef, argnum: isize, val: u32) {
        let ret = veo_args_set_u32(args, argnum as c_int, val as uint32_t);
        if ret < -1 {
            panic!("cannot set veo args");
        }
    }
}


