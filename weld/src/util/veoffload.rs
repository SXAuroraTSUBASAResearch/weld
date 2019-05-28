//! Utility functions for veoffload

use std::os::raw::c_char;
use libc::{uint64_t, int64_t, uint32_t, int32_t, c_int, c_void};

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
    pub fn veo_proc_destroy(PH: VeoProcHandleRef) -> c_int;
    pub fn veo_load_library(PH: VeoProcHandleRef, LibName: *const c_char)
                            -> VeoHandle;
    pub fn veo_get_sym(PH: VeoProcHandleRef,
                       H: VeoHandle,
                       FunName: *const c_char)
                       -> SymHandle;
    pub fn veo_context_open(PH: VeoProcHandleRef) -> VeoThrContextRef;
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

