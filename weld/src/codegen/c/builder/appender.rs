//! Code generation for the appender builder type.
//!
//! Much of this code mirrors the implementation of the `vector` type, and it may be worth merging
//! this module with `llvm::vector` one day. The main difference between a vector and an appender
//! is that an appender has a third capacity field (in addition to the vector's data pointer and
//! size). The appender also contains methods for dynamic resizing.

use llvm_sys;

use std::ffi::CString;
use code_builder::CodeBuilder;

use crate::error::*;

use self::llvm_sys::core::*;
use self::llvm_sys::prelude::*;

use crate::ast::Type;
use crate::codegen::c::intrinsic::Intrinsics;
use crate::codegen::c::CodeGenExt;
use crate::codegen::c::LLVM_VECTOR_WIDTH;
use crate::codegen::c::CContextRef;
use crate::codegen::c::c_u64_type;

pub const POINTER_INDEX: u32 = 0;
pub const SIZE_INDEX: u32 = 1;
pub const CAPACITY_INDEX: u32 = 2;

/// The default Appender capacity.
///
/// This *must* be larger than the `LLVM_VECTOR_WIDTH`.
pub const DEFAULT_CAPACITY: i64 = 16;

pub struct Appender {
    pub appender_ty: LLVMTypeRef,
    pub elem_ty: LLVMTypeRef,
    pub c_elem_ty: String,
    pub name: String,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    ccontext: CContextRef,
    c_new: String,
    c_merge: String,
    c_vmerge: String,
    c_result: String,
}

impl CodeGenExt for Appender {
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

impl Appender {
    pub unsafe fn define<T: AsRef<str>>(
        name: T,
        elem_ty: LLVMTypeRef,
        c_elem_ty: String,
        context: LLVMContextRef,
        module: LLVMModuleRef,
        ccontext: CContextRef,
    ) -> Appender {
        // for C
        let mut def = CodeBuilder::new();
        def.add("typedef struct {");
        def.add(format!("{elem_ty}* data;", elem_ty=c_elem_ty));
        def.add(format!("{u64} size;", u64=c_u64_type(ccontext)));
        def.add(format!("{u64} capacity;", u64=c_u64_type(ccontext)));
        def.add(format!("}} {};", name.as_ref()));
        (*ccontext).prelude_code.add(def.result());
        // for LLVM
        let c_name = CString::new(name.as_ref()).unwrap();
        // An appender is struct with a pointer, size, and capacity.
        let appender = LLVMStructCreateNamed(context, c_name.as_ptr());
        Appender {
            appender_ty: appender,
            elem_ty,
            c_elem_ty,
            name: c_name.into_string().unwrap(),
            context,
            module,
            ccontext,
            c_new: String::new(),
            c_merge: String::new(),
            c_vmerge: String::new(),
            c_result: String::new(),
        }
    }

    /// Returns a pointer to the `index`th element in the appender.
    ///
    /// If the `index` is `None`, thie method returns the base pointer. This method does not
    /// perform any bounds checking.
    unsafe fn c_gen_index(
        &mut self,
        appender: &str,
        index: Option<String>,
    ) -> WeldResult<String> {
        if let Some(index) = index {
            Ok(format!(
                "&({}->data[{}])",
                appender,
                index,
            ))
        } else {
            Ok(format!(
                "{}->data",
                appender,
            ))
        }
    }

    /// Define code for a new appender.
    pub unsafe fn define_new(
        &mut self,
        intrinsics: &mut Intrinsics,
    ) {
        let c_arg_tys = [self.c_u64_type(), self.c_run_handle_type()];
        let c_ret_ty = &self.name.clone();

        // Use C name.
        let name = format!("{}_new", self.name);
        let mut c_code = self.c_define_function(c_ret_ty, &c_arg_tys, name.clone(), false);

        c_code.add("{");
        // let elem_size = self.size_of(self.elem_ty);
        c_code.add(format!("{} ret;", self.name));
        c_code.add(format!(
            "ret.data = {};",
            intrinsics.c_call_weld_run_malloc(
                &self.c_get_run(),
                &format!("{capacity} * {elem_size}",
                         capacity=self.c_get_param(0),
                         elem_size=self.c_size_of(&self.c_elem_ty),
                ),
            ),
        ));
        c_code.add("ret.size = 0;");
        c_code.add(format!("ret.capacity = {};", self.c_get_param(0)));
        c_code.add("return ret;");
        c_code.add("}");
        (*self.ccontext()).prelude_code.add(c_code.result());
        self.c_new = name;
    }
    /// Generates code for a new appender.
    pub unsafe fn c_gen_new(
        &mut self,
        _builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        run: &str,
        capacity: &str,
    ) -> WeldResult<String> {
        if self.c_new.is_empty() {
            self.define_new(intrinsics);
        }
        Ok(format!("{}({}, {})", self.c_new, capacity, run))
    }

    /// Internal merge function generation that supports vectorization.
    ///
    /// Returns an `LLVMValueRef` representing the generated merge function.
    unsafe fn define_merge(
        &mut self,
        intrinsics: &mut Intrinsics,
        vectorized: bool,
    ) -> WeldResult<()> {
        // Number of elements merged in at once.
        let (c_merge_ty, num_elements) = if vectorized {
            (
                self.c_simd_type(&self.c_elem_ty, LLVM_VECTOR_WIDTH),
                LLVM_VECTOR_WIDTH,
            )
        } else {
            (self.c_elem_ty.clone(), 1)
        };

        // use C name
        let name = if vectorized {
            format!("{}_vmerge", self.name)
        } else {
            format!("{}_merge", self.name)
        };

        let c_arg_tys = [
            self.c_pointer_type(&self.name),
            c_merge_ty,
            self.c_run_handle_type(),
        ];
        let c_ret_ty = &self.c_void_type();
        let mut c_code = self.c_define_function(c_ret_ty, &c_arg_tys, name.clone(), true);

        // for C
        c_code.add("{");
        let appender = self.c_get_param(0);
        let merge_value = self.c_get_param(1);
        let run_handle = self.c_get_run();
        c_code.add(format!(
            "{u64} newSize = {app}->size + {num};",
            u64=self.c_u64_type(),
            app=appender,
            num=num_elements,
        ));
        c_code.add(format!(
            "if (newSize > {app}->capacity) {{",
            app=appender,
        ));
        c_code.add(format!(
            "{u64} newCap = {app}->capacity * 2;",
            u64=self.c_u64_type(),
            app=appender,
        ));
        c_code.add(format!(
            "{} = {};",
            format!("{}->data", appender),
            intrinsics.c_call_weld_run_realloc(
                &run_handle,
                &format!("{app}->data", app=appender),
                "newCap",
            ),
        ));
        c_code.add(format!(
            "{app}->capacity = newCap;",
            app=appender,
        ));
        c_code.add("}");
        c_code.add(format!(
            "{app}->data[{app}->size] = {val};",
            app=appender,
            val=merge_value,
        ));
        c_code.add(format!(
            "{app}->size = newSize;",
            app=appender,
        ));
        c_code.add("}");
        (*self.ccontext()).prelude_code.add(c_code.result());
        if vectorized {
            self.c_vmerge = name;
        } else {
            self.c_merge = name;
        }
        Ok(())
    }

    // VE-Weld NO_RESIZE begin
    unsafe fn define_merge_no_resize(
        &mut self,
        //intrinsics: &mut Intrinsics,
        vectorized: bool,
    ) -> WeldResult<()> {
        // Number of elements merged in at once.
        let c_merge_ty = if vectorized {
            self.c_simd_type(&self.c_elem_ty, LLVM_VECTOR_WIDTH)
        } else {
            self.c_elem_ty.clone()
        };

        // use C name
        let name = if vectorized {
            format!("{}_vmerge_no_resize", self.name)
        } else {
            format!("{}_merge_no_resize", self.name)
        };

        let c_arg_tys = [
            self.c_pointer_type(&self.name),
            c_merge_ty,
        ];
        let c_ret_ty = &self.c_void_type();
        let mut c_code = self.c_define_function(c_ret_ty, &c_arg_tys, name.clone(), true);

        // for C
        c_code.add("{");
        let appender = self.c_get_param(0);
        let merge_value = self.c_get_param(1);
        c_code.add(format!(
            "{app}->data[{app}->size++] = {val};",
            app=appender,
            val=merge_value,
        ));
        c_code.add("}");
        (*self.ccontext()).prelude_code.add(c_code.result());

        Ok(())
    }
    // VE-Weld NO_RESIZE end

    /// Generates code to merge a value into an appender.
    pub unsafe fn c_gen_merge(
        &mut self,
        intrinsics: &mut Intrinsics,
        run_arg: &str,
        builder_arg: &str,
        value_arg: &str,
        ty: &Type,
        is_no_resize: bool,    // VE-Weld NO_RESIZE
    ) -> WeldResult<String> {
        use crate::ast::Type::Simd;
        let vectorized = if let Simd(_) = ty { true } else { false };
        if vectorized && self.c_vmerge.is_empty() {
            self.define_merge(intrinsics, true)?;
            self.define_merge_no_resize(true)?;     // VE-Weld NO_RESIZE
        } else if !vectorized && self.c_merge.is_empty() {
            self.define_merge(intrinsics, false)?;
            self.define_merge_no_resize(false)?;     // VE-Weld NO_RESIZE
        }

        if !is_no_resize {  // VE-Weld NO_RESIZE
            Ok(format!(
                "{}(&{}, {}, {})",
                if vectorized { &self.c_vmerge } else { &self.c_merge },
                builder_arg,
                value_arg,
                run_arg,
            ))
        // VE-Weld NO_RESIZE begin
        } else {
            Ok(format!(
                "{}_no_resize(&{}, {})",
                if vectorized { &self.c_vmerge } else { &self.c_merge },
                builder_arg,
                value_arg,
            ))
        }
        // VE-Weld NO_RESIZE end
    }

    /// Generates code to get the result from an appender.
    ///
    /// The Appender's result is a vector.
    pub unsafe fn define_result(
        &mut self,
        c_vector_ty: &str,
    ) -> WeldResult<()> {
        // The vector type that the appender generates.
        let c_arg_tys = [self.c_pointer_type(&self.name)];
        let c_ret_ty = &c_vector_ty;

        // Use C name.
        let name = format!("{}_result", self.name);
        let mut c_code = self.c_define_function(c_ret_ty, &c_arg_tys, name.clone(), false);

        // for C
        c_code.add("{");
        // let elem_size = self.size_of(self.elem_ty);
        c_code.add(format!("{} ret;", c_vector_ty));
        let appender = self.c_get_param(0);
        let pointer = self.c_gen_index(&appender, None)?;
        c_code.add(format!("ret.data = {};", pointer));
        c_code.add(format!("ret.size = {}->size;", appender));
        c_code.add("return ret;");
        c_code.add("}");
        (*self.ccontext()).prelude_code.add(c_code.result());
        self.c_result = name;
        Ok(())
    }

    /// Generates code to get the result from an appender.
    ///
    /// The Appender's result is a vector.
    pub unsafe fn c_gen_result(
        &mut self,
        _builder: LLVMBuilderRef,
        c_vector_ty: &str,
        builder_arg: &str,
    ) -> WeldResult<String> {
        if self.c_result.is_empty() {
            self.define_result(c_vector_ty)?;
        }
        Ok(format!("{}(&{})", self.c_result, builder_arg))
    }
}
