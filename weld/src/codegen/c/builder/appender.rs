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
use self::llvm_sys::LLVMIntPredicate::*;
use self::llvm_sys::LLVMTypeKind;

use crate::ast::Type;
use crate::codegen::c::intrinsic::Intrinsics;
use crate::codegen::c::llvm_exts::*;
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
    new: Option<LLVMValueRef>,
    c_new: String,
    merge: Option<LLVMValueRef>,
    c_merge: String,
    vmerge: Option<LLVMValueRef>,
    c_vmerge: String,
    result: Option<LLVMValueRef>,
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
        let mut layout = [
            LLVMPointerType(elem_ty, 0),
            LLVMInt64TypeInContext(context),
            LLVMInt64TypeInContext(context),
        ];
        let appender = LLVMStructCreateNamed(context, c_name.as_ptr());
        LLVMStructSetBody(appender, layout.as_mut_ptr(), layout.len() as u32, 0);
        Appender {
            appender_ty: appender,
            elem_ty,
            c_elem_ty,
            name: c_name.into_string().unwrap(),
            context,
            module,
            ccontext,
            new: None,
            c_new: String::new(),
            merge: None,
            c_merge: String::new(),
            vmerge: None,
            c_vmerge: String::new(),
            result: None,
            c_result: String::new(),
        }
    }

    /// Returns a pointer to the `index`th element in the appender.
    ///
    /// If the `index` is `None`, thie method returns the base pointer. This method does not
    /// perform any bounds checking.
    unsafe fn gen_index(
        &mut self,
        builder: LLVMBuilderRef,
        appender: LLVMValueRef,
        index: Option<LLVMValueRef>,
    ) -> WeldResult<LLVMValueRef> {
        let pointer = LLVMBuildStructGEP(builder, appender, POINTER_INDEX, c_str!(""));
        let pointer = LLVMBuildLoad(builder, pointer, c_str!(""));
        if let Some(index) = index {
            Ok(LLVMBuildGEP(
                builder,
                pointer,
                [index].as_mut_ptr(),
                1,
                c_str!(""),
            ))
        } else {
            Ok(pointer)
        }
    }
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
        if self.new.is_none() {
            let mut arg_tys = [self.i64_type(), self.run_handle_type()];
            let c_arg_tys = [self.c_u64_type(), self.c_run_handle_type()];
            let ret_ty = self.appender_ty;
            let c_ret_ty = &self.name.clone();

            // Use C name.
            let name = format!("{}_new", self.name);
            let (function, builder, _, mut c_code) = self.define_function(ret_ty, c_ret_ty, &mut arg_tys, &c_arg_tys, name.clone());

            // for C
            c_code.add("{");
            // let elem_size = self.size_of(self.elem_ty);
            c_code.add(format!("{} ret;", self.name));
            let bytes = intrinsics.c_call_weld_run_malloc(
                &mut c_code,
                &self.c_get_run(),
                &format!("{capacity} * {elem_size}",
                         capacity=self.c_get_param(0),
                         elem_size=self.c_size_of(&self.c_elem_ty),
                ),
                None,
            );
            c_code.add(format!("ret->data = {};", bytes));
            c_code.add("ret->size = 0;");
            c_code.add(format!("ret->capacity = {};", self.c_get_param(0)));
            c_code.add("return ret;");
            c_code.add("}");
            (*self.ccontext()).prelude_code.add(c_code.result());
            self.c_new = name;
            // for LLVM
            let capacity = LLVMGetParam(function, 0);
            let run = LLVMGetParam(function, 1);
            let elem_size = self.size_of(self.elem_ty);
            let alloc_size = LLVMBuildMul(builder, elem_size, capacity, c_str!("capacity"));
            let bytes =
                intrinsics.call_weld_run_malloc(builder, run, alloc_size, Some(c_str!("bytes")));
            let elements = LLVMBuildBitCast(
                builder,
                bytes,
                LLVMPointerType(self.elem_ty, 0),
                c_str!("elements"),
            );

            let mut result = LLVMGetUndef(self.appender_ty);
            result = LLVMBuildInsertValue(builder, result, elements, POINTER_INDEX, c_str!(""));
            result = LLVMBuildInsertValue(builder, result, self.i64(0), SIZE_INDEX, c_str!(""));
            result = LLVMBuildInsertValue(builder, result, capacity, CAPACITY_INDEX, c_str!(""));
            LLVMBuildRet(builder, result);

            self.new = Some(function);
            LLVMDisposeBuilder(builder);
        }
    }
    /// Generates code for a new appender.
    pub unsafe fn gen_new(
        &mut self,
        builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        run: LLVMValueRef,
        capacity: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        if self.new.is_none() {
            self.define_new(intrinsics);
        }
        let mut args = [capacity, run];
        Ok(LLVMBuildCall(
            builder,
            self.new.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        ))
    }
    pub unsafe fn c_gen_new(
        &mut self,
        _builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        run: &str,
        capacity: &str,
    ) -> WeldResult<String> {
        if self.new.is_none() {
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
        let (merge_ty, c_merge_ty, num_elements) = if vectorized {
            (
                LLVMVectorType(self.elem_ty, LLVM_VECTOR_WIDTH),
                self.c_simd_type(&self.c_elem_ty, LLVM_VECTOR_WIDTH),
                LLVM_VECTOR_WIDTH,
            )
        } else {
            (self.elem_ty, self.c_elem_ty.clone(), 1)
        };

        // use C name
        let name = if vectorized {
            format!("{}_vmerge", self.name)
        } else {
            format!("{}_merge", self.name)
        };

        let mut arg_tys = [
            LLVMPointerType(self.appender_ty, 0),
            merge_ty,
            self.run_handle_type(),
        ];
        let c_arg_tys = [
            self.c_pointer_type(&self.name),
            c_merge_ty,
            self.c_run_handle_type(),
        ];
        let ret_ty = LLVMVoidTypeInContext(self.context);
        let c_ret_ty = &self.c_void_type();
        let (function, builder, _, mut c_code) = self.define_function(ret_ty, c_ret_ty, &mut arg_tys, &c_arg_tys, name.clone());

        // for C
        c_code.add("{");
        let appender = self.c_get_param(0);
        let merge_value = self.c_get_param(1);
        let run_handle = self.c_get_param(2);
        c_code.add(format!(
            "if ({app}->size + 1 > {app}->capacity) {{",
            app=appender,
        ));
        c_code.add(format!(
            "{u64} newCap = {app}->capacity * 2;",
            u64=self.c_u64_type(),
            app=appender,
        ));
        let _ = intrinsics.c_call_weld_run_realloc(
            &mut c_code,
            &run_handle,
            &format!("{app}->data", app=appender),
            "newCap",
            Some(format!("{}->data", appender)),
        );
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
            "++{app}->size;",
            app=appender,
        ));
        c_code.add("}");
        (*self.ccontext()).prelude_code.add(c_code.result());
        if vectorized {
            self.c_vmerge = name;
        } else {
            self.c_merge = name;
        }
        // for LLVM
        LLVMExtAddAttrsOnFunction(self.context, function, &[LLVMExtAttribute::AlwaysInline]);

        let full_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!("isFull"));
        let finish_block = LLVMAppendBasicBlockInContext(self.context, function, c_str!("finish"));

        // Builder is positioned at the entry block - attempt to merge in value.

        let appender = LLVMGetParam(function, 0);
        let merge_value = LLVMGetParam(function, 1);
        let run_handle = LLVMGetParam(function, 2);

        let size_slot = LLVMBuildStructGEP(builder, appender, SIZE_INDEX, c_str!(""));
        let size = LLVMBuildLoad(builder, size_slot, c_str!("size"));

        let capacity_slot = LLVMBuildStructGEP(builder, appender, CAPACITY_INDEX, c_str!(""));
        let capacity = LLVMBuildLoad(builder, capacity_slot, c_str!("capacity"));

        let new_size = LLVMBuildNSWAdd(
            builder,
            self.i64(i64::from(num_elements)),
            size,
            c_str!("newSize"),
        );

        let full = LLVMBuildICmp(builder, LLVMIntSGT, new_size, capacity, c_str!("full"));
        LLVMBuildCondBr(builder, full, full_block, finish_block);

        // Build the case where the appender is full and we need to alloate more memory.
        LLVMPositionBuilderAtEnd(builder, full_block);
        let new_capacity = LLVMBuildNSWMul(builder, capacity, self.i64(2), c_str!("newCapacity"));
        let elem_size = self.size_of(self.elem_ty);
        let alloc_size = LLVMBuildMul(builder, elem_size, new_capacity, c_str!("allocSize"));
        let base_pointer = self.gen_index(builder, appender, None)?;
        let raw_pointer = LLVMBuildBitCast(
            builder,
            base_pointer,
            LLVMPointerType(self.i8_type(), 0),
            c_str!("rawPtr"),
        );
        let bytes = intrinsics.call_weld_run_realloc(
            builder,
            run_handle,
            raw_pointer,
            alloc_size,
            Some(c_str!("bytes")),
        );
        let typed_bytes =
            LLVMBuildBitCast(builder, bytes, LLVMTypeOf(base_pointer), c_str!("typed"));
        let pointer_slot = LLVMBuildStructGEP(builder, appender, POINTER_INDEX, c_str!(""));
        LLVMBuildStore(builder, typed_bytes, pointer_slot);
        LLVMBuildStore(builder, new_capacity, capacity_slot);
        LLVMBuildBr(builder, finish_block);

        // Build the finish block, which merges the value.
        LLVMPositionBuilderAtEnd(builder, finish_block);

        let mut merge_pointer = self.gen_index(builder, appender, Some(size))?;
        if vectorized {
            merge_pointer = LLVMBuildBitCast(
                builder,
                merge_pointer,
                LLVMPointerType(merge_ty, 0),
                c_str!(""),
            );
        }
        let store_inst = LLVMBuildStore(builder, merge_value, merge_pointer);
        if vectorized {
            LLVMSetAlignment(store_inst, 1);
        }
        LLVMBuildStore(builder, new_size, size_slot);
        LLVMBuildRetVoid(builder);

        LLVMDisposeBuilder(builder);
        if vectorized {
            self.vmerge = Some(function);
        } else {
            self.merge = Some(function);
        }
        Ok(())
    }

    /// Generates code to merge a value into an appender.
    pub unsafe fn gen_merge(
        &mut self,
        builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        run_arg: LLVMValueRef,
        builder_arg: LLVMValueRef,
        value_arg: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        let vectorized = LLVMGetTypeKind(LLVMTypeOf(value_arg)) == LLVMTypeKind::LLVMVectorTypeKind;
        if vectorized && self.vmerge.is_none() {
            self.define_merge(intrinsics, true)?;
        } else if !vectorized && self.merge.is_none() {
            self.define_merge(intrinsics, false)?;
        }

        let mut args = [builder_arg, value_arg, run_arg];
        if vectorized {
            Ok(LLVMBuildCall(
                builder,
                self.vmerge.unwrap(),
                args.as_mut_ptr(),
                args.len() as u32,
                c_str!(""),
            ))
        } else {
            Ok(LLVMBuildCall(
                builder,
                self.merge.unwrap(),
                args.as_mut_ptr(),
                args.len() as u32,
                c_str!(""),
            ))
        }
    }
    pub unsafe fn c_gen_merge(
        &mut self,
        _builder: LLVMBuilderRef,
        intrinsics: &mut Intrinsics,
        run_arg: &str,
        builder_arg: &str,
        value_arg: &str,
        ty: &Type,
    ) -> WeldResult<String> {
        use crate::ast::Type::Simd;
        let vectorized = if let Simd(_) = ty { true } else { false };
        if vectorized && self.vmerge.is_none() {
            self.define_merge(intrinsics, true)?;
        } else if !vectorized && self.merge.is_none() {
            self.define_merge(intrinsics, false)?;
        }

        Ok(format!(
            "{}(&{}, {}, {})",
            if vectorized { &self.c_vmerge } else { &self.c_merge },
            builder_arg,
            value_arg,
            run_arg,
        ))
    }

    /// Generates code to get the result from an appender.
    ///
    /// The Appender's result is a vector.
    pub unsafe fn define_result(
        &mut self,
        vector_ty: LLVMTypeRef,
        c_vector_ty: &str,
    ) -> WeldResult<()> {
        // The vector type that the appender generates.
        use crate::codegen::c::vector;
        if self.result.is_none() {
            let mut arg_tys = [LLVMPointerType(self.appender_ty, 0)];
            let c_arg_tys = [self.c_pointer_type(&self.name)];
            let ret_ty = vector_ty;
            let c_ret_ty = &c_vector_ty;

            // Use C name.
            let name = format!("{}_result", self.name);
            let (function, builder, _, mut c_code) = self.define_function(ret_ty, c_ret_ty, &mut arg_tys, &c_arg_tys, name.clone());

            // for C
            c_code.add("{");
            // let elem_size = self.size_of(self.elem_ty);
            c_code.add(format!("{} ret;", c_vector_ty));
            let appender = self.c_get_param(0);
            let pointer = self.c_gen_index(&appender, None)?;
            c_code.add(format!("ret->data = {};", pointer));
            c_code.add(format!("ret->size = {}->size;", appender));
            c_code.add("return ret;");
            c_code.add("}");
            (*self.ccontext()).prelude_code.add(c_code.result());
            self.c_result = name;
            // for LLVM
            let appender = LLVMGetParam(function, 0);

            let pointer = self.gen_index(builder, appender, None)?;
            let size_slot = LLVMBuildStructGEP(builder, appender, SIZE_INDEX, c_str!(""));
            let size = LLVMBuildLoad(builder, size_slot, c_str!("size"));

            let mut result = LLVMGetUndef(vector_ty);
            result =
                LLVMBuildInsertValue(builder, result, pointer, vector::POINTER_INDEX, c_str!(""));
            result = LLVMBuildInsertValue(builder, result, size, vector::SIZE_INDEX, c_str!(""));
            LLVMBuildRet(builder, result);

            self.result = Some(function);
            LLVMDisposeBuilder(builder);
        }
        Ok(())
    }

    /// Generates code to get the result from an appender.
    ///
    /// The Appender's result is a vector.
    pub unsafe fn gen_result(
        &mut self,
        builder: LLVMBuilderRef,
        vector_ty: LLVMTypeRef,
        c_vector_ty: &str,
        builder_arg: LLVMValueRef,
    ) -> WeldResult<LLVMValueRef> {
        if self.result.is_none() {
            self.define_result(vector_ty, c_vector_ty)?;
        }

        let mut args = [builder_arg];
        Ok(LLVMBuildCall(
            builder,
            self.result.unwrap(),
            args.as_mut_ptr(),
            args.len() as u32,
            c_str!(""),
        ))
    }
    pub unsafe fn c_gen_result(
        &mut self,
        _builder: LLVMBuilderRef,
        vector_ty: LLVMTypeRef,
        c_vector_ty: &str,
        builder_arg: &str,
    ) -> WeldResult<String> {
        if self.result.is_none() {
            self.define_result(vector_ty, c_vector_ty)?;
        }
        Ok(format!("{}(&{})", self.c_result, builder_arg))
    }
}
