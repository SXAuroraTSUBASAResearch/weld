//! Code generation for the merger builder type.

use llvm_sys;

use std::ffi::CString;
use code_builder::CodeBuilder;

use crate::ast::BinOpKind;
use crate::ast::ScalarKind;
use crate::ast::Type;
use crate::ast::Type::Scalar;
use crate::error::*;

use self::llvm_sys::core::*;
use self::llvm_sys::prelude::*;

use crate::codegen::c::numeric::c_gen_binop;
use crate::codegen::c::CodeGenExt;
use crate::codegen::c::LLVM_VECTOR_WIDTH;
use crate::codegen::c::CContextRef;

const SCALAR_INDEX: u32 = 0;
const VECTOR_INDEX: u32 = 1;

/// The merger type.
pub struct Merger {
    pub merger_ty: LLVMTypeRef,
    pub name: String,
    pub c_elem_ty: String,
    pub scalar_kind: ScalarKind,
    pub op: BinOpKind,
    context: LLVMContextRef,
    module: LLVMModuleRef,
    ccontext: CContextRef,
    c_new: String,
    c_merge: String,
    c_vmerge: String,
    c_result: String,
}

impl CodeGenExt for Merger {
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

impl Merger {
    pub unsafe fn define<T: AsRef<str>>(
        name: T,
        op: BinOpKind,
        c_elem_ty: &str,
        scalar_kind: ScalarKind,
        context: LLVMContextRef,
        module: LLVMModuleRef,
        ccontext: CContextRef,
    ) -> Merger {
        // for C
        let mut def = CodeBuilder::new();
        def.add("typedef struct {");
        def.add(format!("{elem_ty} data;", elem_ty=c_elem_ty));
        def.add(format!(
            "{elem_ty} vdata[{size}];",
            elem_ty=c_elem_ty,
            size=LLVM_VECTOR_WIDTH,
        ));
        def.add(format!("}} {};", name.as_ref()));
        (*ccontext).prelude_code.add(def.result());
        // for LLVM
        let c_name = CString::new(name.as_ref()).unwrap();
        let merger = LLVMStructCreateNamed(context, c_name.as_ptr());
        Merger {
            name: name.as_ref().to_string(),
            op,
            merger_ty: merger,
            c_elem_ty: c_elem_ty.to_string(),
            scalar_kind,
            context,
            module,
            ccontext,
            c_new: String::new(),
            c_merge: String::new(),
            c_vmerge: String::new(),
            c_result: String::new(),
        }
    }

    pub unsafe fn define_new(
        &mut self,
    ) -> WeldResult<()> {
        let c_ret_ty = &self.name.clone();
        let c_arg_tys = [self.c_elem_ty.clone()];

        // Use C name.
        let name = format!("{}_new", self.name);
        let mut c_code = self.c_define_function(c_ret_ty, &c_arg_tys, name.clone(), false);

        // for C
        let c_identity = self.c_binop_identity(self.op, self.scalar_kind)?;
        c_code.add("{");
        c_code.add(format!("{} ret;", self.name));
        // Initialize only scalar value using first parameter.
        c_code.add(format!("ret.data = {};", self.c_get_param(0)));
        // Initialize vector data using identity.
        c_code.add(format!("\
            for (int i = 0; i < {}; ++i) {{
                ret.vdata[i] = {};
            }}",
            LLVM_VECTOR_WIDTH,
            c_identity,
        ));
        c_code.add("return ret;");
        c_code.add("}");

        (*self.ccontext()).prelude_code.add(c_code.result());
        self.c_new = name;
        Ok(())
    }
    pub unsafe fn c_gen_new(
        &mut self,
        _builder: LLVMBuilderRef,
        init: &str,
    ) -> WeldResult<String> {
        if self.c_new.is_empty() {
            self.define_new()?;
        }
        Ok(format!("{}({})", self.c_new, init))
    }

    /// Builds the `Merge` function and returns a reference to the function.
    ///
    /// The merge function is similar for the scalar and vector varianthe `gep_index determines
    /// which one is generated.
    unsafe fn define_merge(
        &mut self,
        name: String,
        c_arguments: &[String],
        gep_index: u32,
    ) -> WeldResult<()> {
        let vectorized = gep_index != SCALAR_INDEX;
        let c_ret_ty = &self.c_void_type();
        let mut c_code = self.c_define_function(c_ret_ty, c_arguments, name.clone(), true);

        // for C
        // Load the scalar element, apply the binary operator, and then store it back.
        c_code.add("{");
        if !vectorized {
            let merge = c_gen_binop(
                self.op,
                "p0->data",
                "p1",
                &Scalar(self.scalar_kind),
            )?;
            c_code.add(format!("p0->data = {};", merge));
        } else {
            c_code.add(format!("for (int i = 0; i < {}; ++i) {{",
                LLVM_VECTOR_WIDTH));
            let merge = c_gen_binop(
                self.op,
                "p0->vdata[i]",
                "p1[i]",
                &Scalar(self.scalar_kind),
            )?;
            c_code.add(format!("p0->vdata[i] = {};", merge));
            c_code.add("}");
        }
        c_code.add("}");

        (*self.ccontext()).prelude_code.add(c_code.result());
        if !vectorized {
            self.c_merge = name;
        } else {
            self.c_vmerge = name;
        }
        Ok(())
    }

    pub unsafe fn c_gen_merge(
        &mut self,
        _builder: LLVMBuilderRef,
        builder: &str,
        value: &str,
        ty: &Type,
    ) -> WeldResult<String> {
        use crate::ast::Type::Simd;
        let vectorized = if let Simd(_) = ty { true } else { false };
        if vectorized {
            if self.c_vmerge.is_empty() {
                let c_arg_tys = [
                    self.c_pointer_type(&self.name),
                    self.c_simd_type(&self.c_elem_ty, LLVM_VECTOR_WIDTH as u32),
                ];
                let name = format!("{}_vmerge", self.name);
                self.define_merge(name, &c_arg_tys, VECTOR_INDEX)?;
            }
            Ok(format!("{}(&{}, {})", self.c_vmerge, builder, value))
        } else {
            if self.c_merge.is_empty() {
                let c_arg_tys = [self.c_pointer_type(&self.name), self.c_elem_ty.clone()];
                let name = format!("{}_merge", self.name);
                self.define_merge(name, &c_arg_tys, SCALAR_INDEX)?;
            }
            Ok(format!("{}(&{}, {})", self.c_merge, builder, value))
        }
    }

    pub unsafe fn define_result(
        &mut self,
    ) -> WeldResult<()> {
        let c_ret_ty = &self.c_elem_ty.clone();
        let c_arg_tys = [self.c_pointer_type(&self.name)];

        // Use C name.
        let name = format!("{}_result", self.name);
        let mut c_code = self.c_define_function(c_ret_ty, &c_arg_tys, name.clone(), false);

        // for C
        // Load the scalar element, apply the binary operator, and then store it back.
        c_code.add("{");
        c_code.add(format!("{} ret = p0->data;", self.c_elem_ty));
        c_code.add(format!("for (int i = 0; i < {}; ++i) {{",
            LLVM_VECTOR_WIDTH));
        let merge = c_gen_binop(
            self.op,
            "ret",
            "p0->vdata[i]",
            &Scalar(self.scalar_kind),
        )?;
        c_code.add(format!("ret = {};", merge));
        c_code.add("}");
        c_code.add("return ret;");
        c_code.add("}");

        (*self.ccontext()).prelude_code.add(c_code.result());
        self.c_result = name;
        Ok(())
    }
    pub unsafe fn c_gen_result(
        &mut self,
        _builder: LLVMBuilderRef,
        builder: &str,
    ) -> WeldResult<String> {
        if self.c_result.is_empty() {
            self.define_result()?;
        }
        Ok(format!("{}(&{})", self.c_result, builder))
    }
}
