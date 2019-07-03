//! Traits for code-generating numeric expressions.
//!
//! Specifically, this module provides code generation for the following SIR statements:
//! * `UnaryOp`
//! * `BinaryOp`
//! * `AssignLiteral`
//! * `Cast`
//! * `Negate`

use llvm_sys;

use crate::ast::*;
use crate::error::*;
use crate::sir::*;

use self::llvm_sys::core::*;
use self::llvm_sys::prelude::*;

use crate::codegen::c::intrinsic::Intrinsics;

use super::{CodeGenExt, FunctionContext, CGenerator, LLVM_VECTOR_WIDTH};

/// Generates numeric expresisons.
pub trait NumericExpressionGen {
    /// Generates code for a numeric unary operator.
    ///
    /// This method supports operators over both scalar and SIMD values.
    unsafe fn gen_unaryop(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()>;
    /// Generates code for a numeric binary operator.
    ///
    /// This method supports operators over both scalar and SIMD values.
    unsafe fn gen_binop(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()>;
    /// Generates code for the negation operator.
    unsafe fn gen_negate(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()>;
    /// Generates code for the not operator.
    unsafe fn gen_not(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()>;

    /// Generates a literal.
    ///
    /// This method supports both scalar and SIMD values.
    unsafe fn gen_assign_literal(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()>;
    /// Generates a cast expression.
    unsafe fn gen_cast(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()>;
}

/// Helper trait for generating numeric code.
trait NumericExpressionGenInternal {
    /// Generates the math `Pow` operator.
    unsafe fn gen_pow(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        left: LLVMValueRef,
        right: LLVMValueRef,
        ty: &Type,
    ) -> WeldResult<LLVMValueRef>;
    unsafe fn c_gen_pow(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        left: &str,
        right: &str,
        ty: &Type,
    ) -> WeldResult<String>;
}

impl NumericExpressionGenInternal for CGenerator {
    unsafe fn gen_pow(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        left: LLVMValueRef,
        right: LLVMValueRef,
        ty: &Type,
    ) -> WeldResult<LLVMValueRef> {
        // for LLVM
        use crate::ast::Type::{Scalar, Simd};
        match *ty {
            Scalar(kind) if kind.is_float() => {
                let name = Intrinsics::llvm_numeric("pow", kind, false);
                use crate::ast::ScalarKind::{F32, F64};
                let c_name = match kind {
                    F32 => "powf",
                    F64 => "pow",
                    _ => unreachable!(),
                };
                let ret_ty = LLVMTypeOf(left);
                let c_ret_ty = &self.c_type(&Scalar(kind))? as &str;
                let mut arg_tys = [ret_ty, ret_ty];
                let c_arg_tys = [c_ret_ty, c_ret_ty];
                self.intrinsics.add(&name, &c_name, ret_ty, c_ret_ty, &mut arg_tys, &c_arg_tys);
                self.intrinsics.call(ctx.builder, name, &mut [left, right])
            }
            Simd(kind) if kind.is_float() => {
                let name = Intrinsics::llvm_numeric("pow", kind, false);
                use crate::ast::ScalarKind::{F32, F64};
                let c_name = match kind {
                    F32 => "powf",
                    F64 => "pow",
                    _ => unreachable!(),
                };
                let ret_ty = self.llvm_type(&Scalar(kind))?;
                let c_ret_ty = &self.c_type(&Scalar(kind))? as &str;
                let mut arg_tys = [ret_ty, ret_ty];
                let c_arg_tys = [c_ret_ty, c_ret_ty];
                self.intrinsics.add(&name, &c_name, ret_ty, c_ret_ty, &mut arg_tys, &c_arg_tys);
                // Unroll vector and apply function to each element.
                let mut result = LLVMGetUndef(LLVMVectorType(ret_ty, LLVM_VECTOR_WIDTH));
                for i in 0..LLVM_VECTOR_WIDTH {
                    let base =
                        LLVMBuildExtractElement(ctx.builder, left, self.i32(i as i32), c_str!(""));
                    let power =
                        LLVMBuildExtractElement(ctx.builder, right, self.i32(i as i32), c_str!(""));
                    let value = self
                        .intrinsics
                        .call(ctx.builder, &name, &mut [base, power])?;
                    result = LLVMBuildInsertElement(
                        ctx.builder,
                        result,
                        value,
                        self.i32(i as i32),
                        c_str!(""),
                    );
                }
                Ok(result)
            }
            _ => unreachable!(),
        }
    }
    unsafe fn c_gen_pow(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        left: &str,
        right: &str,
        ty: &Type,
    ) -> WeldResult<String> {
        // for C
        use crate::ast::Type::{Scalar, Simd};
        match *ty {
            Scalar(kind) if kind.is_float() => {
                ctx.body.add(format!("#error gen_pow BinOp pow for scalar {} is not tested yet", self.c_type(ty)?));
                // for LLVM
                let name = Intrinsics::llvm_numeric("pow", kind, false);
                use crate::ast::ScalarKind::{F32, F64};
                let c_name = match kind {
                    F32 => "powf",
                    F64 => "pow",
                    _ => unreachable!(),
                };
                let ret_ty = self.llvm_type(&Scalar(kind))?;
                let c_ret_ty = &self.c_type(&Scalar(kind))? as &str;
                let mut arg_tys = [ret_ty, ret_ty];
                let c_arg_tys = [c_ret_ty, c_ret_ty];
                self.intrinsics.add(&name, &c_name, ret_ty, c_ret_ty, &mut arg_tys, &c_arg_tys);
                let result = (*self.ccontext()).var_ids.next();
                let call = self.intrinsics.c_call(&c_name, &[left, right]);
                ctx.body.add(format!(
                    "{} {} = {};",
                    c_ret_ty,
                    result,
                    call,
                ));
                Ok(result)
            }
            Simd(kind) if kind.is_float() => {
                // for C
                ctx.body.add(format!("#error gen_pow BinOp pow for simd {} is not implemented yet", self.c_type(ty)?));
                // for LLVM
                let name = Intrinsics::llvm_numeric("pow", kind, false);
                use crate::ast::ScalarKind::{F32, F64};
                let c_name = match kind {
                    F32 => "powf",
                    F64 => "pow",
                    _ => unreachable!(),
                };
                let ret_ty = self.llvm_type(&Scalar(kind))?;
                let c_ret_ty = &self.c_type(&Scalar(kind))? as &str;
                let mut arg_tys = [ret_ty, ret_ty];
                let c_arg_tys = [c_ret_ty, c_ret_ty];
                self.intrinsics.add(&name, &c_name, ret_ty, c_ret_ty, &mut arg_tys, &c_arg_tys);
                /*
                // Unroll vector and apply function to each element.
                let mut result = LLVMGetUndef(LLVMVectorType(ret_ty, LLVM_VECTOR_WIDTH));
                for i in 0..LLVM_VECTOR_WIDTH {
                    let base =
                        LLVMBuildExtractElement(ctx.builder, left, self.i32(i as i32), c_str!(""));
                    let power =
                        LLVMBuildExtractElement(ctx.builder, right, self.i32(i as i32), c_str!(""));
                    let value = self
                        .intrinsics
                        .call(ctx.builder, &name, &mut [base, power])?;
                    result = LLVMBuildInsertElement(
                        ctx.builder,
                        result,
                        value,
                        self.i32(i as i32),
                        c_str!(""),
                    );
                }
                */
                Ok("not implemented".to_string() /* result */)
            }
            _ => unreachable!(),
        }
    }
}

trait UnaryOpSupport {
    /// Returns the intrinsic name for a unary op.
    fn llvm_intrinsic(&self) -> Option<&'static str>;
}

impl UnaryOpSupport for UnaryOpKind {
    fn llvm_intrinsic(&self) -> Option<&'static str> {
        use crate::ast::UnaryOpKind::*;
        match *self {
            Exp => Some("exp"),
            Log => Some("log"),
            Sqrt => Some("sqrt"),
            Sin => Some("sin"),
            Cos => Some("cos"),
            _ => None,
        }
    }
}

impl NumericExpressionGen for CGenerator {
    unsafe fn gen_unaryop(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()> {
        use self::UnaryOpSupport;
        use crate::ast::Type::{Scalar, Simd};
        use crate::sir::StatementKind::UnaryOp;
        if let UnaryOp { op, ref child } = statement.kind {
            let ty = ctx.sir_function.symbol_type(child)?;
            let (kind, simd) = match *ty {
                Scalar(kind) => (kind, false),
                Simd(kind) => (kind, true),
                _ => unreachable!(),
            };
            let c_child = ctx.c_get_value(child)?;
            // let child = self.load(ctx.builder, ctx.get_value(child)?)?;

            // Use the LLVM intrinsic if one is available, since LLVM may be able to vectorize it.
            // Otherwise, fall back to the libc math variant and unroll SIMD values manually.
            if let Some(name) = op.llvm_intrinsic() {
                use crate::ast::ScalarKind::{F32, F64};
                use crate::ast::UnaryOpKind::*;
                let c_name = match (op, kind) {
                    (Exp, F32) => "expf",
                    (Log, F32) => "logf",
                    (Sqrt, F32) => "sqrtf",
                    (Sin, F32) => "sinf",
                    (Cos, F32) => "cosf",
                    (Exp, F64) => "exp",
                    (Log, F64) => "log",
                    (Sqrt, F64) => "sqrt",
                    (Sin, F64) => "sin",
                    (Cos, F64) => "cos",
                    _ => unreachable!(),
                };
                let name = Intrinsics::llvm_numeric(name, kind, simd);
                // let ret_ty = LLVMTypeOf(child);
                let ret_ty = self.llvm_type(&Scalar(kind))?;
                let c_ret_ty = &self.c_type(&Scalar(kind))? as &str;
                let mut arg_tys = [ret_ty];
                let c_arg_tys = [c_ret_ty];
                self.intrinsics.add(&name, &c_name, ret_ty, c_ret_ty, &mut arg_tys, &c_arg_tys);
                // for C
                let call = self.intrinsics.c_call(&c_name, &[&c_child]);
                ctx.body.add(format!(
                    "{} = {};",
                    ctx.c_get_value(statement.output.as_ref().unwrap())?,
                    call,
                ));
                // for LLVM
                // self.intrinsics.call(ctx.builder, name, &mut [child])?
            } else {
                use crate::ast::ScalarKind::{F32, F64};
                use crate::ast::UnaryOpKind::*;
                let name = match (op, kind) {
                    (Tan, F32) => "tanf",
                    (ASin, F32) => "asinf",
                    (ACos, F32) => "acosf",
                    (ATan, F32) => "atanf",
                    (Sinh, F32) => "sinhf",
                    (Cosh, F32) => "coshf",
                    (Tanh, F32) => "tanhf",
                    (Erf, F32) => "erff",
                    (Tan, F64) => "tan",
                    (ASin, F64) => "asin",
                    (ACos, F64) => "acos",
                    (ATan, F64) => "atan",
                    (Sinh, F64) => "sinh",
                    (Cosh, F64) => "cosh",
                    (Tanh, F64) => "tanh",
                    (Erf, F64) => "erf",
                    _ => unreachable!(),
                };
                let ret_ty = self.llvm_type(&Scalar(kind))?;
                let c_ret_ty = &self.c_type(&Scalar(kind))? as &str;
                let mut arg_tys = [ret_ty];
                let c_arg_tys = [c_ret_ty];
                self.intrinsics.add(&name, &name, ret_ty, c_ret_ty, &mut arg_tys, &c_arg_tys);
                // If the value is a scalar, just call the intrinsic. If it's a SIMD value, unroll
                // the vector and apply the intrinsic to each element.
                if !simd {
                    // for C
                    let call = self.intrinsics.c_call(&name, &[&c_child]);
                    ctx.body.add(format!(
                        "{} = {};",
                        ctx.c_get_value(statement.output.as_ref().unwrap())?,
                        call,
                    ));
                    // for LLVM
                    // self.intrinsics.call(ctx.builder, name, &mut [child])?
                } else {
                    // for C
                    ctx.body.add(format!("#error UnaryOp of SIMD for {} is not implemented yet", name));
                    // for LLVM
                    /*
                    let mut result = LLVMGetUndef(LLVMVectorType(ret_ty, LLVM_VECTOR_WIDTH));
                    for i in 0..LLVM_VECTOR_WIDTH {
                        let element = LLVMBuildExtractElement(
                            ctx.builder,
                            child,
                            self.i32(i as i32),
                            c_str!(""),
                        );
                        let value = self.intrinsics.call(ctx.builder, &name, &mut [element])?;
                        result = LLVMBuildInsertElement(
                            ctx.builder,
                            result,
                            value,
                            self.i32(i as i32),
                            c_str!(""),
                        );
                    }
                    result
                    */
                }
            };
            /*
            let output = ctx.get_value(statement.output.as_ref().unwrap())?;
            LLVMBuildStore(ctx.builder, result, output);
            */
            Ok(())
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_not(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()> {
        use self::llvm_sys::LLVMIntPredicate::LLVMIntEQ;
        use crate::sir::StatementKind::Not;
        if let Not(ref child) = statement.kind {
            let value = self.load(ctx.builder, ctx.get_value(child)?)?;
            let result = LLVMBuildICmp(ctx.builder, LLVMIntEQ, value, self.bool(false), c_str!(""));
            let result = self.i1_to_bool(ctx.builder, result);
            let output = ctx.get_value(statement.output.as_ref().unwrap())?;
            let _ = LLVMBuildStore(ctx.builder, result, output);
            Ok(())
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_negate(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()> {
        use crate::ast::BinOpKind::Subtract;
        use crate::ast::ScalarKind::{F32, F64};
        use crate::ast::Type::{Scalar, Simd};
        use crate::sir::StatementKind::Negate;
        if let Negate(ref child) = statement.kind {
            let ty = ctx.sir_function.symbol_type(child)?;
            let (kind, simd) = match *ty {
                Scalar(kind) => (kind, false),
                Simd(kind) => (kind, true),
                _ => unreachable!(),
            };

            let mut zero = match kind {
                F32 => LLVMConstReal(self.f32_type(), 0.0),
                F64 => LLVMConstReal(self.f64_type(), 0.0),
                _ => LLVMConstInt(LLVMIntTypeInContext(self.context, kind.bits()), 0, 1),
            };

            if simd {
                zero = LLVMConstVector(
                    [zero; LLVM_VECTOR_WIDTH as usize].as_mut_ptr(),
                    LLVM_VECTOR_WIDTH,
                );
            }

            let child = self.load(ctx.builder, ctx.get_value(child)?)?;
            let result = gen_binop(ctx.builder, Subtract, zero, child, ty)?;
            let output = ctx.get_value(statement.output.as_ref().unwrap())?;
            let _ = LLVMBuildStore(ctx.builder, result, output);
            Ok(())
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_binop(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()> {
        use crate::ast::BinOpKind;
        use crate::ast::Type::{Scalar, Simd, Struct, Vector};
        use crate::sir::StatementKind::BinOp;
        if let BinOp {
            op,
            ref left,
            ref right,
        } = statement.kind
        {
            let ty = ctx.sir_function.symbol_type(left)?;
            match *ty {
                Scalar(_) | Simd(_) => {
                    let c_left = ctx.c_get_value(left)?;
                    let c_right = ctx.c_get_value(right)?;
                    let result = match op {
                        BinOpKind::Pow => self.c_gen_pow(ctx, &c_left, &c_right, ty)?,
                        _ => c_gen_binop(op, &c_left, &c_right, ty)?,
                    };
                    let output = ctx.c_get_value(statement.output.as_ref().unwrap())?;
                    ctx.body.add(format!(
                        "{} = {};",
                        output,
                        result,
                    ));
                }
                Vector(_) | Struct(_) if op.is_comparison() => {
                    // for C
                    ctx.body.add(format!("#error vector/struct BinOp cmp for {} is not implemented yet", self.c_type(ty)?));
                    // for LLVM
                    /*
                    // Note that we assume structs being compared have the same type.
                    let result = match op {
                        BinOpKind::Equal | BinOpKind::NotEqual => {
                            use super::eq::GenEq;
                            let func = self.gen_eq_fn(ty)?;
                            let mut args = [ctx.get_value(left)?, ctx.get_value(right)?];
                            let equal = LLVMBuildCall(
                                ctx.builder,
                                func,
                                args.as_mut_ptr(),
                                args.len() as u32,
                                c_str!(""),
                            );
                            if op == BinOpKind::Equal {
                                equal
                            } else {
                                LLVMBuildNot(ctx.builder, equal, c_str!(""))
                            }
                        }
                        BinOpKind::LessThan | BinOpKind::GreaterThanOrEqual => {
                            use super::cmp::GenCmp;
                            let func = self.gen_cmp_fn(ty)?;
                            let mut args = [ctx.get_value(left)?, ctx.get_value(right)?];
                            let cmp = LLVMBuildCall(
                                ctx.builder,
                                func,
                                args.as_mut_ptr(),
                                args.len() as u32,
                                c_str!(""),
                            );
                            let lt = LLVMBuildICmp(
                                ctx.builder,
                                LLVMIntSLT,
                                cmp,
                                self.i32(0),
                                c_str!(""),
                            );

                            if op == BinOpKind::LessThan {
                                lt
                            } else {
                                LLVMBuildNot(ctx.builder, lt, c_str!(""))
                            }
                        }
                        BinOpKind::GreaterThan | BinOpKind::LessThanOrEqual => {
                            use super::cmp::GenCmp;
                            let func = self.gen_cmp_fn(ty)?;
                            let mut args = [ctx.get_value(left)?, ctx.get_value(right)?];
                            let cmp = LLVMBuildCall(
                                ctx.builder,
                                func,
                                args.as_mut_ptr(),
                                args.len() as u32,
                                c_str!(""),
                            );
                            let gt = LLVMBuildICmp(
                                ctx.builder,
                                LLVMIntSGT,
                                cmp,
                                self.i32(0),
                                c_str!(""),
                            );

                            if op == BinOpKind::GreaterThan {
                                gt
                            } else {
                                LLVMBuildNot(ctx.builder, gt, c_str!(""))
                            }
                        }
                        _ => unreachable!(),
                    };

                    // Extend the `i1` result to a boolean.
                    self.i1_to_bool(ctx.builder, result)
                    */
                }
                // Invalid binary operator.
                _ => unreachable!(),
            };
            /*
            let output = ctx.get_value(statement.output.as_ref().unwrap())?;
            let _ = LLVMBuildStore(ctx.builder, result, output);
            */
            Ok(())
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_assign_literal(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()> {
        use crate::sir::StatementKind::AssignLiteral;
        if let AssignLiteral(ref value) = statement.kind {
            let output = statement.output.as_ref().unwrap();
            let output_type = ctx.sir_function.symbol_type(output)?;
            let mut result = if let LiteralKind::StringLiteral(ref val) = value {
                val.to_string()
            } else {
                self.c_scalar_literal(value)
            };
            if let Type::Simd(_) = output_type {
                result =
                    format!("simd vector of {} is not implemented", result);
                /*
                result = LLVMConstVector(
                    [result; LLVM_VECTOR_WIDTH as usize].as_mut_ptr(),
                    LLVM_VECTOR_WIDTH,
                )
                */
            }
            ctx.body.add(format!(
                "{} = {};",
                ctx.c_get_value(output)?,
                result,
            ));
            Ok(())
        } else {
            unreachable!()
        }
    }

    unsafe fn gen_cast(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()> {
        use crate::sir::StatementKind::Cast;
        let output = &statement.output.clone().unwrap();
        let output_type = ctx.sir_function.symbol_type(output)?;
        if let Cast(ref child, _) = statement.kind {
            let c_output_pointer = ctx.c_get_value(output)?;
            ctx.body.add(format!(
                "{} = ({}){};",
                c_output_pointer,
                &self.c_type(output_type)?,
                ctx.c_get_value(child)?,
            ));
            Ok(())
        } else {
            unreachable!()
        }
    }
}

/// Generates a binary op instruction without intrinsics.
///
/// This function supports code generation for both scalar and SIMD values.
///
/// # Return Types
///
/// If `op.is_comparison()` is true, this function returns a value with type `i1`. Otherwise, this
/// function returns a value of type `LLVMTypeOf(left)`.
pub unsafe fn gen_binop(
    builder: LLVMBuilderRef,
    op: BinOpKind,
    left: LLVMValueRef,
    right: LLVMValueRef,
    ty: &Type,
) -> WeldResult<LLVMValueRef> {
    use self::llvm_sys::LLVMIntPredicate::*;
    use self::llvm_sys::LLVMRealPredicate::*;
    use crate::ast::BinOpKind::*;
    use crate::ast::Type::*;
    let name = c_str!("");
    let result = match *ty {
        Scalar(s) | Simd(s) => match op {
            Add if s.is_integer() => LLVMBuildAdd(builder, left, right, name),
            Add if s.is_float() => LLVMBuildFAdd(builder, left, right, name),

            Subtract if s.is_integer() => LLVMBuildSub(builder, left, right, name),
            Subtract if s.is_float() => LLVMBuildFSub(builder, left, right, name),

            Multiply if s.is_integer() => LLVMBuildMul(builder, left, right, name),
            Multiply if s.is_float() => LLVMBuildFMul(builder, left, right, name),

            Divide if s.is_signed_integer() => LLVMBuildSDiv(builder, left, right, name),
            Divide if s.is_unsigned_integer() => LLVMBuildUDiv(builder, left, right, name),
            Divide if s.is_float() => LLVMBuildFDiv(builder, left, right, name),

            Modulo if s.is_signed_integer() => LLVMBuildSRem(builder, left, right, name),
            Modulo if s.is_unsigned_integer() => LLVMBuildURem(builder, left, right, name),
            Modulo if s.is_float() => LLVMBuildFRem(builder, left, right, name),

            Equal if s.is_integer() || s.is_bool() => {
                LLVMBuildICmp(builder, LLVMIntEQ, left, right, name)
            }
            Equal if s.is_float() => LLVMBuildFCmp(builder, LLVMRealOEQ, left, right, name),

            NotEqual if s.is_integer() || s.is_bool() => {
                LLVMBuildICmp(builder, LLVMIntNE, left, right, name)
            }
            NotEqual if s.is_float() => LLVMBuildFCmp(builder, LLVMRealONE, left, right, name),

            LessThan if s.is_signed_integer() => {
                LLVMBuildICmp(builder, LLVMIntSLT, left, right, name)
            }
            LessThan if s.is_unsigned_integer() => {
                LLVMBuildICmp(builder, LLVMIntULT, left, right, name)
            }
            LessThan if s.is_float() => LLVMBuildFCmp(builder, LLVMRealOLT, left, right, name),

            LessThanOrEqual if s.is_signed_integer() => {
                LLVMBuildICmp(builder, LLVMIntSLE, left, right, name)
            }
            LessThanOrEqual if s.is_unsigned_integer() => {
                LLVMBuildICmp(builder, LLVMIntULE, left, right, name)
            }
            LessThanOrEqual if s.is_float() => {
                LLVMBuildFCmp(builder, LLVMRealOLE, left, right, name)
            }

            GreaterThan if s.is_signed_integer() => {
                LLVMBuildICmp(builder, LLVMIntSGT, left, right, name)
            }
            GreaterThan if s.is_unsigned_integer() => {
                LLVMBuildICmp(builder, LLVMIntUGT, left, right, name)
            }
            GreaterThan if s.is_float() => LLVMBuildFCmp(builder, LLVMRealOGT, left, right, name),

            GreaterThanOrEqual if s.is_signed_integer() => {
                LLVMBuildICmp(builder, LLVMIntSGE, left, right, name)
            }
            GreaterThanOrEqual if s.is_unsigned_integer() => {
                LLVMBuildICmp(builder, LLVMIntUGE, left, right, name)
            }
            GreaterThanOrEqual if s.is_float() => {
                LLVMBuildFCmp(builder, LLVMRealOGE, left, right, name)
            }

            LogicalAnd if s.is_bool() => LLVMBuildAnd(builder, left, right, name),
            BitwiseAnd if s.is_integer() || s.is_bool() => LLVMBuildAnd(builder, left, right, name),

            LogicalOr if s.is_bool() => LLVMBuildOr(builder, left, right, name),
            BitwiseOr if s.is_integer() || s.is_bool() => LLVMBuildOr(builder, left, right, name),

            Xor if s.is_integer() || s.is_bool() => LLVMBuildXor(builder, left, right, name),

            Max => {
                let compare = gen_binop(builder, GreaterThanOrEqual, left, right, ty)?;
                LLVMBuildSelect(builder, compare, left, right, c_str!(""))
            }

            Min => {
                let compare = gen_binop(builder, LessThanOrEqual, left, right, ty)?;
                LLVMBuildSelect(builder, compare, left, right, c_str!(""))
            }

            _ => return compile_err!("Unsupported binary op: {} on {}", op, ty),
        },
        _ => return compile_err!("Unsupported binary op: {} on {}", op, ty),
    };
    Ok(result)
}
pub unsafe fn c_gen_binop(
    op: BinOpKind,
    left: &str,
    right: &str,
    ty: &Type,
) -> WeldResult<String> {
    use crate::ast::BinOpKind::*;
    use crate::ast::Type::*;
    let result = match *ty {
        Scalar(s) | Simd(s) => match op {
            Add if s.is_integer() => format!("{} + {}", left, right),
            Add if s.is_float() => format!("{} + {}", left, right),

            Subtract if s.is_integer() => format!("{} - {}", left, right),
            Subtract if s.is_float() => format!("{} - {}", left, right),

            Multiply if s.is_integer() => format!("{} * {}", left, right),
            Multiply if s.is_float() => format!("{} * {}", left, right),

            Divide if s.is_signed_integer() => format!("{} / {}", left, right),
            Divide if s.is_unsigned_integer() => format!("{} / {}", left, right),
            Divide if s.is_float() => format!("{} / {}", left, right),

            Modulo if s.is_signed_integer() => format!("{} % {}", left, right),
            Modulo if s.is_unsigned_integer() => format!("{} % {}", left, right),
            Modulo if s.is_float() => format!("{} % {}", left, right),

            Equal if s.is_integer() || s.is_bool() => format!("{} == {}", left, right),
            Equal if s.is_float() => format!("{} == {}", left, right),

            NotEqual if s.is_integer() || s.is_bool() => format!("{} != {}", left, right),
            NotEqual if s.is_float() => format!("{} != {}", left, right),

            LessThan if s.is_signed_integer() => format!("{} < {}", left, right),
            LessThan if s.is_unsigned_integer() => format!("{} < {}", left, right),
            LessThan if s.is_float() => format!("{} < {}", left, right),

            LessThanOrEqual if s.is_signed_integer() => format!("{} <= {}", left, right),
            LessThanOrEqual if s.is_unsigned_integer() => format!("{} <= {}", left, right),
            LessThanOrEqual if s.is_float() => format!("{} <= {}", left, right),

            GreaterThan if s.is_signed_integer() => format!("{} > {}", left, right),
            GreaterThan if s.is_unsigned_integer() => format!("{} > {}", left, right),
            GreaterThan if s.is_float() => format!("{} > {}", left, right),

            GreaterThanOrEqual if s.is_signed_integer() => format!("{} >= {}", left, right),
            GreaterThanOrEqual if s.is_unsigned_integer() => format!("{} >= {}", left, right),
            GreaterThanOrEqual if s.is_float() => format!("{} >= {}", left, right),

            LogicalAnd if s.is_bool() => format!("{} && {}", left, right),
            BitwiseAnd if s.is_integer() || s.is_bool() => format!("{} & {}", left, right),

            LogicalOr if s.is_bool() => format!("{} || {}", left, right),
            BitwiseOr if s.is_integer() || s.is_bool() => format!("{} | {}", left, right),

            Xor if s.is_integer() || s.is_bool() => format!("{} ^ {}", left, right),

            Max => format!("{} >= {} ? {} : {}", left, right, left, right),

            Min => format!("{} <= {} ? {} : {}", left, right, left, right),

            _ => return compile_err!("Unsupported binary op: {} on {}", op, ty),
        },
        _ => return compile_err!("Unsupported binary op: {} on {}", op, ty),
    };
    Ok(result)
}
