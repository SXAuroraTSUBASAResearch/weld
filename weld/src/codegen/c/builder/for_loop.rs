//! Code generation for the parallel for loop.
//!
//! This backend currently only generates single threaded code, but its structure is amenable to
//! parallelization (e.g., loops are still divided out into their own function, albeit marked with
//! `alwaysinline` at the moment).
//!
//! The `GenForLoopInternal` is the main workhorse of this module, and provides methods for
//! building a loop, creating bounds checks, loading elements, and so forth.

use llvm_sys;

use std::ffi::CString;

use crate::ast::IterKind::*;
use crate::ast::*;
use crate::error::*;
use crate::runtime::WeldRuntimeErrno;
use crate::sir::*;

use self::llvm_sys::core::*;
use self::llvm_sys::prelude::*;
use self::llvm_sys::{LLVMIntPredicate, LLVMLinkage};

use crate::codegen::c::llvm_exts::LLVMExtAttribute::*;
use crate::codegen::c::llvm_exts::*;
use crate::codegen::c::vector::VectorExt;
use crate::codegen::c::{LLVM_VECTOR_WIDTH, SIR_FUNC_CALL_CONV};

use super::{CodeGenExt, FunctionContext, CGenerator};

/// An internal trait for generating parallel For loops.
pub trait ForLoopGenInternal {
    /// Entry point to generating a for loop.
    ///
    /// This is the only function in the trait that should be called -- all other methods are
    /// helpers.
    unsafe fn gen_for_internal(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        output: &Symbol,
        parfor: &ParallelForData,
    ) -> WeldResult<()>;
    /// Generates bounds checking code for the loop and return number of iterations.
    ///
    /// This function ensures that each iterator will only access in-bounds vector elements and
    /// also ensures that each zipped vector has the same number of consumed elements. If these
    /// checks fail, the generated code raises an error.
    unsafe fn gen_bounds_check(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        parfor: &ParallelForData,
    ) -> WeldResult<(LLVMValueRef, String)>;
    /// Generates the loop body.
    ///
    /// This generates both the loop control flow and the executing body of the loop.
    unsafe fn gen_loop_body_function(
        &mut self,
        program: &SirProgram,
        func: &SirFunction,
        parfor: &ParallelForData,
    ) -> WeldResult<()>;
    /// Generates a bounds check for the given iterator.
    ///
    /// Returns a value representing the number of iterations the iterator will produce.
    unsafe fn gen_iter_bounds_check(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        iterator: &ParallelForIter,
        pass_block: LLVMBasicBlockRef,
        c_pass_block: &str,
        fail_block: LLVMBasicBlockRef,
        c_fail_block: &str,
    ) -> WeldResult<(LLVMValueRef, String)>;
    /// Generates code to check whether the number of iterations in each value is the same.
    ///
    /// If the number of iterations is not the same, the module raises an error and exits.
    unsafe fn gen_check_equal(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        iterations: &[LLVMValueRef],
        c_iterations: &[String],
        pass_block: LLVMBasicBlockRef,
        c_pass_block: &str,
        fail_block: LLVMBasicBlockRef,
        c_fail_block: &str,
    ) -> WeldResult<()>;
    /// Generates code to load potentially zipped elements at index `i` into `e`.
    ///
    /// `e` must be a pointer, and `i` must be a loaded index argument of type `i64`.
    unsafe fn gen_loop_element(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        i: LLVMValueRef,
        c_i: &str,
        e: LLVMValueRef,
        c_e: &str,
        parfor: &ParallelForData,
    ) -> WeldResult<()>;
}

impl ForLoopGenInternal for CGenerator {
    /// Entry point to generating a for loop.
    unsafe fn gen_for_internal(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        output: &Symbol,
        parfor: &ParallelForData,
    ) -> WeldResult<()> {
        let (iterations, c_iterations) = self.gen_bounds_check(ctx, parfor)?;

        let sir_function = &ctx.sir_program.funcs[parfor.body];
        assert!(sir_function.loop_body);

        self.gen_loop_body_function(ctx.sir_program, sir_function, parfor)?;
        let body_function = self.functions[&parfor.body];

        // The parameters of the body function have symbol names that must exist in the current
        // context.
        let mut c_arguments = vec![];
        let mut arguments = vec![];
        for (symbol, _) in sir_function.params.iter() {
            // for C
            c_arguments.push(ctx.c_get_value(symbol)?);
            // for LLVM
            let value = self.load(ctx.builder, ctx.get_value(symbol)?)?;
            arguments.push(value);
        }
        // The body function has an additional arguement representing the number of iterations.
        c_arguments.push(c_iterations);
        arguments.push(iterations);
        // Last argument is always the run handle.
        c_arguments.push(ctx.c_get_run().to_string());
        arguments.push(ctx.get_run());

        // Call the body function, which runs the loop and updates the builder. The updated builder
        // is returned to the current function.
        // for C
        let args_line = self.c_call_args(&c_arguments);
        let c_builder = self.c_call_sir_function(
            &mut ctx.body,
            sir_function,
            &args_line,
            None,
        )?;

        // for LLVM
        let builder = LLVMBuildCall(
            ctx.builder,
            body_function,
            arguments.as_mut_ptr(),
            arguments.len() as u32,
            c_str!(""),
        );
        LLVMSetInstructionCallConv(builder, SIR_FUNC_CALL_CONV);
        // XXX what is parfor.builder now...
        // for C
        ctx.body.add(format!(
            "{} = {};",
            ctx.c_get_value(&parfor.builder)?,
            c_builder,
        ));
        ctx.body.add(format!(
            "{} = {};",
            ctx.c_get_value(&output)?,
            c_builder,
        ));
        // for LLVM
        LLVMBuildStore(ctx.builder, builder, ctx.get_value(&parfor.builder)?);
        LLVMBuildStore(ctx.builder, builder, ctx.get_value(&output)?);

        Ok(())
    }

    /// Generate runtime bounds checking, which looks as follows:
    ///
    /// passed0 = <check bounds of iterator 0>
    /// if passed: goto next, else: goto fail
    /// next:
    /// passed1 = <check bounds of iterator 1>
    /// ...
    /// fail:
    /// raise error
    ///
    /// The bounds check code positions the `FunctionContext` builder after all bounds checking is
    /// complete.
    unsafe fn gen_bounds_check(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        parfor: &ParallelForData,
    ) -> WeldResult<(LLVMValueRef, String)> {
        let mut pass_blocks = vec![];
        for _ in 0..parfor.data.len() {
            pass_blocks.push((
                LLVMAppendBasicBlockInContext(
                    self.context,
                    ctx.llvm_function,
                    c_str!("bounds.check")),
                format!("bounds_check_{}", ctx.bb_index),
            ));
            ctx.bb_index += 1;
        }
        // Jump here if the iterator will cause an array out of bounds error.
        let fail_boundscheck_block =
            LLVMAppendBasicBlockInContext(self.context, ctx.llvm_function, c_str!("bounds.fail"));
        let c_fail_boundscheck_block = format!("bounds_fail_{}", ctx.bb_index);
        ctx.bb_index += 1;
        // Jump here if the zipped vectors produce different numbers of iterations.
        let fail_zip_block =
            LLVMAppendBasicBlockInContext(self.context, ctx.llvm_function, c_str!("bounds.fail"));
        let c_fail_zip_block = format!("bounds_fail_{}", ctx.bb_index);
        ctx.bb_index += 1;
        // Jump here if all checks pass.
        let pass_all_block =
            LLVMAppendBasicBlockInContext(self.context, ctx.llvm_function, c_str!("bounds.passed"));
        let c_pass_all_block = format!("bounds_passed_{}", ctx.bb_index);
        ctx.bb_index += 1;

        if self.conf.enable_bounds_checks {
            info!("Generating bounds checking code")
        } else {
            info!("Omitting bounds checking code")
        }

        let mut num_iterations = vec![];
        let mut c_num_iterations = vec![];
        for (iter, (pass_block, c_pass_block)) in
            parfor.data.iter().zip(pass_blocks)
        {
            let (iterations, c_iterations) = self.gen_iter_bounds_check(
                ctx,
                iter,
                pass_block,
                &c_pass_block,
                fail_boundscheck_block,
                &c_fail_boundscheck_block,
            )?;
            num_iterations.push(iterations);
            c_num_iterations.push(c_iterations);
            LLVMPositionBuilderAtEnd(ctx.builder, pass_block);
            ctx.body.add(format!("{}:", c_pass_block));
        }

        assert!(!num_iterations.is_empty());
        assert!(!c_num_iterations.is_empty());

        // Make sure each iterator produces the same number of iterations.
        if num_iterations.len() > 1 {
            self.gen_check_equal(ctx, &num_iterations, &c_num_iterations, pass_all_block, &c_pass_all_block, fail_zip_block, &c_fail_zip_block)?;
        } else {
            // for C
            ctx.body.add(format!("goto {};", c_pass_all_block));
            // for LLVM
            let _ = LLVMBuildBr(ctx.builder, pass_all_block);
        }

        // for C
        ctx.body.add(format!("{}:", c_fail_boundscheck_block));
        let error = WeldRuntimeErrno::BadIteratorLength.to_string();
        let run = ctx.c_get_run();
        self.intrinsics.c_call_weld_run_set_errno(
            &mut ctx.body, run, &error, None);
        // for LLVM
        LLVMPositionBuilderAtEnd(ctx.builder, fail_boundscheck_block);
        let error = self.i64(WeldRuntimeErrno::BadIteratorLength as i64);
        self.intrinsics
            .call_weld_run_set_errno(ctx.builder, ctx.get_run(), error, None);
        LLVMBuildUnreachable(ctx.builder);

        // for C
        ctx.body.add(format!("{}:", c_fail_zip_block));
        let error = WeldRuntimeErrno::MismatchedZipSize.to_string();
        self.intrinsics.c_call_weld_run_set_errno(
            &mut ctx.body, run, &error, None);
        // for LLVM
        LLVMPositionBuilderAtEnd(ctx.builder, fail_zip_block);
        let error = self.i64(WeldRuntimeErrno::MismatchedZipSize as i64);
        self.intrinsics
            .call_weld_run_set_errno(ctx.builder, ctx.get_run(), error, None);
        LLVMBuildUnreachable(ctx.builder);

        // for C
        ctx.body.add(format!("{}:", c_pass_all_block));
        // for LLVM
        // Bounds check passed - jump to the final block.
        LLVMPositionBuilderAtEnd(ctx.builder, pass_all_block);
        Ok((num_iterations[0], c_num_iterations[0].clone()))
    }

    /// Generate a loop body function.
    ///
    /// A loop body function has the following layout:
    ///
    /// { builders } FuncName(arg1, arg2, ..., iterations, run):
    /// entry:
    ///     alloca all variables except the local builder.
    ///     alias builder argument with parfor.builder_arg
    ///     br loop.begin
    /// loop.begin:
    ///     i = <initializer code>
    ///     br loop.entry
    /// loop.entry:
    ///     if i >= end:
    ///         br loop.exit
    ///     else
    ///         br loop.body
    /// loop.body:
    ///     e = < load elements > based on i
    ///     < generate function body>, replace EndFunction with Br loop.check
    /// loop.check:
    ///     update i
    ///     br loop.entry
    /// loop.exit:
    ///     return { builders }
    unsafe fn gen_loop_body_function(
        &mut self,
        program: &SirProgram,
        func: &SirFunction,
        parfor: &ParallelForData,
    ) -> WeldResult<()> {
        // Construct the return type, which is the builder passed into the function.
        let builders: Vec<Type> = func
            .params
            .values()
            .filter(|v| v.is_builder())
            .cloned()
            .collect();

        // Each loop provides a single builder expression (which could be a struct of builders).
        // The loop's output is by definition derived from this builder.
        assert_eq!(builders.len(), 1);
        let weld_ty = &builders[0];

        // for LLVM
        let mut arg_tys = self.argument_types(func)?;
        // The second-to-last argument is the *total* number of iterations across all threads (in a
        // multi-threaded setting) that this loop will execute for.
        arg_tys.push(self.i64_type());
        let num_iterations_index = (arg_tys.len() - 1) as u32;
        // Last argument is run handle, as always.
        arg_tys.push(self.run_handle_type());

        let ret_ty = self.llvm_type(weld_ty)?;
        let func_ty = LLVMFunctionType(ret_ty, arg_tys.as_mut_ptr(), arg_tys.len() as u32, 0);
        let name = CString::new(format!("f{}_loop", func.id)).unwrap();
        let function = LLVMAddFunction(self.module, name.as_ptr(), func_ty);
        LLVMSetLinkage(function, LLVMLinkage::LLVMPrivateLinkage);

        LLVMExtAddDefaultAttrs(self.context(), function);
        // We always inline this since it will appear only once in the program, so there's no code
        // size cost to doing it.
        //
        // XXX this should only be here in the single threaded setting.
        LLVMExtAddAttrsOnFunction(self.context, function, &[AlwaysInline]);

        self.functions.insert(func.id, function);

        // Create a context for the function.
        let context = &mut FunctionContext::new(self.context, program, func, function);

        // Generate function definition.
        // for C
        let mut c_arg_tys = self.c_argument_types(func)?;
        // The second-to-last argument is the *total* number of iterations across all threads (in a
        // multi-threaded setting) that this loop will execute for.
        c_arg_tys.push(self.c_i64_type());
        let c_num_iterations_index = c_arg_tys.len() - 1;
        // Last argument is run handle, as always.
        c_arg_tys.push(self.c_run_handle_type());

        let c_ret_ty = &self.c_type(weld_ty)?;
        let name = format!("f{}_loop", func.id);
        let args_line = self.c_define_args(&c_arg_tys);
        context.body.add(format!(
            "{} {} {}({})",
            "static inline",            // add inline like what LLVM specifies
            c_ret_ty,
            name,
            args_line,
        ));
        self.c_functions.insert(func.id, name);
        context.body.add("{");
        // Reference to the parameter storing the max number of iterations.
        let max = LLVMGetParam(context.llvm_function, num_iterations_index);
        let c_max = self.c_get_param(c_num_iterations_index);
        // Create the entry basic block, where we define alloca'd variables.
        let entry_bb =
            LLVMAppendBasicBlockInContext(self.context, context.llvm_function, c_str!(""));
        LLVMPositionBuilderAtEnd(context.builder, entry_bb);

        self.gen_allocas(context)?;
        self.gen_store_parameters(context)?;

        // Store the loop induction variable and the builder argument.
        // for C
        context.body.add(format!(
            "{} = {};",
            context.c_get_value(&parfor.builder_arg)?,
            context.c_get_value(&parfor.builder)?,
        ));
        let c_idx = context.c_get_value(&parfor.idx_arg)?;
        context.body.add(format!(
            "for ({} = 0; {} != {}; ++{}) {{",
            c_idx,
            c_idx,
            c_max,
            c_idx,
        ));
        // for LLVM
        LLVMBuildStore(
            context.builder,
            self.load(context.builder, context.get_value(&parfor.builder)?)?,
            context.get_value(&parfor.builder_arg)?,
        );
        LLVMBuildStore(
            context.builder,
            self.i64(0),
            context.get_value(&parfor.idx_arg)?,
        );

        // Add the SIR function basic blocks.
        self.gen_basic_block_defs(context)?;

        // Add the loop end basic blocks.
        let loop_end_bb =
            LLVMAppendBasicBlockInContext(self.context, context.llvm_function, c_str!("loop.end"));
        let loop_exit_bb =
            LLVMAppendBasicBlockInContext(self.context, context.llvm_function, c_str!("loop.exit"));

        // Check whether we need to loop at all.
        let any_iters_cond = LLVMBuildICmp(
            context.builder,
            LLVMIntPredicate::LLVMIntNE,
            max,
            self.i64(0),
            c_str!(""),
        );
        // First body block of the SIR function.
        let first_body_block = context.blocks[&0];
        LLVMBuildCondBr(
            context.builder,
            any_iters_cond,
            first_body_block,
            loop_exit_bb,
        );

        // Build the loop body.
        LLVMPositionBuilderAtEnd(context.builder, first_body_block);

        // Load the loop element.
        let i = self.load(context.builder, context.get_value(&parfor.idx_arg)?)?;
        let c_i = &context.c_get_value(&parfor.idx_arg)?;
        let e = context.get_value(&parfor.data_arg)?;
        let c_e = &context.c_get_value(&parfor.data_arg)?;
        self.gen_loop_element(context, i, c_i, e, c_e, parfor)?;

        // Generate the body - this resembles the usual SIR function generation, but we pass a
        // basic block ID to gen_terminator to change the `EndFunction` terminators to a basic
        // block jump to the end of the loop.
        for bb in func.blocks.iter() {
            // for C
            context.body.add(format!(
                "{}:",
                context.c_get_block(bb.id),
            ));
            // for LLVM
            LLVMPositionBuilderAtEnd(context.builder, context.get_block(bb.id)?);
            for statement in bb.statements.iter() {
                self.gen_statement(context, statement)?;
            }
            let loop_terminator = (loop_end_bb, context.get_value(&parfor.builder_arg)?, context.c_get_value(&parfor.builder_arg)?);
            self.gen_terminator(context, &bb, Some(loop_terminator))?;
        }

        // The EndFunction terminators in the loop body jump to this block.
        LLVMPositionBuilderAtEnd(context.builder, loop_end_bb);

        // Increment the iteration variable i. We don't need to load it again because this block is
        // only reachable from the loop body, which does not mutate i.
        let updated = LLVMBuildNSWAdd(context.builder, i, self.i64(1), c_str!(""));
        LLVMBuildStore(
            context.builder,
            updated,
            context.get_value(&parfor.idx_arg)?,
        );

        // Check whether to continue looping.
        //
        // NOTE: It's important to use `eq` here! LLVM checks for it in its analyses and won't
        // unroll loops without it.
        let finished_cond = LLVMBuildICmp(
            context.builder,
            LLVMIntPredicate::LLVMIntEQ,
            max,
            updated,
            c_str!(""),
        );
        let _ = LLVMBuildCondBr(
            context.builder,
            finished_cond,
            loop_exit_bb,
            first_body_block,
        );
        context.body.add("}");

        // The last basic block loads the updated builder and returns it.
        LLVMPositionBuilderAtEnd(context.builder, loop_exit_bb);
        let updated_builder =
            self.load(context.builder, context.get_value(&parfor.builder_arg)?)?;
        LLVMBuildRet(context.builder, updated_builder);
        context.body.add(format!(
            "return {};",
            context.c_get_value(&parfor.builder_arg)?,
        ));

        context.body.add("}");
        (*self.ccontext()).prelude_code.add(context.body.result());
        Ok(())
    }

    unsafe fn gen_loop_element(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        i: LLVMValueRef,
        c_i: &str,
        e: LLVMValueRef,
        c_e: &str,
        parfor: &ParallelForData,
    ) -> WeldResult<()> {
        let mut values = vec![];
        let mut c_values = vec![];
        for iter in parfor.data.iter() {
            match iter.kind {
                ScalarIter if iter.start.is_some() => {
                    // for C
                    ctx.body.add(format!(
                        "#error gen_loop_element for Part of ScalarIter {} is not implemented yet",
                        ctx.c_get_value(&iter.data)?,
                    ));
                    // for LLVM
                    let start =
                        self.load(ctx.builder, ctx.get_value(iter.start.as_ref().unwrap())?)?;
                    let stride =
                        self.load(ctx.builder, ctx.get_value(iter.stride.as_ref().unwrap())?)?;

                    // Index = (start + stride * i)
                    let tmp = LLVMBuildNSWMul(ctx.builder, stride, i, c_str!(""));
                    let i = LLVMBuildNSWAdd(ctx.builder, start, tmp, c_str!(""));

                    let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                    let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                    let element_pointer = self.gen_at(ctx.builder, vector_type, vector, i)?;
                    let element = self.load(ctx.builder, element_pointer)?;
                    values.push(element);
                }
                ScalarIter => {
                    // for C
                    let vector = &ctx.c_get_value(&iter.data)?;
                    let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                    let element_pointer = self.c_gen_at(
                        ctx.builder, vector_type, vector, c_i)?;
                    let element = ctx.var_ids.next();
                    if let Type::Vector(elem_type) = vector_type {
                        ctx.body.add(format!(
                            "{} {} = *{};",
                            elem_type,
                            element,
                            element_pointer,
                        ));
                    } else {
                        unreachable!()
                    }
                    c_values.push(element);
                    // for LLVM
                    // Iterates over the full vector: Index = i.
                    let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                    let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                    let element_pointer = self.gen_at(ctx.builder, vector_type, vector, i)?;
                    let element = self.load(ctx.builder, element_pointer)?;
                    values.push(element);
                }
                SimdIter if iter.start.is_some() => unreachable!(),
                SimdIter => {
                    // for C
                    ctx.body.add(format!(
                        "#error gen_loop_element for SimdIter {} is not implemented yet",
                        ctx.c_get_value(&iter.data)?,
                    ));
                    // for LLVM
                    let i = LLVMBuildNSWMul(
                        ctx.builder,
                        i,
                        self.i64(i64::from(LLVM_VECTOR_WIDTH)),
                        c_str!(""),
                    );
                    let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                    let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                    let element_pointer = self.gen_vat(ctx.builder, vector_type, vector, i)?;
                    let element = self.load(ctx.builder, element_pointer)?;
                    values.push(element);
                }
                FringeIter if iter.start.is_some() => unreachable!(),
                FringeIter => {
                    // for C
                    ctx.body.add(format!(
                        "#error gen_loop_element for FringeIter {} is not implemented yet",
                        ctx.c_get_value(&iter.data)?,
                    ));
                    // for LLVM
                    let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
                    let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
                    let size = self.gen_size(ctx.builder, vector_type, vector)?;

                    // Start = Len(vector) - Len(vector) % VECTOR_WIDTH
                    // Index = start + i
                    let tmp = LLVMBuildSRem(
                        ctx.builder,
                        size,
                        self.i64(i64::from(LLVM_VECTOR_WIDTH)),
                        c_str!(""),
                    );
                    let start = LLVMBuildNSWSub(ctx.builder, size, tmp, c_str!(""));
                    let i = LLVMBuildNSWAdd(ctx.builder, start, i, c_str!(""));

                    let element_pointer = self.gen_at(ctx.builder, vector_type, vector, i)?;
                    let element = self.load(ctx.builder, element_pointer)?;
                    values.push(element);
                }
                RangeIter => {
                    // for C
                    ctx.body.add(format!(
                        "#error gen_loop_element for RangeIter {} is not implemented yet",
                        ctx.c_get_value(&iter.data)?,
                    ));
                    // for LLVM
                    let start =
                        self.load(ctx.builder, ctx.get_value(iter.start.as_ref().unwrap())?)?;
                    let stride =
                        self.load(ctx.builder, ctx.get_value(iter.stride.as_ref().unwrap())?)?;

                    // Index = (start + stride * i)
                    let tmp = LLVMBuildNSWMul(ctx.builder, stride, i, c_str!(""));
                    let i = LLVMBuildNSWAdd(ctx.builder, start, tmp, c_str!(""));
                    values.push(i);
                }
                NdIter => unimplemented!(), // NdIter Load Element
            }
        }

        assert!(!values.is_empty());

        if values.len() > 1 {
            for (i, value) in values.into_iter().enumerate() {
                let pointer = LLVMBuildStructGEP(ctx.builder, e, i as u32, c_str!(""));
                LLVMBuildStore(ctx.builder, value, pointer);
            }
        } else {
            LLVMBuildStore(ctx.builder, values[0], e);
        }
        if c_values.len() > 1 {
            for (i, value) in c_values.into_iter().enumerate() {
                ctx.body.add(format!(
                    "{}->f{} = {};",
                    c_e,
                    i,
                    value,
                ));
            }
        } else {
            ctx.body.add(format!(
                "{} = {};",
                c_e,
                c_values[0],
            ));
        }
        Ok(())
    }

    unsafe fn gen_iter_bounds_check(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        iter: &ParallelForIter,
        pass_block: LLVMBasicBlockRef,
        c_pass_block: &str,
        fail_block: LLVMBasicBlockRef,
        _c_fail_block: &str,
    ) -> WeldResult<(LLVMValueRef, String)> {
        // for C
        // for LLVM
        let vector = self.load(ctx.builder, ctx.get_value(&iter.data)?)?;
        let c_vector = &ctx.c_get_value(&iter.data)?;
        let vector_type = ctx.sir_function.symbol_type(&iter.data)?;
        let size = self.gen_size(ctx.builder, vector_type, vector)?;
        let c_size = self.c_gen_size(ctx.builder, vector_type, c_vector)?;
        match iter.kind {
            ScalarIter if iter.start.is_some() => {
                // for C
                ctx.body.add(format!(
                    "#error gen_iter_bounds_check for Part of ScalarIter {} is not implemented yet",
                    ctx.c_get_value(&iter.data)?,
                ));
                // for LLVM
                use self::llvm_sys::LLVMIntPredicate::{LLVMIntEQ, LLVMIntSLE, LLVMIntSLT};
                let start = self.load(ctx.builder, ctx.get_value(iter.start.as_ref().unwrap())?)?;
                let stride =
                    self.load(ctx.builder, ctx.get_value(iter.stride.as_ref().unwrap())?)?;
                let end = self.load(ctx.builder, ctx.get_value(iter.end.as_ref().unwrap())?)?;

                let diff = LLVMBuildNSWSub(ctx.builder, end, start, c_str!(""));
                let iterations = LLVMBuildSDiv(ctx.builder, diff, stride, c_str!(""));

                if self.conf.enable_bounds_checks {
                    // Checks required:
                    // start < size
                    // end <= size
                    // start < end
                    // (end - start) % stride == 0
                    // Iterations = (end - start) / stride
                    let start_check =
                        LLVMBuildICmp(ctx.builder, LLVMIntSLT, start, size, c_str!(""));
                    let end_check = LLVMBuildICmp(ctx.builder, LLVMIntSLE, end, size, c_str!(""));
                    let end_start_check =
                        LLVMBuildICmp(ctx.builder, LLVMIntSLT, start, end, c_str!(""));
                    let mod_check = LLVMBuildSRem(ctx.builder, diff, stride, c_str!(""));
                    let mod_check =
                        LLVMBuildICmp(ctx.builder, LLVMIntEQ, mod_check, self.i64(0), c_str!(""));

                    let mut check = LLVMBuildAnd(ctx.builder, start_check, end_check, c_str!(""));
                    check = LLVMBuildAnd(ctx.builder, check, end_start_check, c_str!(""));
                    check = LLVMBuildAnd(ctx.builder, check, mod_check, c_str!(""));
                    let _ = LLVMBuildCondBr(ctx.builder, check, pass_block, fail_block);
                } else {
                    let _ = LLVMBuildBr(ctx.builder, pass_block);
                }

                Ok((iterations, "not implemented".to_string()))
            }
            ScalarIter => {
                // The number of iterations is the size of the vector. No explicit bounds check is
                // necessary here.
                // for C
                ctx.body.add(format!("goto {};", c_pass_block));
                // for LLVM
                let _ = LLVMBuildBr(ctx.builder, pass_block);
                Ok((size, c_size))
            }
            SimdIter if iter.start.is_some() => unreachable!(),
            SimdIter => {
                // for C
                ctx.body.add(format!(
                    "#error gen_iter_bounds_check for SimdIter {} is not implemented yet",
                    ctx.c_get_value(&iter.data)?,
                ));
                // for LLVM
                let iterations = LLVMBuildSDiv(
                    ctx.builder,
                    size,
                    self.i64(i64::from(LLVM_VECTOR_WIDTH)),
                    c_str!(""),
                );
                let _ = LLVMBuildBr(ctx.builder, pass_block);
                Ok((iterations, "not implemented".to_string()))
            }
            FringeIter if iter.start.is_some() => unreachable!(),
            FringeIter => {
                // for C
                ctx.body.add(format!(
                    "#error gen_iter_bounds_check for FringeIter {} is not implemented yet",
                    ctx.c_get_value(&iter.data)?,
                ));
                // for LLVM
                let iterations = LLVMBuildSRem(
                    ctx.builder,
                    size,
                    self.i64(i64::from(LLVM_VECTOR_WIDTH)),
                    c_str!(""),
                );
                let _ = LLVMBuildBr(ctx.builder, pass_block);
                Ok((iterations, "not implemented".to_string()))
            }
            RangeIter => {
                // for C
                ctx.body.add(format!(
                    "#error gen_iter_bounds_check for RangeIter {} is not implemented yet",
                    ctx.c_get_value(&iter.data)?,
                ));
                // for LLVM
                use self::llvm_sys::LLVMIntPredicate::{LLVMIntEQ, LLVMIntSLT};
                let start = self.load(ctx.builder, ctx.get_value(iter.start.as_ref().unwrap())?)?;
                let stride =
                    self.load(ctx.builder, ctx.get_value(iter.stride.as_ref().unwrap())?)?;
                let end = self.load(ctx.builder, ctx.get_value(iter.end.as_ref().unwrap())?)?;
                let diff = LLVMBuildNSWSub(ctx.builder, end, start, c_str!(""));
                let iterations = LLVMBuildSDiv(ctx.builder, diff, stride, c_str!(""));

                if self.conf.enable_bounds_checks {
                    // Checks required:
                    // start < end
                    // (end - start) % stride == 0
                    // Iterations = (end - start) / stride
                    let end_start_check =
                        LLVMBuildICmp(ctx.builder, LLVMIntSLT, start, end, c_str!(""));
                    let mod_check = LLVMBuildSRem(ctx.builder, diff, stride, c_str!(""));
                    let mod_check =
                        LLVMBuildICmp(ctx.builder, LLVMIntEQ, mod_check, self.i64(0), c_str!(""));

                    let check = LLVMBuildAnd(ctx.builder, end_start_check, mod_check, c_str!(""));
                    let _ = LLVMBuildCondBr(ctx.builder, check, pass_block, fail_block);
                } else {
                    let _ = LLVMBuildBr(ctx.builder, pass_block);
                }

                Ok((iterations, "not implemented".to_string()))
            }
            NdIter => unimplemented!(), // NdIter Compute Bounds Check
        }
    }

    unsafe fn gen_check_equal(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        iterations: &[LLVMValueRef],
        c_iterations: &[String],
        pass_block: LLVMBasicBlockRef,
        c_pass_block: &str,
        fail_block: LLVMBasicBlockRef,
        c_fail_block: &str,
    ) -> WeldResult<()> {
        use self::llvm_sys::LLVMIntPredicate::LLVMIntEQ;
        let mut passed = self.i1(true);
        let mut c_passed = "1".to_string();
        if self.conf.enable_bounds_checks {
            // Generate an expression to compare all values.
            // For example,
            //   (iter[0] == iter[1]) && (iter[0] == iter[2]) && ...

            // for C
            for value in c_iterations.iter().skip(1) {
                c_passed = format!(
                    "{} && ({} == {})", c_passed, c_iterations[0], value);
            }
            // for LLVM
            for value in iterations.iter().skip(1) {
                let check =
                    LLVMBuildICmp(ctx.builder, LLVMIntEQ, iterations[0], *value, c_str!(""));
                passed = LLVMBuildAnd(ctx.builder, passed, check, c_str!(""));
            }
        }

        // for C
        ctx.body.add(format!(
            "if ({}) goto {}; else goto {};",
            c_passed,
            c_pass_block,
            c_fail_block,
        ));
        // for LLVM
        LLVMBuildCondBr(ctx.builder, passed, pass_block, fail_block);
        Ok(())
    }
}
