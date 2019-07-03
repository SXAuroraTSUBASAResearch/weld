//! Defines the builder structures and methods over them.
//!
//! Specifically, this module provides an extension trait that generates code for:
//!
//! * `NewBuilder`
//! * `Merge`
//! * `Res`
//!
//! The module additionally encapsulates the builder data structures.

use llvm_sys;

use crate::ast::BinOpKind;
use crate::ast::BuilderKind::*;
use crate::ast::Type::*;
use crate::ast::*;

use crate::error::*;

use crate::sir::StatementKind::*;
use crate::sir::*;

use self::llvm_sys::core::*;
use self::llvm_sys::prelude::*;

// use super::dict;

use super::{CodeGenExt, FunctionContext, CGenerator};

// use super::hash;
use super::numeric;

mod for_loop;

pub mod appender;
pub mod merger;

/// A trait for generating builder code.
///
/// This trait primarily unwraps the builder kind in the statement and muxes functionality to the
/// various builder implementations.
pub trait BuilderExpressionGen {
    /// Merges two pointer values using the provided binary operator.
    ///
    /// Specifically, performs `*builder_value = *builder_value <binop> *merge_value`
    unsafe fn merge_values(
        &mut self,
        builder: LLVMBuilderRef,
        merge_ty: &Type,
        binop: BinOpKind,
        builder_value_pointer: LLVMValueRef,
        merge_value_pointer: LLVMValueRef,
    ) -> WeldResult<()>;
    /// Generates code for the `NewBuilder` statement.
    unsafe fn gen_new_builder(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()>;
    /// Generates code for the `Merge` statement.
    unsafe fn gen_merge(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
        is_no_resize: bool,    // VE-Weld NO_RESIZE
    ) -> WeldResult<()>;
    /// Generates code for the `Result` statement.
    unsafe fn gen_result(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()>;
    /// Generates code for the `ParallelFor` terminator.
    unsafe fn gen_for(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()>;
    /// Generates code to define builder types.
    unsafe fn builder_type(&mut self, builder: &Type) -> WeldResult<LLVMTypeRef>;
    unsafe fn c_builder_type(&mut self, builder: &Type) -> WeldResult<String>;
}

/// Encapsulates the fields of a `NewBuilder` statement.
struct NewBuilderStatement<'a> {
    output: &'a Symbol,
    arg: Option<&'a Symbol>,
    kind: &'a BuilderKind,
}

impl<'a> NewBuilderStatement<'a> {
    fn extract(
        statement: &'a Statement,
        func: &'a SirFunction,
    ) -> WeldResult<NewBuilderStatement<'a>> {
        if statement.output.is_none() {
            unreachable!()
        }
        if let NewBuilder { ref arg, .. } = statement.kind {
            let output = statement.output.as_ref().unwrap();
            let builder_type = func.symbol_type(output)?;
            if let Builder(ref kind, _) = *builder_type {
                let result = NewBuilderStatement {
                    output,
                    arg: arg.as_ref(),
                    kind,
                };
                return Ok(result);
            }
        }
        unreachable!()
    }
}

/// Encapsulates the fields of a `Merge` statement.
struct MergeStatement<'a> {
    builder: &'a Symbol,
    value: &'a Symbol,
    kind: &'a BuilderKind,
}

impl<'a> MergeStatement<'a> {
    fn extract(statement: &'a Statement, func: &'a SirFunction) -> WeldResult<MergeStatement<'a>> {
        if let Merge {
            ref builder,
            ref value,
        } = statement.kind
        {
            let builder_type = func.symbol_type(builder)?;
            if let Builder(ref kind, _) = *builder_type {
                let result = MergeStatement {
                    builder,
                    value,
                    kind,
                };
                return Ok(result);
            }
        }
        unreachable!()
    }
}

/// Encapsulates the fields of a `Res` statement.
struct ResStatement<'a> {
    output: &'a Symbol,
    builder: &'a Symbol,
    kind: &'a BuilderKind,
}

impl<'a> ResStatement<'a> {
    fn extract(statement: &'a Statement, func: &'a SirFunction) -> WeldResult<ResStatement<'a>> {
        if statement.output.is_none() {
            unreachable!()
        }
        if let Res(ref builder) = statement.kind {
            let builder_type = func.symbol_type(builder)?;
            if let Builder(ref kind, _) = *builder_type {
                let result = ResStatement {
                    output: statement.output.as_ref().unwrap(),
                    builder,
                    kind,
                };
                return Ok(result);
            }
        }
        unreachable!()
    }
}

impl BuilderExpressionGen for CGenerator {
    /// Merges two pointer values using the provided binary operator.
    ///
    /// Specifically, performs `*builder_value = *builder_value <binop> *merge_value`
    unsafe fn merge_values(
        &mut self,
        builder: LLVMBuilderRef,
        merge_ty: &Type,
        binop: BinOpKind,
        builder_value_pointer: LLVMValueRef,
        merge_value_pointer: LLVMValueRef,
    ) -> WeldResult<()> {
        match *merge_ty {
            Scalar(_) => {
                let merge_value = self.load(builder, merge_value_pointer)?;
                let builder_value = self.load(builder, builder_value_pointer)?;
                let merged =
                    numeric::gen_binop(builder, binop, builder_value, merge_value, merge_ty)?;
                LLVMBuildStore(builder, merged, builder_value_pointer);
            }
            Struct(ref elems) => {
                for (i, elem) in elems.iter().enumerate() {
                    let builder_elem_pointer =
                        LLVMBuildStructGEP(builder, builder_value_pointer, i as u32, c_str!(""));
                    let builder_value = self.load(builder, builder_elem_pointer)?;
                    let merge_elem_pointer =
                        LLVMBuildStructGEP(builder, merge_value_pointer, i as u32, c_str!(""));
                    let merge_value = self.load(builder, merge_elem_pointer)?;
                    let merged =
                        numeric::gen_binop(builder, binop, builder_value, merge_value, elem)?;
                    LLVMBuildStore(builder, merged, builder_elem_pointer);
                }
            }
            _ => unreachable!(),
        };
        Ok(())
    }

    unsafe fn gen_new_builder(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()> {
        let nb = NewBuilderStatement::extract(statement, ctx.sir_function)?;
        let c_output_pointer = ctx.c_get_value(nb.output)?;
        // let output_pointer = ctx.get_value(nb.output)?;
        match *nb.kind {
            Appender(_) => {
                // for C
                let argument = if let Some(arg) = nb.arg {
                    ctx.c_get_value(arg)?
                } else {
                    format!("{}", appender::DEFAULT_CAPACITY)
                };
                ctx.body.add(format!(
                    "{} = {};",
                    c_output_pointer,
                    {
                        let methods = self.appenders.get_mut(nb.kind).unwrap();
                        let run = ctx.c_get_run();
                        methods.c_gen_new(ctx.builder, &mut self.intrinsics,
                                          run, &argument)?
                    },
                ));

                // for LLVM
                /*
                // The argument is either the provided one or the default capacity.
                let argument = if let Some(arg) = nb.arg {
                    self.load(ctx.builder, ctx.get_value(arg)?)?
                } else {
                    self.i64(appender::DEFAULT_CAPACITY)
                };
                let appender = {
                    let methods = self.appenders.get_mut(nb.kind).unwrap();
                    let run = ctx.get_run();
                    methods.gen_new(ctx.builder, &mut self.intrinsics, run, argument)?
                };
                LLVMBuildStore(ctx.builder, appender, output_pointer);
                */
                Ok(())
            }
            DictMerger(ref key, ref val, _) => {
                // for C
                ctx.body.add(
                    "#error NewBuilder for DictMerger is not implemented yet");

                // for LLVM
                /*
                let dict_type = &Dict(key.clone(), val.clone());
                let default_capacity = self.i64(dict::INITIAL_CAPACITY);
                let dictmerger = {
                    let methods = self.dictionaries.get_mut(dict_type).unwrap();
                    methods.gen_new(
                        ctx.builder,
                        &mut self.intrinsics,
                        default_capacity,
                        ctx.get_run(),
                    )?
                };
                LLVMBuildStore(ctx.builder, dictmerger, output_pointer);
                */
                Ok(())
            }
            GroupMerger(ref key, ref val) => {
                // for C
                ctx.body.add(
                    "#error NewBuilder for GroupMerger is not implemented yet");

                // for LLVM
                /*
                let dict_type = &Dict(key.clone(), Box::new(Vector(val.clone())));
                let default_capacity = self.i64(dict::INITIAL_CAPACITY);
                let groupmerger = {
                    let methods = self.dictionaries.get_mut(dict_type).unwrap();
                    methods.gen_new(
                        ctx.builder,
                        &mut self.intrinsics,
                        default_capacity,
                        ctx.get_run(),
                    )?
                };
                LLVMBuildStore(ctx.builder, groupmerger, output_pointer);
                */
                Ok(())
            }
            Merger(_, _) => {
                // for C
                // The argument is either the provided one or the identity.
                let argument = if let Some(arg) = nb.arg {
                    ctx.c_get_value(arg)?
                } else {
                    let methods = self.mergers.get_mut(nb.kind).unwrap();
                    methods.c_binop_identity(methods.op, methods.scalar_kind)?
                };
                ctx.body.add(format!(
                    "{} = {};",
                    c_output_pointer,
                    {
                        let methods = self.mergers.get_mut(nb.kind).unwrap();
                        methods.c_gen_new(ctx.builder, &argument)?
                    },
                ));

                // for LLVM
                /*
                // The argument is either the provided one or the identity.
                let argument = if let Some(arg) = nb.arg {
                    self.load(ctx.builder, ctx.get_value(arg)?)?
                } else {
                    let methods = self.mergers.get_mut(nb.kind).unwrap();
                    methods.binop_identity(methods.op, methods.scalar_kind)?
                };
                let merger = {
                    let methods = self.mergers.get_mut(nb.kind).unwrap();
                    methods.gen_new(ctx.builder, argument)?
                };
                LLVMBuildStore(ctx.builder, merger, output_pointer);
                */
                Ok(())
            }
            VecMerger(ref elem, _) => {
                // for C
                ctx.body.add(
                    "#error NewBuilder for VecMerger is not implemented yet");

                // for LLVM
                /*
                use super::vector::VectorExt;
                let argument = nb.arg.unwrap();
                let argument = self.load(ctx.builder, ctx.get_value(argument)?)?;
                // XXX This is currently a shallow copy, which isn't quite correct in all cases...
                let builder_value =
                    self.gen_clone(ctx.builder, &Vector(elem.clone()), argument, ctx.get_run())?;
                LLVMBuildStore(ctx.builder, builder_value, output_pointer);
                */
                Ok(())
            }
        }
    }

    unsafe fn gen_merge(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
        is_no_resize: bool,    // VE-Weld NO_RESIZE
    ) -> WeldResult<()> {
        let m = MergeStatement::extract(statement, ctx.sir_function)?;
        // let builder_pointer = ctx.get_value(m.builder)?;
        let c_builder_pointer = ctx.c_get_value(m.builder)?;
        match *m.kind {
            Appender(ref ty) => {
                let methods = self.appenders.get_mut(m.kind).unwrap();
                ctx.body.add(format!(
                    "{};",
                    methods.c_gen_merge(
                        &mut self.intrinsics,
                        ctx.c_get_run(),
                        &c_builder_pointer,
                        &ctx.c_get_value(m.value)?,
                        ty,
                        is_no_resize,    // VE-Weld NO_RESIZE
                    )?,
                ));

                Ok(())
            }
            DictMerger(ref key, ref val, ref binop) => {
                // for C
                ctx.body.add(
                    "#error Merge for DictMerger is not implemented yet");

                // for LLVM
                /*
                use self::hash::*;
                use crate::ast::Type::Scalar;

                // Build the default value that we upsert if the key is not present in the
                // dictionary yet.
                let default = match *val.as_ref() {
                    Scalar(ref kind) => self.binop_identity(*binop, *kind)?,
                    Struct(ref elems) => {
                        let mut default = LLVMGetUndef(self.llvm_type(val)?);
                        for (i, elem) in elems.iter().enumerate() {
                            if let Scalar(ref kind) = *elem {
                                let mut indices = [i as u32];
                                default = LLVMConstInsertValue(
                                    default,
                                    self.binop_identity(*binop, *kind)?,
                                    indices.as_mut_ptr(),
                                    indices.len() as u32,
                                );
                            } else {
                                unreachable!()
                            }
                        }
                        default
                    }
                    _ => unreachable!(),
                };

                // The type of the merge value is {key, value} so use GEP to extract
                // the key and the key pointer.
                let merge_type = ctx.sir_function.symbol_type(m.value)?;
                let (key_pointer, value_pointer) = match *merge_type {
                    // We need this to make sure LLVM doesn't freak out when we use GEP on a
                    // non-struct type or access something out-of-bounds.
                    Struct(ref elems) if elems.len() == 2 => {
                        let merge_value = ctx.get_value(m.value)?;
                        let key_pointer =
                            LLVMBuildStructGEP(ctx.builder, merge_value, 0, c_str!(""));
                        let val_pointer =
                            LLVMBuildStructGEP(ctx.builder, merge_value, 1, c_str!(""));
                        (key_pointer, val_pointer)
                    }
                    _ => unreachable!(),
                };

                let hash = self.gen_hash(key, ctx.builder, key_pointer, None)?;
                let builder_loaded = self.load(ctx.builder, builder_pointer)?;

                let dict_type = &Dict(key.clone(), val.clone());
                let slot_value_pointer = {
                    let methods = self.dictionaries.get_mut(dict_type).unwrap();
                    let slot = methods.gen_upsert(
                        ctx.builder,
                        &mut self.intrinsics,
                        builder_loaded,
                        key_pointer,
                        hash,
                        default,
                        ctx.get_run(),
                    )?;
                    methods.slot_ty.value(ctx.builder, slot)
                };

                // Generate the merge code. We either load the values and add them, or, if the
                // values are structs, we load each element at a time and apply the binop.
                self.merge_values(
                    ctx.builder,
                    val.as_ref(),
                    *binop,
                    slot_value_pointer,
                    value_pointer,
                )
                */
                Ok(())
            }
            GroupMerger(ref key, ref value) => {
                // for C
                ctx.body.add(
                    "#error Merge for GroupMerger is not implemented yet");

                // for LLVM
                /*
                use self::dict::GroupingDict;
                use self::hash::*;
                // The merge value is a {K, V} struct.
                let merge_value_ptr = ctx.get_value(m.value)?;
                let key_pointer = LLVMBuildStructGEP(ctx.builder, merge_value_ptr, 0, c_str!(""));
                let hash = self.gen_hash(key, ctx.builder, key_pointer, None)?;

                let val_pointer = LLVMBuildStructGEP(ctx.builder, merge_value_ptr, 1, c_str!(""));
                let val = self.load(ctx.builder, val_pointer).unwrap();

                let builder_loaded = self.load(ctx.builder, builder_pointer)?;

                let dict_type = &Dict(key.clone(), Box::new(Vector(value.clone())));
                let methods = self.dictionaries.get_mut(dict_type).unwrap();
                let _ = methods.gen_merge_grouped(
                    ctx.builder,
                    &mut self.intrinsics,
                    self.vectors.get_mut(value).unwrap(),
                    builder_loaded,
                    key_pointer,
                    hash,
                    val,
                    ctx.get_run(),
                )?;
                */
                Ok(())
            }
            Merger(ref ty, _) => {
                let c_merge_value = ctx.c_get_value(m.value)?;
                let methods = self.mergers.get_mut(m.kind).unwrap();
                let merge = methods.c_gen_merge(ctx.builder, &c_builder_pointer, &c_merge_value, ty)?;
                ctx.body.add(format!(
                    "{};",
                    merge,
                ));
                Ok(())
            }
            VecMerger(ref elem, ref binop) => {
                // for C
                ctx.body.add(
                    "#error Merge for VecMerger is not implemented yet");

                // for LLVM
                /*
                use super::vector::VectorExt;
                // The type of the merge value is {index, value} so use GEP to extract
                // the key and the key pointer.
                let merge_type = ctx.sir_function.symbol_type(m.value)?;
                let builder_loaded = self.load(ctx.builder, builder_pointer)?;
                let (index_pointer, value_pointer) = match *merge_type {
                    // We need this to make sure LLVM doesn't freak out when we use GEP on a
                    // non-struct type or access something out-of-bounds.
                    Struct(ref elems) if elems.len() == 2 => {
                        let merge_value = ctx.get_value(m.value)?;
                        let index_pointer =
                            LLVMBuildStructGEP(ctx.builder, merge_value, 0, c_str!(""));
                        let val_pointer =
                            LLVMBuildStructGEP(ctx.builder, merge_value, 1, c_str!(""));
                        (index_pointer, val_pointer)
                    }
                    _ => unreachable!(),
                };
                let index = self.load(ctx.builder, index_pointer)?;
                let slot_value_pointer =
                    self.gen_at(ctx.builder, &Vector(elem.clone()), builder_loaded, index)?;

                // Generate the merge code. We either load the values and add them, or, if the
                // values are structs, we load each element at a time and apply the binop.
                self.merge_values(
                    ctx.builder,
                    elem.as_ref(),
                    *binop,
                    slot_value_pointer,
                    value_pointer,
                )
                */
                Ok(())
            }
        }
    }

    unsafe fn gen_result(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()> {
        let m = ResStatement::extract(statement, ctx.sir_function)?;
        // let output_pointer = ctx.get_value(m.output)?;
        let c_output_pointer = ctx.c_get_value(m.output)?;
        // let builder_pointer = ctx.get_value(m.builder)?;
        let c_builder_pointer = ctx.c_get_value(m.builder)?;
        match *m.kind {
            Appender(ref elem_type) => {
                let vector = &Vector(elem_type.clone());
                let c_vector_type = &self.c_type(vector)?.to_string();
                // for C
                let result = {
                    let methods = self.appenders.get_mut(m.kind).unwrap();
                    methods.c_gen_result(ctx.builder, c_vector_type, &c_builder_pointer)?
                };
                ctx.body.add(format!(
                    "{} = {};",
                    c_output_pointer,
                    result,
                ));

                // for LLVM
                /*
                let result = {
                    let methods = self.appenders.get_mut(m.kind).unwrap();
                    methods.gen_result(ctx.builder, vector_type, c_vector_type, builder_pointer)?
                };
                LLVMBuildStore(ctx.builder, result, output_pointer);
                */
                Ok(())
            }
            DictMerger(_, _, _) | GroupMerger(_, _) => {
                // for C
                ctx.body.add(
                    "#error Res for DictMerger/GroupMerger is not implemented yet");

                // for LLVM
                // A dictmerger just updates a dictionary in-place, so return the produced
                // dictionary.
                /*
                let builder_loaded = self.load(ctx.builder, builder_pointer)?;
                LLVMBuildStore(ctx.builder, builder_loaded, output_pointer);
                */
                Ok(())
            }
            Merger(_, _) => {
                // for C
                let result = {
                    let methods = self.mergers.get_mut(m.kind).unwrap();
                    methods.c_gen_result(ctx.builder, &c_builder_pointer)?
                };
                ctx.body.add(format!(
                    "{} = {};",
                    c_output_pointer,
                    result,
                ));

                // for LLVM
                /*
                let result = {
                    let methods = self.mergers.get_mut(m.kind).unwrap();
                    methods.gen_result(ctx.builder, builder_pointer)?
                };
                LLVMBuildStore(ctx.builder, result, output_pointer);
                */
                Ok(())
            }
            VecMerger(_, _) => {
                // for C
                ctx.body.add(
                    "#error Res for VecMerger is not implemented yet");

                // for LLVM
                // VecMergers update a vector in place, so just return the produced vector.
                /*
                let builder_loaded = self.load(ctx.builder, builder_pointer)?;
                LLVMBuildStore(ctx.builder, builder_loaded, output_pointer);
                */
                Ok(())
            }
        }
    }

    unsafe fn gen_for(
        &mut self,
        ctx: &mut FunctionContext<'_>,
        statement: &Statement,
    ) -> WeldResult<()> {
        use self::for_loop::ForLoopGenInternal;
        if statement.output.is_none() {
            unreachable!()
        }
        if let ParallelFor(ref parfor) = statement.kind {
            let output = statement.output.as_ref().unwrap();
            self.gen_for_internal(ctx, output, parfor)
        } else {
            unreachable!()
        }
    }

    unsafe fn builder_type(&mut self, builder: &Type) -> WeldResult<LLVMTypeRef> {
        if let Builder(ref kind, _) = *builder {
            match *kind {
                Appender(ref elem_type) => {
                    if !self.appenders.contains_key(kind) {
                        let c_elem_type = self.c_type(elem_type)?.to_string();
                        let llvm_elem_type = self.llvm_type(elem_type)?;
                        let appender = appender::Appender::define(
                            "appender",
                            llvm_elem_type,
                            c_elem_type,
                            self.context,
                            self.module,
                            self.ccontext,
                        );
                        self.appenders.insert(kind.clone(), appender);
                    }
                    Ok(self.appenders[kind].appender_ty)
                }
                DictMerger(ref key, ref value, _) => {
                    let dict_type = &Dict(key.clone(), value.clone());
                    self.llvm_type(dict_type)
                }
                GroupMerger(ref key, ref value) => {
                    // GroupMerger is backed by dictionary, but the value type is a vector.
                    let dict_type = &Dict(key.clone(), Box::new(Vector(value.clone())));
                    self.llvm_type(dict_type)
                }
                Merger(ref elem_type, ref binop) => {
                    if !self.mergers.contains_key(kind) {
                        let scalar_kind = if let Scalar(ref kind) = *elem_type.as_ref() {
                            *kind
                        } else {
                            unreachable!()
                        };
                        let c_elem_type = &self.c_type(elem_type)?.to_string();
                        let merger = merger::Merger::define(
                            "merger",
                            *binop,
                            c_elem_type,
                            scalar_kind,
                            self.context,
                            self.module,
                            self.ccontext,
                        );
                        self.mergers.insert(kind.clone(), merger);
                    }
                    Ok(self.mergers[kind].merger_ty)
                }
                VecMerger(ref elem, _) => {
                    let vec_type = &Vector(elem.clone());
                    self.llvm_type(vec_type)
                }
            }
        } else {
            unreachable!()
        }
    }
    unsafe fn c_builder_type(&mut self, builder: &Type) -> WeldResult<String> {
        if let Builder(ref kind, _) = *builder {
            match *kind {
                Appender(ref elem_type) => {
                    if !self.appenders.contains_key(kind) {
                        let name = format!("appender{}", self.appender_index);
                        self.appender_index += 1;
                        let c_elem_type = self.c_type(elem_type)?.to_string();
                        let llvm_elem_type = self.llvm_type(elem_type)?;
                        let appender = appender::Appender::define(
                            name,
                            llvm_elem_type,
                            c_elem_type,
                            self.context,
                            self.module,
                            self.ccontext,
                        );
                        self.appenders.insert(kind.clone(), appender);
                    }
                    Ok(self.appenders[kind].name.clone())
                }
                DictMerger(ref key, ref value, _) => {
                    let dict_type = &Dict(key.clone(), value.clone());
                    self.c_type(dict_type)
                }
                GroupMerger(ref key, ref value) => {
                    // GroupMerger is backed by dictionary, but the value type is a vector.
                    let dict_type = &Dict(key.clone(), Box::new(Vector(value.clone())));
                    self.c_type(dict_type)
                }
                Merger(ref elem_type, ref binop) => {
                    if !self.mergers.contains_key(kind) {
                        let scalar_kind = if let Scalar(ref kind) = *elem_type.as_ref() {
                            *kind
                        } else {
                            unreachable!()
                        };
                        let name = format!("merger{}", self.merger_index);
                        self.merger_index += 1;
                        let c_elem_type = &self.c_type(elem_type)?.to_string();
                        let merger = merger::Merger::define(
                            name,
                            *binop,
                            c_elem_type,
                            scalar_kind,
                            self.context,
                            self.module,
                            self.ccontext,
                        );
                        self.mergers.insert(kind.clone(), merger);
                    }
                    Ok(self.mergers[kind].name.clone())
                }
                VecMerger(ref elem, _) => {
                    let vec_type = &Vector(elem.clone());
                    self.c_type(vec_type)
                }
            }
        } else {
            unreachable!()
        }
    }
}
