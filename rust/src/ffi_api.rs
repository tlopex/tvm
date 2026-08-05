/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

//! Ergonomic, invariant-preserving wrappers over stubgen's typed globals.
//!
//! Generated object handles are opaque and all field reads are fallible,
//! reflection-backed getters. Constructors in this module always call the
//! canonical typed C++ global; generic reflected builders are not exposed.

use crate::generated::ir::{
    self, BaseFunc, DictAttrs, Expr, GlobalVar, IRModule, IntImm, Op, Span, Type,
};
use crate::generated::tirx::transform as tirx_transform;
use crate::generated::tirx::{
    self, AssertStmt, Buffer, Evaluate, For, IfThenElse, IterVar, PrimFunc, SeqStmt, Stmt,
    StringImm, TilePrimitiveCall, Var,
};
use crate::generated::transform::{self as transform, Pass, PassContext, PassInfo};
use crate::PrimExpr;
use tvm_ffi::{
    object::ObjectRef, Any, AnyValue, AnyView, Array, DLDataType, DLDataTypeExt, Error, Function,
    Map, ObjectRefCast, ObjectRefCore, Result, String as FfiString,
};

pub(crate) fn require_defined<T>(value: Option<T>, context: &str) -> Result<T> {
    value.ok_or_else(|| {
        Error::new(
            tvm_ffi::error::TYPE_ERROR,
            &format!("{context} unexpectedly returned an undefined ObjectRef"),
            "",
        )
    })
}

/// Construct an `ir.IntImm` through the validating C++ constructor.
pub fn int_imm(dtype: DLDataType, value: i64, span: Option<&Span>) -> Result<IntImm> {
    require_defined(ir::int_imm(dtype, value, span.cloned())?, "ir.IntImm")
}

/// Convenience form of [`int_imm`] for TVM dtype strings such as `"int32"`.
pub fn int_imm_from_str(dtype: &str, value: i64, span: Option<&Span>) -> Result<IntImm> {
    int_imm(DLDataType::try_from_str(dtype)?, value, span)
}

/// Add the `PrimExpr` refinement after checking the expression's reflected type.
///
/// This accepts generated primitive nodes such as `IntImm` and `Var` through
/// their ordinary conversion to `Expr`. Passing a non-primitive expression is
/// a recoverable type error.
pub fn prim_expr<E>(expr: E) -> Result<PrimExpr>
where
    E: Into<Expr>,
{
    PrimExpr::try_from_base(expr.into())
}

/// Construct a `tirx.Evaluate` through the validating C++ constructor.
pub fn evaluate(value: &crate::generated::ir::Expr, span: Option<&Span>) -> Result<Evaluate> {
    require_defined(
        tirx::evaluate(Some(value.clone()), span.cloned())?,
        "tirx.Evaluate",
    )
}

/// Construct a TIRx string literal through C++.
pub fn string_imm(value: &str, span: Option<&Span>) -> Result<StringImm> {
    require_defined(
        tirx::string_imm(FfiString::from(value), span.cloned())?,
        "tirx.StringImm",
    )
}

/// Construct a scalar TIRx variable through C++.
pub fn var(name: &str, dtype: DLDataType, span: Option<&Span>) -> Result<Var> {
    require_defined(
        tirx::var(FfiString::from(name), AnyView::from(&dtype), span.cloned())?,
        "tirx.Var",
    )
}

/// Convenience form of [`var`] for a dtype string.
pub fn var_from_str(name: &str, dtype: &str, span: Option<&Span>) -> Result<Var> {
    var(name, DLDataType::try_from_str(dtype)?, span)
}

/// Construct an `AssertStmt` through C++, including its boolean type check.
pub fn assert_stmt(
    condition: &PrimExpr,
    error_kind: &StringImm,
    message_parts: &[StringImm],
    span: Option<&Span>,
) -> Result<AssertStmt> {
    let message_parts = Array::new(message_parts.iter().cloned().map(Some).collect());
    require_defined(
        tirx::assert_stmt(
            Some(condition.clone()),
            Some(error_kind.clone()),
            message_parts,
            span.cloned(),
        )?,
        "tirx.AssertStmt",
    )
}

/// Rebuild a `tirx.IfThenElse` through the canonical C++ constructor.
pub fn if_then_else(
    condition: &PrimExpr,
    then_case: &Stmt,
    else_case: Option<&Stmt>,
    span: Option<&Span>,
) -> Result<IfThenElse> {
    require_defined(
        tirx::if_then_else(
            Some(condition.clone()),
            Some(then_case.clone()),
            else_case.cloned(),
            span.cloned(),
        )?,
        "tirx.IfThenElse",
    )
}

/// Rebuild a `tirx.For` through the canonical constructor.
#[allow(clippy::too_many_arguments)]
pub fn for_loop(
    loop_var: &Var,
    min: &PrimExpr,
    extent: &PrimExpr,
    kind: i64,
    body: &Stmt,
    thread_binding: Option<&IterVar>,
    annotations: &Map<FfiString, AnyValue>,
    step: Option<&PrimExpr>,
    span: Option<&Span>,
) -> Result<For> {
    require_defined(
        tirx::r#for(
            Some(loop_var.clone()),
            Some(min.clone()),
            Some(extent.clone()),
            kind,
            Some(body.clone()),
            thread_binding.cloned(),
            Some(annotations.clone()),
            step.cloned(),
            span.cloned(),
        )?,
        "tirx.For",
    )
}

/// Rebuild a tile primitive call while preserving type-erased scalar/config values.
#[allow(clippy::too_many_arguments)]
pub fn tile_primitive_call(
    op: &Op,
    args: &Array<AnyValue>,
    workspace: &Map<FfiString, Option<Buffer>>,
    config: &Map<FfiString, AnyValue>,
    dispatch_token: Option<&FfiString>,
    scope: Option<&crate::generated::tirx::ExecScope>,
) -> Result<TilePrimitiveCall> {
    require_defined(
        tirx::tile_primitive_call(
            op.clone(),
            args.clone(),
            workspace.clone(),
            config.clone(),
            dispatch_token.cloned(),
            scope.cloned(),
        )?,
        "tirx.TilePrimitiveCall",
    )
}

fn is_evaluate_zero(stmt: &Stmt) -> Result<bool> {
    let Some(evaluate) = crate::visitor::try_downcast::<_, Evaluate>(stmt) else {
        return Ok(false);
    };
    let value = require_defined(evaluate.value()?, "Evaluate::value")?;
    let Some(value) = crate::visitor::try_downcast::<_, IntImm>(&value) else {
        return Ok(false);
    };
    Ok(value.value()? == 0)
}

fn append_flattened(stmt: Stmt, output: &mut Vec<Stmt>) -> Result<()> {
    if let Some(sequence) = crate::visitor::try_downcast::<_, SeqStmt>(&stmt) {
        let children = sequence.seq()?;
        for (index, child) in (&children).into_iter().enumerate() {
            let child = require_defined(child, &format!("SeqStmt::seq[{index}]"))?;
            append_flattened(child, output)?;
        }
    } else if !is_evaluate_zero(&stmt)? {
        output.push(stmt);
    }
    Ok(())
}

fn make_seq_stmt(stmts: Vec<Stmt>, span: Option<&Span>) -> Result<SeqStmt> {
    let stmts = Array::new(stmts.into_iter().map(Some).collect());
    require_defined(tirx::seq_stmt(stmts, span.cloned())?, "tirx.SeqStmt")
}

/// Normalize a sequence with the semantics of C++ `SeqStmt::Flatten`.
///
/// Nested `SeqStmt`s are recursively flattened and `Evaluate(IntImm(0))`
/// no-ops are discarded.  An empty result becomes `Evaluate(0)`, a singleton
/// is returned directly, and only two-or-more statements are passed to the C++
/// `tirx.SeqStmt` constructor. Thus this provides a total ergonomic wrapper
/// around the C++ constructor's strict two-or-more-element invariant.
/// `span` is used only when a replacement node must be created.
pub fn normalize_seq<I>(stmts: I, span: Option<&Span>) -> Result<Stmt>
where
    I: IntoIterator<Item = Stmt>,
{
    let roots: Vec<Stmt> = stmts.into_iter().collect();
    let original_sequence = if roots.len() == 1 {
        crate::visitor::try_downcast::<_, SeqStmt>(&roots[0])
    } else {
        None
    };

    let mut flattened = Vec::new();
    for stmt in roots {
        append_flattened(stmt, &mut flattened)?;
    }

    match flattened.len() {
        0 => {
            let zero = int_imm_from_str("int32", 0, span)?;
            let zero = crate::generated::ir::Expr::from(zero);
            Ok(Stmt::from(evaluate(&zero, span)?))
        }
        1 => Ok(flattened.pop().expect("length checked above")),
        _ => {
            // Match C++ Flatten's COW behavior for an already-normalized single
            // SeqStmt input.
            if let Some(original) = original_sequence {
                let children = original.seq()?;
                let mut unchanged = children.len() == flattened.len();
                if unchanged {
                    for (index, (before, after)) in
                        (&children).into_iter().zip(flattened.iter()).enumerate()
                    {
                        let before = require_defined(before, &format!("SeqStmt::seq[{index}]"))?;
                        if !before.same_as(after) {
                            unchanged = false;
                            break;
                        }
                    }
                }
                if unchanged {
                    return Ok(original.into());
                }
            }
            Ok(Stmt::from(make_seq_stmt(flattened, span)?))
        }
    }
}

/// Construct canonical empty attributes through `ir.IRModule` defaults.
pub fn empty_dict_attrs() -> Result<DictAttrs> {
    let module = require_defined(
        ir::ir_module(Map::new(), None, Map::new())?,
        "ir.IRModule(empty attrs)",
    )?;
    module.attrs()
}

/// Construct an `ir.GlobalVar` through C++.
pub fn global_var(name: &str) -> Result<GlobalVar> {
    require_defined(ir::global_var(FfiString::from(name))?, "ir.GlobalVar")
}

/// Construct a one-function `IRModule` through the canonical C++ constructor.
///
pub fn ir_module_with_prim_func(name: &str, func: &PrimFunc) -> Result<IRModule> {
    let global_var = global_var(name)?;
    let base_func = BaseFunc::from(func.clone());
    let functions = Map::from_iter([(Some(global_var), Some(base_func))]);
    require_defined(ir::ir_module(functions, None, Map::new())?, "ir.IRModule")
}

/// Construct TVM's canonical void type (`TupleType([])`) through C++.
pub fn void_type() -> Result<Type> {
    let fields = Array::<Type>::new(vec![]);
    Ok(ir::tuple_type(fields, None)?.into())
}

/// Construct a `PrimFunc` through C++, allowing the usual missing return type
/// and attrs while requiring an explicitly typed (possibly empty) buffer map.
pub fn prim_func(
    params: &Array<Option<Var>>,
    body: &Stmt,
    ret_type: Option<&Type>,
    buffer_map: &Map<Option<Var>, Option<Buffer>>,
    attrs: Option<&DictAttrs>,
    span: Option<&Span>,
) -> Result<PrimFunc> {
    let ret_type = match ret_type {
        Some(ret_type) => ret_type.clone(),
        None => void_type()?,
    };
    let attrs = match attrs {
        Some(attrs) => attrs.clone(),
        None => empty_dict_attrs()?,
    };
    require_defined(
        tirx::prim_func(
            params.clone(),
            Some(body.clone()),
            ret_type,
            buffer_map.clone(),
            attrs,
            span.cloned(),
        )?,
        "tirx.PrimFunc",
    )
}

/// Construct a parameterless `PrimFunc` with an empty buffer map.
///
/// The two empty containers are passed as their raw FFI handles.  This avoids
/// forcing Rust to resolve `tirx.Var` merely to describe the element type of an
/// empty container, which matters with TVM builds that register types lazily.
pub fn prim_func_without_params(body: &Stmt, span: Option<&Span>) -> Result<PrimFunc> {
    prim_func(&Array::new(vec![]), body, None, &Map::new(), None, span)
}

/// Rebuild a `PrimFunc` with a replacement body via its C++ constructor.
///
/// This preserves params, return type, buffer map, attrs, and span.  The C++
/// constructor also recomputes the function type, which a field-wise native
/// Rust allocation would fail to do.
pub fn prim_func_with_body(func: &PrimFunc, body: &Stmt) -> Result<PrimFunc> {
    let params = func.params()?;
    let ret_type = func.ret_type()?;
    let buffer_map = func.buffer_map()?;
    let attrs = func.attrs()?;
    let span = func.span()?;
    prim_func(
        &params,
        body,
        Some(&ret_type),
        &buffer_map,
        Some(&attrs),
        span.as_ref(),
    )
}

fn required_array(required: &[&str]) -> Array<FfiString> {
    Array::new(required.iter().map(|name| FfiString::from(*name)).collect())
}

/// Construct `transform.PassInfo` through the C++ constructor.
pub fn pass_info(
    opt_level: i32,
    name: &str,
    required: &[&str],
    traceable: bool,
) -> Result<PassInfo> {
    require_defined(
        transform::pass_info(
            i64::from(opt_level),
            FfiString::from(name),
            required_array(required),
            traceable,
        )?,
        "transform.PassInfo",
    )
}

fn callback_arity(args: &[AnyView<'_>], expected: usize) -> Result<()> {
    if args.len() == expected {
        Ok(())
    } else {
        Err(Error::new(
            tvm_ffi::error::VALUE_ERROR,
            &format!(
                "Rust pass callback expected {expected} arguments, got {}",
                args.len()
            ),
            "",
        ))
    }
}

fn callback_arg<T>(args: &[AnyView<'_>], index: usize) -> Result<T>
where
    T: TryFrom<Any, Error = Error>,
{
    let value = args.get(index).ok_or_else(|| {
        Error::new(
            tvm_ffi::error::VALUE_ERROR,
            &format!("Rust pass callback is missing argument #{index}"),
            "",
        )
    })?;
    Any::from(*value).try_into()
}

/// Wrap a typed Rust function as a TVM `PrimFuncPass`.
///
/// tvm-ffi's shared `extern "C"` trampoline converts both ordinary Rust errors
/// and panics into TVM errors before control returns to C++.
pub fn create_prim_func_pass<F>(
    transform: F,
    opt_level: i32,
    name: &str,
    required: &[&str],
    traceable: bool,
) -> Result<Pass>
where
    F: Fn(PrimFunc, IRModule, PassContext) -> Result<PrimFunc> + Send + Sync + 'static,
{
    let info = pass_info(opt_level, name, required, traceable)?;
    create_prim_func_pass_with_info(transform, &info)
}

/// Variant of [`create_prim_func_pass`] using an existing `PassInfo`.
pub fn create_prim_func_pass_with_info<F>(transform: F, info: &PassInfo) -> Result<Pass>
where
    F: Fn(PrimFunc, IRModule, PassContext) -> Result<PrimFunc> + Send + Sync + 'static,
{
    // C++ calls this function with `RValueRef<PrimFunc>` as argument zero.
    // tvm-ffi's Rust typed callback decoder does not currently recognize the
    // special kTVMFFIObjectRValueRef type index.  Converting its AnyView to an
    // owned Any uses TVMFFIAnyViewToOwnedAny, which consumes that rvalue (or
    // copies a normal lvalue), after which the ordinary typed conversion is
    // correct.  The shared callback trampoline also contains any panic raised
    // while decoding an already-consumed or malformed rvalue.
    let callback = Function::from_packed(move |args| {
        callback_arity(args, 3)?;
        let func = callback_arg::<PrimFunc>(args, 0)?;
        let module = callback_arg::<IRModule>(args, 1)?;
        let context = callback_arg::<PassContext>(args, 2)?;
        transform(func, module, context).map(Any::from)
    });

    let pass = require_defined(
        tirx_transform::create_prim_func_pass(callback, Some(info.clone()))?,
        "tirx.transform.CreatePrimFuncPass",
    )?;
    Ok(pass.into())
}

/// Wrap a typed Rust function as a TVM `ModulePass`.
///
/// This is the module-level counterpart needed by analysis passes such as
/// VerifySSA.  It follows the same panic-containment rule as PrimFunc passes.
pub fn create_module_pass<F>(
    transform: F,
    opt_level: i32,
    name: &str,
    required: &[&str],
    traceable: bool,
) -> Result<Pass>
where
    F: Fn(IRModule, PassContext) -> Result<IRModule> + Send + Sync + 'static,
{
    let info = pass_info(opt_level, name, required, traceable)?;
    let callback = Function::from_packed(move |args| {
        callback_arity(args, 2)?;
        let module = callback_arg::<IRModule>(args, 0)?;
        let context = callback_arg::<PassContext>(args, 1)?;
        transform(module, context).map(Any::from)
    });

    let pass = require_defined(
        transform::make_module_pass(callback, Some(info))?,
        "transform.MakeModulePass",
    )?;
    Ok(pass.into())
}

/// Construct a TVM `Sequential` pass through its registered global function.
pub fn sequential(
    passes: Vec<Pass>,
    opt_level: i32,
    name: &str,
    required: &[&str],
    traceable: bool,
) -> Result<Pass> {
    let passes = Array::new(passes.into_iter().map(Some).collect());
    let name = FfiString::from(name);
    let required = required_array(required);
    let opt_level = i64::from(opt_level);
    transform::sequential_packed(&[
        AnyView::from(&passes),
        AnyView::from(&opt_level),
        AnyView::from(&name),
        AnyView::from(&required),
        AnyView::from(&traceable),
    ])?
    .try_into()
}

/// Execute a pass through `transform.RunPass`.
pub fn run_pass(pass: &Pass, module: &IRModule) -> Result<IRModule> {
    let module: ObjectRef = module.clone().try_cast()?;
    require_defined(
        transform::run_pass(Some(pass.clone()), module)?,
        "transform.RunPass",
    )
}
