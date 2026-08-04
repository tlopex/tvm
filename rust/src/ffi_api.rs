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

//! Safe, typed wrappers around TVM global functions used by Rust IR passes.
//!
//! The stub generator currently emits layout mirrors and, for some objects,
//! native all-fields constructors.  Those constructors do not execute the C++
//! validation/normalization logic and cannot see unreflected trailing state.
//! This module therefore constructs every TVM IR/pass object through its
//! registered C++ global function.  In particular, it never calls a
//! `generated::*::new` function.

use std::any::Any as StdAny;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::generated::ir::{
    BaseFunc, DictAttrs, DictAttrsObj, GlobalVar, IRModule, IntImm, IntImmObj, Span, TupleType,
    Type,
};
use crate::generated::tirx::{
    AssertStmt, AttrStmt, Buffer, BufferRegion, Evaluate, EvaluateObj, For, IfThenElse, IterVar,
    MatchBufferRegion, PrimFunc, SBlock, SBlockRealize, SeqStmt, SeqStmtObj, Stmt, StringImm, Var,
    While,
};
use crate::generated::transform::{Pass, PassContext, PassInfo};
use tvm_ffi::{
    object::ObjectRef, Any, AnyView, Array, DLDataType, DLDataTypeExt, Error, Function, Map,
    ObjectCore, Optional, Result, String as FfiString,
};

/// Look up and call a global function.
///
/// Lookup is intentionally retryable instead of being cached in a panicking
/// `LazyLock`: callers may load `libtvm_compiler` after this Rust crate.  A
/// successful lookup is cheap relative to constructing an IR node, and this
/// keeps load-order failures in TVM's normal `Result` error channel.
fn call_global(name: &str, args: &[AnyView<'_>]) -> Result<Any> {
    Function::get_global(name)?.call_packed(args)
}

fn span_view<'a>(span: Option<&'a Span>, none: &'a Any) -> AnyView<'a> {
    match span {
        Some(span) => AnyView::from(span),
        None => AnyView::from(none),
    }
}

/// Construct an `ir.IntImm` through the validating C++ constructor.
pub fn int_imm(dtype: DLDataType, value: i64, span: Option<&Span>) -> Result<IntImm> {
    let none = Any::new();
    call_global(
        "ir.IntImm",
        &[
            AnyView::from(&dtype),
            AnyView::from(&value),
            span_view(span, &none),
        ],
    )?
    .try_into()
}

/// Convenience form of [`int_imm`] for TVM dtype strings such as `"int32"`.
pub fn int_imm_from_str(dtype: &str, value: i64, span: Option<&Span>) -> Result<IntImm> {
    int_imm(DLDataType::try_from_str(dtype)?, value, span)
}

/// Construct a `tirx.Evaluate` through the validating C++ constructor.
pub fn evaluate(value: &crate::generated::ir::Expr, span: Option<&Span>) -> Result<Evaluate> {
    let none = Any::new();
    call_global(
        "tirx.Evaluate",
        &[AnyView::from(value), span_view(span, &none)],
    )?
    .try_into()
}

/// Construct a TIRx string literal through C++.
pub fn string_imm(value: &str, span: Option<&Span>) -> Result<StringImm> {
    let none = Any::new();
    let value = FfiString::from(value);
    call_global(
        "tirx.StringImm",
        &[AnyView::from(&value), span_view(span, &none)],
    )?
    .try_into()
}

/// Construct a scalar TIRx variable through C++.
pub fn var(name: &str, dtype: DLDataType, span: Option<&Span>) -> Result<Var> {
    let none = Any::new();
    let name = FfiString::from(name);
    call_global(
        "tirx.Var",
        &[
            AnyView::from(&name),
            AnyView::from(&dtype),
            span_view(span, &none),
        ],
    )?
    .try_into()
}

/// Convenience form of [`var`] for a dtype string.
pub fn var_from_str(name: &str, dtype: &str, span: Option<&Span>) -> Result<Var> {
    var(name, DLDataType::try_from_str(dtype)?, span)
}

/// Construct an `AssertStmt` through C++, including its boolean type check.
pub fn assert_stmt(
    condition: &crate::generated::ir::Expr,
    error_kind: &StringImm,
    message_parts: &[StringImm],
    span: Option<&Span>,
) -> Result<AssertStmt> {
    let none = Any::new();
    let message_parts = Array::new(message_parts.to_vec());
    call_global(
        "tirx.AssertStmt",
        &[
            AnyView::from(condition),
            AnyView::from(error_kind),
            AnyView::from(&message_parts),
            span_view(span, &none),
        ],
    )?
    .try_into()
}

/// Rebuild a `tirx.AttrStmt` through the canonical C++ constructor.
pub fn attr_stmt(
    node: &Any,
    attr_key: &FfiString,
    value: &crate::generated::ir::Expr,
    body: &Stmt,
    span: Option<&Span>,
) -> Result<AttrStmt> {
    let none = Any::new();
    call_global(
        "tirx.AttrStmt",
        &[
            AnyView::from(node),
            AnyView::from(attr_key),
            AnyView::from(value),
            AnyView::from(body),
            span_view(span, &none),
        ],
    )?
    .try_into()
}

/// Rebuild a `tirx.IfThenElse` through the canonical C++ constructor.
pub fn if_then_else(
    condition: &crate::generated::ir::Expr,
    then_case: &Stmt,
    else_case: Option<&Stmt>,
    span: Option<&Span>,
) -> Result<IfThenElse> {
    let none = Any::new();
    let else_view = else_case
        .map(AnyView::from)
        .unwrap_or_else(|| AnyView::from(&none));
    call_global(
        "tirx.IfThenElse",
        &[
            AnyView::from(condition),
            AnyView::from(then_case),
            else_view,
            span_view(span, &none),
        ],
    )?
    .try_into()
}

/// Rebuild a `tirx.For` without interpreting its type-erased annotations.
///
/// The generated annotation marker is currently `Map<String, ObjectRef>` even
/// though C++ declares `Map<String, Any>`.  Passing the unchanged map handle
/// preserves scalar metadata; reading it through that generated marker is not
/// safe and is intentionally avoided here.
#[allow(clippy::too_many_arguments)]
pub fn for_loop(
    loop_var: &Var,
    min: &crate::generated::ir::Expr,
    extent: &crate::generated::ir::Expr,
    kind: i32,
    body: &Stmt,
    thread_binding: &Optional<IterVar>,
    annotations: &Map<FfiString, ObjectRef>,
    step: &Optional<crate::generated::ir::Expr>,
    span: Option<&Span>,
) -> Result<For> {
    let none = Any::new();
    let thread_binding = thread_binding.get();
    let thread_binding_view = thread_binding
        .as_ref()
        .map(AnyView::from)
        .unwrap_or_else(|| AnyView::from(&none));
    let step = step.get();
    let step_view = step
        .as_ref()
        .map(AnyView::from)
        .unwrap_or_else(|| AnyView::from(&none));
    call_global(
        "tirx.For",
        &[
            AnyView::from(loop_var),
            AnyView::from(min),
            AnyView::from(extent),
            AnyView::from(&kind),
            AnyView::from(body),
            thread_binding_view,
            AnyView::from(annotations),
            step_view,
            span_view(span, &none),
        ],
    )?
    .try_into()
}

/// Rebuild a `tirx.While` through the canonical C++ constructor.
pub fn while_loop(
    condition: &crate::generated::ir::Expr,
    body: &Stmt,
    span: Option<&Span>,
) -> Result<While> {
    let none = Any::new();
    call_global(
        "tirx.While",
        &[
            AnyView::from(condition),
            AnyView::from(body),
            span_view(span, &none),
        ],
    )?
    .try_into()
}

/// Rebuild a TIRx block, preserving all non-statement fields verbatim.
#[allow(clippy::too_many_arguments)]
pub fn sblock(
    iter_vars: &Array<IterVar>,
    reads: &Array<BufferRegion>,
    writes: &Array<BufferRegion>,
    name_hint: &FfiString,
    body: &Stmt,
    init: Option<&Stmt>,
    alloc_buffers: &Array<Buffer>,
    match_buffers: &Array<MatchBufferRegion>,
    annotations: &Map<FfiString, ObjectRef>,
    span: Option<&Span>,
) -> Result<SBlock> {
    let none = Any::new();
    let init_view = init
        .map(AnyView::from)
        .unwrap_or_else(|| AnyView::from(&none));
    call_global(
        "tirx.SBlock",
        &[
            AnyView::from(iter_vars),
            AnyView::from(reads),
            AnyView::from(writes),
            AnyView::from(name_hint),
            AnyView::from(body),
            init_view,
            AnyView::from(alloc_buffers),
            AnyView::from(match_buffers),
            AnyView::from(annotations),
            span_view(span, &none),
        ],
    )?
    .try_into()
}

/// Rebuild a `tirx.SBlockRealize` through the canonical C++ constructor.
pub fn sblock_realize(
    iter_values: &Array<crate::generated::ir::Expr>,
    predicate: &crate::generated::ir::Expr,
    block: &SBlock,
    span: Option<&Span>,
) -> Result<SBlockRealize> {
    let none = Any::new();
    call_global(
        "tirx.SBlockRealize",
        &[
            AnyView::from(iter_values),
            AnyView::from(predicate),
            AnyView::from(block),
            span_view(span, &none),
        ],
    )?
    .try_into()
}

fn is_evaluate_zero(stmt: &Stmt) -> bool {
    let Some(evaluate) = stmt.downcast::<EvaluateObj>() else {
        return false;
    };
    evaluate
        .value
        .downcast::<IntImmObj>()
        .is_some_and(|value| value.value == 0)
}

fn append_flattened(stmt: Stmt, output: &mut Vec<Stmt>) {
    if let Some(sequence) = stmt.downcast::<SeqStmtObj>() {
        // Clone the children before recursively consuming them; they remain
        // pointer-identical object references.
        let children: Vec<Stmt> = sequence.seq.iter().collect();
        for child in children {
            append_flattened(child, output);
        }
    } else if !is_evaluate_zero(&stmt) {
        output.push(stmt);
    }
}

fn make_seq_stmt(stmts: Vec<Stmt>, span: Option<&Span>) -> Result<SeqStmt> {
    let none = Any::new();
    let stmts = Array::new(stmts);
    call_global(
        "tirx.SeqStmt",
        &[AnyView::from(&stmts), span_view(span, &none)],
    )?
    .try_into()
}

/// Normalize a sequence with the semantics of C++ `SeqStmt::Flatten`.
///
/// Nested `SeqStmt`s are recursively flattened and `Evaluate(IntImm(0))`
/// no-ops are discarded.  An empty result becomes `Evaluate(0)`, a singleton
/// is returned directly, and only two-or-more statements are passed to the C++
/// `tirx.SeqStmt` constructor.  Thus this function cannot create the invalid
/// zero- or one-element `SeqStmt`s permitted by the generated native builder.
/// `span` is used only when a replacement node must be created.
pub fn normalize_seq<I>(stmts: I, span: Option<&Span>) -> Result<Stmt>
where
    I: IntoIterator<Item = Stmt>,
{
    let roots: Vec<Stmt> = stmts.into_iter().collect();
    let original_sequence = if roots.len() == 1 && roots[0].downcast::<SeqStmtObj>().is_some() {
        Some(roots[0].clone())
    } else {
        None
    };

    let mut flattened = Vec::new();
    for stmt in roots {
        append_flattened(stmt, &mut flattened);
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
                let original_node = original
                    .downcast::<SeqStmtObj>()
                    .expect("candidate checked above");
                let unchanged = original_node.seq.len() == flattened.len()
                    && original_node
                        .seq
                        .iter()
                        .zip(flattened.iter())
                        .all(|(before, after)| before.same_as(after));
                if unchanged {
                    return Ok(original);
                }
            }
            Ok(Stmt::from(make_seq_stmt(flattened, span)?))
        }
    }
}

/// Construct an empty, defined `DictAttrs` through its reflected C++
/// `__ffi_init__` hook.
///
/// A general `Map<String, Any>` wrapper cannot be exposed until stubgen stops
/// narrowing `Any` container elements to `ObjectRef`, but the empty map needed
/// by canonical `PrimFunc` construction is unambiguous.
pub fn empty_dict_attrs() -> Result<DictAttrs> {
    let dict = Map::<FfiString, ObjectRef>::new();
    let init = crate::ffi_compat::from_type_method(DictAttrsObj::type_index(), "__ffi_init__")?;
    init.call_packed(&[AnyView::from(&dict)])?.try_into()
}

/// Construct an `ir.GlobalVar` through C++.
pub fn global_var(name: &str) -> Result<GlobalVar> {
    let name = FfiString::from(name);
    call_global("ir.GlobalVar", &[AnyView::from(&name)])?.try_into()
}

/// Construct a one-function `IRModule` through the canonical C++ constructor.
///
/// Keeping this focused helper in the prototype is enough to execute Rust pass
/// callbacks end-to-end without exposing more incorrectly narrowed
/// `Map<String, Any>` markers.  A generated SDK should eventually provide the
/// general typed `IRModule` constructor.
pub fn ir_module_with_prim_func(name: &str, func: &PrimFunc) -> Result<IRModule> {
    let global_var = global_var(name)?;
    let base_func = BaseFunc::from(func.clone());
    let functions = call_global(
        "ffi.Map",
        &[AnyView::from(&global_var), AnyView::from(&base_func)],
    )?;
    let global_infos = call_global("ffi.Map", &[])?;
    let attrs = empty_dict_attrs()?;
    call_global(
        "ir.IRModule",
        &[
            AnyView::from(&functions),
            AnyView::from(&attrs),
            AnyView::from(&global_infos),
        ],
    )?
    .try_into()
}

/// Construct TVM's canonical void type (`TupleType([])`) through C++.
pub fn void_type() -> Result<Type> {
    let none = Any::new();
    let fields = Array::<Type>::new(vec![]);
    let tuple: TupleType = call_global(
        "ir.TupleType",
        &[AnyView::from(&fields), AnyView::from(&none)],
    )?
    .try_into()?;
    Ok(tuple.into())
}

/// Construct a `PrimFunc` through C++, allowing the usual missing return type
/// and attrs while requiring an explicitly typed (possibly empty) buffer map.
pub fn prim_func(
    params: &Array<Var>,
    body: &Stmt,
    ret_type: Option<&Type>,
    buffer_map: &Map<Var, Buffer>,
    attrs: Option<&DictAttrs>,
    span: Option<&Span>,
) -> Result<PrimFunc> {
    let none = Any::new();
    let default_ret_type;
    let ret_type = match ret_type {
        Some(ret_type) => AnyView::from(ret_type),
        None => {
            default_ret_type = void_type()?;
            AnyView::from(&default_ret_type)
        }
    };
    let default_attrs;
    let attrs = match attrs {
        Some(attrs) => AnyView::from(attrs),
        None => {
            default_attrs = empty_dict_attrs()?;
            AnyView::from(&default_attrs)
        }
    };
    call_global(
        "tirx.PrimFunc",
        &[
            AnyView::from(params),
            AnyView::from(body),
            ret_type,
            AnyView::from(buffer_map),
            attrs,
            span_view(span, &none),
        ],
    )?
    .try_into()
}

/// Construct a parameterless `PrimFunc` with an empty buffer map.
///
/// The two empty containers are passed as their raw FFI handles.  This avoids
/// forcing Rust to resolve `tirx.Var` merely to describe the element type of an
/// empty container, which matters with TVM builds that register types lazily.
pub fn prim_func_without_params(body: &Stmt, span: Option<&Span>) -> Result<PrimFunc> {
    let none = Any::new();
    let params = Array::<Var>::new(vec![]);
    let buffer_map = call_global("ffi.Map", &[])?;
    let ret_type = void_type()?;
    let attrs = empty_dict_attrs()?;
    call_global(
        "tirx.PrimFunc",
        &[
            AnyView::from(&params),
            AnyView::from(body),
            AnyView::from(&ret_type),
            AnyView::from(&buffer_map),
            AnyView::from(&attrs),
            span_view(span, &none),
        ],
    )?
    .try_into()
}

/// Rebuild a `PrimFunc` with a replacement body via its C++ constructor.
///
/// This preserves params, return type, buffer map, attrs, and span.  The C++
/// constructor also recomputes the function type, which a field-wise native
/// Rust allocation would fail to do.
pub fn prim_func_with_body(func: &PrimFunc, body: &Stmt) -> Result<PrimFunc> {
    let none = Any::new();
    call_global(
        "tirx.PrimFunc",
        &[
            AnyView::from(&func.params),
            AnyView::from(body),
            AnyView::from(&func.ret_type),
            AnyView::from(&func.buffer_map),
            AnyView::from(&func.attrs),
            span_view(Some(&func.span), &none),
        ],
    )?
    .try_into()
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
    let name = FfiString::from(name);
    let required = required_array(required);
    call_global(
        "transform.PassInfo",
        &[
            AnyView::from(&opt_level),
            AnyView::from(&name),
            AnyView::from(&required),
            AnyView::from(&traceable),
        ],
    )?
    .try_into()
}

fn panic_message(payload: Box<dyn StdAny + Send>) -> std::string::String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<std::string::String>() {
        message.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

fn callback_panic(kind: &str, payload: Box<dyn StdAny + Send>) -> Error {
    Error::new(
        tvm_ffi::error::RUNTIME_ERROR,
        &format!("panic in Rust {kind} callback: {}", panic_message(payload)),
        "",
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
/// The callback is panic-contained before entering tvm-ffi's `extern "C"`
/// trampoline, so both ordinary Rust errors and panics reach C++ as TVM errors.
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
    // correct.  Keep decoding inside catch_unwind as the conversion shim in
    // the current runtime can panic on an already-consumed rvalue.
    let callback = Function::from_packed(move |args| {
        match catch_unwind(AssertUnwindSafe(|| -> Result<Any> {
            callback_arity(args, 3)?;
            let func = callback_arg::<PrimFunc>(args, 0)?;
            let module = callback_arg::<IRModule>(args, 1)?;
            let context = callback_arg::<PassContext>(args, 2)?;
            transform(func, module, context).map(Any::from)
        })) {
            Ok(result) => result,
            Err(payload) => Err(callback_panic("PrimFuncPass", payload)),
        }
    });

    call_global(
        "tirx.transform.CreatePrimFuncPass",
        &[AnyView::from(&callback), AnyView::from(info)],
    )?
    .try_into()
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
        match catch_unwind(AssertUnwindSafe(|| -> Result<Any> {
            callback_arity(args, 2)?;
            let module = callback_arg::<IRModule>(args, 0)?;
            let context = callback_arg::<PassContext>(args, 1)?;
            transform(module, context).map(Any::from)
        })) {
            Ok(result) => result,
            Err(payload) => Err(callback_panic("ModulePass", payload)),
        }
    });

    call_global(
        "transform.MakeModulePass",
        &[AnyView::from(&callback), AnyView::from(&info)],
    )?
    .try_into()
}

/// Construct a TVM `Sequential` pass through its registered global function.
pub fn sequential(
    passes: Vec<Pass>,
    opt_level: i32,
    name: &str,
    required: &[&str],
    traceable: bool,
) -> Result<Pass> {
    let passes = Array::new(passes);
    let name = FfiString::from(name);
    let required = required_array(required);
    call_global(
        "transform.Sequential",
        &[
            AnyView::from(&passes),
            AnyView::from(&opt_level),
            AnyView::from(&name),
            AnyView::from(&required),
            AnyView::from(&traceable),
        ],
    )?
    .try_into()
}

/// Execute a pass through `transform.RunPass`.
pub fn run_pass(pass: &Pass, module: &IRModule) -> Result<IRModule> {
    call_global(
        "transform.RunPass",
        &[AnyView::from(pass), AnyView::from(module)],
    )?
    .try_into()
}
