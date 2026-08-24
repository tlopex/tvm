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

use std::collections::HashMap;

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{
    structural_map, structural_mutate, Any, Array, DefRegionKind, Error, Function, MutateCallbacks,
    MutateContext, ObjectArc, ObjectRefCast, ObjectRefCore, Result, String, WalkOrder,
    RUNTIME_ERROR,
};

use crate::ir::{BaseFunc, Expr, IRModule, IntImm, PrimType, Var};
use crate::relax::RelaxFunction;
use crate::tirx::{Add, AssertStmtObj, Evaluate, For as TirFor, Mul, PrimFunc, SeqStmt, Stmt, Sub};

/// Opaque Rust view of TVM's `PassNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "transform.Pass"]
pub struct PassObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to a TVM pass.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Pass {
    data: ObjectArc<PassObj>,
}

impl std::ops::Deref for Pass {
    type Target = PassObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Opaque Rust view of TVM's `PassContextNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "transform.PassContext"]
#[type_final]
pub struct PassContextObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to the active TVM pass context.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct PassContext {
    data: ObjectArc<PassContextObj>,
}

impl std::ops::Deref for PassContext {
    type Target = PassContextObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl Pass {
    /// Run this pass on an IRModule using TVM's current PassContext.
    ///
    /// This consumes the Rust handle.  The current Rust FFI still transports it
    /// as an lvalue, so C++ may perform one copy-on-write step at the boundary.
    pub fn run(&self, module: IRModule) -> Result<IRModule> {
        crate::global_function!("transform.RunPass")?
            .call_tuple_with_len::<2, _>((self, module))?
            .try_into()
    }
}

/// Replace every `AssertStmt` in a PrimFunc with `Evaluate(0)`.
pub fn skip_assert_prim_func(func: PrimFunc) -> Result<PrimFunc> {
    let mut mapper = AssertSkipper;
    structural_map(func, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

struct AssertSkipper;

#[tvm_ffi::dispatch(map)]
impl AssertSkipper {
    fn map_assert(&mut self, _value: &AssertStmtObj) -> Result<Any> {
        Ok(Any::from(Evaluate::from_i64(0)?))
    }

    fn map_sequence(&mut self, value: SeqStmt) -> Result<Any> {
        let mut flattened = Vec::new();
        for statement in value.statements()?.iter() {
            if let Ok(sequence) = statement.clone().try_cast::<SeqStmt>() {
                flattened.extend(sequence.statements()?.iter());
            } else if !is_evaluate_zero(&statement)? {
                flattened.push(statement);
            }
        }
        match flattened.len() {
            0 => Ok(Any::from(Evaluate::from_i64(0)?)),
            1 => Ok(Any::from(flattened.pop().unwrap())),
            _ => Ok(Any::from(SeqStmt::new(flattened)?)),
        }
    }
}

fn is_evaluate_zero(statement: &Stmt) -> Result<bool> {
    let Ok(evaluate) = statement.clone().try_cast::<Evaluate>() else {
        return Ok(false);
    };
    Ok(int_value(&evaluate.value()?)? == Some(0))
}

/// Build the Rust implementation of `tirx.SkipAssert` as a normal TVM pass.
pub fn skip_assert() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.RustSkipAssert",
        0,
        Vec::new(),
        false,
        |func, _module, _context| skip_assert_prim_func(func),
    )
}

/// Increment every integer literal in an expression while preserving its dtype.
pub fn increment_int_immediates(expr: crate::ir::Expr) -> Result<crate::ir::Expr> {
    let mut mapper = IncrementIntImmediates;
    structural_map(expr, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

struct IncrementIntImmediates;

#[tvm_ffi::dispatch(map)]
impl IncrementIntImmediates {
    fn map_integer(&mut self, value: IntImm) -> Result<Any> {
        let dtype = value.ty()?.try_cast::<PrimType>()?.dtype()?;
        Ok(Any::from(IntImm::from_dtype(dtype, value.value()? + 1)?))
    }
}

/// Remove additions whose left or right operand is integer zero.
pub fn simplify_add_zero_expr(expr: crate::ir::Expr) -> Result<crate::ir::Expr> {
    let mut mapper = AddZeroSimplifier;
    structural_map(expr, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

/// Apply the add-zero simplifier throughout a PrimFunc.
pub fn simplify_add_zero_prim_func(func: PrimFunc) -> Result<PrimFunc> {
    let mut mapper = AddZeroSimplifier;
    structural_map(func, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

/// Apply the add-zero simplifier to every function reachable from a module.
pub fn simplify_add_zero_module(module: IRModule) -> Result<IRModule> {
    let mut mapper = AddZeroSimplifier;
    let functions = module.functions()?;
    // Module_Add mutates its IRModuleNode.  Copy that node once so callers that
    // retained another handle to `module` do not observe this pass's updates.
    let mut output = module.copy_for_update()?;
    for (global_var, function) in functions.iter() {
        let mapped = structural_map(function, &mut mapper, WalkOrder::PostOrder)?;
        output = output.update_function_owned(&global_var, &BaseFunc::try_from(mapped)?)?;
    }
    Ok(output)
}

/// Build add-zero simplification as a TVM PrimFunc pass.
pub fn simplify_add_zero() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.RustSimplifyAddZero",
        0,
        Vec::new(),
        false,
        |func, _module, _context| simplify_add_zero_prim_func(func),
    )
}

/// Build add-zero simplification as a module pass.
pub fn simplify_add_zero_module_pass() -> Result<Pass> {
    create_module_pass(
        "tirx.RustSimplifyAddZeroModule",
        0,
        Vec::new(),
        false,
        |module, _context| simplify_add_zero_module(module),
    )
}

struct AddZeroSimplifier;

#[tvm_ffi::dispatch(map)]
impl AddZeroSimplifier {
    fn map_add(&mut self, value: Add) -> Result<Any> {
        let lhs = value.lhs()?;
        let rhs = value.rhs()?;
        if int_value(&lhs)? == Some(0) {
            return Ok(Any::from(rhs));
        }
        if int_value(&rhs)? == Some(0) {
            return Ok(Any::from(lhs));
        }
        Ok(Any::from(value))
    }
}

fn int_value(expr: &crate::ir::Expr) -> Result<Option<i64>> {
    match expr.clone().try_cast::<IntImm>() {
        Ok(value) => Ok(Some(value.value()?)),
        Err(_) => Ok(None),
    }
}

/// Simplify arithmetic identity operations using framework-controlled mapping.
///
/// The post-order map rewrites children first, then removes `x + 0`, `0 + x`,
/// `x - 0`, `x * 1`, and `1 * x` throughout the expression graph.
pub fn simplify_neutral_elements_expr(expr: Expr) -> Result<Expr> {
    let mut mapper = NeutralElementSimplifier;
    structural_map(expr, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

/// Apply neutral-element simplification throughout a TIR PrimFunc.
pub fn simplify_neutral_elements_prim_func(func: PrimFunc) -> Result<PrimFunc> {
    let mut mapper = NeutralElementSimplifier;
    structural_map(func, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

struct NeutralElementSimplifier;

#[tvm_ffi::dispatch(map)]
impl NeutralElementSimplifier {
    fn map_add(&mut self, value: Add) -> Result<Any> {
        let lhs = value.lhs()?;
        let rhs = value.rhs()?;
        if int_value(&lhs)? == Some(0) {
            return Ok(Any::from(rhs));
        }
        if int_value(&rhs)? == Some(0) {
            return Ok(Any::from(lhs));
        }
        Ok(Any::from(value))
    }

    fn map_subtract(&mut self, value: Sub) -> Result<Any> {
        let rhs = value.rhs()?;
        if int_value(&rhs)? == Some(0) {
            return Ok(Any::from(value.lhs()?));
        }
        Ok(Any::from(value))
    }

    fn map_multiply(&mut self, value: Mul) -> Result<Any> {
        let lhs = value.lhs()?;
        let rhs = value.rhs()?;
        if int_value(&lhs)? == Some(1) {
            return Ok(Any::from(rhs));
        }
        if int_value(&rhs)? == Some(1) {
            return Ok(Any::from(lhs));
        }
        Ok(Any::from(value))
    }
}

#[derive(Default)]
struct LoopBodyMutationState {
    depth: usize,
}

/// Simplify neutral arithmetic operations only while inside a loop body.
///
/// This uses callback-controlled mutation because the state transition must
/// surround one specific child (`For.body`).  Bounds of a top-level loop are
/// left unchanged, while bounds of a nested loop are transformed because that
/// loop itself occurs inside its parent's body.
pub fn simplify_neutral_elements_in_loop_bodies(statement: Stmt) -> Result<Stmt> {
    let mut mutator = MutateCallbacks::new(
        LoopBodyMutationState::default(),
        (
            mutate_loop,
            mutate_scoped_add,
            mutate_scoped_subtract,
            mutate_scoped_multiply,
        ),
    );
    structural_mutate(statement, &mut mutator)?.try_into()
}

fn mutate_loop(
    value: TirFor,
    mutator: &mut MutateContext<'_, LoopBodyMutationState>,
) -> Result<Any> {
    let loop_var =
        Var::try_from(mutator.mutate_with(&value.loop_var()?, DefRegionKind::Recursive)?)?;
    let minimum = Expr::try_from(mutator.mutate(&value.minimum()?)?)?;
    let extent = Expr::try_from(mutator.mutate(&value.extent()?)?)?;

    mutator.state_mut().depth += 1;
    let body_result = mutator.mutate(&value.body()?);
    mutator.state_mut().depth -= 1;
    let body = Stmt::try_from(body_result?)?;

    let thread_binding =
        Option::<crate::tirx::IterVar>::try_from(mutator.mutate(&value.thread_binding()?)?)?;
    let annotations = mutator.mutate(&value.annotations()?)?;
    let step = value
        .step()?
        .map(|step| mutator.mutate(&step).and_then(Expr::try_from))
        .transpose()?;
    let span = value.span()?;

    Ok(Any::from(TirFor::with_metadata(
        &loop_var,
        &minimum,
        &extent,
        value.kind()?,
        &body,
        thread_binding.as_ref(),
        &annotations,
        step.as_ref(),
        span.as_ref(),
    )?))
}

fn mutate_scoped_add(
    _value: Add,
    mutator: &mut MutateContext<'_, LoopBodyMutationState>,
) -> Result<Any> {
    let value = Add::try_from(mutator.default_mutate()?)?;
    if mutator.state().depth == 0 {
        return Ok(Any::from(value));
    }
    let lhs = value.lhs()?;
    let rhs = value.rhs()?;
    if int_value(&lhs)? == Some(0) {
        return Ok(Any::from(rhs));
    }
    if int_value(&rhs)? == Some(0) {
        return Ok(Any::from(lhs));
    }
    Ok(Any::from(value))
}

fn mutate_scoped_subtract(
    _value: Sub,
    mutator: &mut MutateContext<'_, LoopBodyMutationState>,
) -> Result<Any> {
    let value = Sub::try_from(mutator.default_mutate()?)?;
    if mutator.state().depth > 0 && int_value(&value.rhs()?)? == Some(0) {
        return Ok(Any::from(value.lhs()?));
    }
    Ok(Any::from(value))
}

fn mutate_scoped_multiply(
    _value: Mul,
    mutator: &mut MutateContext<'_, LoopBodyMutationState>,
) -> Result<Any> {
    let value = Mul::try_from(mutator.default_mutate()?)?;
    if mutator.state().depth == 0 {
        return Ok(Any::from(value));
    }
    let lhs = value.lhs()?;
    let rhs = value.rhs()?;
    if int_value(&lhs)? == Some(1) {
        return Ok(Any::from(rhs));
    }
    if int_value(&rhs)? == Some(1) {
        return Ok(Any::from(lhs));
    }
    Ok(Any::from(value))
}

/// Alpha-rename every variable definition and let structural mapping remap uses.
///
/// Free variables are preserved.  Function parameters, loop variables, and
/// binding variables receive `suffix`; uses resolve through the map operation's
/// invocation-local identity table rather than through string matching.
pub fn rename_bound_variables(expr: Expr, suffix: &str) -> Result<Expr> {
    let mut mapper = BoundVariableRenamer {
        suffix: suffix.to_owned(),
    };
    structural_map(expr, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

/// Alpha-rename definitions in one Relax function.
pub fn rename_bound_variables_function(
    function: RelaxFunction,
    suffix: &str,
) -> Result<RelaxFunction> {
    rename_bound_variables(function.into(), suffix)?.try_cast()
}

/// Build alpha-renaming as a normal TVM Relax FunctionPass.
pub fn rename_bound_variables_pass(suffix: &str) -> Result<Pass> {
    let suffix = suffix.to_owned();
    create_relax_function_pass(
        "relax.RustRenameBoundVariables",
        0,
        Vec::new(),
        false,
        move |function, _module, _context| rename_bound_variables_function(function, &suffix),
    )
}

struct BoundVariableRenamer {
    suffix: std::string::String,
}

#[tvm_ffi::dispatch(map)]
impl BoundVariableRenamer {
    fn map_variable(&mut self, value: Var, kind: DefRegionKind) -> Result<Any> {
        if kind == DefRegionKind::None {
            return Ok(Any::from(value));
        }
        let name = format!("{}{}", value.name()?.as_str(), self.suffix);
        Ok(Any::from(Var::with_type(&name, &value.ty()?)?))
    }
}

#[derive(Default)]
struct UnitLoopEliminationState {
    replacements: HashMap<usize, Expr>,
}

/// Type-erased view used only to read the common map header.
#[repr(C)]
#[derive(ObjectRef, Clone)]
struct UntypedMap {
    data: ObjectArc<tvm_ffi::collections::map::MapObj>,
}

impl std::ops::Deref for UntypedMap {
    type Target = tvm_ffi::collections::map::MapObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Eliminate unannotated unit loops and substitute their variables.
///
/// This is the unit-loop portion of C++ `tirx.transform.LowerTIRxOpaque`.
/// Substitution is based on object identity, so buffer indices and all other
/// uses of the loop variable receive the mapped loop minimum.
pub fn eliminate_unit_loops_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let mut mutator = MutateCallbacks::new(
        UnitLoopEliminationState::default(),
        (eliminate_unit_loop, substitute_unit_loop_variable),
    );
    structural_mutate(function, &mut mutator)?.try_into()
}

/// Build unit-loop elimination as a normal TVM PrimFunc pass.
pub fn eliminate_unit_loops() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.RustEliminateUnitLoops",
        0,
        Vec::new(),
        false,
        |function, _module, _context| eliminate_unit_loops_prim_func(function),
    )
}

fn eliminate_unit_loop(
    value: TirFor,
    mutator: &mut MutateContext<'_, UnitLoopEliminationState>,
) -> Result<Any> {
    let minimum = Expr::try_from(mutator.mutate(&value.minimum()?)?)?;
    let extent = Expr::try_from(mutator.mutate(&value.extent()?)?)?;
    let annotations = value.annotations()?;
    let kind = value.kind()?;
    let should_eliminate = kind != crate::tirx::ForKind::ThreadBinding
        && int_value(&extent)? == Some(1)
        && untyped_map_is_empty(annotations.clone())?;

    if should_eliminate {
        let key = object_identity(&value.loop_var()?);
        let previous = mutator
            .state_mut()
            .replacements
            .insert(key, minimum.clone());
        let body_result = mutator.mutate(&value.body()?);
        match previous {
            Some(previous) => {
                mutator.state_mut().replacements.insert(key, previous);
            }
            None => {
                mutator.state_mut().replacements.remove(&key);
            }
        }
        return body_result;
    }

    let loop_var =
        Var::try_from(mutator.mutate_with(&value.loop_var()?, DefRegionKind::Recursive)?)?;
    let body = Stmt::try_from(mutator.mutate(&value.body()?)?)?;
    let thread_binding =
        Option::<crate::tirx::IterVar>::try_from(mutator.mutate(&value.thread_binding()?)?)?;
    let annotations = mutator.mutate(&annotations)?;
    let step = value
        .step()?
        .map(|step| mutator.mutate(&step).and_then(Expr::try_from))
        .transpose()?;
    let span = value.span()?;

    Ok(Any::from(TirFor::with_metadata(
        &loop_var,
        &minimum,
        &extent,
        kind,
        &body,
        thread_binding.as_ref(),
        &annotations,
        step.as_ref(),
        span.as_ref(),
    )?))
}

fn substitute_unit_loop_variable(
    value: Var,
    mutator: &mut MutateContext<'_, UnitLoopEliminationState>,
) -> Result<Any> {
    if let Some(replacement) = mutator.state().replacements.get(&object_identity(&value)) {
        return Ok(Any::from(replacement.clone()));
    }
    mutator.default_mutate()
}

fn object_identity<T: ObjectRefCore>(value: &T) -> usize {
    // The pointer is used only as a non-dereferenced identity key while
    // `value` and the owning traversal root keep the object alive.
    unsafe { ObjectArc::as_raw(T::data(value)) as usize }
}

fn untyped_map_is_empty(value: Any) -> Result<bool> {
    // This checked cast verifies only the common `ffi.Map` runtime type.  No
    // annotation key or heterogeneous value is interpreted.
    Ok(UntypedMap::try_from(value)?.size == 0)
}

/// Construct a TVM PrimFunc pass backed by a Rust callback.
///
/// The callback signature mirrors C++ `CreatePrimFuncPass`: the function is
/// passed as an ABI rvalue reference, followed by the module and pass context.
pub fn create_prim_func_pass<F>(
    name: &str,
    opt_level: i64,
    required: Vec<&str>,
    traceable: bool,
    pass_func: F,
) -> Result<Pass>
where
    F: Fn(PrimFunc, IRModule, PassContext) -> Result<PrimFunc> + 'static,
{
    let pass_func = Function::from_packed(move |args| {
        if args.len() != 3 {
            return Err(Error::new(
                RUNTIME_ERROR,
                "Rust PrimFunc pass expected (PrimFunc, IRModule, PassContext)",
                "",
            ));
        }
        // PrimFunc passes receive their first argument as an ABI RValueRef.
        // Owning the view normalizes that transport wrapper to the object.
        let func = PrimFunc::try_from(Any::from(args[0]))?;
        let module = IRModule::try_from(args[1])?;
        let context = PassContext::try_from(args[2])?;
        Ok(Any::from(pass_func(func, module, context)?))
    });

    let pass_info = create_pass_info(name, opt_level, required, traceable)?;

    crate::global_function!("tirx.transform.CreatePrimFuncPass")?
        .call_packed(&[
            tvm_ffi::AnyView::from(&pass_func),
            tvm_ffi::AnyView::from(&pass_info),
        ])?
        .try_into()
}

/// Construct a TVM Relax FunctionPass backed by a Rust callback.
pub fn create_relax_function_pass<F>(
    name: &str,
    opt_level: i64,
    required: Vec<&str>,
    traceable: bool,
    pass_func: F,
) -> Result<Pass>
where
    F: Fn(RelaxFunction, IRModule, PassContext) -> Result<RelaxFunction> + 'static,
{
    let pass_func = Function::from_packed(move |args| {
        if args.len() != 3 {
            return Err(Error::new(
                RUNTIME_ERROR,
                "Rust Relax function pass expected (Function, IRModule, PassContext)",
                "",
            ));
        }
        let function = RelaxFunction::try_from(Any::from(args[0]))?;
        let module = IRModule::try_from(args[1])?;
        let context = PassContext::try_from(args[2])?;
        Ok(Any::from(pass_func(function, module, context)?))
    });
    let pass_info = create_pass_info(name, opt_level, required, traceable)?;

    crate::global_function!("relax.transform.MakeFunctionPass")?
        .call_packed(&[
            tvm_ffi::AnyView::from(&pass_func),
            tvm_ffi::AnyView::from(&pass_info),
        ])?
        .try_into()
}

/// Construct a TVM module pass backed by a Rust callback.
pub fn create_module_pass<F>(
    name: &str,
    opt_level: i64,
    required: Vec<&str>,
    traceable: bool,
    pass_func: F,
) -> Result<Pass>
where
    F: Fn(IRModule, PassContext) -> Result<IRModule> + 'static,
{
    let pass_func = Function::from_packed(move |args| {
        if args.len() != 2 {
            return Err(Error::new(
                RUNTIME_ERROR,
                "Rust module pass expected (IRModule, PassContext)",
                "",
            ));
        }
        // Module passes receive their first argument through the RValueRef ABI.
        let module = IRModule::try_from(Any::from(args[0]))?;
        let context = PassContext::try_from(args[1])?;
        Ok(Any::from(pass_func(module, context)?))
    });
    let pass_info = create_pass_info(name, opt_level, required, traceable)?;

    crate::global_function!("transform.MakeModulePass")?
        .call_packed(&[
            tvm_ffi::AnyView::from(&pass_func),
            tvm_ffi::AnyView::from(&pass_info),
        ])?
        .try_into()
}

fn create_pass_info(
    name: &str,
    opt_level: i64,
    required: Vec<&str>,
    traceable: bool,
) -> Result<Any> {
    let required = Array::<String>::new(required.into_iter().map(String::from).collect());
    let name = String::from(name);
    crate::global_function!("transform.PassInfo")?.call_packed(&[
        tvm_ffi::AnyView::from(&opt_level),
        tvm_ffi::AnyView::from(&name),
        tvm_ffi::AnyView::from(&required),
        tvm_ffi::AnyView::from(&traceable),
    ])
}
