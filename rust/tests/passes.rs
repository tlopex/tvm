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

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tvm_ffi::{
    AnyValue, AnyView, Array, DLDataType, DLDataTypeExt, Error, Map, Module, ObjectRefCore, Result,
};
use tvm_tirx_bindings::analyzer::{Analyzer, ProofStrength};
use tvm_tirx_bindings::ffi_api;
use tvm_tirx_bindings::generated::ir::{self, Expr, IRModule, IntImm};
use tvm_tirx_bindings::generated::tirx::{
    self, Add, AssertStmt, Buffer, Evaluate, For, Layout, PrimFunc, Stmt, TilePrimitiveCall, Var,
};
use tvm_tirx_bindings::mutator::{rewrite_stmt, StmtExprRewriter};
use tvm_tirx_bindings::passes::{
    simplify_stmt_expressions, skip_assert, skip_assert_pass, verify_ssa, verify_ssa_pass,
};
use tvm_tirx_bindings::visitor::{try_downcast, try_downcast_exact, StmtExprVisitor};
use tvm_tirx_bindings::PrimExpr;

fn compiler_library() -> PathBuf {
    if let Some(path) = std::env::var_os("TVM_COMPILER_LIB") {
        return PathBuf::from(path);
    }
    let build_dir = std::env::var_os("TVM_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../build"));
    build_dir.join("lib/libtvm_compiler.so")
}

fn load_compiler() -> Result<Module> {
    let path = compiler_library();
    assert!(
        path.is_file(),
        "runtime pass tests require a matching compiler; set TVM_BUILD_DIR or TVM_COMPILER_LIB (missing {})",
        path.display()
    );
    Module::load_from_file(path.to_string_lossy())
}

fn int_expr(dtype: &str, value: i64) -> Result<PrimExpr> {
    ffi_api::prim_expr(ffi_api::int_imm_from_str(dtype, value, None)?)
}

fn equal(lhs: PrimExpr, rhs: PrimExpr) -> Result<PrimExpr> {
    ffi_api::prim_expr(
        tirx::eq(Some(lhs), Some(rhs), None)?.expect("tirx.EQ must return a defined expression"),
    )
}

fn test_buffer(data: &Var, shape: PrimExpr) -> Result<Buffer> {
    let dtype = ir::prim_type(DLDataType::try_from_str("float32")?)?;
    let strides = Array::<Option<PrimExpr>>::new(vec![]);
    let axis_separators = Array::<Option<IntImm>>::new(vec![]);
    let no_span = None::<ir::Span>;
    let no_layout = None::<Layout>;
    tirx::buffer_packed(&[
        AnyView::from(data),
        AnyView::from(&dtype),
        AnyView::from(&Array::new(vec![Some(shape)])),
        AnyView::from(&strides),
        AnyView::from(&int_expr("int64", 0)?),
        AnyView::from(&tvm_ffi::String::from("test_buffer")),
        AnyView::from(&0_i64),
        AnyView::from(&0_i64),
        AnyView::from(&tvm_ffi::String::from("default")),
        AnyView::from(&axis_separators),
        AnyView::from(&no_span),
        AnyView::from(&no_layout),
    ])?
    .try_into()
}

fn evaluated_int(stmt: &Stmt) -> Result<Option<i64>> {
    let Some(evaluate) = try_downcast::<_, Evaluate>(stmt) else {
        return Ok(None);
    };
    let Some(value) = evaluate.value()? else {
        return Ok(None);
    };
    let Some(value) = try_downcast::<_, IntImm>(&value) else {
        return Ok(None);
    };
    Ok(Some(value.value()?))
}

fn only_prim_func(module: &IRModule) -> Result<PrimFunc> {
    let functions = module.functions()?;
    assert_eq!(functions.len(), 1, "test module must contain one function");
    let base_func = functions
        .values()
        .next()
        .expect("test module must contain one function")
        .expect("test module function must be defined");
    Ok(
        try_downcast::<_, PrimFunc>(&base_func)
            .expect("test module function must be tirx.PrimFunc"),
    )
}

fn expect_error<T>(result: Result<T>) -> Error {
    match result {
        Ok(_) => panic!("expected operation to fail"),
        Err(error) => error,
    }
}

#[derive(Default)]
struct IntImmCounter {
    count: usize,
}

impl StmtExprVisitor for IntImmCounter {
    fn visit_int_imm(&mut self, _node: &IntImm) -> Result<()> {
        self.count += 1;
        Ok(())
    }
}

struct ReplaceIntImm {
    from: i64,
    to: PrimExpr,
}

impl StmtExprRewriter for ReplaceIntImm {
    fn rewrite_expr(&mut self, expr: Expr) -> Result<Option<Expr>> {
        let Some(node) = try_downcast_exact::<_, IntImm>(&expr) else {
            return Ok(None);
        };
        Ok((node.value()? == self.from).then(|| self.to.clone().into_base()))
    }
}

struct PanickingRewriter;

impl StmtExprRewriter for PanickingRewriter {
    fn rewrite_expr(&mut self, _expr: Expr) -> Result<Option<Expr>> {
        panic!("intentional scoped rewriter panic")
    }
}

#[derive(Default)]
struct TilePayloadCounter {
    int_imms: usize,
    vars: usize,
    assertions: usize,
}

impl StmtExprVisitor for TilePayloadCounter {
    fn visit_int_imm(&mut self, _node: &IntImm) -> Result<()> {
        self.int_imms += 1;
        Ok(())
    }

    fn visit_var(&mut self, _node: &Var) -> Result<()> {
        self.vars += 1;
        Ok(())
    }

    fn visit_assert_stmt(&mut self, _node: &AssertStmt) -> Result<()> {
        self.assertions += 1;
        Ok(())
    }
}

#[test]
fn reflected_getters_preserve_nullability_and_identity() -> Result<()> {
    let _compiler = load_compiler()?;
    let seven = int_expr("int32", 7)?;
    let evaluate_seven: Stmt = ffi_api::evaluate(&seven, None)?.into();
    let func = ffi_api::prim_func_without_params(&evaluate_seven, None)?;

    assert!(func.span()?.is_none());
    let body = func.body()?.expect("PrimFunc::body must be defined");
    assert!(body.same_as(&evaluate_seven));
    Ok(())
}

#[test]
fn exact_downcast_distinguishes_dynamic_type_from_parent() -> Result<()> {
    let _compiler = load_compiler()?;
    let seven = ffi_api::int_imm_from_str("int32", 7, None)?;
    let seven_expr: Expr = seven.into();
    // Runtime subtype casts and compiler dispatch are intentionally distinct:
    // an IntImm is an Expr, but its exact dynamic type is still IntImm.
    assert!(try_downcast::<_, Expr>(&seven_expr).is_some());
    assert!(try_downcast_exact::<_, Expr>(&seven_expr).is_none());
    let exact = try_downcast_exact::<_, IntImm>(&seven_expr)
        .expect("the exact dynamic type must remain IntImm");
    assert!(exact.same_as(&seven_expr));
    Ok(())
}

#[test]
fn visitor_visits_each_call_argument_once() -> Result<()> {
    let _compiler = load_compiler()?;
    let one = int_expr("int32", 1)?;
    let two = int_expr("int32", 2)?;
    let op: Expr = ir::get_op(tvm_ffi::String::from("tirx.webgpu.subgroup_shuffle"))?.into();
    let call_type = ir::expr_type(Some(one.as_base().clone()))?;
    let call: Expr = ir::call(
        call_type,
        Some(op),
        Array::new(vec![Some(one.clone().into_base()), Some(two.into_base())]),
        None,
        Array::new(vec![]),
        None,
    )?
    .expect("ir.Call must return a defined expression")
    .into();

    let mut visitor = IntImmCounter::default();
    visitor.visit_expr(&call)?;
    assert_eq!(visitor.count, 2);
    Ok(())
}

#[test]
fn mutator_rewrites_expressions_in_statements_and_nested_expressions() -> Result<()> {
    let _compiler = load_compiler()?;
    let one = int_expr("int32", 1)?;
    let two = int_expr("int32", 2)?;
    let nine = int_expr("int32", 9)?;
    let sum = ffi_api::prim_expr(
        tirx::add(Some(one.clone()), Some(two.clone()), None)?
            .expect("tirx.Add must return a defined expression"),
    )?;
    let body: Stmt = ffi_api::evaluate(sum.as_base(), None)?.into();
    let loop_var = ffi_api::var_from_str("i", "int32", None)?;
    let loop_stmt: Stmt = ffi_api::for_loop(
        &loop_var,
        &one,
        &two,
        0,
        &body,
        None,
        &Map::new(),
        None,
        None,
    )?
    .into();

    let rewritten = rewrite_stmt(&loop_stmt, &mut ReplaceIntImm { from: 1, to: nine })?;
    let rewritten_for = try_downcast::<_, For>(&rewritten).expect("result must remain a For");
    let min = rewritten_for.min()?.expect("For::min must be defined");
    let min = try_downcast::<_, IntImm>(min.as_base()).expect("For::min must remain IntImm");
    assert_eq!(min.value()?, 9);

    let body = rewritten_for.body()?.expect("For::body must be defined");
    let evaluate = try_downcast::<_, Evaluate>(&body).expect("body must remain Evaluate");
    let sum = evaluate.value()?.expect("Evaluate::value must be defined");
    let sum = try_downcast::<_, Add>(&sum).expect("nested expression must remain Add");
    let lhs = sum.a()?.expect("Add::a must be defined");
    let lhs = try_downcast::<_, IntImm>(lhs.as_base()).expect("Add::a must remain IntImm");
    assert_eq!(lhs.value()?, 9);
    Ok(())
}

#[test]
fn mutator_contains_panics_at_the_scoped_ffi_boundary() -> Result<()> {
    let _compiler = load_compiler()?;
    let statement: Stmt = ffi_api::evaluate(int_expr("int32", 1)?.as_base(), None)?.into();

    let error = expect_error(rewrite_stmt(&statement, &mut PanickingRewriter));

    assert_eq!(error.kind().as_str(), "RuntimeError");
    assert!(error
        .message()
        .contains("intentional scoped rewriter panic"));
    Ok(())
}

#[test]
fn simplify_pass_rewrites_nested_arithmetic() -> Result<()> {
    let _compiler = load_compiler()?;
    let one = int_expr("int32", 1)?;
    let two = int_expr("int32", 2)?;
    let sum = ffi_api::prim_expr(
        tirx::add(Some(one), Some(two), None)?.expect("tirx.Add must return a defined expression"),
    )?;
    let statement: Stmt = ffi_api::evaluate(sum.as_base(), None)?.into();

    let simplified = simplify_stmt_expressions(&statement)?;
    assert_eq!(evaluated_int(&simplified)?, Some(3));
    Ok(())
}

#[test]
fn skip_assert_rewrites_assertions_and_preserves_the_remaining_body() -> Result<()> {
    let _compiler = load_compiler()?;
    let true_value = int_expr("bool", 1)?;
    let seven = int_expr("int32", 7)?;
    let evaluate_seven: Stmt = ffi_api::evaluate(&seven, None)?.into();

    let error_kind = ffi_api::string_imm("ValueError", None)?;
    let message = ffi_api::string_imm("expected true", None)?;
    let assertion: Stmt = ffi_api::assert_stmt(&true_value, &error_kind, &[message], None)?.into();
    let body = ffi_api::normalize_seq(vec![assertion.clone(), evaluate_seven.clone()], None)?;

    let skipped = skip_assert(&body)?;
    assert_eq!(evaluated_int(&skipped)?, Some(7));
    assert!(skipped.same_as(&evaluate_seven));

    let func = ffi_api::prim_func_without_params(&body, None)?;
    let module = ffi_api::ir_module_with_prim_func("main", &func)?;
    let output = ffi_api::run_pass(&skip_assert_pass()?, &module)?;
    let output_body = only_prim_func(&output)?
        .body()?
        .expect("transformed PrimFunc::body must be defined");
    assert!(output_body.same_as(&evaluate_seven));
    Ok(())
}

#[test]
fn tile_statement_mutator_rewrites_nested_statements_and_preserves_metadata() -> Result<()> {
    let _compiler = load_compiler()?;
    let true_value = int_expr("bool", 1)?;
    let error_kind = ffi_api::string_imm("ValueError", None)?;
    let message = ffi_api::string_imm("expected true", None)?;
    let assertion: Stmt = ffi_api::assert_stmt(&true_value, &error_kind, &[message], None)?.into();

    let tile_op = ir::get_op(tvm_ffi::String::from("tirx.tile.zero"))?;
    let tile_args = Array::new(vec![
        AnyValue::from_value(tile_op.clone()),
        AnyValue::from_value(assertion.clone()),
    ]);
    let workspace = Map::<tvm_ffi::String, Option<Buffer>>::new();
    let config_key = tvm_ffi::String::from("nested_stmt");
    let config: Map<tvm_ffi::String, AnyValue> =
        [(config_key.clone(), AnyValue::from_value(assertion))]
            .into_iter()
            .collect();
    let tile = ffi_api::tile_primitive_call(&tile_op, &tile_args, &workspace, &config, None, None)?;
    let tile_stmt: Stmt = tile.into();

    let rewritten_stmt = skip_assert(&tile_stmt)?;
    let rewritten = try_downcast::<_, TilePrimitiveCall>(&rewritten_stmt)
        .expect("rewritten statement must remain a TilePrimitiveCall");
    assert!(rewritten.op()?.same_as(&tile_op));

    let nested_arg = rewritten
        .args()?
        .get(1)?
        .try_as::<tvm_ffi::object::ObjectRef>()
        .expect("TilePrimitiveCall argument must remain an object");
    let nested_arg = try_downcast::<_, Stmt>(&nested_arg)
        .expect("TilePrimitiveCall argument must remain a statement");
    assert_eq!(evaluated_int(&nested_arg)?, Some(0));

    let nested_config = rewritten
        .config()?
        .get(&config_key)?
        .expect("TilePrimitiveCall config key must remain present")
        .try_as::<tvm_ffi::object::ObjectRef>()
        .expect("TilePrimitiveCall config value must remain an object");
    let nested_config = try_downcast::<_, Stmt>(&nested_config)
        .expect("TilePrimitiveCall config value must remain a statement");
    assert_eq!(evaluated_int(&nested_config)?, Some(0));
    Ok(())
}

#[test]
fn tile_visitor_traverses_selected_type_erased_payloads() -> Result<()> {
    let _compiler = load_compiler()?;
    let zero = int_expr("int32", 0)?;
    let one = int_expr("int32", 1)?;
    let domain = ir::range_from_min_extent(Some(zero.clone()), Some(one), None)?
        .expect("ir.Range_from_min_extent must return a defined range");
    let var = ffi_api::var_from_str("tile_axis", "int32", None)?;
    let iter_var = tirx::iter_var(Some(domain), Some(var), 0, tvm_ffi::String::from(""), None)?
        .expect("tirx.IterVar must return a defined value");

    let tile_op = ir::get_op(tvm_ffi::String::from("tirx.tile.zero"))?;
    assert!(try_downcast::<_, PrimExpr>(&tile_op).is_none());
    let error = expect_error(ffi_api::prim_expr(tile_op.clone()));
    assert_eq!(error.kind().as_str(), "TypeError");
    assert!(error.message().contains("ir.Type"));
    assert!(error.message().contains("ir.PrimType"));

    let assertion: Stmt = ffi_api::assert_stmt(
        &int_expr("bool", 1)?,
        &ffi_api::string_imm("ValueError", None)?,
        &[ffi_api::string_imm("expected true", None)?],
        None,
    )?
    .into();
    let args = Array::new(vec![
        AnyValue::from_value(zero),
        AnyValue::from_value(7_i64),
        AnyValue::from_value(iter_var),
        AnyValue::from_value(tile_op.clone()),
        AnyValue::from_value(assertion.clone()),
    ]);
    let config = [(
        tvm_ffi::String::from("nested"),
        AnyValue::from_value(assertion),
    )]
    .into_iter()
    .collect();
    let tile: Stmt = ffi_api::tile_primitive_call(
        &tile_op,
        &args,
        &Map::<tvm_ffi::String, Option<Buffer>>::new(),
        &config,
        None,
        None,
    )?
    .into();

    let mut visitor = TilePayloadCounter::default();
    visitor.visit_stmt(&tile)?;
    // Any::as<PrimExpr>() is strict. Unlike packed-argument conversion, it
    // deliberately ignores scalar and PrimExprConvertible fallbacks.
    assert_eq!(visitor.int_imms, 1);
    assert_eq!(visitor.vars, 0);
    assert_eq!(visitor.assertions, 2);
    Ok(())
}

#[test]
fn verify_ssa_rejects_duplicate_definitions_through_module_pass() -> Result<()> {
    let _compiler = load_compiler()?;
    let zero = int_expr("int32", 0)?;
    let body: Stmt = ffi_api::evaluate(zero.as_base(), None)?.into();

    let valid = ffi_api::prim_func_without_params(&body, None)?;
    assert!(verify_ssa(&valid)?);
    let valid_module = ffi_api::ir_module_with_prim_func("main", &valid)?;
    let verified = ffi_api::run_pass(&verify_ssa_pass()?, &valid_module)?;
    assert!(verified.same_as(&valid_module));

    let var = ffi_api::var_from_str("duplicate", "int32", None)?;
    let params = Array::new(vec![Some(var.clone()), Some(var)]);
    let invalid = ffi_api::prim_func(
        &params,
        &body,
        None,
        &Map::<Option<Var>, Option<Buffer>>::new(),
        None,
        None,
    )?;
    assert!(!verify_ssa(&invalid)?);

    let module = ffi_api::ir_module_with_prim_func("main", &invalid)?;
    let error = expect_error(ffi_api::run_pass(&verify_ssa_pass()?, &module));
    assert_eq!(error.kind().as_str(), "RuntimeError");
    assert!(error.message().contains("not in SSA form"));
    Ok(())
}

#[test]
fn verify_ssa_distinguishes_equal_and_unequal_duplicate_let_bindings() -> Result<()> {
    let _compiler = load_compiler()?;
    let var = ffi_api::var_from_str("shared_let", "int32", None)?;
    let var_expr = ffi_api::prim_expr(var.clone())?;
    let one = int_expr("int32", 1)?;
    let two = int_expr("int32", 2)?;

    let make_let = |value: PrimExpr| -> Result<PrimExpr> {
        let node = tirx::r#let(Some(var.clone()), Some(value), Some(var_expr.clone()), None)?
            .expect("tirx.Let must return a defined expression");
        ffi_api::prim_expr(node)
    };
    let make_func = |lhs: PrimExpr, rhs: PrimExpr| -> Result<PrimFunc> {
        let sum = tirx::add(Some(lhs), Some(rhs), None)?
            .expect("tirx.Add must return a defined expression");
        let sum = ffi_api::prim_expr(sum)?;
        let body: Stmt = ffi_api::evaluate(sum.as_base(), None)?.into();
        ffi_api::prim_func_without_params(&body, None)
    };

    let equal = make_func(make_let(one.clone())?, make_let(one.clone())?)?;
    assert!(verify_ssa(&equal)?);

    // Establish the first definition with a different value while retaining
    // the exact same Var object in both Let nodes.
    let different = make_func(make_let(one)?, make_let(two)?)?;
    assert!(!verify_ssa(&different)?);
    Ok(())
}

#[test]
fn verify_ssa_matches_cpp_at_statement_and_buffer_definition_sites() -> Result<()> {
    let _compiler = load_compiler()?;
    let zero = int_expr("int32", 0)?;
    let one = int_expr("int32", 1)?;
    let leaf: Stmt = ffi_api::evaluate(zero.as_base(), None)?.into();

    let loop_var = ffi_api::var_from_str("i", "int32", None)?;
    let make_loop = |body: &Stmt| -> Result<Stmt> {
        Ok(ffi_api::for_loop(
            &loop_var,
            &zero,
            &one,
            0,
            body,
            None,
            &Map::new(),
            None,
            None,
        )?
        .into())
    };
    let inner_loop = make_loop(&leaf)?;
    let duplicate_loop = make_loop(&inner_loop)?;

    let bound_var = ffi_api::var_from_str("bound", "int32", None)?;
    let bind: Stmt = tirx::bind(
        Some(bound_var.clone()),
        Some(zero.clone().into_base()),
        None,
    )?
    .expect("tirx.Bind must return a defined statement")
    .into();
    let duplicate_bind = ffi_api::normalize_seq(vec![bind.clone(), bind], None)?;

    let element_type = ir::prim_type(DLDataType::try_from_str("float32")?)?;
    let pointer_type = ir::pointer_type(element_type.into(), tvm_ffi::String::from("global"))?;
    let data = tirx::var(
        tvm_ffi::String::from("data"),
        AnyView::from(&pointer_type),
        None,
    )?
    .expect("tirx.Var must return a defined pointer variable");
    let shape_var = ffi_api::var_from_str("n", "int32", None)?;
    let shape = ffi_api::prim_expr(shape_var.clone())?;
    let buffer = test_buffer(&data, shape)?;
    let alloc: Stmt = tirx::alloc_buffer(Some(buffer.clone()), None, None)?
        .expect("tirx.AllocBuffer must return a defined statement")
        .into();
    let duplicate_alloc = ffi_api::normalize_seq(vec![alloc.clone(), alloc], None)?;

    // Buffer descriptors are match scopes in the canonical verifier.  Their
    // data and shape variables may therefore repeat PrimFunc parameters.
    let buffer_params = Array::new(vec![Some(data.clone()), Some(shape_var)]);
    let buffer_map = [(Some(data), Some(buffer))].into_iter().collect();
    let buffer_map_func = ffi_api::prim_func(&buffer_params, &leaf, None, &buffer_map, None, None)?;

    let cases = [
        (
            "duplicate For loop variable",
            ffi_api::prim_func_without_params(&duplicate_loop, None)?,
            false,
        ),
        (
            "duplicate Bind variable",
            ffi_api::prim_func_without_params(&duplicate_bind, None)?,
            false,
        ),
        (
            "duplicate AllocBuffer data variable",
            ffi_api::prim_func_without_params(&duplicate_alloc, None)?,
            false,
        ),
        ("buffer_map match scope", buffer_map_func, true),
    ];
    for (name, func, expected) in cases {
        let rust = verify_ssa(&func)?;
        let cpp = tirx::analysis::verify_ssa(Some(func))?;
        assert_eq!(rust, cpp, "Rust/C++ mismatch for {name}");
        assert_eq!(rust, expected, "unexpected SSA result for {name}");
    }
    Ok(())
}

#[test]
fn analyzer_fork_copies_existing_facts_without_aliasing_future_bindings() -> Result<()> {
    let _compiler = load_compiler()?;
    let zero = int_expr("int32", 0)?;
    let one = int_expr("int32", 1)?;
    let x_var = ffi_api::var_from_str("x", "int32", None)?;
    let y_var = ffi_api::var_from_str("y", "int32", None)?;
    let x = ffi_api::prim_expr(x_var.clone())?;
    let y = ffi_api::prim_expr(y_var.clone())?;
    let x_is_zero = equal(x, zero.clone())?;
    let y_is_one = equal(y, one.clone())?;

    let analyzer = Analyzer::new()?;
    assert!(!analyzer.can_prove(&x_is_zero, ProofStrength::Default)?);
    analyzer.bind_expr(&x_var, &zero, false)?;
    let fork = analyzer.fork()?;
    assert!(fork.can_prove(&x_is_zero, ProofStrength::Default)?);
    analyzer.bind_expr(&y_var, &one, false)?;
    assert!(analyzer.can_prove(&y_is_one, ProofStrength::Default)?);
    assert!(!fork.can_prove(&y_is_one, ProofStrength::Default)?);
    Ok(())
}

#[test]
fn analyzer_constraint_scopes_restore_after_success_error_and_panic() -> Result<()> {
    let _compiler = load_compiler()?;
    let zero = int_expr("int32", 0)?;
    let one = int_expr("int32", 1)?;
    let x = ffi_api::prim_expr(ffi_api::var_from_str("x", "int32", None)?)?;
    let y = ffi_api::prim_expr(ffi_api::var_from_str("y", "int32", None)?)?;
    let x_is_zero = equal(x, zero)?;
    let y_is_one = equal(y, one)?;

    let scoped = Analyzer::new()?;
    assert!(!scoped.can_prove(&x_is_zero, ProofStrength::Default)?);
    scoped.with_constraint(&x_is_zero, |outer| {
        assert!(outer.can_prove(&x_is_zero, ProofStrength::Default)?);
        let error = expect_error(outer.fork());
        assert_eq!(error.kind().as_str(), "ValueError");
        assert!(error.message().contains("active constraint scope"));

        outer.with_constraint(&y_is_one, |inner| {
            assert!(inner.can_prove(&x_is_zero, ProofStrength::Default)?);
            assert!(inner.can_prove(&y_is_one, ProofStrength::Default)?);
            Ok(())
        })?;
        assert!(!outer.can_prove(&y_is_one, ProofStrength::Default)?);
        Ok(())
    })?;
    assert!(!scoped.can_prove(&x_is_zero, ProofStrength::Default)?);

    let body_error = scoped.with_constraint(&x_is_zero, |inside| -> Result<()> {
        assert!(inside.can_prove(&x_is_zero, ProofStrength::Default)?);
        Err(Error::new(
            tvm_ffi::error::VALUE_ERROR,
            "intentional constraint body error",
            "",
        ))
    });
    let error = expect_error(body_error);
    assert_eq!(error.kind().as_str(), "ValueError");
    assert!(!scoped.can_prove(&x_is_zero, ProofStrength::Default)?);

    let panic = catch_unwind(AssertUnwindSafe(|| {
        scoped.with_constraint(&x_is_zero, |inside| -> Result<()> {
            assert!(inside.can_prove(&x_is_zero, ProofStrength::Default)?);
            panic!("intentional constraint body panic");
        })
    }));
    assert!(panic.is_err());
    assert!(!scoped.can_prove(&x_is_zero, ProofStrength::Default)?);

    // A fresh scope proves both that panic unwinding ran the native recovery
    // callback and that the wrapper's depth accounting returned to zero.
    scoped.with_constraint(&x_is_zero, |inside| {
        assert!(inside.can_prove(&x_is_zero, ProofStrength::Default)?);
        Ok(())
    })?;
    Ok(())
}

#[test]
fn prim_func_pass_callback_round_trips_owned_rvalues() -> Result<()> {
    let _compiler = load_compiler()?;
    let body: Stmt = ffi_api::evaluate(int_expr("int32", 7)?.as_base(), None)?.into();
    let func = ffi_api::prim_func_without_params(&body, None)?;
    let module = ffi_api::ir_module_with_prim_func("main", &func)?;

    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&calls);
    let identity_pass = ffi_api::create_prim_func_pass(
        move |func, _module, _context| {
            callback_calls.fetch_add(1, Ordering::Relaxed);
            Ok(func)
        },
        0,
        "tirx.RustIdentityCallbackTest",
        &[],
        false,
    )?;
    let output = ffi_api::run_pass(&identity_pass, &module)?;
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert!(only_prim_func(&output)?.same_as(&func));

    Ok(())
}

#[test]
fn sequential_runs_rust_pass_callbacks_in_order() -> Result<()> {
    let _compiler = load_compiler()?;
    let body: Stmt = ffi_api::evaluate(int_expr("int32", 7)?.as_base(), None)?.into();
    let func = ffi_api::prim_func_without_params(&body, None)?;
    let module = ffi_api::ir_module_with_prim_func("main", &func)?;

    let order = Arc::new(AtomicUsize::new(0));
    let first_order = Arc::clone(&order);
    let first = ffi_api::create_prim_func_pass(
        move |func, _module, _context| {
            assert_eq!(first_order.fetch_add(1, Ordering::SeqCst), 0);
            Ok(func)
        },
        0,
        "tirx.RustSequentialFirstTest",
        &[],
        false,
    )?;
    let second_order = Arc::clone(&order);
    let second = ffi_api::create_prim_func_pass(
        move |func, _module, _context| {
            assert_eq!(second_order.fetch_add(1, Ordering::SeqCst), 1);
            Ok(func)
        },
        0,
        "tirx.RustSequentialSecondTest",
        &[],
        false,
    )?;
    let sequential = ffi_api::sequential(
        vec![first, second],
        0,
        "tirx.RustSequentialOrderTest",
        &[],
        false,
    )?;
    let output = ffi_api::run_pass(&sequential, &module)?;
    assert_eq!(order.load(Ordering::SeqCst), 2);
    assert!(only_prim_func(&output)?.same_as(&func));
    Ok(())
}

#[test]
fn pass_callbacks_contain_panics_for_both_pass_kinds() -> Result<()> {
    let _compiler = load_compiler()?;
    let body: Stmt = ffi_api::evaluate(int_expr("int32", 0)?.as_base(), None)?.into();
    let func = ffi_api::prim_func_without_params(&body, None)?;
    let module = ffi_api::ir_module_with_prim_func("main", &func)?;

    let prim_func_panic = ffi_api::create_prim_func_pass(
        |_func, _module, _context| panic!("intentional PrimFuncPass panic"),
        0,
        "tirx.RustPrimFuncPanicContainmentTest",
        &[],
        false,
    )?;
    let error = expect_error(ffi_api::run_pass(&prim_func_panic, &module));
    assert_eq!(error.kind().as_str(), "RuntimeError");
    assert!(error.message().contains("intentional PrimFuncPass panic"));

    let module_panic = ffi_api::create_module_pass(
        |_module, _context| panic!("intentional ModulePass panic"),
        0,
        "tirx.RustModulePanicContainmentTest",
        &[],
        false,
    )?;
    let error = expect_error(ffi_api::run_pass(&module_panic, &module));
    assert_eq!(error.kind().as_str(), "RuntimeError");
    assert!(error.message().contains("intentional ModulePass panic"));
    Ok(())
}
