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

use std::path::PathBuf;

use tvm_ffi::{AnyValue, Array, Error, Map, Module, ObjectRefCore, Result};
use tvm_tirx_bindings::analyzer::{Analyzer, ProofStrength};
use tvm_tirx_bindings::ffi_api;
use tvm_tirx_bindings::generated::ir::{self, Expr, IRModule, IntImm};
use tvm_tirx_bindings::generated::tirx::{
    self, Buffer, Evaluate, PrimFunc, Stmt, TilePrimitiveCall, Var,
};
use tvm_tirx_bindings::passes::{
    remove_no_op_conservative, skip_assert, skip_assert_pass, verify_ssa, verify_ssa_pass,
};
use tvm_tirx_bindings::visitor::{try_downcast, StmtExprVisitor};
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

#[derive(Default)]
struct TilePayloadCounter {
    int_imms: usize,
    vars: usize,
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
fn statement_passes_rewrite_only_proven_safe_cases() -> Result<()> {
    let _compiler = load_compiler()?;
    let true_value = int_expr("bool", 1)?;
    let false_value = int_expr("bool", 0)?;
    let seven = int_expr("int32", 7)?;
    let evaluate_seven: Stmt = ffi_api::evaluate(&seven, None)?.into();

    let error_kind = ffi_api::string_imm("ValueError", None)?;
    let message = ffi_api::string_imm("expected true", None)?;
    let assertion: Stmt = ffi_api::assert_stmt(&true_value, &error_kind, &[message], None)?.into();
    let body = ffi_api::normalize_seq(vec![assertion.clone(), evaluate_seven.clone()], None)?;

    let skipped = skip_assert(&body)?;
    assert_eq!(evaluated_int(&skipped)?, Some(7));

    let dead_if: Stmt = ffi_api::if_then_else(&false_value, &evaluate_seven, None, None)?.into();
    let simplified = remove_no_op_conservative(&dead_if)?;
    assert_eq!(evaluated_int(&simplified)?, Some(0));

    let op = ir::get_op(tvm_ffi::String::from("tirx.webgpu.subgroup_shuffle"))?;
    let op_expr: Expr = op.clone().into();
    let evaluate_op: Stmt = ffi_api::evaluate(&op_expr, None)?.into();
    let retained = remove_no_op_conservative(&evaluate_op)?;
    assert!(retained.same_as(&evaluate_op));

    // Owning downcasts and reflection-backed getters retain the exact C++
    // object identity; they never allocate a mirrored Rust object.
    let retained_evaluate =
        try_downcast::<_, Evaluate>(&retained).expect("retained statement must be Evaluate");
    let retained_op = retained_evaluate
        .value()?
        .expect("Evaluate::value must be defined");
    assert!(retained_op.same_as(&op_expr));

    let call_type = ir::expr_type(Some(seven.as_base().clone()))?;
    let call = ir::call(
        call_type,
        Some(op_expr.clone()),
        Array::new(vec![]),
        None,
        Array::new(vec![]),
        None,
    )?
    .expect("ir.Call must return a defined expression");
    let call = ffi_api::prim_expr(call)?;
    let loop_var = ffi_api::var_from_str("i", "int32", None)?;
    let zero = int_expr("int32", 0)?;
    let annotations = Map::<tvm_ffi::String, AnyValue>::new();
    let dead_loop: Stmt = ffi_api::for_loop(
        &loop_var,
        &zero,
        &zero,
        0,
        &evaluate_seven,
        None,
        &annotations,
        Some(&call),
        None,
    )?
    .into();
    let preserved_step_loop = remove_no_op_conservative(&dead_loop)?;
    assert!(preserved_step_loop.same_as(&dead_loop));
    Ok(())
}

#[test]
fn tile_payload_roles_match_cpp_traversal() -> Result<()> {
    let _compiler = load_compiler()?;
    let true_value = int_expr("bool", 1)?;
    let error_kind = ffi_api::string_imm("ValueError", None)?;
    let message = ffi_api::string_imm("expected true", None)?;
    let assertion: Stmt = ffi_api::assert_stmt(&true_value, &error_kind, &[message], None)?.into();

    let tile_op = ir::get_op(tvm_ffi::String::from("tirx.tile.zero"))?;
    assert!(try_downcast::<_, PrimExpr>(&tile_op).is_none());
    let error = expect_error(ffi_api::prim_expr(tile_op.clone()));
    assert_eq!(error.kind().as_str(), "TypeError");
    assert!(error.message().contains("ir.Type"));
    assert!(error.message().contains("ir.PrimType"));
    let tile_args = Array::new(vec![
        // `Op` derives from Expr but is not a PrimExpr. C++ ignores it.
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

    // VerifySSA uses the expression+statement visitor. A visitor that treats
    // every Expr as PrimExpr fails on the Op payload above.
    let func = ffi_api::prim_func_without_params(&tile_stmt, None)?;
    assert!(verify_ssa(&func)?);

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
fn tile_payload_uses_strict_cpp_casts() -> Result<()> {
    let _compiler = load_compiler()?;
    let zero = int_expr("int32", 0)?;
    let one = int_expr("int32", 1)?;
    let domain = ir::range_from_min_extent(Some(zero.clone()), Some(one), None)?
        .expect("ir.Range_from_min_extent must return a defined range");
    let var = ffi_api::var_from_str("tile_axis", "int32", None)?;
    let iter_var = tirx::iter_var(Some(domain), Some(var), 0, tvm_ffi::String::from(""), None)?
        .expect("tirx.IterVar must return a defined value");

    let tile_op = ir::get_op(tvm_ffi::String::from("tirx.tile.zero"))?;
    let args = Array::new(vec![
        AnyValue::from_value(zero),
        AnyValue::from_value(7_i64),
        AnyValue::from_value(iter_var),
        AnyValue::from_value(tile_op.clone()),
    ]);
    let tile: Stmt = ffi_api::tile_primitive_call(
        &tile_op,
        &args,
        &Map::<tvm_ffi::String, Option<Buffer>>::new(),
        &Map::<tvm_ffi::String, AnyValue>::new(),
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
    Ok(())
}

#[test]
fn verify_ssa_rejects_duplicate_definitions_through_module_pass() -> Result<()> {
    let _compiler = load_compiler()?;
    let zero = int_expr("int32", 0)?;
    let body: Stmt = ffi_api::evaluate(zero.as_base(), None)?.into();

    let valid = ffi_api::prim_func_without_params(&body, None)?;
    assert!(verify_ssa(&valid)?);

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
fn verify_ssa_accepts_only_equal_duplicate_let_bindings() -> Result<()> {
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
fn analyzer_forks_facts_and_balances_nested_constraints() -> Result<()> {
    let _compiler = load_compiler()?;
    let zero = int_expr("int32", 0)?;
    let one = int_expr("int32", 1)?;
    let x_var = ffi_api::var_from_str("x", "int32", None)?;
    let y_var = ffi_api::var_from_str("y", "int32", None)?;
    let x = ffi_api::prim_expr(x_var.clone())?;
    let y = ffi_api::prim_expr(y_var.clone())?;
    let x_is_zero = ffi_api::prim_expr(
        tirx::eq(Some(x), Some(zero.clone()), None)?
            .expect("tirx.EQ must return a defined expression"),
    )?;
    let y_is_one = ffi_api::prim_expr(
        tirx::eq(Some(y), Some(one.clone()), None)?
            .expect("tirx.EQ must return a defined expression"),
    )?;

    let analyzer = Analyzer::new()?;
    assert!(!analyzer.can_prove(&x_is_zero, ProofStrength::Default)?);
    analyzer.bind_expr(&x_var, &zero, false)?;
    let fork = analyzer.fork()?;
    assert!(fork.can_prove(&x_is_zero, ProofStrength::Default)?);
    analyzer.bind_expr(&y_var, &one, false)?;
    assert!(analyzer.can_prove(&y_is_one, ProofStrength::Default)?);
    assert!(!fork.can_prove(&y_is_one, ProofStrength::Default)?);

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
    Ok(())
}

#[test]
fn pass_callback_abi_handles_success_and_panic() -> Result<()> {
    let _compiler = load_compiler()?;
    let true_value = int_expr("bool", 1)?;
    let seven = int_expr("int32", 7)?;
    let assertion: Stmt = ffi_api::assert_stmt(
        &true_value,
        &ffi_api::string_imm("ValueError", None)?,
        &[ffi_api::string_imm("expected true", None)?],
        None,
    )?
    .into();
    let evaluate_seven: Stmt = ffi_api::evaluate(&seven, None)?.into();
    let body = ffi_api::normalize_seq(vec![assertion, evaluate_seven], None)?;
    let func = ffi_api::prim_func_without_params(&body, None)?;
    let module = ffi_api::ir_module_with_prim_func("main", &func)?;

    let skipped_module = ffi_api::run_pass(&skip_assert_pass()?, &module)?;
    let skipped_func = only_prim_func(&skipped_module)?;
    let skipped_body = skipped_func
        .body()?
        .expect("transformed PrimFunc::body must be defined");
    assert_eq!(evaluated_int(&skipped_body)?, Some(7));

    let panic_pass = ffi_api::create_module_pass(
        |_module, _context| panic!("intentional Rust pass panic"),
        0,
        "tirx.RustPanicContainmentTest",
        &[],
        false,
    )?;
    let error = expect_error(ffi_api::run_pass(&panic_pass, &module));
    assert_eq!(error.kind().as_str(), "RuntimeError");
    assert!(error.message().contains("intentional Rust pass panic"));
    Ok(())
}
