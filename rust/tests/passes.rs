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

use tvm_ffi::{AnyValue, Array, Map, Module, Result};
use tvm_tirx_bindings::analyzer::{Analyzer, ProofStrength};
use tvm_tirx_bindings::ffi_api;
use tvm_tirx_bindings::generated::ir::{self, Expr, IRModule, IntImm};
use tvm_tirx_bindings::generated::tirx::{
    self, Buffer, Evaluate, PrimFunc, Stmt, TilePrimitiveCall,
};
use tvm_tirx_bindings::passes::{
    remove_no_op_conservative, remove_no_op_conservative_pass, skip_assert, skip_assert_pass,
    verify_ssa, verify_ssa_pass,
};
use tvm_tirx_bindings::visitor::try_downcast;

fn compiler_library() -> PathBuf {
    if let Some(path) = std::env::var_os("TVM_COMPILER_LIB") {
        return PathBuf::from(path);
    }
    let build_dir = std::env::var_os("TVM_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../build"));
    build_dir.join("lib/libtvm_compiler.so")
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

#[test]
fn generated_bindings_drive_real_rust_passes() -> Result<()> {
    let compiler_library = compiler_library();
    assert!(
        compiler_library.is_file(),
        "runtime pass test requires a matching compiler; set TVM_BUILD_DIR or TVM_COMPILER_LIB (missing {})",
        compiler_library.display()
    );
    let _compiler = Module::load_from_file(compiler_library.to_string_lossy())?;

    let true_value: Expr = ffi_api::int_imm_from_str("bool", 1, None)?.into();
    let false_value: Expr = ffi_api::int_imm_from_str("bool", 0, None)?.into();
    let seven: Expr = ffi_api::int_imm_from_str("int32", 7, None)?.into();
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

    // A non-TIRx Expr kind is outside this prototype visitor.  Conservative
    // RemoveNoOp must retain it rather than fail the pass or erase evaluation.
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

    // Heterogeneous TilePrimitiveCall arguments can contain nested statements.
    // The Rust mutator must rewrite those statements and preserve unrelated
    // object handles such as the operator.
    let tile_op = ir::get_op(tvm_ffi::String::from("tirx.tile.zero"))?;
    let tile_args = Array::new(vec![AnyValue::from_value(assertion.clone())]);
    let workspace = Map::<tvm_ffi::String, Option<Buffer>>::new();
    let config_key = tvm_ffi::String::from("nested_stmt");
    let config: Map<tvm_ffi::String, AnyValue> =
        [(config_key.clone(), AnyValue::from_value(assertion.clone()))]
            .into_iter()
            .collect();
    let tile = ffi_api::tile_primitive_call(&tile_op, &tile_args, &workspace, &config, None, None)?;
    let tile_stmt: Stmt = tile.clone().into();
    let rewritten_tile_stmt = skip_assert(&tile_stmt)?;
    assert!(!rewritten_tile_stmt.same_as(&tile_stmt));
    let rewritten_tile = try_downcast::<_, TilePrimitiveCall>(&rewritten_tile_stmt)
        .expect("rewritten statement must remain a TilePrimitiveCall");
    assert!(rewritten_tile.op()?.same_as(&tile_op));
    let rewritten_args = rewritten_tile.args()?;
    let nested = rewritten_args
        .get(0)?
        .try_as::<tvm_ffi::object::ObjectRef>()
        .expect("TilePrimitiveCall argument must remain an object");
    let nested = try_downcast::<_, Stmt>(&nested)
        .expect("TilePrimitiveCall argument must remain a statement");
    assert_eq!(evaluated_int(&nested)?, Some(0));
    let rewritten_config = rewritten_tile.config()?;
    let nested = rewritten_config
        .get(&config_key)?
        .expect("TilePrimitiveCall config key must remain present")
        .try_as::<tvm_ffi::object::ObjectRef>()
        .expect("TilePrimitiveCall config value must remain an object");
    let nested = try_downcast::<_, Stmt>(&nested)
        .expect("TilePrimitiveCall config value must remain a statement");
    assert_eq!(evaluated_int(&nested)?, Some(0));

    // An unknown call in an explicit step cannot be extracted into a single
    // Evaluate without changing its execution count. Keep the loop intact.
    let call_type = ir::expr_type(Some(seven.clone()))?;
    let call = ir::call(
        call_type,
        Some(op_expr.clone()),
        Array::new(vec![]),
        None,
        Array::new(vec![]),
        None,
    )?
    .expect("ir.Call must return a defined expression");
    let call: Expr = call.into();
    let loop_var = ffi_api::var_from_str("i", "int32", None)?;
    let zero: Expr = ffi_api::int_imm_from_str("int32", 0, None)?.into();
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

    let func = ffi_api::prim_func_without_params(&body, None)?;
    assert!(verify_ssa(&func)?);

    // Nullable reflected fields decode as Option and do not require a forged
    // null ObjectArc.  A required field getter keeps pointer identity.
    assert!(func.span()?.is_none());
    let func_body = func.body()?.expect("PrimFunc::body must be defined");
    assert!(func_body.same_as(&body));

    // The high-level Analyzer exposes simplification, explicit deep-copy, and
    // balanced constraint scopes without leaking its mutable raw state.
    let zero: Expr = ffi_api::int_imm_from_str("int32", 0, None)?.into();
    let x: Expr = ffi_api::var_from_str("x", "int32", None)?.into();
    let x_plus_zero: Expr = tirx::add(Some(x.clone()), Some(zero.clone()), None)?
        .expect("tirx.Add must return a defined expression")
        .into();
    let analyzer = Analyzer::new()?;
    let simplified = analyzer.simplify_default(&x_plus_zero)?;
    assert!(analyzer.can_prove_equal(&simplified, &x)?);
    let fork = analyzer.fork()?;
    assert!(fork.can_prove_equal(&simplified, &x)?);

    let x_is_zero: Expr = tirx::eq(Some(x.clone()), Some(zero), None)?
        .expect("tirx.EQ must return a defined expression")
        .into();
    analyzer.with_constraint(&x_is_zero, |scoped| {
        assert!(scoped.can_prove(&x_is_zero, ProofStrength::Default)?);
        assert!(scoped.fork().is_err());
        scoped.with_constraint(&x_is_zero, |nested| {
            assert!(nested.can_prove(&x_is_zero, ProofStrength::Default)?);
            Ok(())
        })?;
        Ok(())
    })?;
    assert!(analyzer.can_prove_equal(&x, &x)?);

    // Exercise the C++ -> Rust callback boundary, including the RValueRef
    // representation used by both pass factories.  Merely constructing these
    // Pass objects would not detect an incompatible Rust callback decoder.
    let module = ffi_api::ir_module_with_prim_func("main", &func)?;
    let _source_map = module.source_map()?;
    let skip_pass = skip_assert_pass()?;
    let skipped_module = ffi_api::run_pass(&skip_pass, &module)?;
    let skipped_func = only_prim_func(&skipped_module)?;
    let skipped_body = skipped_func
        .body()?
        .expect("transformed PrimFunc::body must be defined");
    assert_eq!(evaluated_int(&skipped_body)?, Some(7));

    let dead_func = ffi_api::prim_func_without_params(&dead_if, None)?;
    let dead_module = ffi_api::ir_module_with_prim_func("main", &dead_func)?;
    let remove_pass = remove_no_op_conservative_pass()?;
    let simplified_module = ffi_api::run_pass(&remove_pass, &dead_module)?;
    let simplified_func = only_prim_func(&simplified_module)?;
    let simplified_body = simplified_func
        .body()?
        .expect("transformed PrimFunc::body must be defined");
    assert_eq!(evaluated_int(&simplified_body)?, Some(0));

    let verify_pass = verify_ssa_pass()?;
    let verified_module = ffi_api::run_pass(&verify_pass, &module)?;
    assert!(verify_ssa(&only_prim_func(&verified_module)?)?);

    // A panic must become a recoverable TVM error instead of unwinding across
    // the C ABI callback trampoline.
    let panic_pass = ffi_api::create_module_pass(
        |_module, _context| panic!("intentional Rust pass panic"),
        0,
        "tirx.RustPanicContainmentTest",
        &[],
        false,
    )?;
    assert!(ffi_api::run_pass(&panic_pass, &module).is_err());
    Ok(())
}
