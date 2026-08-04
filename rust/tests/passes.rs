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

use tvm_ffi::{Any, AnyView, Function, Module, ObjectArc, ObjectRefCore, Result, TypeIndex};
use tvm_tirx_bindings::ffi_api;
use tvm_tirx_bindings::generated::ir::{Expr, IRModule, IntImmObj, Op, Range, RangeObj, SpanObj};
use tvm_tirx_bindings::generated::tirx::{EvaluateObj, PrimFunc, Stmt};
use tvm_tirx_bindings::passes::{
    remove_no_op_conservative, remove_no_op_conservative_pass, skip_assert, skip_assert_pass,
    verify_ssa, verify_ssa_pass,
};

fn compiler_library() -> PathBuf {
    if let Some(path) = std::env::var_os("TVM_COMPILER_LIB") {
        return PathBuf::from(path);
    }
    let build_dir = std::env::var_os("TVM_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../build"));
    build_dir.join("lib/libtvm_compiler.so")
}

fn evaluated_int(stmt: &Stmt) -> Option<i64> {
    stmt.downcast::<EvaluateObj>()
        .and_then(|node| node.value.downcast::<IntImmObj>())
        .map(|node| node.value)
}

fn only_prim_func(module: &IRModule) -> PrimFunc {
    let base_func = module
        .functions
        .values()
        .next()
        .expect("test module must contain one function");
    Any::from(base_func)
        .try_as::<PrimFunc>()
        .expect("test module function must be tirx.PrimFunc")
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
    let body = ffi_api::normalize_seq(vec![assertion, evaluate_seven.clone()], None)?;

    let skipped = skip_assert(&body)?;
    assert_eq!(evaluated_int(&skipped), Some(7));

    let dead_if: Stmt = ffi_api::if_then_else(&false_value, &evaluate_seven, None, None)?.into();
    let simplified = remove_no_op_conservative(&dead_if)?;
    assert_eq!(evaluated_int(&simplified), Some(0));

    // A non-TIRx Expr kind is outside this prototype visitor.  Conservative
    // RemoveNoOp must retain it rather than fail the pass or erase evaluation.
    let get_op = Function::get_global("ir.GetOp")?;
    let op_name = tvm_ffi::String::from("tirx.webgpu.subgroup_shuffle");
    let op: Op = get_op.call_packed(&[AnyView::from(&op_name)])?.try_into()?;
    let op_expr: Expr = op.into();
    let evaluate_op: Stmt = ffi_api::evaluate(&op_expr, None)?.into();
    let retained = remove_no_op_conservative(&evaluate_op)?;
    assert!(retained.same_as(&evaluate_op));

    let func = ffi_api::prim_func_without_params(&body, None)?;
    assert!(verify_ssa(&func)?);

    // C++ stores an absent Span as a null ObjectRef in-place.  The generated
    // field must preserve that layout, encode as FFI None, and downcast safely.
    assert!(func.span.is_null());
    assert_eq!(
        AnyView::from(&func.span).type_index(),
        TypeIndex::kTVMFFINone as i32
    );
    assert!(func.span.downcast::<SpanObj>().is_none());

    // Nullable object fields remain ordinary generated Rust field types.  The
    // visitor turns an undefined Range into a contextual Error, not a panic.
    struct NullRangeVisitor;
    impl tvm_tirx_bindings::visitor::StmtExprVisitor for NullRangeVisitor {}
    let null_range = <Range as ObjectRefCore>::from_data(unsafe {
        ObjectArc::<RangeObj>::from_raw(std::ptr::null())
    });
    let error = tvm_tirx_bindings::visitor::StmtExprVisitor::visit_range(
        &mut NullRangeVisitor,
        &null_range,
        "test IterVar::dom",
    )
    .expect_err("undefined Range must fail closed");
    assert!(error.message().contains("test IterVar::dom"));

    // Exercise the C++ -> Rust callback boundary, including the RValueRef
    // representation used by both pass factories.  Merely constructing these
    // Pass objects would not detect an incompatible Rust callback decoder.
    let module = ffi_api::ir_module_with_prim_func("main", &func)?;
    assert!(module.source_map.is_defined());
    let skip_pass = skip_assert_pass()?;
    let skipped_module = ffi_api::run_pass(&skip_pass, &module)?;
    assert_eq!(
        evaluated_int(&only_prim_func(&skipped_module).body),
        Some(7)
    );

    let dead_func = ffi_api::prim_func_without_params(&dead_if, None)?;
    let dead_module = ffi_api::ir_module_with_prim_func("main", &dead_func)?;
    let remove_pass = remove_no_op_conservative_pass()?;
    let simplified_module = ffi_api::run_pass(&remove_pass, &dead_module)?;
    assert_eq!(
        evaluated_int(&only_prim_func(&simplified_module).body),
        Some(0)
    );

    let verify_pass = verify_ssa_pass()?;
    let verified_module = ffi_api::run_pass(&verify_pass, &module)?;
    assert!(verify_ssa(&only_prim_func(&verified_module))?);

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
