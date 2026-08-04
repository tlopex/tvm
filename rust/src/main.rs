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

//! End-to-end demo of the safe pass-writing layer over generated bindings.

use std::path::PathBuf;

use tvm_ffi::{Module, Result};
use tvm_tirx_bindings::ffi_api;
use tvm_tirx_bindings::generated::ir::{self, Expr, IntImm};
use tvm_tirx_bindings::generated::tirx::{Evaluate, Stmt};
use tvm_tirx_bindings::passes::{
    remove_no_op_conservative, skip_assert, skip_assert_pass, verify_ssa,
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

fn main() -> Result<()> {
    let compiler_library = compiler_library();
    let _compiler = Module::load_from_file(compiler_library.to_string_lossy())?;

    let forty_two: Expr = ffi_api::int_imm_from_str("int64", 42, None)?.into();
    let forty_two_value =
        try_downcast::<_, IntImm>(&forty_two).expect("IntImm constructor must return IntImm");
    println!("IntImm.value = {}", forty_two_value.value()?);

    let condition: Expr = ffi_api::int_imm_from_str("bool", 1, None)?.into();
    let value: Expr = ffi_api::int_imm_from_str("int32", 7, None)?.into();
    let evaluate: Stmt = ffi_api::evaluate(&value, None)?.into();
    let if_then_else = ffi_api::if_then_else(&condition, &evaluate, None, None)?;
    println!(
        "IfThenElse.else_case.has_value() = {}",
        if_then_else.else_case()?.is_some()
    );

    let op = ir::get_op(tvm_ffi::String::from("tirx.webgpu.subgroup_shuffle"))?;
    println!("Op.num_inputs = {}", op.num_inputs()?);

    let error_kind = ffi_api::string_imm("ValueError", None)?;
    let message = ffi_api::string_imm("expected true", None)?;
    let assertion: Stmt = ffi_api::assert_stmt(&condition, &error_kind, &[message], None)?.into();
    let body = ffi_api::normalize_seq(vec![assertion, evaluate], None)?;
    let skipped = skip_assert(&body)?;
    println!(
        "SkipAssert result = Evaluate({:?})",
        evaluated_int(&skipped)?
    );

    let pure_evaluate: Stmt = ffi_api::evaluate(&forty_two, None)?.into();
    let no_op = remove_no_op_conservative(&pure_evaluate)?;
    println!("RemoveNoOp result = Evaluate({:?})", evaluated_int(&no_op)?);

    let func = ffi_api::prim_func_without_params(&body, None)?;
    println!("VerifySSA = {}", verify_ssa(&func)?);
    let _pass = skip_assert_pass()?;

    println!("OK: safe C++ construction + Rust visitor/mutator/pass callback");
    Ok(())
}
