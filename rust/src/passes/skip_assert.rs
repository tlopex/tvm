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

//! Rust implementation of TIRx's `SkipAssert` transform.

use crate::ffi_api;
use crate::generated::tirx::{AssertStmt, PrimFunc, Stmt};
use crate::generated::transform::Pass;
use crate::mutator::StatementMutator;
use tvm_ffi::{ObjectRefCore, Result};

struct AssertSkipper;

impl StatementMutator for AssertSkipper {
    fn mutate_assert_stmt(&mut self, _original: &Stmt, _node: &AssertStmt) -> Result<Stmt> {
        let zero = ffi_api::int_imm_from_str("int32", 0, None)?;
        let zero = crate::generated::ir::Expr::from(zero);
        Ok(ffi_api::evaluate(&zero, None)?.into())
    }
}

/// Replace every `AssertStmt` in a supported statement tree with `Evaluate(0)`.
pub fn skip_assert(stmt: &Stmt) -> Result<Stmt> {
    AssertSkipper.mutate_stmt(stmt)
}

/// Apply [`skip_assert`] to a `PrimFunc` body through the canonical C++
/// `PrimFunc` constructor.
pub fn skip_assert_prim_func(func: &PrimFunc) -> Result<PrimFunc> {
    let old_body = ffi_api::require_defined(func.body()?, "PrimFunc::body")?;
    let body = skip_assert(&old_body)?;
    if body.same_as(&old_body) {
        Ok(func.clone())
    } else {
        ffi_api::prim_func_with_body(func, &body)
    }
}

/// Package the Rust implementation as a normal TVM `PrimFuncPass`.
///
/// The distinct name makes it possible to compare this implementation with
/// C++ `tirx.transform.SkipAssert` in the same pipeline.
pub fn skip_assert_pass() -> Result<Pass> {
    ffi_api::create_prim_func_pass(
        |func, _module, _context| skip_assert_prim_func(&func),
        0,
        "tirx.RustSkipAssert",
        &[],
        false,
    )
}
