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

//! Post-order arithmetic simplification implemented through the Rust mutator.

use crate::analyzer::Analyzer;
use crate::ffi_api;
use crate::generated::ir::Expr;
use crate::generated::tirx::{PrimFunc, Stmt};
use crate::generated::transform::Pass;
use crate::mutator::{rewrite_stmt, StmtExprRewriter};
use crate::visitor::try_downcast;
use crate::PrimExpr;
use tvm_ffi::{ObjectRefCore, Result};

struct ExpressionSimplifier {
    analyzer: Analyzer,
}

impl StmtExprRewriter for ExpressionSimplifier {
    fn rewrite_expr(&mut self, expr: Expr) -> Result<Option<Expr>> {
        let Some(prim_expr) = try_downcast::<_, PrimExpr>(&expr) else {
            return Ok(None);
        };
        let simplified = self.analyzer.simplify_default(&prim_expr)?.into_base();
        Ok((!expr.same_as(&simplified)).then_some(simplified))
    }
}

/// Simplify every primitive expression reachable through the canonical TIRx mutator.
pub fn simplify_stmt_expressions(stmt: &Stmt) -> Result<Stmt> {
    rewrite_stmt(
        stmt,
        &mut ExpressionSimplifier {
            analyzer: Analyzer::new()?,
        },
    )
}

/// Simplify a `PrimFunc` body while preserving pointer identity when unchanged.
pub fn simplify_prim_func(func: &PrimFunc) -> Result<PrimFunc> {
    let old_body = ffi_api::require_defined(func.body()?, "PrimFunc::body")?;
    let body = simplify_stmt_expressions(&old_body)?;
    if body.same_as(&old_body) {
        Ok(func.clone())
    } else {
        ffi_api::prim_func_with_body(func, &body)
    }
}

/// Package [`simplify_prim_func`] as a normal TVM `PrimFuncPass`.
pub fn simplify_pass() -> Result<Pass> {
    ffi_api::create_prim_func_pass(
        |func, _module, _context| simplify_prim_func(&func),
        0,
        "tirx.RustSimplifyExpressions",
        &[],
        false,
    )
}
