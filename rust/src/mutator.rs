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

//! Stateful Rust rewrites over the canonical TIRx statement mutator.
//!
//! TVM already owns the complete statement/expression reconstruction rules,
//! including buffer remapping and newly registered node kinds.  This module
//! keeps those rules in one place and exposes only the rewrite policy to Rust.
//! The callback is scoped to the synchronous `tirx.IRTransform` call, so it may
//! safely borrow thread-bound generated handles and ordinary `&mut` state.

use crate::ffi_api;
use crate::generated::ir::Expr;
use crate::generated::tirx::{self, Stmt};
use tvm_ffi::{Any, AnyView, Error, Function, Result};

/// Post-order rewrite policy for a TIRx tree.
///
/// Returning `None` preserves the node produced after rewriting its children.
/// Returning `Some` replaces that node.  The driver guarantees that expression
/// callbacks receive expressions and statement callbacks receive statements.
pub trait StmtExprRewriter {
    fn rewrite_expr(&mut self, _expr: Expr) -> Result<Option<Expr>> {
        Ok(None)
    }

    fn rewrite_stmt(&mut self, _stmt: Stmt) -> Result<Option<Stmt>> {
        Ok(None)
    }
}

fn rewrite_node<R: StmtExprRewriter + ?Sized>(
    rewriter: &mut R,
    args: &[AnyView<'_>],
) -> Result<Any> {
    let [node] = args else {
        return Err(Error::new(
            tvm_ffi::error::VALUE_ERROR,
            &format!("TIRx rewriter expected one argument, got {}", args.len()),
            "",
        ));
    };
    let node = Any::from(*node);
    if let Some(stmt) = node.try_as::<Stmt>() {
        return Ok(match rewriter.rewrite_stmt(stmt)? {
            Some(replacement) => Any::from(replacement),
            None => Any::new(),
        });
    }
    if let Some(expr) = node.try_as::<Expr>() {
        return Ok(match rewriter.rewrite_expr(expr)? {
            Some(replacement) => Any::from(replacement),
            None => Any::new(),
        });
    }
    Err(Error::new(
        tvm_ffi::error::TYPE_ERROR,
        "tirx.IRTransform callback received neither Stmt nor Expr",
        "",
    ))
}

/// Rewrite a statement tree in post-order while preserving identity for
/// unchanged nodes.
///
/// Unlike a registry callback, `rewriter` need not be `Send`, `Sync`, or
/// `'static`.  `tirx.IRTransform` invokes its callback synchronously and never
/// retains it; the temporary FFI function is destroyed before this function
/// returns.
pub fn rewrite_stmt<R: StmtExprRewriter + ?Sized>(stmt: &Stmt, rewriter: &mut R) -> Result<Stmt> {
    let mut callback = |args: &[AnyView<'_>]| rewrite_node(rewriter, args);
    Function::with_scoped_packed(&mut callback, |function| {
        // `tirx.IRTransform` is documented and implemented as a synchronous,
        // stack-local traversal. The runtime helper additionally rejects any
        // late, recursive, or cross-thread invocation before borrowed state is
        // touched.
        ffi_api::require_defined(
            tirx::ir_transform(Some(stmt.clone()), None, Some(function), None)?,
            "tirx.IRTransform",
        )
    })
}
