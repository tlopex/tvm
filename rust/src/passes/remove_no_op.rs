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

//! A conservative first Rust phase of TIRx `RemoveNoOp`.
//!
//! This pass removes operations that can be proven locally without an
//! `arith::Analyzer`: call-free `Evaluate` expressions, literal-false control
//! flow, non-positive literal loop extents, and debug-skip regions.  Unknown
//! calls are always retained.  It is deliberately named `Conservative` and is
//! not presented as equivalent to C++ `tirx.RemoveNoOp`, which additionally
//! uses effect metadata, contextual proofs, and buffer-load/store equality.

use crate::ffi_api;
use crate::generated::ir::{Call, Expr, IntImm};
use crate::generated::tirx::{AttrStmt, Evaluate, For, IfThenElse, PrimFunc, Stmt, While};
use crate::generated::transform::Pass;
use crate::mutator::StatementMutator;
use crate::visitor::{try_downcast, StmtExprVisitor};
use tvm_ffi::{ObjectRefCore, Result};

struct UnknownCallFinder {
    found: bool,
}

impl StmtExprVisitor for UnknownCallFinder {
    fn visit_call(&mut self, _node: &Call) -> Result<()> {
        // Without a generated Op effect-kind wrapper, retaining every Call is
        // the only conservative choice.  Its arguments need not be inspected
        // once the enclosing expression is known to contain a Call.
        self.found = true;
        Ok(())
    }
}

fn contains_unknown_call(expr: &Expr) -> bool {
    let mut finder = UnknownCallFinder { found: false };
    match finder.visit_expr(expr) {
        Ok(()) => finder.found,
        // An Expr kind outside the generated TIRx semantic visitor is not
        // evidence that evaluation is pure.  Keep the enclosing statement
        // rather than making a conservative pass fail or erase behavior.
        Err(_) => true,
    }
}

fn literal_int(expr: &Expr) -> Result<Option<i64>> {
    let Some(node) = try_downcast::<_, IntImm>(expr) else {
        return Ok(None);
    };
    Ok(Some(node.value()?))
}

fn is_no_op(stmt: &Stmt) -> Result<bool> {
    let Some(node) = try_downcast::<_, Evaluate>(stmt) else {
        return Ok(false);
    };
    let value = ffi_api::require_defined(node.value()?, "Evaluate::value")?;
    Ok(literal_int(&value)? == Some(0))
}

fn make_no_op() -> Result<Stmt> {
    let zero = ffi_api::int_imm_from_str("int32", 0, None)?;
    let zero = Expr::from(zero);
    Ok(ffi_api::evaluate(&zero, None)?.into())
}

struct ConservativeNoOpRemover;

impl ConservativeNoOpRemover {
    fn preserve_expr_effects(&mut self, expressions: &[&Expr]) -> Result<Stmt> {
        let mut effects = Vec::with_capacity(expressions.len());
        for expr in expressions {
            let evaluate: Stmt = ffi_api::evaluate(expr, None)?.into();
            effects.push(self.mutate_stmt(&evaluate)?);
        }
        ffi_api::normalize_seq(effects, None)
    }
}

impl StatementMutator for ConservativeNoOpRemover {
    fn mutate_evaluate(&mut self, original: &Stmt, node: &Evaluate) -> Result<Stmt> {
        let value = ffi_api::require_defined(node.value()?, "Evaluate::value")?;
        if contains_unknown_call(&value) {
            Ok(original.clone())
        } else {
            make_no_op()
        }
    }

    fn mutate_attr_stmt(&mut self, original: &Stmt, node: &AttrStmt) -> Result<Stmt> {
        let attr_node = node.node()?;
        let attr_key = node.attr_key()?;
        let value = ffi_api::require_defined(node.value()?, "AttrStmt::value")?;
        let old_body = ffi_api::require_defined(node.body()?, "AttrStmt::body")?;
        let span = node.span()?;

        if attr_key.as_str() == "pragma_debug_skip_region" {
            return make_no_op();
        }

        let body = self.mutate_stmt(&old_body)?;
        if is_no_op(&body)? {
            self.preserve_expr_effects(&[&value])
        } else if body.same_as(&old_body) {
            Ok(original.clone())
        } else {
            Ok(ffi_api::attr_stmt(&attr_node, &attr_key, &value, &body, span.as_ref())?.into())
        }
    }

    fn mutate_if_then_else(&mut self, original: &Stmt, node: &IfThenElse) -> Result<Stmt> {
        let condition = ffi_api::require_defined(node.condition()?, "IfThenElse::condition")?;
        let old_then = ffi_api::require_defined(node.then_case()?, "IfThenElse::then_case")?;
        let old_else = node.else_case()?;
        let span = node.span()?;

        if let Some(condition_value) = literal_int(&condition)? {
            return if condition_value != 0 {
                self.mutate_stmt(&old_then)
            } else if let Some(else_case) = old_else.as_ref() {
                self.mutate_stmt(else_case)
            } else {
                make_no_op()
            };
        }

        let then_case = self.mutate_stmt(&old_then)?;
        let else_case = match old_else.as_ref() {
            Some(stmt) => Some(self.mutate_stmt(stmt)?),
            None => None,
        };
        let no_op_then = is_no_op(&then_case)?;
        let no_op_else = match else_case.as_ref() {
            Some(stmt) => is_no_op(stmt)?,
            None => true,
        };
        if no_op_then && no_op_else {
            return self.preserve_expr_effects(&[&condition]);
        }

        let else_unchanged = match (old_else.as_ref(), else_case.as_ref()) {
            (None, None) => true,
            (Some(before), Some(after)) => before.same_as(after),
            _ => false,
        };
        if then_case.same_as(&old_then) && else_unchanged {
            Ok(original.clone())
        } else {
            Ok(
                ffi_api::if_then_else(&condition, &then_case, else_case.as_ref(), span.as_ref())?
                    .into(),
            )
        }
    }

    fn mutate_for(&mut self, original: &Stmt, node: &For) -> Result<Stmt> {
        let loop_var = ffi_api::require_defined(node.loop_var()?, "For::loop_var")?;
        let min = ffi_api::require_defined(node.min()?, "For::min")?;
        let extent = ffi_api::require_defined(node.extent()?, "For::extent")?;
        let kind = node.kind()?;
        let old_body = ffi_api::require_defined(node.body()?, "For::body")?;
        let thread_binding = node.thread_binding()?;
        let annotations = node.annotations()?;
        let step = node.step()?;
        let span = node.span()?;

        // An explicit step participates in loop execution rather than being a
        // one-time bound. If it may call unknown code, extracting it into one
        // Evaluate would change its execution count. Keep the loop unchanged.
        if step
            .as_ref()
            .is_some_and(|value| contains_unknown_call(value))
        {
            return Ok(original.clone());
        }

        if literal_int(&extent)?.is_some_and(|extent| extent <= 0) {
            // The loop body is dead, but evaluating its bounds may still
            // contain an unknown call. Preserve those effects conservatively.
            return self.preserve_expr_effects(&[&min, &extent]);
        }

        let body = self.mutate_stmt(&old_body)?;
        if is_no_op(&body)? {
            return self.preserve_expr_effects(&[&min, &extent]);
        }
        if body.same_as(&old_body) {
            Ok(original.clone())
        } else {
            Ok(ffi_api::for_loop(
                &loop_var,
                &min,
                &extent,
                kind,
                &body,
                thread_binding.as_ref(),
                &annotations,
                step.as_ref(),
                span.as_ref(),
            )?
            .into())
        }
    }

    fn mutate_while(&mut self, original: &Stmt, node: &While) -> Result<Stmt> {
        let condition = ffi_api::require_defined(node.condition()?, "While::condition")?;
        let old_body = ffi_api::require_defined(node.body()?, "While::body")?;
        let span = node.span()?;

        if literal_int(&condition)? == Some(0) {
            return make_no_op();
        }
        let body = self.mutate_stmt(&old_body)?;
        if body.same_as(&old_body) {
            Ok(original.clone())
        } else {
            Ok(ffi_api::while_loop(&condition, &body, span.as_ref())?.into())
        }
    }
}

/// Conservatively remove locally evident no-ops from a statement tree.
pub fn remove_no_op_conservative(stmt: &Stmt) -> Result<Stmt> {
    ConservativeNoOpRemover.mutate_stmt(stmt)
}

/// Apply [`remove_no_op_conservative`] to a `PrimFunc` body.
pub fn remove_no_op_conservative_prim_func(func: &PrimFunc) -> Result<PrimFunc> {
    let old_body = ffi_api::require_defined(func.body()?, "PrimFunc::body")?;
    let body = remove_no_op_conservative(&old_body)?;
    if body.same_as(&old_body) {
        Ok(func.clone())
    } else {
        ffi_api::prim_func_with_body(func, &body)
    }
}

/// Package the conservative Rust implementation as a `PrimFuncPass`.
pub fn remove_no_op_conservative_pass() -> Result<Pass> {
    ffi_api::create_prim_func_pass(
        |func, _module, _context| remove_no_op_conservative_prim_func(&func),
        0,
        "tirx.RustRemoveNoOpConservative",
        &[],
        false,
    )
}
