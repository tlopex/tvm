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
use crate::generated::ir::{CallObj, Expr, IntImmObj};
use crate::generated::tirx::{
    AttrStmtObj, EvaluateObj, ForObj, IfThenElseObj, PrimFunc, Stmt, WhileObj,
};
use crate::generated::transform::Pass;
use crate::mutator::StatementMutator;
use crate::visitor::StmtExprVisitor;
use tvm_ffi::Result;

struct UnknownCallFinder {
    found: bool,
}

impl StmtExprVisitor for UnknownCallFinder {
    fn visit_call(&mut self, _node: &CallObj) -> Result<()> {
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

fn literal_int(expr: &Expr) -> Option<i64> {
    expr.downcast::<IntImmObj>().map(|node| node.value)
}

fn is_no_op(stmt: &Stmt) -> bool {
    stmt.downcast::<EvaluateObj>()
        .and_then(|node| literal_int(&node.value))
        == Some(0)
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
    fn mutate_evaluate(&mut self, original: &Stmt, node: &EvaluateObj) -> Result<Stmt> {
        if contains_unknown_call(&node.value) {
            Ok(original.clone())
        } else {
            make_no_op()
        }
    }

    fn mutate_attr_stmt(&mut self, original: &Stmt, node: &AttrStmtObj) -> Result<Stmt> {
        if node.attr_key.as_str() == "pragma_debug_skip_region" {
            return make_no_op();
        }

        let body = self.mutate_stmt(&node.body)?;
        if is_no_op(&body) {
            self.preserve_expr_effects(&[&node.value])
        } else if body.same_as(&node.body) {
            Ok(original.clone())
        } else {
            Ok(ffi_api::attr_stmt(
                &node.node,
                &node.attr_key,
                &node.value,
                &body,
                Some(&node.span),
            )?
            .into())
        }
    }

    fn mutate_if_then_else(&mut self, original: &Stmt, node: &IfThenElseObj) -> Result<Stmt> {
        if let Some(condition) = literal_int(&node.condition) {
            return if condition != 0 {
                self.mutate_stmt(&node.then_case)
            } else if let Some(else_case) = node.else_case.get() {
                self.mutate_stmt(&else_case)
            } else {
                make_no_op()
            };
        }

        let then_case = self.mutate_stmt(&node.then_case)?;
        let old_else = node.else_case.get();
        let else_case = match old_else.as_ref() {
            Some(stmt) => Some(self.mutate_stmt(stmt)?),
            None => None,
        };
        let no_op_then = is_no_op(&then_case);
        let no_op_else = else_case.as_ref().is_none_or(is_no_op);
        if no_op_then && no_op_else {
            return self.preserve_expr_effects(&[&node.condition]);
        }

        let else_unchanged = match (old_else.as_ref(), else_case.as_ref()) {
            (None, None) => true,
            (Some(before), Some(after)) => before.same_as(after),
            _ => false,
        };
        if then_case.same_as(&node.then_case) && else_unchanged {
            Ok(original.clone())
        } else {
            Ok(ffi_api::if_then_else(
                &node.condition,
                &then_case,
                else_case.as_ref(),
                Some(&node.span),
            )?
            .into())
        }
    }

    fn mutate_for(&mut self, original: &Stmt, node: &ForObj) -> Result<Stmt> {
        if literal_int(&node.extent).is_some_and(|extent| extent <= 0) {
            return make_no_op();
        }

        let body = self.mutate_stmt(&node.body)?;
        if is_no_op(&body) {
            return self.preserve_expr_effects(&[&node.min, &node.extent]);
        }
        if body.same_as(&node.body) {
            Ok(original.clone())
        } else {
            Ok(ffi_api::for_loop(
                &node.loop_var,
                &node.min,
                &node.extent,
                node.kind,
                &body,
                &node.thread_binding,
                &node.annotations,
                &node.step,
                Some(&node.span),
            )?
            .into())
        }
    }

    fn mutate_while(&mut self, original: &Stmt, node: &WhileObj) -> Result<Stmt> {
        if literal_int(&node.condition) == Some(0) {
            return make_no_op();
        }
        let body = self.mutate_stmt(&node.body)?;
        if body.same_as(&node.body) {
            Ok(original.clone())
        } else {
            Ok(ffi_api::while_loop(&node.condition, &body, Some(&node.span))?.into())
        }
    }
}

/// Conservatively remove locally evident no-ops from a statement tree.
pub fn remove_no_op_conservative(stmt: &Stmt) -> Result<Stmt> {
    ConservativeNoOpRemover.mutate_stmt(stmt)
}

/// Apply [`remove_no_op_conservative`] to a `PrimFunc` body.
pub fn remove_no_op_conservative_prim_func(func: &PrimFunc) -> Result<PrimFunc> {
    let body = remove_no_op_conservative(&func.body)?;
    if body.same_as(&func.body) {
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
