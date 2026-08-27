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

use tvm_ffi::{structural_map, Any, ObjectRefCast, Result, WalkOrder};

use super::utils::{int_value, LazyAnalyzer};
use super::{create_prim_func_pass, Pass};
use crate::analysis::side_effect;
use crate::ir::Expr;
use crate::tirx::{Evaluate, For as TirFor, IfThenElse, PrimFunc, SeqStmt, Stmt};

/// Remove control flow proven inactive by literals or TVM's arithmetic analyzer.
pub fn simplify_known_control_flow_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let mut mapper = KnownControlFlowSimplifier::default();
    structural_map(function, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

/// Build analyzed control-flow simplification as a normal TVM PrimFunc pass.
pub fn simplify_known_control_flow() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.RustSimplifyKnownControlFlow",
        0,
        Vec::new(),
        false,
        |function, _module, _context| simplify_known_control_flow_prim_func(function),
    )
}

#[derive(Default)]
struct KnownControlFlowSimplifier {
    analyzer: LazyAnalyzer,
}

impl KnownControlFlowSimplifier {
    fn known_integer(&mut self, expression: &Expr) -> Result<Option<i64>> {
        if let Some(value) = int_value(expression)? {
            return Ok(Some(value));
        }
        int_value(&self.analyzer.get()?.simplify(expression)?)
    }

    fn preserve_update_or_no_op(&self, expression: &Expr) -> Result<Stmt> {
        if side_effect(expression)?.may_update_state() {
            Ok(Evaluate::new(expression.clone())?.into())
        } else {
            no_op()
        }
    }
}

#[tvm_ffi::dispatch(map)]
impl KnownControlFlowSimplifier {
    fn map_evaluation(&mut self, value: Evaluate) -> Result<Any> {
        let expression = value.value()?;
        if int_value(&expression)?.is_some() {
            Ok(Any::from(no_op()?))
        } else if side_effect(&expression)?.may_update_state() {
            Ok(Any::from(value))
        } else {
            Ok(Any::from(no_op()?))
        }
    }

    fn map_conditional(&mut self, value: IfThenElse) -> Result<Any> {
        let condition_expr = value.condition()?;
        let then_case = value.then_case()?;
        let else_case = value.else_case()?;
        if let Some(condition) = self.known_integer(&condition_expr)? {
            let selected = if condition != 0 {
                then_case.clone()
            } else {
                match &else_case {
                    Some(else_case) => else_case.clone(),
                    None => no_op()?,
                }
            };
            return Ok(Any::from(canonical_no_op(selected)?));
        }

        let no_op_then = is_no_op(&then_case)?;
        let no_op_else = match &else_case {
            Some(else_case) => is_no_op(else_case)?,
            None => true,
        };
        if no_op_then && no_op_else {
            return Ok(Any::from(self.preserve_update_or_no_op(&condition_expr)?));
        }
        if let Some(else_case) = &else_case {
            if is_no_op(else_case)? {
                return Ok(Any::from(Stmt::from(IfThenElse::with_span(
                    &condition_expr,
                    &then_case,
                    None,
                    value.span()?.as_ref(),
                )?)));
            }
        }
        Ok(Any::from(value))
    }

    fn map_loop(&mut self, value: TirFor) -> Result<Any> {
        let minimum = value.min()?;
        let extent = value.extent()?;
        let body = value.body()?;
        if self.known_integer(&extent)? == Some(0) {
            return Ok(Any::from(no_op()?));
        }
        if is_no_op(&body)?
            && self.known_integer(&minimum)?.is_some()
            && self.known_integer(&extent)?.is_some()
        {
            return Ok(Any::from(no_op()?));
        }
        Ok(Any::from(value))
    }

    fn map_sequence(&mut self, value: SeqStmt) -> Result<Any> {
        Ok(Any::from(Stmt::sequence_with_span(
            value.seq()?.iter().collect(),
            value.span()?.as_ref(),
        )?))
    }
}

fn no_op() -> Result<Stmt> {
    Ok(Evaluate::from_i64(0)?.into())
}

fn canonical_no_op(statement: Stmt) -> Result<Stmt> {
    if is_no_op(&statement)? {
        no_op()
    } else {
        Ok(statement)
    }
}

fn is_no_op(statement: &Stmt) -> Result<bool> {
    let Ok(evaluate) = statement.clone().try_cast::<Evaluate>() else {
        return Ok(false);
    };
    Ok(int_value(&evaluate.value()?)?.is_some())
}
