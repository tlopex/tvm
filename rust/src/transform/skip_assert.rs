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

use super::utils::int_value;
use super::{create_prim_func_pass, Pass};
use crate::tirx::{AssertStmtObj, Evaluate, PrimFunc, SeqStmt, Stmt};

/// Replace every `AssertStmt` in a PrimFunc with `Evaluate(0)`.
pub fn skip_assert_prim_func(func: PrimFunc) -> Result<PrimFunc> {
    let mut mapper = AssertSkipper;
    structural_map(func, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

struct AssertSkipper;

#[tvm_ffi::dispatch(map)]
impl AssertSkipper {
    fn map_assert(&mut self, _value: &AssertStmtObj) -> Result<Any> {
        Ok(Any::from(Evaluate::from_i64(0)?))
    }

    fn map_sequence(&mut self, value: SeqStmt) -> Result<Any> {
        let mut flattened = Vec::new();
        for statement in value.seq.iter() {
            if let Ok(sequence) = statement.clone().try_cast::<SeqStmt>() {
                flattened.extend(sequence.seq.iter());
            } else if !is_evaluate_zero(&statement)? {
                flattened.push(statement);
            }
        }
        match flattened.len() {
            0 => Ok(Any::from(Evaluate::from_i64(0)?)),
            1 => Ok(Any::from(flattened.pop().unwrap())),
            _ => Ok(Any::from(SeqStmt::new(flattened)?)),
        }
    }
}

fn is_evaluate_zero(statement: &Stmt) -> Result<bool> {
    let Ok(evaluate) = statement.clone().try_cast::<Evaluate>() else {
        return Ok(false);
    };
    Ok(int_value(&evaluate.value)? == Some(0))
}

/// Build the Rust implementation of `tirx.SkipAssert` as a normal TVM pass.
pub fn skip_assert() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.RustSkipAssert",
        0,
        Vec::new(),
        false,
        |func, _module, _context| skip_assert_prim_func(func),
    )
}
