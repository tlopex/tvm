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

use tvm_ffi::{structural_map, Any, Error, ObjectRefCast, Result, WalkOrder, VALUE_ERROR};

use crate::ir::{Expr, IntImm, PrimType};

/// Increment every integer literal in an expression while preserving its dtype.
pub fn increment_int_immediates(expr: Expr) -> Result<Expr> {
    let mut mapper = IncrementIntImmediates;
    structural_map(expr, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

struct IncrementIntImmediates;

#[tvm_ffi::dispatch(map)]
impl IncrementIntImmediates {
    fn map_integer(&mut self, value: IntImm) -> Result<Any> {
        let dtype = value.ty()?.try_cast::<PrimType>()?.dtype()?;
        let span = value.span()?;
        let current = value.value()?;
        let incremented = current.checked_add(1).ok_or_else(|| {
            Error::new(
                VALUE_ERROR,
                &format!(
                    "cannot increment integer literal {} without overflow",
                    current
                ),
                "",
            )
        })?;
        Ok(Any::from(IntImm::from_dtype_with_span(
            dtype,
            incremented,
            span.as_ref(),
        )?))
    }
}
