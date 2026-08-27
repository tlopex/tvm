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

use tvm_ffi::{structural_map, Any, DLDataType, DLDataTypeCode, ObjectRefCast, Result, WalkOrder};

use super::{create_prim_func_pass, Pass};
use crate::ir::{Expr, IntImm, PrimType, Span};
use crate::tirx::{Add, Mul, PrimFunc, Sub};

/// Fold binary integer expressions with checked Rust arithmetic.
pub fn fold_integer_constants_expr(expr: Expr) -> Result<Expr> {
    let mut mapper = IntegerConstantFolder;
    structural_map(expr, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

/// Fold binary integer constants throughout a TIR PrimFunc.
pub fn fold_integer_constants_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let mut mapper = IntegerConstantFolder;
    structural_map(function, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

/// Build checked integer constant folding as a normal TVM PrimFunc pass.
pub fn fold_integer_constants() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.RustFoldIntegerConstants",
        0,
        Vec::new(),
        false,
        |function, _module, _context| fold_integer_constants_prim_func(function),
    )
}

struct IntegerConstantFolder;

impl IntegerConstantFolder {
    fn fold_or_keep(
        &mut self,
        original: Expr,
        lhs: &Expr,
        rhs: &Expr,
        span: Option<&Span>,
        operation: fn(i64, i64) -> Option<i64>,
    ) -> Result<Any> {
        let Some((lhs, rhs, dtype)) = matching_integer_literals(lhs, rhs)? else {
            return Ok(Any::from(original));
        };
        let Some(value) = operation(lhs.value()?, rhs.value()?) else {
            // Keep the native expression when the operation does not fit the
            // storage used by IntImm; silently wrapping here would change IR.
            return Ok(Any::from(original));
        };
        Ok(Any::from(IntImm::from_dtype_with_span(dtype, value, span)?))
    }
}

#[tvm_ffi::dispatch(map)]
impl IntegerConstantFolder {
    fn map_add(&mut self, value: Add) -> Result<Any> {
        let lhs = value.a()?;
        let rhs = value.b()?;
        let span = value.span()?;
        self.fold_or_keep(
            value.clone().into(),
            &lhs,
            &rhs,
            span.as_ref(),
            i64::checked_add,
        )
    }

    fn map_subtract(&mut self, value: Sub) -> Result<Any> {
        let lhs = value.a()?;
        let rhs = value.b()?;
        let span = value.span()?;
        self.fold_or_keep(
            value.clone().into(),
            &lhs,
            &rhs,
            span.as_ref(),
            i64::checked_sub,
        )
    }

    fn map_multiply(&mut self, value: Mul) -> Result<Any> {
        let lhs = value.a()?;
        let rhs = value.b()?;
        let span = value.span()?;
        self.fold_or_keep(
            value.clone().into(),
            &lhs,
            &rhs,
            span.as_ref(),
            i64::checked_mul,
        )
    }
}

fn matching_integer_literals(
    lhs: &Expr,
    rhs: &Expr,
) -> Result<Option<(IntImm, IntImm, DLDataType)>> {
    let Ok(lhs) = lhs.clone().try_cast::<IntImm>() else {
        return Ok(None);
    };
    let Ok(rhs) = rhs.clone().try_cast::<IntImm>() else {
        return Ok(None);
    };
    let lhs_dtype = lhs.ty()?.try_cast::<PrimType>()?.dtype()?;
    let rhs_dtype = rhs.ty()?.try_cast::<PrimType>()?.dtype()?;
    if lhs_dtype != rhs_dtype
        || (lhs_dtype.code != DLDataTypeCode::kDLInt as u8
            && lhs_dtype.code != DLDataTypeCode::kDLUInt as u8)
    {
        return Ok(None);
    }
    Ok(Some((lhs, rhs, lhs_dtype)))
}
