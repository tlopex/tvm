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

use crate::generated::ir::{Expr, PrimType};
use tvm_ffi::{ObjectRefCast, Result, TypedExprType};

/// An IR expression whose reflected result type is a primitive scalar/vector type.
pub type PrimExpr = tvm_ffi::TypedExpr<Expr, PrimType>;

impl TypedExprType<Expr> for PrimType {
    fn type_of(expr: &Expr) -> Result<Self> {
        expr.ty()?.try_cast()
    }
}
