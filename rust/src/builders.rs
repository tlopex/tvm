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

//! Convenience builders layered above the handwritten IR bindings.

use tvm_ffi::Result;

use crate::ir::{Expr, IntImm};
use crate::tirx::Add;

impl Expr {
    /// Construct an integer literal expression.
    pub fn int(dtype: &str, value: i64) -> Result<Self> {
        Ok(IntImm::new(dtype, value)?.into())
    }

    /// Construct an addition expression.
    pub fn add<L, R>(lhs: L, rhs: R) -> Result<Self>
    where
        L: Into<Expr>,
        R: Into<Expr>,
    {
        Ok(Add::new(lhs, rhs)?.into())
    }
}
