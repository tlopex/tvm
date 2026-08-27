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

use tvm_ffi::{structural_map, Any, DefRegionKind, ObjectRefCast, Result, WalkOrder};

use super::{create_relax_function_pass, Pass};
use crate::ir::{Expr, Var};
use crate::relax::RelaxFunction;

/// Alpha-rename every variable definition and let structural mapping remap uses.
///
/// Free variables are preserved.  Function parameters, loop variables, and
/// binding variables receive `suffix`; uses resolve through the map operation's
/// invocation-local identity table rather than through string matching.
pub fn rename_bound_variables(expr: Expr, suffix: &str) -> Result<Expr> {
    let mut mapper = BoundVariableRenamer {
        suffix: suffix.to_owned(),
    };
    structural_map(expr, &mut mapper, WalkOrder::PostOrder)?.try_into()
}

/// Alpha-rename definitions in one Relax function.
pub fn rename_bound_variables_function(
    function: RelaxFunction,
    suffix: &str,
) -> Result<RelaxFunction> {
    rename_bound_variables(function.into(), suffix)?.try_cast()
}

/// Build alpha-renaming as a normal TVM Relax FunctionPass.
pub fn rename_bound_variables_pass(suffix: &str) -> Result<Pass> {
    let suffix = suffix.to_owned();
    create_relax_function_pass(
        "relax.RustRenameBoundVariables",
        0,
        Vec::new(),
        false,
        move |function, _module, _context| rename_bound_variables_function(function, &suffix),
    )
}

struct BoundVariableRenamer {
    suffix: std::string::String,
}

#[tvm_ffi::dispatch(map)]
impl BoundVariableRenamer {
    fn map_variable(&mut self, value: Var, kind: DefRegionKind) -> Result<Any> {
        if kind == DefRegionKind::None {
            return Ok(Any::from(value));
        }
        let name = format!("{}{}", value.name()?.as_str(), self.suffix);
        Ok(Any::from(Var::with_type_and_span(
            &name,
            value.ty()?,
            value.span()?.as_ref(),
        )?))
    }
}
