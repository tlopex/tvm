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

use std::collections::HashMap;

use tvm_ffi::{
    structural_mutate, Any, AnyCompatible, AnyMap, DefRegionKind, MutateCallbacks, MutateContext,
    ObjectArc, ObjectRefCore, Result, String,
};

use super::utils::int_value;
use super::{create_prim_func_pass, Pass};
use crate::ir::{Expr, Var};
use crate::tirx::{For as TirFor, PrimFunc, Stmt};

#[derive(Default)]
struct UnitLoopEliminationState {
    replacements: HashMap<usize, Expr>,
}

/// Eliminate unannotated unit loops and substitute their variables.
///
/// This is the unit-loop portion of C++ `tirx.transform.LowerTIRxOpaque`.
/// Substitution is based on object identity, so buffer indices and all other
/// uses of the loop variable receive the mapped loop minimum.
pub fn eliminate_unit_loops_prim_func(function: PrimFunc) -> Result<PrimFunc> {
    let mut mutator = MutateCallbacks::new(
        UnitLoopEliminationState::default(),
        (eliminate_unit_loop, substitute_unit_loop_variable),
    );
    structural_mutate(function, &mut mutator)?.try_into()
}

/// Build unit-loop elimination as a normal TVM PrimFunc pass.
pub fn eliminate_unit_loops() -> Result<Pass> {
    create_prim_func_pass(
        "tirx.RustEliminateUnitLoops",
        0,
        Vec::new(),
        false,
        |function, _module, _context| eliminate_unit_loops_prim_func(function),
    )
}

fn eliminate_unit_loop(
    value: TirFor,
    mutator: &mut MutateContext<'_, UnitLoopEliminationState>,
) -> Result<Any> {
    let minimum = Expr::try_from(mutator.mutate(&value.min)?)?;
    let extent = Expr::try_from(mutator.mutate(&value.extent)?)?;
    let annotations = value.annotations.clone();
    let kind = value.kind;
    let should_eliminate = kind != crate::tirx::ForKind::ThreadBinding
        && int_value(&extent) == Some(1)
        && annotations.is_empty();

    if should_eliminate {
        let key = object_identity(&value.loop_var);
        let previous = mutator
            .state_mut()
            .replacements
            .insert(key, minimum.clone());
        let body_result = mutator.mutate(&value.body);
        match previous {
            Some(previous) => {
                mutator.state_mut().replacements.insert(key, previous);
            }
            None => {
                mutator.state_mut().replacements.remove(&key);
            }
        }
        return body_result;
    }

    let loop_var = Var::try_from(mutator.mutate_with(&value.loop_var, DefRegionKind::Recursive)?)?;
    let body = Stmt::try_from(mutator.mutate(&value.body)?)?;
    let thread_binding =
        Option::<crate::tirx::IterVar>::try_from(mutator.mutate(&value.thread_binding)?)?;
    let annotations = AnyMap::<String>::try_from(mutator.mutate(&annotations)?)?;
    let step = value
        .step
        .as_ref()
        .map(|step| mutator.mutate(step).and_then(Expr::try_from))
        .transpose()?;

    Ok(Any::from(TirFor::with_metadata(
        &loop_var,
        &minimum,
        &extent,
        kind,
        &body,
        thread_binding.as_ref(),
        &annotations,
        step.as_ref(),
        value.span.as_ref(),
    )?))
}

fn substitute_unit_loop_variable(
    value: Var,
    mutator: &mut MutateContext<'_, UnitLoopEliminationState>,
) -> Result<Any> {
    if let Some(replacement) = mutator.state().replacements.get(&object_identity(&value)) {
        return Ok(replacement.to_any());
    }
    mutator.default_mutate()
}

fn object_identity<T: ObjectRefCore>(value: &T) -> usize {
    // The pointer is used only as a non-dereferenced identity key while
    // `value` and the owning traversal root keep the object alive.
    unsafe { ObjectArc::as_raw(T::data(value)) as usize }
}
