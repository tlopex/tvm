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

//! Rust port of TIRx's `VerifySSA` analysis.
//!
//! This is intentionally implemented with reflection-backed generated field
//! getters and [`crate::visitor::StmtExprVisitor`], rather than delegating to
//! the existing C++ pass.  It therefore doubles as an executable test of the
//! generated owning wrappers and the hand-written semantic traversal.

use std::collections::HashMap;

use crate::ffi_api;
use crate::generated::ir::{Expr, IRModule};
use crate::generated::tirx::{analysis, AllocBuffer, Bind, Buffer, For, Let, PrimFunc, Var};
use crate::generated::transform::Pass;
use crate::visitor::{try_downcast, try_downcast_exact, walk_expr, walk_stmt, StmtExprVisitor};
use crate::PrimExpr;
use tvm_ffi::{Array, Error, ObjectArc, ObjectRefCore, Result};

fn object_identity<R: ObjectRefCore>(value: &R) -> usize {
    unsafe { ObjectArc::as_raw(<R as ObjectRefCore>::data(value)) as *const () as usize }
}

fn expr_deep_equal(lhs: &Expr, rhs: &PrimExpr) -> Result<bool> {
    analysis::expr_deep_equal(
        Some(PrimExpr::try_from_base(lhs.clone())?),
        Some(rhs.clone()),
    )
}

fn type_error(message: impl AsRef<str>) -> Error {
    Error::new(tvm_ffi::error::TYPE_ERROR, message.as_ref(), "")
}

fn required_var_item(values: &Array<Option<Var>>, index: usize, path: &str) -> Result<Var> {
    let value = values.get(index).map_err(|error| {
        type_error(format!(
            "cannot decode required IR array element `{path}[{index}]`: {error}"
        ))
    })?;
    ffi_api::require_defined(value, &format!("{path}[{index}]"))
}

struct SsaVerifier {
    definitions: HashMap<usize, Expr>,
    in_match_scope: bool,
    is_ssa: bool,
}

impl SsaVerifier {
    fn new() -> Self {
        Self {
            definitions: HashMap::new(),
            in_match_scope: false,
            is_ssa: true,
        }
    }

    fn mark_definition(&mut self, var: &Var, value: Expr, allow_duplicate: bool) {
        let identity = object_identity(var);
        match self.definitions.entry(identity) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(value);
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                if !allow_duplicate {
                    self.is_ssa = false;
                }
            }
        }
    }

    /// Match-buffer definitions are the one place where variables occurring
    /// inside a buffer descriptor are definitions rather than uses.  Keep this
    /// separate from the visitor's normal `visit_buffer_def`, matching the C++
    /// `SSAVerifier::DefineBuffer` implementation.
    fn define_buffer(&mut self, buffer: &Buffer) -> Result<()> {
        let previous_scope = self.in_match_scope;
        self.in_match_scope = true;
        let result = (|| {
            let data = ffi_api::require_defined(buffer.data()?, "Buffer::data")?;
            let shape = buffer.shape()?;
            let strides = buffer.strides()?;
            let elem_offset =
                ffi_api::require_defined(buffer.elem_offset()?, "Buffer::elem_offset")?;

            let data: Expr = data.into();
            self.visit_expr(&data)?;
            self.visit_prim_expr_array(&shape, "Buffer::shape")?;
            if strides.is_defined() {
                self.visit_prim_expr_array(&strides, "Buffer::strides")?;
            }
            self.visit_prim_expr(&elem_offset)
        })();
        self.in_match_scope = previous_scope;
        result
    }

    fn run(&mut self, func: &PrimFunc) -> Result<()> {
        let params = func.params()?;
        let buffer_map = func.buffer_map()?;
        let body = ffi_api::require_defined(func.body()?, "PrimFunc::body")?;

        for index in 0..params.len() {
            let param = required_var_item(&params, index, "PrimFunc::params")?;
            let value: Expr = param.clone().into();
            self.mark_definition(&param, value, false);
        }
        for (index, (var, buffer)) in (&buffer_map).into_iter().enumerate() {
            let _var =
                ffi_api::require_defined(var, &format!("PrimFunc::buffer_map[{index}].key"))?;
            let buffer =
                ffi_api::require_defined(buffer, &format!("PrimFunc::buffer_map[{index}].value"))?;
            self.define_buffer(&buffer)?;
        }
        self.visit_stmt(&body)
    }
}

impl StmtExprVisitor for SsaVerifier {
    fn visit_expr(&mut self, expr: &Expr) -> Result<()> {
        if !self.is_ssa {
            return Ok(());
        }

        // Intercept variables before normal dispatch so the definition table
        // retains the owning reference and therefore its pointer identity.
        if let Some(var) = try_downcast_exact::<_, Var>(expr) {
            if self.in_match_scope {
                self.mark_definition(&var, expr.clone(), true);
            }
            return Ok(());
        }
        walk_expr(self, expr)
    }

    fn visit_stmt(&mut self, stmt: &crate::generated::tirx::Stmt) -> Result<()> {
        if self.is_ssa {
            walk_stmt(self, stmt)
        } else {
            Ok(())
        }
    }

    fn visit_let(&mut self, node: &Let) -> Result<()> {
        let var = ffi_api::require_defined(node.var()?, "Let::var")?;
        let value = ffi_api::require_defined(node.value()?, "Let::value")?;
        let body = ffi_api::require_defined(node.body()?, "Let::body")?;
        let identity = object_identity(&var);
        if let Some(previous) = self.definitions.get(&identity) {
            if !expr_deep_equal(previous, &value)? {
                self.is_ssa = false;
                return Ok(());
            }
        } else {
            // Keep the unrefined owning handle in the table.  Removing the
            // static refinement does not change the object reference, so the
            // definition and pointer-identity semantics remain unchanged.
            self.definitions.insert(identity, value.clone().into_base());
        }
        self.visit_prim_expr(&value)?;
        self.visit_prim_expr(&body)
    }

    fn visit_bind(&mut self, node: &Bind) -> Result<()> {
        let var = ffi_api::require_defined(node.var()?, "Bind::var")?;
        let value = ffi_api::require_defined(node.value()?, "Bind::value")?;
        self.mark_definition(&var, value.clone(), false);
        self.visit_expr(&value)
    }

    fn visit_for(&mut self, node: &For) -> Result<()> {
        let loop_var = ffi_api::require_defined(node.loop_var()?, "For::loop_var")?;
        let min = ffi_api::require_defined(node.min()?, "For::min")?;
        let extent = ffi_api::require_defined(node.extent()?, "For::extent")?;
        let step = node.step()?;
        let body = ffi_api::require_defined(node.body()?, "For::body")?;

        let value: Expr = loop_var.clone().into();
        self.mark_definition(&loop_var, value, false);
        self.visit_prim_expr(&min)?;
        self.visit_prim_expr(&extent)?;
        if let Some(step) = step {
            self.visit_prim_expr(&step)?;
        }
        self.visit_stmt(&body)
    }

    fn visit_alloc_buffer(&mut self, node: &AllocBuffer) -> Result<()> {
        let buffer = ffi_api::require_defined(node.buffer()?, "AllocBuffer::buffer")?;
        let data = ffi_api::require_defined(buffer.data()?, "AllocBuffer::buffer.data")?;
        let value: Expr = data.clone().into();
        self.mark_definition(&data, value, false);
        self.visit_buffer_def(&buffer, true)
    }
}

/// Return whether `func` satisfies TIRx's (slightly relaxed) SSA rule.
///
/// Repeated `Let` definitions of the same variable are accepted only when the
/// bound values are deeply equal, matching the canonical C++ implementation.
pub fn verify_ssa(func: &PrimFunc) -> Result<bool> {
    let mut verifier = SsaVerifier::new();
    verifier.run(func)?;
    Ok(verifier.is_ssa)
}

/// Validate SSA form and return a normal TVM FFI error on failure.
pub fn verify_ssa_or_error(func: &PrimFunc) -> Result<()> {
    if verify_ssa(func)? {
        Ok(())
    } else {
        Err(tvm_ffi::Error::new(
            tvm_ffi::error::RUNTIME_ERROR,
            "TIRx PrimFunc is not in SSA form",
            "",
        ))
    }
}

/// Validate every TIRx `PrimFunc` in an IRModule and return the module
/// pointer-identically on success.
pub fn verify_ssa_module(module: &IRModule) -> Result<IRModule> {
    let functions = module.functions()?;
    for (index, (global_var, base_func)) in (&functions).into_iter().enumerate() {
        let _global_var =
            ffi_api::require_defined(global_var, &format!("IRModule::functions[{index}].key"))?;
        let base_func =
            ffi_api::require_defined(base_func, &format!("IRModule::functions[{index}].value"))?;
        if let Some(func) = try_downcast::<_, PrimFunc>(&base_func) {
            verify_ssa_or_error(&func)?;
        }
    }
    Ok(module.clone())
}

/// Package the Rust verifier as a normal TVM `ModulePass`.
pub fn verify_ssa_pass() -> Result<Pass> {
    ffi_api::create_module_pass(
        |module, _context| verify_ssa_module(&module),
        0,
        "tirx.RustVerifySSA",
        &[],
        false,
    )
}
