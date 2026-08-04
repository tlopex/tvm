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
//! This is intentionally implemented with the generated field bindings and
//! [`crate::visitor::StmtExprVisitor`], rather than delegating to the existing
//! C++ pass.  It therefore doubles as an executable test of the generated
//! inheritance/layout information and of the hand-written semantic traversal.

use std::collections::HashMap;

use crate::ffi_api;
use crate::generated::ir::{Expr, IRModule};
use crate::generated::tirx::{
    AllocBufferObj, BindObj, Buffer, ForObj, LetObj, PrimFunc, Var, VarObj,
};
use crate::generated::transform::Pass;
use crate::visitor::{try_downcast, walk_expr, walk_stmt, StmtExprVisitor};
use tvm_ffi::{Any, Function, ObjectArc, ObjectRefCast, ObjectRefCore, Result};

fn object_identity<R: ObjectRefCore>(value: &R) -> usize {
    unsafe { ObjectArc::as_raw(<R as ObjectRefCore>::data(value)) as *const () as usize }
}

fn expr_deep_equal(lhs: &Expr, rhs: &Expr) -> Result<bool> {
    let result = Function::get_global("tirx.analysis.expr_deep_equal")?.call_tuple((lhs, rhs))?;
    bool::try_from(result)
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
        self.in_match_scope = true;
        let data: Expr = buffer.data.clone().into();
        self.visit_expr(&data)?;
        self.visit_expr_array(&buffer.shape)?;
        if buffer.strides.is_defined() {
            self.visit_expr_array(&buffer.strides)?;
        }
        self.visit_expr(&buffer.elem_offset)?;
        self.in_match_scope = false;
        Ok(())
    }

    fn run(&mut self, func: &PrimFunc) -> Result<()> {
        for param in &func.params {
            let value: Expr = param.clone().into();
            self.mark_definition(&param, value, false);
        }
        for (_, buffer) in &func.buffer_map {
            self.define_buffer(&buffer)?;
        }
        self.visit_stmt(&func.body)
    }
}

impl StmtExprVisitor for SsaVerifier {
    fn visit_expr(&mut self, expr: &Expr) -> Result<()> {
        if !self.is_ssa {
            return Ok(());
        }

        // `visit_var` receives only `&VarObj`, which is sufficient for reading
        // fields but cannot be cloned into an owning Expr.  Intercepting here
        // preserves the original object reference for the definition table.
        if try_downcast::<_, VarObj>(expr).is_some() {
            if self.in_match_scope {
                let var = expr.clone().try_cast::<Var>()?;
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

    fn visit_let(&mut self, node: &LetObj) -> Result<()> {
        let identity = object_identity(&node.var);
        if let Some(previous) = self.definitions.get(&identity) {
            if !expr_deep_equal(previous, &node.value)? {
                self.is_ssa = false;
                return Ok(());
            }
        } else {
            self.definitions.insert(identity, node.value.clone());
        }
        self.visit_expr(&node.value)?;
        self.visit_expr(&node.body)
    }

    fn visit_bind(&mut self, node: &BindObj) -> Result<()> {
        self.mark_definition(&node.var, node.value.clone(), false);
        self.visit_expr(&node.value)
    }

    fn visit_for(&mut self, node: &ForObj) -> Result<()> {
        let value: Expr = node.loop_var.clone().into();
        self.mark_definition(&node.loop_var, value, false);
        self.visit_expr(&node.min)?;
        self.visit_expr(&node.extent)?;
        if let Some(step) = node.step.get() {
            self.visit_expr(&step)?;
        }
        self.visit_stmt(&node.body)
    }

    fn visit_alloc_buffer(&mut self, node: &AllocBufferObj) -> Result<()> {
        let value: Expr = node.buffer.data.clone().into();
        self.mark_definition(&node.buffer.data, value, false);
        self.visit_buffer_def(&node.buffer, true)
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
    for (_, base_func) in &module.functions {
        let any = Any::from(base_func);
        if let Some(func) = any.try_as::<PrimFunc>() {
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
