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

//! Stateful arithmetic analysis with explicit aliasing and scope semantics.
//!
//! The generated `arith::Analyzer` handle has ordinary reference-counted
//! `Clone`, which aliases mutable C++ state. This wrapper deliberately does not
//! implement `Clone`; [`Analyzer::fork`] is the named deep-copy operation.

use std::cell::Cell;

use crate::ffi_api::require_defined;
use crate::generated::arith::{self, Analyzer as RawAnalyzer};
use crate::generated::ir::{Expr, Range};
use crate::generated::tirx::Var;
use tvm_ffi::{AnyView, Function, Result};

/// Strength used by [`Analyzer::can_prove`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i64)]
pub enum ProofStrength {
    /// Use TVM's default native analyzers.
    #[default]
    Default = 0,
    /// Also use symbolic-bound/Z3 fallback when available.
    SymbolicBound = 1,
}

/// Ergonomic owner of one mutable C++ arithmetic-analyzer state.
///
/// This type is conservatively `!Send + !Sync`, inherited from the generated
/// handle. Reflection metadata does not prove that C++ methods are thread safe.
pub struct Analyzer {
    inner: RawAnalyzer,
    active_constraints: Cell<usize>,
    poisoned: Cell<bool>,
}

impl Analyzer {
    /// Create an analyzer with empty accumulated facts.
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: arith::analyzer()?,
            active_constraints: Cell::new(0),
            poisoned: Cell::new(false),
        })
    }

    fn ensure_usable(&self) -> Result<()> {
        if self.poisoned.get() {
            Err(tvm_ffi::Error::new(
                tvm_ffi::error::RUNTIME_ERROR,
                "arithmetic Analyzer is poisoned after a constraint-exit failure",
                "",
            ))
        } else {
            Ok(())
        }
    }

    /// Borrow the generated handle for APIs not yet covered by this SDK.
    ///
    /// Cloning this raw handle aliases the same mutable state. Prefer
    /// [`Analyzer::fork`] when independent state is intended. Calling raw
    /// constraint APIs also bypasses this wrapper's LIFO and poison checks.
    pub fn as_raw(&self) -> &RawAnalyzer {
        &self.inner
    }

    /// Deep-copy all accumulated facts into independent analyzer state.
    ///
    /// Forking is rejected while [`Analyzer::with_constraint`] is active,
    /// because native scope-recovery callbacks belong to the original state.
    pub fn fork(&self) -> Result<Self> {
        self.ensure_usable()?;
        if self.active_constraints.get() != 0 {
            return Err(tvm_ffi::Error::new(
                tvm_ffi::error::VALUE_ERROR,
                "cannot fork an arithmetic Analyzer inside an active constraint scope",
                "",
            ));
        }
        Ok(Self {
            inner: arith::analyzer_clone(self.inner.clone())?,
            active_constraints: Cell::new(0),
            poisoned: Cell::new(false),
        })
    }

    /// Run TVM's alternating rewrite/canonical simplify pipeline.
    pub fn simplify(&self, expr: &Expr, steps: i64) -> Result<Expr> {
        self.ensure_usable()?;
        require_defined(
            arith::analyzer_simplify(self.inner.clone(), Some(expr.clone()), steps)?,
            "arith.AnalyzerSimplify",
        )
    }

    /// Simplify with TVM's default two pipeline steps.
    pub fn simplify_default(&self, expr: &Expr) -> Result<Expr> {
        self.simplify(expr, 2)
    }

    /// Apply rewrite simplification only.
    pub fn rewrite_simplify(&self, expr: &Expr) -> Result<Expr> {
        self.ensure_usable()?;
        require_defined(
            arith::analyzer_rewrite_simplify(self.inner.clone(), Some(expr.clone()))?,
            "arith.AnalyzerRewriteSimplify",
        )
    }

    /// Apply canonical simplification only.
    pub fn canonical_simplify(&self, expr: &Expr) -> Result<Expr> {
        self.ensure_usable()?;
        require_defined(
            arith::analyzer_canonical_simplify(self.inner.clone(), Some(expr.clone()))?,
            "arith.AnalyzerCanonicalSimplify",
        )
    }

    /// Return whether TVM can prove `expr` under the accumulated facts.
    pub fn can_prove(&self, expr: &Expr, strength: ProofStrength) -> Result<bool> {
        self.ensure_usable()?;
        arith::analyzer_can_prove(self.inner.clone(), Some(expr.clone()), strength as i64)
    }

    /// Return whether TVM can prove two expressions equal.
    pub fn can_prove_equal(&self, lhs: &Expr, rhs: &Expr) -> Result<bool> {
        self.ensure_usable()?;
        arith::analyzer_can_prove_equal(self.inner.clone(), Some(lhs.clone()), Some(rhs.clone()))
    }

    /// Bind a variable to an expression.
    pub fn bind_expr(&self, var: &Var, expr: &Expr, allow_override: bool) -> Result<()> {
        self.ensure_usable()?;
        arith::analyzer_bind_packed(&[
            AnyView::from(&self.inner),
            AnyView::from(var),
            AnyView::from(expr),
            AnyView::from(&allow_override),
        ])?;
        Ok(())
    }

    /// Bind a variable to a half-open range.
    pub fn bind_range(&self, var: &Var, range: &Range, allow_override: bool) -> Result<()> {
        self.ensure_usable()?;
        arith::analyzer_bind_packed(&[
            AnyView::from(&self.inner),
            AnyView::from(var),
            AnyView::from(range),
            AnyView::from(&allow_override),
        ])?;
        Ok(())
    }

    fn constraint_scope<'a>(&'a self, constraint: &Expr) -> Result<ConstraintGuard<'a>> {
        self.ensure_usable()?;
        let next_depth = self
            .active_constraints
            .get()
            .checked_add(1)
            .ok_or_else(|| {
                tvm_ffi::Error::new(
                    tvm_ffi::error::RUNTIME_ERROR,
                    "arithmetic Analyzer constraint depth overflow",
                    "",
                )
            })?;
        let exit = require_defined(
            arith::analyzer_enter_constraint_context(self.inner.clone(), Some(constraint.clone()))?,
            "arith.AnalyzerEnterConstraintContext",
        )?;
        self.active_constraints.set(next_depth);
        Ok(ConstraintGuard {
            analyzer: self,
            exit: Some(exit),
        })
    }

    /// Run `body` under a constraint and always execute the native exit callback.
    ///
    /// If `body` panics, normal Rust unwinding drops the guard and still exits
    /// the C++ scope before the panic continues.
    pub fn with_constraint<T>(
        &self,
        constraint: &Expr,
        body: impl FnOnce(&Analyzer) -> Result<T>,
    ) -> Result<T> {
        let guard = self.constraint_scope(constraint)?;
        let result = body(self);
        let exit_result = guard.exit();
        match (result, exit_result) {
            (Err(error), _) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Ok(value), Ok(())) => Ok(value),
        }
    }
}

// This guard is intentionally private. Exposing two shared-borrow guards would
// let safe Rust drop an outer native scope before an inner one, violating the
// C++ analyzers' strict LIFO recovery contract.
struct ConstraintGuard<'a> {
    analyzer: &'a Analyzer,
    exit: Option<Function>,
}

impl ConstraintGuard<'_> {
    fn finish(&mut self) -> Result<()> {
        if let Some(exit) = self.exit.take() {
            let result = exit.call_packed(&[]).map(|_| ());
            let depth = self.analyzer.active_constraints.get();
            self.analyzer
                .active_constraints
                .set(depth.saturating_sub(1));
            if result.is_err() {
                self.analyzer.poisoned.set(true);
            }
            result?;
        }
        Ok(())
    }

    fn exit(mut self) -> Result<()> {
        self.finish()
    }
}

impl Drop for ConstraintGuard<'_> {
    fn drop(&mut self) {
        // The C++ exit closure only releases a scope object and is expected not
        // to fail. Drop cannot return errors, so a failure poisons the wrapper;
        // the non-panicking path calls `exit` and returns that error directly.
        let _ = self.finish();
    }
}
