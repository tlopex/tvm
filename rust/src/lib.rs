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

//! Rust-first pass SDK over deterministic, reflection-backed TVM bindings.
//!
//! `src/generated/` is reproduced by `regen.sh`; handwritten modules add
//! invariant-preserving constructors, traversal, mutation, analysis, and pass
//! callback ergonomics without mirroring C++ object layouts.
//!
//! Reflection does not publish a thread-safety contract, so generated handles
//! are thread-bound by default:
//!
//! ```compile_fail
//! fn require_send<T: Send>() {}
//! require_send::<tvm_tirx_bindings::generated::ir::Expr>();
//! ```
//!
//! ```compile_fail
//! fn require_sync<T: Sync>() {}
//! require_sync::<tvm_tirx_bindings::generated::ir::Expr>();
//! ```

pub mod analyzer;
pub mod ffi_api;
pub mod generated;
pub mod mutator;
pub mod passes;
pub mod visitor;

mod prim_expr;

/// An IR expression whose reflected result type is a primitive scalar/vector type.
///
/// Unlike an ordinary [`generated::ir::Expr`], this refinement is checked when
/// the value crosses the FFI boundary or is created with
/// [`tvm_ffi::TypedExpr::try_from_base`].
pub use prim_expr::PrimExpr;
