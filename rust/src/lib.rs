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

//! Minimal handwritten TVM IR bindings used to develop Rust analyses and passes.
//!
//! Nodes use ABI-complete `#[repr(C)]` Rust layouts and are allocated by Rust.
//! Type-specific validation, derived fields, and formerly virtual behavior use
//! registered C ABI function tables rather than packed constructors or the C++
//! object ABI. All objects share the FFI header and runtime type table for
//! ownership, checked casts, and structural traversal.

#[doc(hidden)]
pub mod abi;
pub mod analysis;
pub mod ir;
pub mod relax;
pub mod tirx;
pub mod transform;

pub use tvm_ffi;
