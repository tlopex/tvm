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
//! Ordinary data nodes use ABI-complete `#[repr(C)]` Rust layouts and can be
//! allocated by Rust.  Opaque wrappers and registered C++ constructors are
//! retained only where C++ owns hidden state, virtual dispatch, interning, or
//! other semantic invariants.  Both forms use the common FFI object header and
//! runtime type table for checked casts and structural traversal.

macro_rules! global_function {
    ($name:literal) => {{
        static FUNCTION: std::sync::OnceLock<tvm_ffi::Function> = std::sync::OnceLock::new();

        if let Some(function) = FUNCTION.get() {
            Ok::<&'static tvm_ffi::Function, tvm_ffi::Error>(function)
        } else {
            // Do not cache failures: the dynamic TVM library may be loaded and
            // register this function before a later call.
            let function = tvm_ffi::Function::get_global($name)?;
            let _ = FUNCTION.set(function);
            Ok(FUNCTION
                .get()
                .expect("a successful global-function lookup must populate its cache"))
        }
    }};
}

pub(crate) use global_function;

macro_rules! reflected_field {
    ($object:expr, $name:literal) => {{
        static ACCESSOR: std::sync::OnceLock<$crate::reflection::FieldAccessor> =
            std::sync::OnceLock::new();
        $crate::reflection::get_reflected_field($object, $name, &ACCESSOR)
    }};
}

pub(crate) use reflected_field;

pub mod analysis;
pub mod ir;
pub mod relax;
pub mod tirx;
pub mod transform;

mod reflection;

pub use tvm_ffi;
