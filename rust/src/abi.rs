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

//! Small helpers shared by the handwritten opaque TVM object bindings.

use std::sync::OnceLock;

use tvm_ffi::{Error, FieldGetter, ObjectCore, Result};

/// Read one native reflected field, resolving its metadata only once.
pub(crate) fn reflected_field<N, T>(
    object: &N,
    cache: &'static OnceLock<FieldGetter>,
    field_name: &'static str,
) -> Result<T>
where
    N: ObjectCore,
    T: TryFrom<tvm_ffi::Any, Error = Error>,
{
    let getter = if let Some(getter) = cache.get() {
        getter
    } else {
        let candidate = FieldGetter::new(N::type_index(), field_name)?;
        let _ = cache.set(candidate);
        cache
            .get()
            .expect("a FieldGetter was inserted into this OnceLock")
    };
    getter.get(object)
}

/// Define typed accessors backed by TVM's registered field getters.
macro_rules! reflected_fields {
    ($object:ty { $($(#[$meta:meta])* $method:ident => $name:literal : $field_type:ty),* $(,)? }) => {
        impl $object {
            $(
                $(#[$meta])*
                pub fn $method(&self) -> ::tvm_ffi::Result<$field_type> {
                    static GETTER: ::std::sync::OnceLock<::tvm_ffi::FieldGetter> =
                        ::std::sync::OnceLock::new();
                    $crate::abi::reflected_field(self, &GETTER, $name)
                }
            )*
        }
    };
}

pub(crate) use reflected_fields;
