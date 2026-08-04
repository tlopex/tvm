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

//! Compat shims for APIs emitted by the historical Rust stub generator but
//! absent from the current tvm-ffi runtime fork. The generated call site in
//! `generated/ir/mod.rs` is postprocessed to call here; the long-term fix is
//! to move the method/TypeAttr fallback into the rebased generator/runtime.

use tvm_ffi::tvm_ffi_sys::{
    TVMFFIByteArray, TVMFFIGetTypeAttrColumn, TVMFFIGetTypeInfo, TVMFFITypeIndex,
};
use tvm_ffi::{AnyCompatible, Error, Function, Result};

/// Fork `Function::from_type_method_cached`: look up a reflected type method
/// (e.g. `__ffi_init__`) by runtime type index, memoized in a thread-local cell.
pub fn from_type_method_cached(
    cell: &'static std::thread::LocalKey<std::cell::OnceCell<Function>>,
    type_index: i32,
    method_name: &str,
) -> Result<Function> {
    cell.with(|c| {
        if let Some(f) = c.get() {
            return Ok(f.clone());
        }
        let f = from_type_method(type_index, method_name)?;
        let _ = c.set(f.clone());
        Ok(f)
    })
}

/// Fork `Function::from_type_method`: fetch a constructor hook from the type's
/// reflection table as an `ffi::Function`.
///
/// Explicit type methods take precedence.  Auto-generated dataclass
/// constructors such as `__ffi_init__`, however, are registered only in a
/// `TypeAttrColumn`, so that column is the required fallback (matching the
/// Python binding's lookup order).
pub fn from_type_method(type_index: i32, method_name: &str) -> Result<Function> {
    let type_error = |msg: String| Error::new(tvm_ffi::error::TYPE_ERROR, &msg, "");
    unsafe {
        let info = TVMFFIGetTypeInfo(type_index);
        if info.is_null() {
            return Err(type_error(format!(
                "no type info for type_index `{type_index}`"
            )));
        }
        let info = &*info;
        for i in 0..info.num_methods {
            let mi = &*info.methods.add(i as usize);
            if mi.name.as_str() == method_name {
                if !<Function as AnyCompatible>::check_any_strict(&mi.method) {
                    return Err(type_error(format!(
                        "method `{method_name}` on type_index `{type_index}` is not a Function"
                    )));
                }
                return Ok(<Function as AnyCompatible>::copy_from_any_view_after_check(
                    &mi.method,
                ));
            }
        }

        let attr_name = TVMFFIByteArray::from_str(method_name);
        let column = TVMFFIGetTypeAttrColumn(&attr_name);
        if !column.is_null() {
            let column = &*column;
            let index = type_index - column.begin_index;
            if index >= 0 && index < column.size && !column.data.is_null() {
                let attr = &*column.data.add(index as usize);
                if attr.type_index != TVMFFITypeIndex::kTVMFFINone as i32 {
                    if !<Function as AnyCompatible>::check_any_strict(attr) {
                        return Err(type_error(format!(
                            "type attribute `{method_name}` on type_index `{type_index}` is not a Function"
                        )));
                    }
                    return Ok(<Function as AnyCompatible>::copy_from_any_view_after_check(
                        attr,
                    ));
                }
            }
        }
    }
    Err(type_error(format!(
        "method `{method_name}` not found on type_index `{type_index}`"
    )))
}
