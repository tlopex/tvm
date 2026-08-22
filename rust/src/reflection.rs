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

//! Checked access to fields registered in TVM's runtime reflection table.

use std::sync::OnceLock;

use tvm_ffi::tvm_ffi_sys::{TVMFFIFieldGetter, TVMFFIFieldInfo, TVMFFIGetTypeInfo, TVMFFITypeInfo};
use tvm_ffi::{Any, Error, ObjectCore, Result, ATTRIBUTE_ERROR, RUNTIME_ERROR};

/// Stable copy of the only field metadata needed by a generated getter.
pub(crate) struct FieldAccessor {
    offset: usize,
    getter: TVMFFIFieldGetter,
}

/// Read one reflected field from a borrowed TVM object node.
///
/// The field getter returns an owning [`Any`].  Generated bindings can convert
/// it to their declared Rust field type with `TryFrom`.
pub(crate) fn get_reflected_field<O: ObjectCore>(
    object: &O,
    field_name: &str,
    cache: &'static OnceLock<FieldAccessor>,
) -> Result<Any> {
    let accessor = if let Some(accessor) = cache.get() {
        accessor
    } else {
        let resolved = resolve_field::<O>(field_name)?;
        let _ = cache.set(resolved);
        cache
            .get()
            .expect("a resolved reflected field must populate its cache")
    };

    let address = unsafe {
        (object as *const O as *const u8)
            .add(accessor.offset)
            .cast_mut()
            .cast()
    };
    let mut result = Any::new();
    if unsafe { (accessor.getter)(address, Any::as_data_ptr(&mut result)) } != 0 {
        return Err(Error::from_raised());
    }
    Ok(result)
}

fn resolve_field<O: ObjectCore>(field_name: &str) -> Result<FieldAccessor> {
    let type_index = O::type_index();
    let type_info = unsafe { TVMFFIGetTypeInfo(type_index) };
    if type_info.is_null() {
        return Err(Error::new(
            RUNTIME_ERROR,
            &format!("no runtime type information for type index {type_index}"),
            "",
        ));
    }

    let field = unsafe { find_field(type_info, field_name) }.ok_or_else(|| {
        Error::new(
            ATTRIBUTE_ERROR,
            &format!(
                "type `{}` has no reflected field `{field_name}`",
                O::TYPE_KEY
            ),
            "",
        )
    })?;
    let getter = field.getter.ok_or_else(|| {
        Error::new(
            ATTRIBUTE_ERROR,
            &format!(
                "reflected field `{}.{field_name}` has no getter",
                O::TYPE_KEY
            ),
            "",
        )
    })?;
    let offset = usize::try_from(field.offset).map_err(|_| {
        Error::new(
            RUNTIME_ERROR,
            &format!(
                "reflected field `{}.{field_name}` has an invalid offset",
                O::TYPE_KEY
            ),
            "",
        )
    })?;

    Ok(FieldAccessor { offset, getter })
}

unsafe fn find_field(
    type_info: *const TVMFFITypeInfo,
    field_name: &str,
) -> Option<&'static TVMFFIFieldInfo> {
    // Slot zero is the root Object.  Reflection fields are searched in the
    // same base-to-derived order used by TVM's C++ ForEachFieldInfo helper.
    for depth in 1..(*type_info).type_depth {
        let ancestor = *(*type_info).type_acenstors.add(depth as usize);
        if let Some(field) = find_field_at_level(ancestor, field_name) {
            return Some(field);
        }
    }
    find_field_at_level(type_info, field_name)
}

unsafe fn find_field_at_level(
    type_info: *const TVMFFITypeInfo,
    field_name: &str,
) -> Option<&'static TVMFFIFieldInfo> {
    if type_info.is_null() || (*type_info).fields.is_null() {
        return None;
    }
    let fields = std::slice::from_raw_parts((*type_info).fields, (*type_info).num_fields as usize);
    fields
        .iter()
        .find(|field| field.name.as_str() == field_name)
        .map(|field| &*(field as *const TVMFFIFieldInfo))
}
