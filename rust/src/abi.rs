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

//! Machine-checkable layouts for Rust-allocated TVM FFI objects.

use std::sync::atomic::{AtomicBool, Ordering};

use tvm_ffi::tvm_ffi_sys::{TVMFFIGetTypeInfo, TVMFFITypeInfo};
use tvm_ffi::{
    get_constructor_recipe, get_native_object_layout, Any, AnyView, Error, Function, Map,
    ObjectArc, ObjectCore, Result, String, CONSTRUCTOR_PREPARE_METHOD, TYPE_ERROR, VALUE_ERROR,
};

/// Generated shape of one semantic constructor-preparation recipe.
///
/// The native reflected method validates and derives values, while this Rust-side
/// contract makes the ordered inputs and complete output-key set part of
/// the generated binding. A returned-map mismatch therefore fails explicitly
/// instead of silently ignoring a newly derived physical field.
pub(crate) trait ConstructorRecipe: ObjectCore {
    const INPUTS: &'static [&'static str];
    const DERIVED_FIELDS: &'static [&'static str];
}

/// Run a semantic constructor recipe through its reflected static type method.
///
/// Every recipe returns the same representation: a map from physical field
/// names to derived values. The caller combines those values with constructor
/// inputs and defaults, then allocates the complete object in Rust.
pub(crate) fn prepare_constructor<N: ConstructorRecipe>(
    args: &[AnyView<'_>],
) -> Result<Map<String, Any>> {
    if args.len() != N::INPUTS.len() {
        return Err(Error::new(
            TYPE_ERROR,
            &format!(
                "generated recipe for `{}` expects {} arguments, but received {}",
                N::TYPE_KEY,
                N::INPUTS.len(),
                args.len()
            ),
            "",
        ));
    }
    let native_recipe = get_constructor_recipe(N::type_index())?.ok_or_else(|| {
        Error::new(
            TYPE_ERROR,
            &format!(
                "native type `{}` does not publish a semantic constructor recipe",
                N::TYPE_KEY
            ),
            "",
        )
    })?;
    if native_recipe.version != 1 || native_recipe.method.as_str() != CONSTRUCTOR_PREPARE_METHOD {
        return Err(Error::new(
            TYPE_ERROR,
            &format!(
                "native constructor recipe for `{}` is unsupported",
                N::TYPE_KEY
            ),
            "",
        ));
    }
    let native_inputs = native_recipe
        .inputs
        .iter()
        .map(|name| name.as_str().to_owned())
        .collect::<Vec<_>>();
    let native_derived = native_recipe
        .derived_fields
        .iter()
        .map(|name| name.as_str().to_owned())
        .collect::<Vec<_>>();
    if native_inputs != N::INPUTS || native_derived != N::DERIVED_FIELDS {
        return Err(Error::new(
            TYPE_ERROR,
            &format!(
                "generated constructor recipe for `{}` does not match the loaded native recipe",
                N::TYPE_KEY
            ),
            "",
        ));
    }
    let fields = Map::<String, Any>::try_from(
        Function::from_type_method(N::type_index(), CONSTRUCTOR_PREPARE_METHOD)?
            .call_packed(args)?,
    )?;
    if fields.len() != N::DERIVED_FIELDS.len() {
        return Err(Error::new(
            TYPE_ERROR,
            &format!(
                "generated recipe for `{}` expects {} derived fields, but its native method returned {}",
                N::TYPE_KEY,
                N::DERIVED_FIELDS.len(),
                fields.len()
            ),
            "",
        ));
    }
    for (name, _) in fields.iter() {
        if !N::DERIVED_FIELDS
            .iter()
            .any(|expected| *expected == name.as_str())
        {
            return Err(Error::new(
                TYPE_ERROR,
                &format!(
                    "native constructor preparation for `{}` returned unexpected field `{}`",
                    N::TYPE_KEY,
                    name.as_str()
                ),
                "",
            ));
        }
    }
    Ok(fields)
}

/// Read and cast one derived field returned by a constructor recipe.
pub(crate) fn prepared_field<T>(
    fields: &Map<String, Any>,
    owner_type_key: &'static str,
    field_name: &'static str,
) -> Result<T>
where
    T: TryFrom<Any, Error = Error>,
{
    let value = fields.get(&String::from(field_name))?.ok_or_else(|| {
        Error::new(
            VALUE_ERROR,
            &format!(
                "type `{owner_type_key}` constructor preparation did not return `{field_name}`"
            ),
            "",
        )
    })?;
    T::try_from(value)
}

/// Rust's representation of one field that is also registered for C++ reflection.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldLayout {
    pub name: &'static str,
    pub offset: usize,
    pub size: usize,
    pub alignment: usize,
}

/// Physical layout emitted for an ABI-complete object definition.
///
/// Opaque handles deliberately do not implement this trait.  Implementing it
/// proves layout knowledge and permits lossless Rust allocation from complete
/// fields, but it does not by itself provide validation, defaults, or derived
/// fields for a semantic convenience constructor.
#[doc(hidden)]
pub trait ObjectLayout {
    const SIZE: usize;
    const ALIGNMENT: usize;
    const FIELDS: &'static [FieldLayout];

    #[doc(hidden)]
    fn loaded_layout_was_validated() -> &'static AtomicBool;

    /// Validate the complete Rust prefix inherited by this object.
    ///
    /// A derived object's total size and direct-field offsets are not enough
    /// to prove that the nested Rust base has the native parent layout.  The
    /// generated implementation overrides this hook whenever the parent is
    /// another generated object.
    #[doc(hidden)]
    fn validate_parent_layout() -> std::result::Result<(), std::string::String> {
        Ok(())
    }
}

/// Marker for a complete native type that may be a final Rust allocation.
///
/// Layout-only base prefixes deliberately do not implement this trait.  In
/// particular, knowing the bytes of a behavior base is not permission to
/// create a standalone object without registered concrete behavior methods.
#[doc(hidden)]
pub trait RustAllocatable: ObjectLayout {}

/// Allocate only an object whose complete physical layout is part of the Rust contract.
///
/// Generated constructors use this entry instead of calling `ObjectArc::new`
/// directly, so an opaque/header-only binding cannot accidentally gain a Rust
/// allocator without first supplying machine-checked layout evidence.
pub(crate) fn allocate_object<N>(value: N) -> ObjectArc<N>
where
    N: ObjectCore + RustAllocatable,
{
    validate_loaded_layout::<N>().unwrap_or_else(|message| {
        panic!(
            "cannot allocate `{}` in Rust because the loaded native layout is incompatible: {message}",
            N::TYPE_KEY
        )
    });
    ObjectArc::new(value)
}

fn runtime_type_info<N: ObjectCore>(
) -> std::result::Result<&'static TVMFFITypeInfo, std::string::String> {
    let type_index = N::type_index();
    let pointer = unsafe { TVMFFIGetTypeInfo(type_index) };
    if pointer.is_null() {
        return Err(format!(
            "the runtime returned no type information for type index {type_index}"
        ));
    }
    Ok(unsafe { &*pointer })
}

/// Check every object-layout property exposed by the loaded TVM runtime.
///
/// This runs once per Rust object type before its first direct allocation.  A
/// mismatch is fatal because writing a Rust value with a different native
/// layout would make subsequent C++ field access memory-unsafe.
#[doc(hidden)]
pub fn validate_loaded_layout<N>() -> std::result::Result<(), std::string::String>
where
    N: ObjectCore + ObjectLayout,
{
    if N::loaded_layout_was_validated().load(Ordering::Acquire) {
        return Ok(());
    }

    N::validate_parent_layout()?;
    let info = runtime_type_info::<N>()?;
    if info.type_index != N::type_index() {
        return Err(format!(
            "type index differs (Rust {}, native {})",
            N::type_index(),
            info.type_index
        ));
    }
    if info.type_key.as_str() != N::TYPE_KEY {
        return Err(format!(
            "type key differs (Rust `{}`, native `{}`)",
            N::TYPE_KEY,
            info.type_key.as_str()
        ));
    }
    if info.type_depth != N::TYPE_DEPTH {
        return Err(format!(
            "inheritance depth differs (Rust {}, native {})",
            N::TYPE_DEPTH,
            info.type_depth
        ));
    }
    let native_parent_key = if info.type_depth == 0 {
        None
    } else {
        if info.type_acenstors.is_null() {
            return Err("the runtime published a null ancestor table".to_owned());
        }
        let parent = unsafe { *info.type_acenstors.add((info.type_depth - 1) as usize) };
        if parent.is_null() {
            return Err("the runtime published a null direct parent".to_owned());
        }
        Some(unsafe { (*parent).type_key.as_str() })
    };
    if native_parent_key != N::TYPE_PARENT_KEY {
        return Err(format!(
            "direct parent differs (Rust {:?}, native {:?})",
            N::TYPE_PARENT_KEY,
            native_parent_key
        ));
    }
    if info.metadata.is_null() {
        return Err("the runtime did not publish object metadata".to_owned());
    }

    let certificate = get_native_object_layout(N::type_index())
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "the runtime did not certify a complete native object layout".to_owned())?;
    if certificate.version != 1 {
        return Err(format!(
            "native layout certificate version {} is unsupported",
            certificate.version
        ));
    }
    if certificate.alignment != N::ALIGNMENT {
        return Err(format!(
            "total alignment differs (Rust {}, native {})",
            N::ALIGNMENT,
            certificate.alignment
        ));
    }
    if certificate.is_final != N::TYPE_FINAL {
        return Err(format!(
            "finality differs (Rust {}, native {})",
            N::TYPE_FINAL,
            certificate.is_final
        ));
    }
    if certificate.field_count != N::FIELDS.len() {
        return Err(format!(
            "certified field count differs (Rust {}, native {})",
            N::FIELDS.len(),
            certificate.field_count
        ));
    }

    let native_size = unsafe { (*info.metadata).total_size };
    if native_size <= 0 {
        return Err("the runtime did not publish a fixed object size".to_owned());
    }
    if native_size as usize != N::SIZE {
        return Err(format!(
            "total size differs (Rust {}, native {})",
            N::SIZE,
            native_size
        ));
    }
    if info.num_fields < 0 {
        return Err(format!(
            "the runtime published a negative field count ({})",
            info.num_fields
        ));
    }
    if info.num_fields as usize != N::FIELDS.len() {
        return Err(format!(
            "direct field count differs (Rust {}, native {})",
            N::FIELDS.len(),
            info.num_fields
        ));
    }
    if info.num_fields != 0 && info.fields.is_null() {
        return Err("the runtime published a null field table".to_owned());
    }

    let native_fields = if info.num_fields == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(info.fields, info.num_fields as usize) }
    };
    for rust_field in N::FIELDS {
        let mut matching_fields = native_fields
            .iter()
            .filter(|field| field.name.as_str() == rust_field.name);
        let native_field = matching_fields
            .next()
            .ok_or_else(|| format!("native field `{}` is missing", rust_field.name))?;
        if matching_fields.next().is_some() {
            return Err(format!(
                "native field `{}` is registered more than once",
                rust_field.name
            ));
        }
        let native_name = native_field.name.as_str();
        if native_field.offset < 0 || native_field.size < 0 || native_field.alignment <= 0 {
            return Err(format!(
                "field `{native_name}` has invalid native layout values (offset {}, size {}, alignment {})",
                native_field.offset, native_field.size, native_field.alignment
            ));
        }
        if native_field.offset as usize != rust_field.offset
            || native_field.size as usize != rust_field.size
            || native_field.alignment as usize != rust_field.alignment
        {
            return Err(format!(
                "field `{native_name}` layout differs (Rust offset/size/alignment {}/{}/{}, native {}/{}/{})",
                rust_field.offset,
                rust_field.size,
                rust_field.alignment,
                native_field.offset,
                native_field.size,
                native_field.alignment
            ));
        }
        let native_end = (native_field.offset as usize)
            .checked_add(native_field.size as usize)
            .ok_or_else(|| format!("field `{native_name}` extent overflows usize"))?;
        if native_end > N::SIZE {
            return Err(format!(
                "field `{native_name}` ends at byte {native_end}, beyond object size {}",
                N::SIZE
            ));
        }
    }

    let expected_fingerprint = layout_fingerprint::<N>();
    if certificate.fingerprint.as_str() != expected_fingerprint {
        return Err(format!(
            "layout fingerprint differs (Rust {expected_fingerprint}, native {})",
            certificate.fingerprint.as_str()
        ));
    }

    N::loaded_layout_was_validated().store(true, Ordering::Release);
    Ok(())
}

fn layout_fingerprint<N: ObjectCore + ObjectLayout>() -> std::string::String {
    fn add_bytes(state: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *state ^= u64::from(*byte);
            *state = state.wrapping_mul(1_099_511_628_211);
        }
    }
    fn add_integer(state: &mut u64, value: u64) {
        add_bytes(state, &value.to_le_bytes());
    }
    fn add_string(state: &mut u64, value: &str) {
        add_integer(state, value.len() as u64);
        add_bytes(state, value.as_bytes());
    }

    let mut state = 14_695_981_039_346_656_037_u64;
    add_integer(&mut state, 1);
    add_string(&mut state, N::TYPE_KEY);
    add_string(&mut state, N::TYPE_PARENT_KEY.unwrap_or(""));
    add_integer(&mut state, N::SIZE as u64);
    add_integer(&mut state, N::ALIGNMENT as u64);
    add_integer(&mut state, u64::from(N::TYPE_FINAL));
    add_integer(&mut state, N::FIELDS.len() as u64);
    let mut fields = N::FIELDS.to_vec();
    fields.sort_by(|lhs, rhs| {
        lhs.offset
            .cmp(&rhs.offset)
            .then_with(|| lhs.name.cmp(rhs.name))
    });
    for field in fields {
        add_string(&mut state, field.name);
        add_integer(&mut state, field.offset as u64);
        add_integer(&mut state, field.size as u64);
        add_integer(&mut state, field.alignment as u64);
    }
    format!("{state:016x}")
}

macro_rules! impl_object_layout {
    ($object:ty $(: $parent:ty)? { $($name:literal => $field:ident : $field_type:ty),* $(,)? }) => {
        impl $crate::abi::ObjectLayout for $object {
            const SIZE: usize = ::std::mem::size_of::<$object>();
            const ALIGNMENT: usize = ::std::mem::align_of::<$object>();
            const FIELDS: &'static [$crate::abi::FieldLayout] = &[
                $(
                    $crate::abi::FieldLayout {
                        name: $name,
                        offset: ::std::mem::offset_of!($object, $field),
                        size: ::std::mem::size_of::<$field_type>(),
                        alignment: ::std::mem::align_of::<$field_type>(),
                    },
                )*
            ];

            fn loaded_layout_was_validated() -> &'static ::std::sync::atomic::AtomicBool {
                static VALIDATED: ::std::sync::atomic::AtomicBool =
                    ::std::sync::atomic::AtomicBool::new(false);
                &VALIDATED
            }

            $crate::abi::impl_parent_layout_validation!($($parent)?);
        }
    };
}

pub(crate) use impl_object_layout;

macro_rules! impl_parent_layout_validation {
    () => {};
    ($parent:ty) => {
        fn validate_parent_layout() -> ::std::result::Result<(), ::std::string::String> {
            $crate::abi::validate_loaded_layout::<$parent>()
        }
    };
}

pub(crate) use impl_parent_layout_validation;

macro_rules! impl_rust_allocatable {
    ($($object:ty),+ $(,)?) => {
        $(impl $crate::abi::RustAllocatable for $object {})+
    };
}

pub(crate) use impl_rust_allocatable;
