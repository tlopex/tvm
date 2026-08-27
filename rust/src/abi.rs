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
    Any, AnyView, Error, Function, Map, ObjectArc, ObjectCore, Result, String, TYPE_ERROR,
    VALUE_ERROR,
};

const CONSTRUCTOR_PREPARE_METHOD: &str = "__ffi_prepare__";

/// Generated shape of one semantic constructor-preparation recipe.
///
/// The native reflected method validates and derives values, while this Rust-side
/// contract makes the expected input arity and complete output-key set part of
/// the generated binding. A returned-map mismatch therefore fails explicitly
/// instead of silently ignoring a newly derived physical field.
pub(crate) trait ConstructorRecipe: ObjectCore {
    const NUM_INPUTS: usize;
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
    if args.len() != N::NUM_INPUTS {
        return Err(Error::new(
            TYPE_ERROR,
            &format!(
                "generated recipe for `{}` expects {} arguments, but received {}",
                N::TYPE_KEY,
                N::NUM_INPUTS,
                args.len()
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
fn validate_loaded_layout<N>() -> std::result::Result<(), std::string::String>
where
    N: ObjectCore + RustAllocatable,
{
    if N::loaded_layout_was_validated().load(Ordering::Acquire) {
        return Ok(());
    }

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

    N::loaded_layout_was_validated().store(true, Ordering::Release);
    Ok(())
}

macro_rules! impl_object_layout {
    ($object:ty { $($name:literal => $field:ident : $field_type:ty),* $(,)? }) => {
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
        }
    };
}

pub(crate) use impl_object_layout;

macro_rules! impl_rust_allocatable {
    ($($object:ty),+ $(,)?) => {
        $(impl $crate::abi::RustAllocatable for $object {})+
    };
}

pub(crate) use impl_rust_allocatable;

/// Implement an infallible, ownership-preserving object-reference upcast.
///
/// Each listed source object must embed the target object as its offset-zero
/// base prefix. The handwritten declarations mirror TVM's registered
/// inheritance tree; stubgen can emit the same relation directly.
macro_rules! impl_object_upcast {
    ($($source:ty => $target:ty),+ $(,)?) => {
        $(
            impl From<$source> for $target {
                #[inline]
                fn from(value: $source) -> Self {
                    let data = <$source as tvm_ffi::ObjectRefCore>::into_data(value);
                    let data = unsafe {
                        tvm_ffi::ObjectArc::from_raw(
                            tvm_ffi::ObjectArc::into_raw(data)
                                .cast::<<$target as tvm_ffi::ObjectRefCore>::ContainerType>(),
                        )
                    };
                    <$target as tvm_ffi::ObjectRefCore>::from_data(data)
                }
            }

            impl From<&$source> for $target {
                #[inline]
                fn from(value: &$source) -> Self {
                    value.clone().into()
                }
            }
        )+
    };
}

pub(crate) use impl_object_upcast;

/// Let an owning constructor accept either an owned base handle or a borrow.
///
/// Converting an owned handle moves it without touching the reference count;
/// converting a borrow explicitly creates the owning handle that a stored
/// object field requires.
macro_rules! impl_object_borrow_to_owned {
    ($($reference:ty),+ $(,)?) => {
        $(
            impl From<&$reference> for $reference {
                #[inline]
                fn from(value: &$reference) -> Self {
                    value.clone()
                }
            }
        )+
    };
}

pub(crate) use impl_object_borrow_to_owned;
