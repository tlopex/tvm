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

use tvm_ffi::{
    Any, AnyMap, AnyView, Error, Function, ObjectArc, ObjectCore, Result, String, TYPE_ERROR,
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
) -> Result<AnyMap<String>> {
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
    let fields = AnyMap::<String>::try_from(
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
    fields: &AnyMap<String>,
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
    ObjectArc::new(value)
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
