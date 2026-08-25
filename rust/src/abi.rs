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

use std::sync::atomic::{AtomicPtr, Ordering};

use tvm_ffi::tvm_ffi_sys::{
    TVMFFIAny, TVMFFIByteArray, TVMFFIGetTypeAttrColumn, TVMFFIGetTypeInfo, TVMFFITypeAttrColumn,
    TVMFFITypeIndex,
};
use tvm_ffi::{
    Any, AnyMap, AnyView, Error, ObjectArc, ObjectCore, Result, String, TYPE_ERROR, VALUE_ERROR,
};

type FConstructorPrepare = unsafe extern "C" fn(*const TVMFFIAny, i32) -> TVMFFIAny;

/// Rust mirror of `TVMIRConstructorVTable`.
#[repr(C)]
struct ConstructorVTable {
    num_args: i32,
    prepare: Option<FConstructorPrepare>,
}

static CONSTRUCTOR_VTABLE_COLUMN: AtomicPtr<TVMFFITypeAttrColumn> =
    AtomicPtr::new(std::ptr::null_mut());

/// Generated shape of one semantic constructor-preparation recipe.
///
/// The native table validates and derives values, while this Rust-side
/// contract makes the expected input arity and complete output-key set part of
/// the generated binding.  A version mismatch therefore fails explicitly
/// instead of silently ignoring a newly derived physical field.
pub(crate) trait ConstructorRecipe: ObjectCore {
    const NUM_INPUTS: usize;
    const DERIVED_FIELDS: &'static [&'static str];
}

/// Copy the raw cell represented by a borrowed `AnyView`.
///
/// The returned cell remains borrowed: callers must not drop it as an owning
/// `Any`, and every referenced value must remain alive through the ABI call.
pub(crate) fn borrowed_raw(value: AnyView<'_>) -> TVMFFIAny {
    const {
        assert!(std::mem::size_of::<AnyView<'static>>() == std::mem::size_of::<TVMFFIAny>());
        assert!(std::mem::align_of::<AnyView<'static>>() == std::mem::align_of::<TVMFFIAny>());
    }
    // SAFETY: AnyView is repr(C), starts with exactly one TVMFFIAny, and its
    // remaining PhantomData field has size zero. The assertions guard both
    // size and alignment.
    unsafe { *(&value as *const AnyView<'_>).cast::<TVMFFIAny>() }
}

/// Take ownership of one result returned by an Expected-style C ABI hook.
pub(crate) unsafe fn result_from_raw(raw: TVMFFIAny) -> Result<Any> {
    let value = Any::from_raw_ffi_any(raw);
    if value.type_index() != TVMFFITypeIndex::kTVMFFIError as i32 {
        return Ok(value);
    }
    match Error::try_from(value) {
        Ok(error) | Err(error) => Err(error),
    }
}

/// Fetch an immutable C ABI vtable stored as an opaque type attribute.
///
/// Registered attribute columns and vtables must remain valid for the process
/// lifetime, matching the structural visit/mutate hook protocol.
pub(crate) fn opaque_type_vtable<T>(
    cache: &'static AtomicPtr<TVMFFITypeAttrColumn>,
    attr_name: &'static str,
    type_index: i32,
) -> Result<&'static T> {
    let mut column = cache.load(Ordering::Acquire);
    if column.is_null() {
        let name = unsafe { TVMFFIByteArray::from_str(attr_name) };
        column = unsafe { TVMFFIGetTypeAttrColumn(&name).cast_mut() };
        if column.is_null() {
            return Err(Error::new(
                TYPE_ERROR,
                &format!("type attribute `{attr_name}` is not registered"),
                "",
            ));
        }
        cache.store(column, Ordering::Release);
    }

    let attr = unsafe {
        let column = &*column;
        let index = type_index - column.begin_index;
        if index < 0 || index >= column.size || column.data.is_null() {
            None
        } else {
            Some(*column.data.add(index as usize))
        }
    }
    .ok_or_else(|| {
        let type_key = runtime_type_key(type_index);
        Error::new(
            TYPE_ERROR,
            &format!("type `{type_key}` does not register `{attr_name}`"),
            "",
        )
    })?;

    if attr.type_index != TVMFFITypeIndex::kTVMFFIOpaquePtr as i32 {
        let type_key = runtime_type_key(type_index);
        return Err(Error::new(
            TYPE_ERROR,
            &format!("type `{type_key}` must register `{attr_name}` as an opaque C ABI vtable"),
            "",
        ));
    }
    let pointer = unsafe { attr.data_union.v_ptr.cast::<T>() };
    unsafe { pointer.as_ref() }.ok_or_else(|| {
        let type_key = runtime_type_key(type_index);
        Error::new(
            TYPE_ERROR,
            &format!("type `{type_key}` registers a null `{attr_name}` vtable"),
            "",
        )
    })
}

fn runtime_type_key(type_index: i32) -> std::string::String {
    unsafe {
        let info = TVMFFIGetTypeInfo(type_index);
        if info.is_null() {
            format!("<type_index {type_index}>")
        } else {
            (*info).type_key.as_str().to_owned()
        }
    }
}

/// Run a semantic constructor recipe through its direct C ABI table.
///
/// Every recipe returns the same representation: a map from physical field
/// names to derived values. The caller combines those values with constructor
/// inputs and defaults, then allocates the complete object in Rust.
pub(crate) fn prepare_constructor<N: ConstructorRecipe>(
    args: &[AnyView<'_>],
) -> Result<AnyMap<String>> {
    let vtable = opaque_type_vtable::<ConstructorVTable>(
        &CONSTRUCTOR_VTABLE_COLUMN,
        "__constructor_vtable__",
        N::type_index(),
    )?;
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
    let count = i32::try_from(N::NUM_INPUTS).map_err(|_| {
        Error::new(
            VALUE_ERROR,
            "constructor preparation argument count does not fit i32",
            "",
        )
    })?;
    if count != vtable.num_args {
        return Err(Error::new(
            TYPE_ERROR,
            &format!(
                "generated recipe for `{}` has {} arguments, but its native table declares {}",
                N::TYPE_KEY,
                N::NUM_INPUTS,
                vtable.num_args,
            ),
            "",
        ));
    }
    let callback = vtable.prepare.ok_or_else(|| {
        Error::new(
            TYPE_ERROR,
            &format!(
                "type `{}` has no constructor preparation entry",
                N::TYPE_KEY
            ),
            "",
        )
    })?;
    let raw_args = args.iter().copied().map(borrowed_raw).collect::<Vec<_>>();
    let raw = unsafe { callback(raw_args.as_ptr(), count) };
    let fields = AnyMap::<String>::try_from(unsafe { result_from_raw(raw) }?)?;
    if fields.len() != N::DERIVED_FIELDS.len() {
        return Err(Error::new(
            TYPE_ERROR,
            &format!(
                "generated recipe for `{}` expects {} derived fields, but its native table returned {}",
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
/// create a standalone object without a registered concrete behavior table.
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
