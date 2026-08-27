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

//! Shared helpers used by generated TVM FFI bindings.

use tvm_ffi::{
    get_constructor_recipe, Any, AnyView, Error, Function, Map, ObjectCore, Result, String,
    CONSTRUCTOR_PREPARE_METHOD, TYPE_ERROR, VALUE_ERROR,
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
