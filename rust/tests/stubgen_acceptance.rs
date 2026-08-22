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

//! Minimal acceptance tests for generated TVM Rust IR bindings.
//!
//! These tests intentionally use only the small handwritten binding slice that
//! stubgen should replace first.  Once those bindings are generated, deleting
//! their handwritten definitions must not require changing this file.

use std::path::PathBuf;
use std::sync::OnceLock;

use tvm::ir::{Expr, IntImm, Var, VarObj};
use tvm::tirx::{Add, AddObj, Evaluate, PrimFunc};
use tvm::tvm_ffi::{
    structural_map, structural_walk, Any, DefRegionKind, Module, ObjectRefCast, Result, WalkOrder,
    WalkResult,
};

static TVM_COMPILER: OnceLock<Module> = OnceLock::new();

fn load_tvm_compiler() {
    TVM_COMPILER.get_or_init(|| {
        let default = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("build")
            .join("lib")
            .join("libtvm_compiler.so");
        let library = std::env::var_os("TVM_COMPILER_LIBRARY")
            .map(PathBuf::from)
            .unwrap_or(default);
        Module::load_from_file(library.to_string_lossy()).unwrap()
    });
}

fn sample_function() -> PrimFunc {
    let variable = Var::new("x", "int32").unwrap();
    let zero = Expr::int("int32", 0).unwrap();
    let value = Expr::add(&variable.clone().into(), &zero).unwrap();
    let body = Evaluate::new(&value).unwrap();
    PrimFunc::new(vec![variable], &body).unwrap()
}

#[test]
fn generated_bindings_support_structural_walk() {
    load_tvm_compiler();
    let function = sample_function();
    let mut additions = 0;
    let mut integer_literals = Vec::new();
    let mut variable_regions = Vec::new();

    structural_walk(
        &function,
        (
            |_: &AddObj| {
                additions += 1;
                WalkResult::Advance
            },
            |value: &tvm::ir::IntImmObj| -> Result<WalkResult> {
                integer_literals.push(value.value()?);
                Ok(WalkResult::Advance)
            },
            |_: &VarObj, kind: DefRegionKind| {
                variable_regions.push(kind);
                WalkResult::Advance
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap();

    assert_eq!(additions, 1);
    assert_eq!(integer_literals, vec![0]);
    assert_eq!(
        variable_regions,
        vec![DefRegionKind::Recursive, DefRegionKind::None]
    );
}

#[test]
fn generated_bindings_support_structural_map() {
    load_tvm_compiler();
    let original = sample_function();

    let mapped = structural_map(
        original.clone(),
        |addition: Add| -> Result<Any> {
            let lhs = addition.lhs()?;
            let rhs = addition.rhs()?;
            let rhs_is_zero = rhs
                .clone()
                .try_cast::<IntImm>()
                .ok()
                .map(|value| value.value())
                .transpose()?
                == Some(0);
            Ok(Any::from(if rhs_is_zero { lhs } else { addition.into() }))
        },
        WalkOrder::PostOrder,
    )
    .and_then(PrimFunc::try_from)
    .unwrap();

    let mapped_value = mapped
        .body()
        .unwrap()
        .try_cast::<Evaluate>()
        .unwrap()
        .value()
        .unwrap();
    assert_eq!(
        mapped_value
            .try_cast::<Var>()
            .unwrap()
            .name()
            .unwrap()
            .as_str(),
        "x"
    );

    let original_value = original
        .body()
        .unwrap()
        .try_cast::<Evaluate>()
        .unwrap()
        .value()
        .unwrap();
    assert!(original_value.try_cast::<Add>().is_ok());
}
