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

use tvm::ir::{BaseFuncObj, Expr, ExprObj, IntImm, IntImmObj, PrimType, Var, VarObj};
use tvm::tirx::{Add, AddObj, Evaluate, EvaluateObj, PrimFunc, PrimFuncObj, StmtObj};
use tvm::tvm_ffi::tvm_ffi_sys::{
    TVMFFIFieldFlagBitMask, TVMFFIFieldInfo, TVMFFIGetTypeInfo, TVMFFITypeInfo,
};
use tvm::tvm_ffi::{
    structural_map, structural_walk, Any, AnyCompatible, AnyView, DefRegionKind, Function, Module,
    Object, ObjectArc, ObjectCore, ObjectRefCast, ObjectRefCore, Result, WalkOrder, WalkResult,
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
    let variable_expr = Expr::from(variable.clone());
    let zero = Expr::from(IntImm::new("int32", 0).unwrap());
    let value = Expr::from(Add::new(&variable_expr, &zero).unwrap());
    let body = Evaluate::new(&value).unwrap();
    PrimFunc::new(vec![variable], &body).unwrap()
}

fn runtime_type_info<N: ObjectCore>() -> &'static TVMFFITypeInfo {
    let pointer = unsafe { TVMFFIGetTypeInfo(N::type_index()) };
    assert!(!pointer.is_null(), "missing type info for {}", N::TYPE_KEY);
    unsafe { &*pointer }
}

fn direct_fields<N: ObjectCore>() -> &'static [TVMFFIFieldInfo] {
    let info = runtime_type_info::<N>();
    if info.num_fields == 0 {
        return &[];
    }
    assert!(!info.fields.is_null(), "missing fields for {}", N::TYPE_KEY);
    unsafe { std::slice::from_raw_parts(info.fields, info.num_fields as usize) }
}

fn direct_field<N: ObjectCore>(name: &str) -> &'static TVMFFIFieldInfo {
    direct_fields::<N>()
        .iter()
        .find(|field| field.name.as_str() == name)
        .unwrap_or_else(|| panic!("missing reflected field {}.{name}", N::TYPE_KEY))
}

fn assert_type_contract<N: ObjectCore, P: ObjectCore>(
    expected_final: bool,
    expected_fields: &[&str],
) {
    let info = runtime_type_info::<N>();
    assert_eq!(info.type_index, N::type_index());
    assert_eq!(info.type_key.as_str(), N::TYPE_KEY);
    assert_eq!(info.type_depth, N::TYPE_DEPTH);
    assert_eq!(N::TYPE_DEPTH, P::TYPE_DEPTH + 1);
    assert_eq!(N::TYPE_FINAL, expected_final);

    assert!(!info.type_acenstors.is_null());
    let parent = unsafe { *info.type_acenstors.add(P::TYPE_DEPTH as usize) };
    assert!(!parent.is_null());
    assert_eq!(unsafe { (*parent).type_index }, P::type_index());

    let fields = direct_fields::<N>();
    assert_eq!(
        fields
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>(),
        expected_fields
    );
    for field in fields {
        assert!(
            field.getter.is_some(),
            "reflected field {}.{} has no getter",
            N::TYPE_KEY,
            field.name.as_str()
        );
    }
}

fn assert_field_flag<N: ObjectCore>(name: &str, flag: TVMFFIFieldFlagBitMask) {
    assert_ne!(direct_field::<N>(name).flags & flag as i64, 0);
}

fn object_pointer<O: ObjectRefCore>(value: &O) -> *const () {
    unsafe { ObjectArc::as_raw(O::data(value)).cast() }
}

fn cpp_reflected_field<O: ObjectRefCore>(value: &O, name: &str) -> Any {
    let field = direct_field::<O::ContainerType>(name);
    let getter = field
        .getter
        .expect("reflected field must have a C ABI getter");
    let object = object_pointer(value).cast::<u8>();
    let address = unsafe { object.add(field.offset as usize).cast_mut().cast() };
    let mut result = Any::new();
    assert_eq!(unsafe { getter(address, Any::as_data_ptr(&mut result)) }, 0);
    result
}

fn assert_cpp_structural_equal<L: AnyCompatible, R: AnyCompatible>(lhs: &L, rhs: &R) {
    let equal = Function::get_global("ffi.StructuralEqual")
        .unwrap()
        .call_packed(&[
            AnyView::from(lhs),
            AnyView::from(rhs),
            AnyView::from(&false),
            AnyView::from(&false),
        ])
        .unwrap();
    assert!(bool::try_from(equal).unwrap());
}

#[test]
fn minimal_bindings_match_the_runtime_contract() {
    load_tvm_compiler();

    assert_type_contract::<ExprObj, Object>(false, &["span", "ty"]);
    assert_type_contract::<VarObj, ExprObj>(false, &["name"]);
    assert_type_contract::<IntImmObj, ExprObj>(true, &["value"]);
    assert_type_contract::<AddObj, ExprObj>(true, &["a", "b"]);
    assert_type_contract::<StmtObj, Object>(false, &["span"]);
    assert_type_contract::<EvaluateObj, StmtObj>(true, &["value"]);
    assert_type_contract::<BaseFuncObj, ExprObj>(false, &["attrs"]);
    assert_type_contract::<PrimFuncObj, BaseFuncObj>(true, &["params", "ret_type", "body"]);

    assert_field_flag::<ExprObj>(
        "span",
        TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashIgnore,
    );
    assert_field_flag::<VarObj>(
        "name",
        TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashIgnore,
    );
    assert_field_flag::<StmtObj>(
        "span",
        TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashIgnore,
    );
    assert_field_flag::<PrimFuncObj>(
        "params",
        TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefRecursive,
    );
}

#[test]
fn direct_and_semantic_constructors_round_trip() {
    load_tvm_compiler();
    // These constructors intentionally remain registered-function overrides:
    // PrimType is interned, TypeMissing is a singleton, and PrimFunc derives
    // additional semantic state.
    for name in ["ir.PrimType", "ir.TypeMissing", "tirx.PrimFunc"] {
        Function::get_global(name)
            .unwrap_or_else(|error| panic!("missing semantic constructor `{name}`: {error}"));
    }

    let function = sample_function();
    let parameter = function.params().unwrap().get(0).unwrap();
    assert_eq!(parameter.name().unwrap().as_str(), "x");
    assert!(parameter.span().unwrap().is_none());
    assert_eq!(
        parameter
            .ty()
            .unwrap()
            .try_cast::<PrimType>()
            .unwrap()
            .dtype()
            .unwrap()
            .bits,
        32
    );

    let body = function.body().unwrap().try_cast::<Evaluate>().unwrap();
    assert!(body.span().unwrap().is_none());
    let addition = body.value().unwrap().try_cast::<Add>().unwrap();
    let lhs = addition.lhs().unwrap().try_cast::<Var>().unwrap();
    let rhs = addition.rhs().unwrap().try_cast::<IntImm>().unwrap();
    assert_eq!(object_pointer(&lhs), object_pointer(&parameter));
    assert_eq!(rhs.value().unwrap(), 0);
    assert!(rhs.span().unwrap().is_none());

    function.attrs().unwrap();
    function.ret_type().unwrap();
    function.ty().unwrap();
    assert!(function.span().unwrap().is_none());
    assert!(IntImm::new("int8", 128).is_err());
}

#[test]
fn rust_allocated_nodes_are_consumed_by_cpp_abi() {
    load_tvm_compiler();

    let lhs = Expr::from(IntImm::new("int32", 7).unwrap());
    let rhs = Expr::from(IntImm::new("int32", 9).unwrap());
    let rust_add = Add::new(&lhs, &rhs).unwrap();

    // Invoke the field getter registered by C++ on a Rust allocation.  A wrong
    // Rust field offset or representation would return the wrong object here.
    let reflected_lhs = Expr::try_from(cpp_reflected_field(&rust_add, "a")).unwrap();
    assert_eq!(object_pointer(&reflected_lhs), object_pointer(&lhs));

    // Build the equivalent node through C++ and compare through C++'s
    // structural-equality implementation.
    let none = ();
    let cpp_lhs = Function::get_global("ir.IntImm")
        .unwrap()
        .call_packed(&[
            AnyView::from(&PrimType::new("int32").unwrap().dtype().unwrap()),
            AnyView::from(&7_i64),
            AnyView::from(&none),
        ])
        .unwrap();
    let cpp_rhs = Function::get_global("ir.IntImm")
        .unwrap()
        .call_packed(&[
            AnyView::from(&PrimType::new("int32").unwrap().dtype().unwrap()),
            AnyView::from(&9_i64),
            AnyView::from(&none),
        ])
        .unwrap();
    let cpp_add: Add = Function::get_global("tirx.Add")
        .unwrap()
        .call_packed(&[
            AnyView::from(&cpp_lhs),
            AnyView::from(&cpp_rhs),
            AnyView::from(&none),
        ])
        .unwrap()
        .try_into()
        .unwrap();
    assert_cpp_structural_equal(&rust_add, &cpp_add);
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
    let mapped_variable = mapped_value.try_cast::<Var>().unwrap();
    assert_eq!(mapped_variable.name().unwrap().as_str(), "x");
    assert_eq!(
        object_pointer(&mapped_variable),
        object_pointer(&mapped.params().unwrap().get(0).unwrap())
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
