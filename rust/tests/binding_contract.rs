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

//! Runtime conformance checks for the complete handwritten IR surface.

use tvm::abi::ObjectLayout;
use tvm::ir::{
    Attrs, AttrsObj, BaseFunc, BaseFuncObj, Call, CallObj, DictAttrs, DictAttrsObj,
    DummyGlobalInfo, DummyGlobalInfoObj, Expr, ExprObj, GlobalInfo, GlobalInfoObj, GlobalVar,
    GlobalVarObj, IRModule, IRModuleObj, IntImm, IntImmObj, PrimExprConvertibleObj, PrimType,
    PrimTypeObj, Range, RangeObj, Source, SourceMap, SourceMapObj, SourceName, SourceNameObj,
    SourceObj, Span, SpanObj, TupleType, TupleTypeObj, Type, TypeObj, Var, VarObj,
};
use tvm::relax::{
    Binding, BindingBlock, BindingBlockObj, BindingObj, If as RelaxIf, IfObj as RelaxIfObj,
    RelaxFunction, RelaxFunctionObj, SeqExpr, SeqExprObj, Tuple as RelaxTuple, TupleObj,
    VarBinding, VarBindingObj,
};
use tvm::tirx::{
    Add, AddObj, AssertStmt, AssertStmtObj, Axis, AxisObj, BufferLoad, BufferLoadObj, BufferRegion,
    BufferRegionObj, BufferStore, BufferStoreObj, BufferType, BufferTypeObj, DataProducerObj,
    Evaluate, EvaluateObj, For, ForKind, ForObj, IfThenElse, IfThenElseObj, Iter, IterObj, IterVar,
    IterVarObj, IterVarType, Layout, LayoutObj, MatchBufferRegion, MatchBufferRegionObj, Mul,
    MulObj, PrimFunc, PrimFuncObj, SBlock, SBlockObj, SBlockRealize, SBlockRealizeObj, SeqStmt,
    SeqStmtObj, Stmt, StmtObj, StringImm, StringImmObj, Sub, SubObj, TileLayout, TileLayoutObj,
};
use tvm::tvm_ffi::tvm_ffi_sys::{TVMFFIFieldFlagBitMask, TVMFFISEqHashKind};
use tvm::tvm_ffi::{AnyMap, Array, DLDataType, Map, Object, ObjectCore, String};

mod common;
use common::{direct_fields, load_tvm_compiler, runtime_type_info};

const DEFAULT: i64 = TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskHasDefault as i64;
const IGNORE: i64 = TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64;
const DEF_RECURSIVE: i64 =
    TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefRecursive as i64;

const SCHEMA_ANY_MAP: &str = r#"{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"Any"}]}"#;
const SCHEMA_ARRAY_BINDING: &str = r#"{"type":"ffi.Array","args":[{"type":"relax.expr.Binding"}]}"#;
const SCHEMA_ARRAY_BINDING_BLOCK: &str =
    r#"{"type":"ffi.Array","args":[{"type":"relax.expr.BindingBlock"}]}"#;
const SCHEMA_ARRAY_BUFFER_REGION: &str =
    r#"{"type":"ffi.Array","args":[{"type":"tirx.BufferRegion"}]}"#;
const SCHEMA_ARRAY_EXPR: &str = r#"{"type":"ffi.Array","args":[{"type":"ir.Expr"}]}"#;
const SCHEMA_ARRAY_GLOBAL_INFO_MAP: &str = r#"{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ffi.Array","args":[{"type":"ir.GlobalInfo"}]}]}"#;
const SCHEMA_ARRAY_INT: &str = r#"{"type":"ffi.Array","args":[{"type":"int"}]}"#;
const SCHEMA_ARRAY_ITER: &str = r#"{"type":"ffi.Array","args":[{"type":"tirx.Iter"}]}"#;
const SCHEMA_ARRAY_ITER_VAR: &str = r#"{"type":"ffi.Array","args":[{"type":"tirx.IterVar"}]}"#;
const SCHEMA_ARRAY_MATCH_BUFFER: &str =
    r#"{"type":"ffi.Array","args":[{"type":"tirx.MatchBufferRegion"}]}"#;
const SCHEMA_ARRAY_RANGE: &str = r#"{"type":"ffi.Array","args":[{"type":"ir.Range"}]}"#;
const SCHEMA_ARRAY_STMT: &str = r#"{"type":"ffi.Array","args":[{"type":"tirx.Stmt"}]}"#;
const SCHEMA_ARRAY_STRING_IMM: &str = r#"{"type":"ffi.Array","args":[{"type":"tirx.StringImm"}]}"#;
const SCHEMA_ARRAY_TYPE: &str = r#"{"type":"ffi.Array","args":[{"type":"ir.Type"}]}"#;
const SCHEMA_ARRAY_VAR: &str = r#"{"type":"ffi.Array","args":[{"type":"ir.Var"}]}"#;
const SCHEMA_ATTRS: &str = r#"{"type":"ir.Attrs"}"#;
const SCHEMA_AXIS: &str = r#"{"type":"tirx.Axis"}"#;
const SCHEMA_BOOL: &str = r#"{"type":"bool"}"#;
const SCHEMA_BUFFER_REGION: &str = r#"{"type":"tirx.BufferRegion"}"#;
const SCHEMA_DICT_ATTRS: &str = r#"{"type":"ir.DictAttrs"}"#;
const SCHEMA_DTYPE: &str = r#"{"type":"DataType"}"#;
const SCHEMA_EXPR: &str = r#"{"type":"ir.Expr"}"#;
const SCHEMA_INT: &str = r#"{"type":"int"}"#;
const SCHEMA_LAYOUT_OPTIONAL: &str = r#"{"type":"Optional","args":[{"type":"tirx.Layout"}]}"#;
const SCHEMA_MAP_AXIS_EXPR: &str =
    r#"{"type":"ffi.Map","args":[{"type":"tirx.Axis"},{"type":"ir.Expr"}]}"#;
const SCHEMA_MAP_FUNCTIONS: &str =
    r#"{"type":"ffi.Map","args":[{"type":"ir.GlobalVar"},{"type":"ir.BaseFunc"}]}"#;
const SCHEMA_MAP_GLOBAL_VAR: &str =
    r#"{"type":"ffi.Map","args":[{"type":"ffi.String"},{"type":"ir.GlobalVar"}]}"#;
const SCHEMA_MAP_SOURCE: &str =
    r#"{"type":"ffi.Map","args":[{"type":"ir.SourceName"},{"type":"ir.Source"}]}"#;
const SCHEMA_OPTIONAL_EXPR: &str = r#"{"type":"Optional","args":[{"type":"ir.Expr"}]}"#;
const SCHEMA_OPTIONAL_ITER_VAR: &str = r#"{"type":"Optional","args":[{"type":"tirx.IterVar"}]}"#;
const SCHEMA_OPTIONAL_STMT: &str = r#"{"type":"Optional","args":[{"type":"tirx.Stmt"}]}"#;
const SCHEMA_PRIM_TYPE: &str = r#"{"type":"ir.PrimType"}"#;
const SCHEMA_RANGE: &str = r#"{"type":"ir.Range"}"#;
const SCHEMA_SBLOCK: &str = r#"{"type":"tirx.SBlock"}"#;
const SCHEMA_SEQ_EXPR: &str = r#"{"type":"relax.expr.SeqExpr"}"#;
const SCHEMA_SOURCE_MAP: &str = r#"{"type":"ir.SourceMap"}"#;
const SCHEMA_SOURCE_NAME: &str = r#"{"type":"ir.SourceName"}"#;
const SCHEMA_SPAN: &str = r#"{"type":"ir.Span"}"#;
const SCHEMA_STMT: &str = r#"{"type":"tirx.Stmt"}"#;
const SCHEMA_STRING: &str = r#"{"type":"ffi.String"}"#;
const SCHEMA_STRING_IMM: &str = r#"{"type":"tirx.StringImm"}"#;
const SCHEMA_TYPE: &str = r#"{"type":"ir.Type"}"#;
const SCHEMA_VAR: &str = r#"{"type":"ir.Var"}"#;

fn assert_contract<N: ObjectCore, P: ObjectCore>(
    expected_final: bool,
    expected_structural_kind: Option<TVMFFISEqHashKind>,
    expected_fields: &[(&str, i64, &str)],
) {
    let info = runtime_type_info::<N>();
    assert_eq!(
        info.type_index,
        N::type_index(),
        "{} type index",
        N::TYPE_KEY
    );
    assert_eq!(info.type_key.as_str(), N::TYPE_KEY);
    assert_eq!(info.type_depth, N::TYPE_DEPTH, "{} depth", N::TYPE_KEY);
    assert_eq!(
        N::TYPE_DEPTH,
        P::TYPE_DEPTH + 1,
        "{} parent depth",
        N::TYPE_KEY
    );
    assert_eq!(N::TYPE_FINAL, expected_final, "{} finality", N::TYPE_KEY);

    assert!(!info.type_acenstors.is_null());
    let parent = unsafe { *info.type_acenstors.add(P::TYPE_DEPTH as usize) };
    assert!(!parent.is_null());
    assert_eq!(
        unsafe { (*parent).type_index },
        P::type_index(),
        "{} parent",
        N::TYPE_KEY
    );

    let actual_fields = direct_fields::<N>();
    assert_eq!(
        actual_fields.len(),
        expected_fields.len(),
        "{} field count",
        N::TYPE_KEY
    );
    for (field, (expected_name, expected_flags, expected_schema)) in
        actual_fields.iter().zip(expected_fields)
    {
        assert_eq!(
            field.name.as_str(),
            *expected_name,
            "{} field name",
            N::TYPE_KEY
        );
        assert_eq!(
            field.flags,
            *expected_flags,
            "{}.{} flags",
            N::TYPE_KEY,
            expected_name
        );
        if expected_flags & DEFAULT != 0 {
            // Expr.ty uses the exact `ir.Type` missing-type sentinel.  The
            // other defaults in this slice are nullable object metadata and
            // therefore use `None`.
            let expected_type_index = if N::TYPE_KEY == ExprObj::TYPE_KEY && *expected_name == "ty"
            {
                TypeObj::type_index()
            } else {
                tvm::tvm_ffi::tvm_ffi_sys::TVMFFITypeIndex::kTVMFFINone as i32
            };
            assert_eq!(
                field.default_value_or_factory.type_index,
                expected_type_index,
                "{}.{} default value",
                N::TYPE_KEY,
                expected_name
            );
            if expected_type_index == TypeObj::type_index() {
                assert!(
                    !unsafe { field.default_value_or_factory.data_union.v_obj }.is_null(),
                    "{}.{} missing-type default has a null object",
                    N::TYPE_KEY,
                    expected_name
                );
            }
        }
        assert!(
            field.getter.is_some(),
            "reflected field {}.{} has no getter",
            N::TYPE_KEY,
            expected_name
        );
        let expected_metadata = format!(
            "{{\"type_schema\":\"{}\"}}",
            expected_schema.replace('"', "\\\"")
        );
        assert_eq!(
            field.metadata.as_str(),
            expected_metadata,
            "{}.{} type schema",
            N::TYPE_KEY,
            expected_name
        );
    }

    let actual_structural_kind = if info.metadata.is_null() {
        None
    } else {
        Some(unsafe { (*info.metadata).structural_eq_hash_kind })
    };
    assert_eq!(
        actual_structural_kind,
        expected_structural_kind.map(|kind| kind as i32),
        "{} structural kind",
        N::TYPE_KEY
    );
}

fn assert_layout<N: ObjectCore + ObjectLayout, P>() {
    let info = runtime_type_info::<N>();
    assert!(
        !info.metadata.is_null(),
        "missing metadata for {}",
        N::TYPE_KEY
    );
    let cpp_size = unsafe { (*info.metadata).total_size };
    assert_ne!(cpp_size, 0, "missing C++ object size for {}", N::TYPE_KEY);
    assert_eq!(
        N::SIZE,
        cpp_size as usize,
        "{} total object size",
        N::TYPE_KEY
    );

    let cpp_fields = direct_fields::<N>();
    assert_eq!(
        N::FIELDS.len(),
        cpp_fields.len(),
        "{} physical/reflected field count",
        N::TYPE_KEY
    );
    for rust_field in N::FIELDS {
        let cpp_field = cpp_fields
            .iter()
            .find(|field| field.name.as_str() == rust_field.name)
            .unwrap_or_else(|| panic!("missing C++ field {}.{}", N::TYPE_KEY, rust_field.name));
        assert_eq!(
            rust_field.offset,
            cpp_field.offset as usize,
            "{}.{} offset",
            N::TYPE_KEY,
            rust_field.name
        );
        assert_eq!(
            rust_field.size,
            cpp_field.size as usize,
            "{}.{} size",
            N::TYPE_KEY,
            rust_field.name
        );
        assert_eq!(
            rust_field.alignment,
            cpp_field.alignment as usize,
            "{}.{} alignment",
            N::TYPE_KEY,
            rust_field.name
        );
    }

    let registered_alignment = cpp_fields
        .iter()
        .fold(std::mem::align_of::<P>(), |current, field| {
            current.max(field.alignment as usize)
        });
    assert_eq!(
        N::ALIGNMENT,
        registered_alignment,
        "{} alignment implied by its parent and registered fields",
        N::TYPE_KEY
    );
}

fn assert_no_reflection_creator<N: ObjectCore>() {
    let info = runtime_type_info::<N>();
    assert!(
        !info.metadata.is_null(),
        "missing metadata for {}",
        N::TYPE_KEY
    );
    assert!(
        unsafe { (*info.metadata).creator }.is_none(),
        "behavior-only base {} must not be reflection-constructible",
        N::TYPE_KEY
    );
}

#[test]
fn all_handwritten_objects_match_runtime_metadata() {
    load_tvm_compiler();
    use TVMFFISEqHashKind::{
        kTVMFFISEqHashKindDAGNode as Dag, kTVMFFISEqHashKindFreeVar as FreeVar,
        kTVMFFISEqHashKindTreeNode as Tree, kTVMFFISEqHashKindUnsupported as Unsupported,
    };

    assert_contract::<ExprObj, Object>(
        false,
        Some(Tree),
        &[
            ("span", DEFAULT | IGNORE, SCHEMA_SPAN),
            ("ty", DEFAULT, SCHEMA_TYPE),
        ],
    );
    assert_contract::<BaseFuncObj, ExprObj>(false, Some(Tree), &[("attrs", 0, SCHEMA_DICT_ATTRS)]);
    assert_contract::<GlobalVarObj, ExprObj>(
        true,
        Some(FreeVar),
        &[("name_hint", 0, SCHEMA_STRING)],
    );
    assert_contract::<VarObj, ExprObj>(false, Some(FreeVar), &[("name", IGNORE, SCHEMA_STRING)]);
    assert_contract::<SourceNameObj, Object>(true, Some(Tree), &[("name", 0, SCHEMA_STRING)]);
    assert_contract::<SourceObj, Object>(
        true,
        Some(Unsupported),
        &[
            ("source_name", 0, SCHEMA_SOURCE_NAME),
            ("source", 0, SCHEMA_STRING),
            ("line_map", 0, SCHEMA_ARRAY_INT),
        ],
    );
    assert_contract::<SourceMapObj, Object>(
        true,
        Some(Tree),
        &[("source_map", 0, SCHEMA_MAP_SOURCE)],
    );
    assert_contract::<SpanObj, Object>(
        false,
        Some(Tree),
        &[
            ("source_name", 0, SCHEMA_SOURCE_NAME),
            ("line", 0, SCHEMA_INT),
            ("column", 0, SCHEMA_INT),
            ("end_line", 0, SCHEMA_INT),
            ("end_column", 0, SCHEMA_INT),
        ],
    );
    assert_contract::<PrimExprConvertibleObj, Object>(false, Some(Unsupported), &[]);
    assert_contract::<DataProducerObj, PrimExprConvertibleObj>(false, Some(Unsupported), &[]);
    assert_contract::<RangeObj, Object>(
        true,
        Some(Tree),
        &[
            ("min", 0, SCHEMA_EXPR),
            ("extent", 0, SCHEMA_EXPR),
            ("span", IGNORE, SCHEMA_SPAN),
        ],
    );
    assert_contract::<CallObj, ExprObj>(
        true,
        Some(Tree),
        &[
            ("op", 0, SCHEMA_EXPR),
            ("args", 0, SCHEMA_ARRAY_EXPR),
            ("attrs", 0, SCHEMA_ATTRS),
            ("ty_args", 0, SCHEMA_ARRAY_TYPE),
        ],
    );
    assert_contract::<TypeObj, Object>(
        false,
        Some(Tree),
        &[("span", DEFAULT | IGNORE, SCHEMA_SPAN)],
    );
    assert_contract::<PrimTypeObj, TypeObj>(true, Some(Tree), &[("dtype", 0, SCHEMA_DTYPE)]);
    assert_contract::<TupleTypeObj, TypeObj>(true, Some(Tree), &[("fields", 0, SCHEMA_ARRAY_TYPE)]);
    assert_contract::<IntImmObj, ExprObj>(true, Some(Tree), &[("value", 0, SCHEMA_INT)]);
    assert_contract::<AttrsObj, Object>(false, Some(Tree), &[]);
    assert_contract::<DictAttrsObj, AttrsObj>(true, Some(Tree), &[("__dict__", 0, SCHEMA_ANY_MAP)]);
    assert_contract::<GlobalInfoObj, Object>(false, Some(Tree), &[]);
    assert_contract::<DummyGlobalInfoObj, GlobalInfoObj>(true, Some(Tree), &[]);
    assert_contract::<IRModuleObj, Object>(
        true,
        Some(Tree),
        &[
            ("functions", 0, SCHEMA_MAP_FUNCTIONS),
            ("global_var_map_", 0, SCHEMA_MAP_GLOBAL_VAR),
            ("source_map", 0, SCHEMA_SOURCE_MAP),
            ("attrs", 0, SCHEMA_DICT_ATTRS),
            ("global_infos", 0, SCHEMA_ARRAY_GLOBAL_INFO_MAP),
        ],
    );

    assert_contract::<AddObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<SubObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<MulObj, ExprObj>(
        true,
        Some(Tree),
        &[("a", 0, SCHEMA_EXPR), ("b", 0, SCHEMA_EXPR)],
    );
    assert_contract::<StringImmObj, ExprObj>(true, Some(Tree), &[("value", 0, SCHEMA_STRING)]);
    assert_contract::<StmtObj, Object>(false, Some(Tree), &[("span", IGNORE, SCHEMA_SPAN)]);
    assert_contract::<AssertStmtObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("condition", 0, SCHEMA_EXPR),
            ("error_kind", 0, SCHEMA_STRING_IMM),
            ("message_parts", 0, SCHEMA_ARRAY_STRING_IMM),
        ],
    );
    assert_contract::<EvaluateObj, StmtObj>(true, Some(Tree), &[("value", 0, SCHEMA_EXPR)]);
    assert_contract::<SeqStmtObj, StmtObj>(true, Some(Tree), &[("seq", 0, SCHEMA_ARRAY_STMT)]);
    assert_contract::<IfThenElseObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("condition", 0, SCHEMA_EXPR),
            ("then_case", 0, SCHEMA_STMT),
            ("else_case", 0, SCHEMA_OPTIONAL_STMT),
        ],
    );
    assert_contract::<ForObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("loop_var", DEF_RECURSIVE, SCHEMA_VAR),
            ("min", 0, SCHEMA_EXPR),
            ("extent", 0, SCHEMA_EXPR),
            ("kind", 0, SCHEMA_INT),
            ("body", 0, SCHEMA_STMT),
            ("thread_binding", 0, SCHEMA_OPTIONAL_ITER_VAR),
            ("annotations", 0, SCHEMA_ANY_MAP),
            ("step", 0, SCHEMA_OPTIONAL_EXPR),
        ],
    );
    assert_contract::<PrimFuncObj, BaseFuncObj>(
        true,
        Some(Tree),
        &[
            ("params", DEF_RECURSIVE, SCHEMA_ARRAY_VAR),
            ("ret_type", 0, SCHEMA_TYPE),
            ("body", 0, SCHEMA_STMT),
        ],
    );
    assert_contract::<LayoutObj, Object>(false, Some(Tree), &[]);
    assert_contract::<AxisObj, Object>(
        true,
        Some(Tree),
        &[
            ("name", 0, SCHEMA_STRING),
            ("registry_index", IGNORE, SCHEMA_INT),
        ],
    );
    assert_contract::<IterObj, Object>(
        true,
        Some(Tree),
        &[
            ("extent", 0, SCHEMA_EXPR),
            ("stride", 0, SCHEMA_EXPR),
            ("axis", 0, SCHEMA_AXIS),
        ],
    );
    assert_contract::<TileLayoutObj, LayoutObj>(
        true,
        Some(Tree),
        &[
            ("shard", 0, SCHEMA_ARRAY_ITER),
            ("replica", 0, SCHEMA_ARRAY_ITER),
            ("offset", 0, SCHEMA_MAP_AXIS_EXPR),
        ],
    );
    assert_contract::<BufferTypeObj, TypeObj>(
        true,
        Some(Tree),
        &[
            ("dtype", 0, SCHEMA_PRIM_TYPE),
            ("storage_scope", 0, SCHEMA_STRING),
            ("shape", DEF_RECURSIVE, SCHEMA_ARRAY_EXPR),
            ("strides", DEF_RECURSIVE, SCHEMA_ARRAY_EXPR),
            ("elem_offset", DEF_RECURSIVE, SCHEMA_EXPR),
            ("data_alignment", 0, SCHEMA_INT),
            ("offset_factor", 0, SCHEMA_INT),
            ("layout", 0, SCHEMA_LAYOUT_OPTIONAL),
            ("allocated_addr", 0, SCHEMA_ARRAY_EXPR),
        ],
    );
    assert_contract::<BufferLoadObj, ExprObj>(
        true,
        Some(Tree),
        &[
            ("buffer", DEF_RECURSIVE, SCHEMA_VAR),
            ("indices", 0, SCHEMA_ARRAY_EXPR),
            ("predicate", 0, SCHEMA_OPTIONAL_EXPR),
        ],
    );
    assert_contract::<BufferStoreObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("buffer", DEF_RECURSIVE, SCHEMA_VAR),
            ("value", 0, SCHEMA_EXPR),
            ("indices", 0, SCHEMA_ARRAY_EXPR),
            ("predicate", 0, SCHEMA_OPTIONAL_EXPR),
        ],
    );
    assert_contract::<BufferRegionObj, PrimExprConvertibleObj>(
        true,
        Some(Tree),
        &[
            ("buffer", DEF_RECURSIVE, SCHEMA_VAR),
            ("region", 0, SCHEMA_ARRAY_RANGE),
        ],
    );
    assert_contract::<MatchBufferRegionObj, Object>(
        true,
        Some(Tree),
        &[
            ("buffer", DEF_RECURSIVE, SCHEMA_VAR),
            ("source", 0, SCHEMA_BUFFER_REGION),
        ],
    );
    assert_contract::<IterVarObj, PrimExprConvertibleObj>(
        true,
        Some(Tree),
        &[
            ("dom", 0, SCHEMA_RANGE),
            ("var", DEF_RECURSIVE, SCHEMA_VAR),
            ("iter_type", 0, SCHEMA_INT),
            ("thread_tag", 0, SCHEMA_STRING),
            ("span", DEFAULT | IGNORE, SCHEMA_SPAN),
        ],
    );
    assert_contract::<SBlockObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("iter_vars", DEF_RECURSIVE, SCHEMA_ARRAY_ITER_VAR),
            ("reads", 0, SCHEMA_ARRAY_BUFFER_REGION),
            ("writes", 0, SCHEMA_ARRAY_BUFFER_REGION),
            ("name_hint", IGNORE, SCHEMA_STRING),
            ("alloc_buffers", DEF_RECURSIVE, SCHEMA_ARRAY_VAR),
            ("match_buffers", 0, SCHEMA_ARRAY_MATCH_BUFFER),
            ("annotations", 0, SCHEMA_ANY_MAP),
            ("init", 0, SCHEMA_OPTIONAL_STMT),
            ("body", 0, SCHEMA_STMT),
        ],
    );
    assert_contract::<SBlockRealizeObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("iter_values", 0, SCHEMA_ARRAY_EXPR),
            ("predicate", 0, SCHEMA_EXPR),
            ("block", 0, SCHEMA_SBLOCK),
        ],
    );

    assert_contract::<TupleObj, ExprObj>(true, Some(Tree), &[("fields", 0, SCHEMA_ARRAY_EXPR)]);
    assert_contract::<RelaxIfObj, ExprObj>(
        true,
        Some(Dag),
        &[
            ("cond", 0, SCHEMA_EXPR),
            ("true_branch", 0, SCHEMA_SEQ_EXPR),
            ("false_branch", 0, SCHEMA_SEQ_EXPR),
        ],
    );
    assert_contract::<BindingObj, Object>(
        false,
        Some(Tree),
        &[
            ("span", IGNORE, SCHEMA_SPAN),
            ("var", DEF_RECURSIVE, SCHEMA_VAR),
        ],
    );
    assert_contract::<VarBindingObj, BindingObj>(true, Some(Tree), &[("value", 0, SCHEMA_EXPR)]);
    assert_contract::<BindingBlockObj, Object>(
        false,
        Some(Tree),
        &[
            ("bindings", 0, SCHEMA_ARRAY_BINDING),
            ("span", DEFAULT | IGNORE, SCHEMA_SPAN),
        ],
    );
    assert_contract::<SeqExprObj, ExprObj>(
        true,
        Some(Tree),
        &[
            ("blocks", 0, SCHEMA_ARRAY_BINDING_BLOCK),
            ("body", 0, SCHEMA_EXPR),
        ],
    );
    assert_contract::<RelaxFunctionObj, BaseFuncObj>(
        true,
        Some(Dag),
        &[
            ("params", DEF_RECURSIVE, SCHEMA_ARRAY_VAR),
            ("body", 0, SCHEMA_SEQ_EXPR),
            ("ret_ty", 0, SCHEMA_TYPE),
            ("is_pure", 0, SCHEMA_BOOL),
        ],
    );
}

#[test]
fn abi_complete_object_layout_evidence_matches_cpp() {
    load_tvm_compiler();

    assert_no_reflection_creator::<PrimExprConvertibleObj>();
    assert_no_reflection_creator::<DataProducerObj>();
    assert_no_reflection_creator::<LayoutObj>();

    assert_layout::<ExprObj, Object>();
    assert_layout::<BaseFuncObj, ExprObj>();
    assert_layout::<GlobalVarObj, ExprObj>();
    assert_layout::<VarObj, ExprObj>();
    assert_layout::<SourceNameObj, Object>();
    assert_layout::<SourceObj, Object>();
    assert_layout::<SourceMapObj, Object>();
    assert_layout::<SpanObj, Object>();
    assert_layout::<RangeObj, Object>();
    assert_layout::<PrimExprConvertibleObj, Object>();
    assert_layout::<DataProducerObj, PrimExprConvertibleObj>();
    assert_layout::<CallObj, ExprObj>();
    assert_layout::<TypeObj, Object>();
    assert_layout::<PrimTypeObj, TypeObj>();
    assert_layout::<TupleTypeObj, TypeObj>();
    assert_layout::<IntImmObj, ExprObj>();
    assert_layout::<AttrsObj, Object>();
    assert_layout::<DictAttrsObj, AttrsObj>();
    assert_layout::<GlobalInfoObj, Object>();
    assert_layout::<DummyGlobalInfoObj, GlobalInfoObj>();
    assert_layout::<IRModuleObj, Object>();

    assert_layout::<AddObj, ExprObj>();
    assert_layout::<SubObj, ExprObj>();
    assert_layout::<MulObj, ExprObj>();
    assert_layout::<StringImmObj, ExprObj>();
    assert_layout::<StmtObj, Object>();
    assert_layout::<AssertStmtObj, StmtObj>();
    assert_layout::<EvaluateObj, StmtObj>();
    assert_layout::<SeqStmtObj, StmtObj>();
    assert_layout::<IfThenElseObj, StmtObj>();
    assert_layout::<ForObj, StmtObj>();
    assert_layout::<PrimFuncObj, BaseFuncObj>();
    assert_layout::<AxisObj, Object>();
    assert_layout::<IterObj, Object>();
    assert_layout::<LayoutObj, Object>();
    assert_layout::<TileLayoutObj, LayoutObj>();
    assert_layout::<BufferTypeObj, TypeObj>();
    assert_layout::<BufferLoadObj, ExprObj>();
    assert_layout::<BufferStoreObj, StmtObj>();
    assert_layout::<BufferRegionObj, PrimExprConvertibleObj>();
    assert_layout::<MatchBufferRegionObj, Object>();
    assert_layout::<IterVarObj, PrimExprConvertibleObj>();
    assert_layout::<SBlockObj, StmtObj>();
    assert_layout::<SBlockRealizeObj, StmtObj>();

    assert_layout::<TupleObj, ExprObj>();
    assert_layout::<RelaxIfObj, ExprObj>();
    assert_layout::<BindingObj, Object>();
    assert_layout::<VarBindingObj, BindingObj>();
    assert_layout::<BindingBlockObj, Object>();
    assert_layout::<SeqExprObj, ExprObj>();
    assert_layout::<RelaxFunctionObj, BaseFuncObj>();
}

macro_rules! assert_complete_allocator {
    ($constructor:path : fn($($argument:ty),* $(,)?) -> $output:ty) => {
        let _: fn($($argument),*) -> $output = $constructor;
    };
}

#[test]
fn complete_field_allocators_follow_owned_native_field_order() {
    assert_eq!(std::mem::size_of::<ForKind>(), std::mem::size_of::<i32>());
    assert_eq!(
        std::mem::size_of::<IterVarType>(),
        std::mem::size_of::<i32>()
    );
    assert_eq!(ForKind::from_raw(99).as_raw(), 99);
    assert_eq!(IterVarType::from_raw(99).as_raw(), 99);
    assert_eq!(ForKind::try_from(99_i64).unwrap().as_raw(), 99);
    assert_eq!(IterVarType::try_from(99_i64).unwrap().as_raw(), 99);
    assert!(ForKind::try_from(i64::from(i32::MAX) + 1).is_err());
    assert!(IterVarType::try_from(i64::from(i32::MIN) - 1).is_err());

    assert_complete_allocator!(SourceName::from_complete_fields: fn(String) -> SourceName);
    assert_complete_allocator!(Source::from_complete_fields: fn(SourceName, String, Array<i64>) -> Source);
    assert_complete_allocator!(SourceMap::from_complete_fields: fn(Map<SourceName, Source>) -> SourceMap);
    assert_complete_allocator!(Span::from_complete_fields: fn(SourceName, i32, i32, i32, i32) -> Span);
    assert_complete_allocator!(Range::from_complete_fields: fn(Expr, Expr, Option<Span>) -> Range);
    assert_complete_allocator!(Type::from_complete_fields: fn(Option<Span>) -> Type);
    assert_complete_allocator!(TupleType::from_complete_fields: fn(Option<Span>, Array<Type>) -> TupleType);
    assert_complete_allocator!(DummyGlobalInfo::from_complete_fields: fn() -> DummyGlobalInfo);
    assert_complete_allocator!(IntImm::from_complete_fields: fn(Option<Span>, Type, i64) -> IntImm);
    assert_complete_allocator!(PrimType::from_complete_fields: fn(Option<Span>, DLDataType) -> PrimType);
    assert_complete_allocator!(Var::from_complete_fields: fn(Option<Span>, Type, String) -> Var);
    assert_complete_allocator!(GlobalVar::from_complete_fields: fn(Option<Span>, Type, String) -> GlobalVar);
    assert_complete_allocator!(Call::from_complete_fields: fn(Option<Span>, Type, Expr, Array<Expr>, Option<Attrs>, Array<Type>) -> Call);
    assert_complete_allocator!(IRModule::from_complete_fields: fn(Map<GlobalVar, BaseFunc>, SourceMap, DictAttrs, Map<String, Array<GlobalInfo>>, Map<String, GlobalVar>) -> IRModule);
    assert_complete_allocator!(DictAttrs::from_complete_fields: fn(AnyMap<String>) -> DictAttrs);

    assert_complete_allocator!(Add::from_complete_fields: fn(Option<Span>, Type, Expr, Expr) -> Add);
    assert_complete_allocator!(Sub::from_complete_fields: fn(Option<Span>, Type, Expr, Expr) -> Sub);
    assert_complete_allocator!(Mul::from_complete_fields: fn(Option<Span>, Type, Expr, Expr) -> Mul);
    assert_complete_allocator!(StringImm::from_complete_fields: fn(Option<Span>, Type, String) -> StringImm);
    assert_complete_allocator!(AssertStmt::from_complete_fields: fn(Option<Span>, Expr, StringImm, Array<StringImm>) -> AssertStmt);
    assert_complete_allocator!(Evaluate::from_complete_fields: fn(Option<Span>, Expr) -> Evaluate);
    assert_complete_allocator!(SeqStmt::from_complete_fields: fn(Option<Span>, Array<Stmt>) -> SeqStmt);
    assert_complete_allocator!(IfThenElse::from_complete_fields: fn(Option<Span>, Expr, Stmt, Option<Stmt>) -> IfThenElse);
    assert_complete_allocator!(For::from_complete_fields: fn(Option<Span>, Var, Expr, Expr, ForKind, Stmt, Option<IterVar>, AnyMap<String>, Option<Expr>) -> For);
    assert_complete_allocator!(PrimFunc::from_complete_fields: fn(Option<Span>, Type, DictAttrs, Array<Var>, Type, Stmt) -> PrimFunc);

    assert_complete_allocator!(Iter::from_complete_fields: fn(Expr, Expr, Axis) -> Iter);
    assert_complete_allocator!(TileLayout::from_complete_fields: fn(Array<Iter>, Array<Iter>, Map<Axis, Expr>) -> TileLayout);
    assert_complete_allocator!(BufferType::from_complete_fields: fn(Option<Span>, PrimType, String, Array<Expr>, Array<Expr>, Expr, i32, i32, Option<Layout>, Array<Expr>) -> BufferType);
    assert_complete_allocator!(BufferLoad::from_complete_fields: fn(Option<Span>, Type, Var, Array<Expr>, Option<Expr>) -> BufferLoad);
    assert_complete_allocator!(BufferStore::from_complete_fields: fn(Option<Span>, Var, Expr, Array<Expr>, Option<Expr>) -> BufferStore);
    assert_complete_allocator!(BufferRegion::from_complete_fields: fn(Var, Array<Range>) -> BufferRegion);
    assert_complete_allocator!(MatchBufferRegion::from_complete_fields: fn(Var, BufferRegion) -> MatchBufferRegion);
    assert_complete_allocator!(IterVar::from_complete_fields: fn(Option<Range>, Var, IterVarType, String, Option<Span>) -> IterVar);
    assert_complete_allocator!(SBlock::from_complete_fields: fn(Option<Span>, Array<IterVar>, Array<BufferRegion>, Array<BufferRegion>, String, Array<Var>, Array<MatchBufferRegion>, AnyMap<String>, Option<Stmt>, Stmt) -> SBlock);
    assert_complete_allocator!(SBlockRealize::from_complete_fields: fn(Option<Span>, Array<Expr>, Expr, SBlock) -> SBlockRealize);

    assert_complete_allocator!(RelaxTuple::from_complete_fields: fn(Option<Span>, Type, Array<Expr>) -> RelaxTuple);
    assert_complete_allocator!(RelaxIf::from_complete_fields: fn(Option<Span>, Type, Expr, SeqExpr, SeqExpr) -> RelaxIf);
    assert_complete_allocator!(VarBinding::from_complete_fields: fn(Option<Span>, Var, Expr) -> VarBinding);
    assert_complete_allocator!(BindingBlock::from_complete_fields: fn(Array<Binding>, Option<Span>) -> BindingBlock);
    assert_complete_allocator!(SeqExpr::from_complete_fields: fn(Option<Span>, Type, Array<BindingBlock>, Expr) -> SeqExpr);
    assert_complete_allocator!(RelaxFunction::from_complete_fields: fn(Option<Span>, Type, DictAttrs, Array<Var>, SeqExpr, Type, bool) -> RelaxFunction);
}
