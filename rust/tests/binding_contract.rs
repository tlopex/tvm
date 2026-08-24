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

use std::path::PathBuf;
use std::sync::OnceLock;

use tvm::ir::{
    AttrsObj, BaseFuncObj, CallObj, DictAttrsObj, ExprObj, GlobalVarObj, IRModuleObj, IntImmObj,
    PrimExprConvertibleObj, PrimTypeObj, RangeObj, SourceNameObj, SpanObj, TypeObj, VarObj,
};
use tvm::relax::{
    BindingBlockObj, BindingObj, IfObj as RelaxIfObj, RelaxFunctionObj, SeqExprObj, TupleObj,
    VarBindingObj,
};
use tvm::tirx::{
    AddObj, AssertStmtObj, BufferLoadObj, BufferRegionObj, BufferStoreObj, BufferTypeObj,
    EvaluateObj, ForObj, IfThenElseObj, IterVarObj, MatchBufferRegionObj, MulObj, PrimFuncObj,
    SBlockObj, SBlockRealizeObj, SeqStmtObj, StmtObj, StringImmObj, SubObj,
};
use tvm::tvm_ffi::tvm_ffi_sys::{
    TVMFFIFieldFlagBitMask, TVMFFIFieldInfo, TVMFFIGetTypeInfo, TVMFFISEqHashKind, TVMFFITypeInfo,
};
use tvm::tvm_ffi::{Module, Object, ObjectCore};

static TVM_COMPILER: OnceLock<Module> = OnceLock::new();

const DEFAULT: i64 = TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskHasDefault as i64;
const IGNORE: i64 = TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashIgnore as i64;
const DEF_RECURSIVE: i64 =
    TVMFFIFieldFlagBitMask::kTVMFFIFieldFlagBitMaskSEqHashDefRecursive as i64;

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

fn assert_contract<N: ObjectCore, P: ObjectCore>(
    expected_final: bool,
    expected_structural_kind: Option<TVMFFISEqHashKind>,
    expected_fields: &[(&str, i64)],
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

    let actual_fields = direct_fields::<N>()
        .iter()
        .map(|field| {
            assert!(
                field.getter.is_some(),
                "reflected field {}.{} has no getter",
                N::TYPE_KEY,
                field.name.as_str()
            );
            (field.name.as_str(), field.flags)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual_fields,
        expected_fields,
        "{} direct fields",
        N::TYPE_KEY
    );

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

#[test]
fn all_handwritten_objects_match_runtime_metadata() {
    load_tvm_compiler();
    use TVMFFISEqHashKind::{
        kTVMFFISEqHashKindDAGNode as Dag, kTVMFFISEqHashKindFreeVar as FreeVar,
        kTVMFFISEqHashKindTreeNode as Tree,
    };

    assert_contract::<ExprObj, Object>(
        false,
        Some(Tree),
        &[("span", DEFAULT | IGNORE), ("ty", DEFAULT)],
    );
    assert_contract::<BaseFuncObj, ExprObj>(false, Some(Tree), &[("attrs", 0)]);
    assert_contract::<GlobalVarObj, ExprObj>(true, Some(FreeVar), &[("name_hint", 0)]);
    assert_contract::<VarObj, ExprObj>(false, Some(FreeVar), &[("name", IGNORE)]);
    assert_contract::<SourceNameObj, Object>(true, Some(Tree), &[("name", 0)]);
    assert_contract::<SpanObj, Object>(
        false,
        Some(Tree),
        &[
            ("source_name", 0),
            ("line", 0),
            ("column", 0),
            ("end_line", 0),
            ("end_column", 0),
        ],
    );
    assert_contract::<PrimExprConvertibleObj, Object>(false, None, &[]);
    assert_contract::<RangeObj, Object>(
        true,
        Some(Tree),
        &[("min", 0), ("extent", 0), ("span", IGNORE)],
    );
    assert_contract::<CallObj, ExprObj>(
        true,
        Some(Tree),
        &[("op", 0), ("args", 0), ("attrs", 0), ("ty_args", 0)],
    );
    assert_contract::<TypeObj, Object>(false, Some(Tree), &[("span", DEFAULT | IGNORE)]);
    assert_contract::<PrimTypeObj, TypeObj>(true, Some(Tree), &[("dtype", 0)]);
    assert_contract::<IntImmObj, ExprObj>(true, Some(Tree), &[("value", 0)]);
    assert_contract::<AttrsObj, Object>(false, Some(Tree), &[]);
    assert_contract::<DictAttrsObj, AttrsObj>(true, Some(Tree), &[("__dict__", 0)]);
    assert_contract::<IRModuleObj, Object>(
        true,
        Some(Tree),
        &[
            ("functions", 0),
            ("global_var_map_", 0),
            ("source_map", 0),
            ("attrs", 0),
            ("global_infos", 0),
        ],
    );

    assert_contract::<AddObj, ExprObj>(true, Some(Tree), &[("a", 0), ("b", 0)]);
    assert_contract::<SubObj, ExprObj>(true, Some(Tree), &[("a", 0), ("b", 0)]);
    assert_contract::<MulObj, ExprObj>(true, Some(Tree), &[("a", 0), ("b", 0)]);
    assert_contract::<StringImmObj, ExprObj>(true, Some(Tree), &[("value", 0)]);
    assert_contract::<StmtObj, Object>(false, Some(Tree), &[("span", IGNORE)]);
    assert_contract::<AssertStmtObj, StmtObj>(
        true,
        Some(Tree),
        &[("condition", 0), ("error_kind", 0), ("message_parts", 0)],
    );
    assert_contract::<EvaluateObj, StmtObj>(true, Some(Tree), &[("value", 0)]);
    assert_contract::<SeqStmtObj, StmtObj>(true, Some(Tree), &[("seq", 0)]);
    assert_contract::<IfThenElseObj, StmtObj>(
        true,
        Some(Tree),
        &[("condition", 0), ("then_case", 0), ("else_case", 0)],
    );
    assert_contract::<ForObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("loop_var", DEF_RECURSIVE),
            ("min", 0),
            ("extent", 0),
            ("kind", 0),
            ("body", 0),
            ("thread_binding", 0),
            ("annotations", 0),
            ("step", 0),
        ],
    );
    assert_contract::<PrimFuncObj, BaseFuncObj>(
        true,
        Some(Tree),
        &[("params", DEF_RECURSIVE), ("ret_type", 0), ("body", 0)],
    );
    assert_contract::<BufferTypeObj, TypeObj>(
        true,
        Some(Tree),
        &[
            ("dtype", 0),
            ("storage_scope", 0),
            ("shape", DEF_RECURSIVE),
            ("strides", DEF_RECURSIVE),
            ("elem_offset", DEF_RECURSIVE),
            ("data_alignment", 0),
            ("offset_factor", 0),
            ("layout", 0),
            ("allocated_addr", 0),
        ],
    );
    assert_contract::<BufferLoadObj, ExprObj>(
        true,
        Some(Tree),
        &[("buffer", DEF_RECURSIVE), ("indices", 0), ("predicate", 0)],
    );
    assert_contract::<BufferStoreObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("buffer", DEF_RECURSIVE),
            ("value", 0),
            ("indices", 0),
            ("predicate", 0),
        ],
    );
    assert_contract::<BufferRegionObj, PrimExprConvertibleObj>(
        true,
        Some(Tree),
        &[("buffer", DEF_RECURSIVE), ("region", 0)],
    );
    assert_contract::<MatchBufferRegionObj, Object>(
        true,
        Some(Tree),
        &[("buffer", DEF_RECURSIVE), ("source", 0)],
    );
    assert_contract::<IterVarObj, PrimExprConvertibleObj>(
        true,
        Some(Tree),
        &[
            ("dom", 0),
            ("var", DEF_RECURSIVE),
            ("iter_type", 0),
            ("thread_tag", 0),
        ],
    );
    assert_contract::<SBlockObj, StmtObj>(
        true,
        Some(Tree),
        &[
            ("iter_vars", DEF_RECURSIVE),
            ("reads", 0),
            ("writes", 0),
            ("name_hint", IGNORE),
            ("alloc_buffers", DEF_RECURSIVE),
            ("match_buffers", 0),
            ("annotations", 0),
            ("init", 0),
            ("body", 0),
        ],
    );
    assert_contract::<SBlockRealizeObj, StmtObj>(
        true,
        Some(Tree),
        &[("iter_values", 0), ("predicate", 0), ("block", 0)],
    );

    assert_contract::<TupleObj, ExprObj>(true, Some(Tree), &[("fields", 0)]);
    assert_contract::<RelaxIfObj, ExprObj>(
        true,
        Some(Dag),
        &[("cond", 0), ("true_branch", 0), ("false_branch", 0)],
    );
    assert_contract::<BindingObj, Object>(
        false,
        Some(Tree),
        &[("span", IGNORE), ("var", DEF_RECURSIVE)],
    );
    assert_contract::<VarBindingObj, BindingObj>(true, Some(Tree), &[("value", 0)]);
    assert_contract::<BindingBlockObj, Object>(
        false,
        Some(Tree),
        &[("bindings", 0), ("span", DEFAULT | IGNORE)],
    );
    assert_contract::<SeqExprObj, ExprObj>(true, Some(Tree), &[("blocks", 0), ("body", 0)]);
    assert_contract::<RelaxFunctionObj, BaseFuncObj>(
        true,
        Some(Dag),
        &[
            ("params", DEF_RECURSIVE),
            ("body", 0),
            ("ret_ty", 0),
            ("is_pure", 0),
        ],
    );
}
