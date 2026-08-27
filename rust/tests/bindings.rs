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

use tvm::ir::{
    Call, DictAttrs, Expr, GlobalVar, IRModule, IntImm, PrimExprConvertible, PrimType, Range,
    SourceMap, SourceName, Span, TupleType, Type, Var,
};
use tvm::relax::{
    Binding, BindingBlock, If as RelaxIf, RelaxFunction, SeqExpr, Tuple as RelaxTuple, VarBinding,
};
use tvm::tirx::{
    Add, AssertStmt, Axis, BufferLoad, BufferRegion, BufferStore, BufferType, Evaluate, For,
    ForKind, IfThenElse, Iter, IterVar, IterVarType, Layout, MatchBufferRegion, PrimFunc, SBlock,
    SBlockRealize, SeqStmt, Stmt, StringImm, TileLayout,
};
use tvm::tvm_ffi::{Any, Array, Function, Map, ObjectRefCast, ObjectRefCore, String as FfiString};

mod common;
use common::{assert_structural_equal, load_tvm_compiler, object_pointer};

fn int(dtype: &str, value: i64) -> Expr {
    IntImm::new(dtype, value).unwrap().into()
}

#[test]
fn core_ir_uses_native_constructors_and_reflected_accessors() {
    load_tvm_compiler();

    let source_map = SourceMap::new().unwrap();
    let source_name = source_map.add("example.tvm", "first\nsecond\n").unwrap();
    assert_eq!(source_name.name().unwrap().as_str(), "example.tvm");
    let source = source_map
        .source_map()
        .unwrap()
        .get(&source_name)
        .unwrap()
        .unwrap();
    assert_eq!(source.source().unwrap().as_str(), "first\nsecond\n");

    let span = Span::new(source_name.clone(), 2, 2, 1, 6).unwrap();
    assert!(span.source_name().unwrap().same_as(&source_name));
    assert_eq!(span.line().unwrap(), 2);
    assert_eq!(span.end_line().unwrap(), 2);
    assert_eq!(span.column().unwrap(), 1);
    assert_eq!(span.end_column().unwrap(), 6);

    let int_type = PrimType::new("int32").unwrap();
    assert_eq!(int_type.dtype().unwrap().bits, 32);
    let literal = IntImm::from_dtype_with_span(int_type.dtype().unwrap(), 7, Some(&span)).unwrap();
    assert_eq!(literal.value().unwrap(), 7);
    assert!(literal.span().unwrap().unwrap().same_as(&span));
    assert!(
        literal
            .ty()
            .unwrap()
            .try_cast::<PrimType>()
            .unwrap()
            .dtype()
            .unwrap()
            == int_type.dtype().unwrap()
    );

    let variable = Var::with_type_and_span("x", int_type.clone(), Some(&span)).unwrap();
    assert_eq!(variable.name().unwrap().as_str(), "x");
    assert!(variable.span().unwrap().unwrap().same_as(&span));
    let global = GlobalVar::new("callee").unwrap();
    let call = Call::new(
        int_type.clone(),
        global.clone(),
        vec![literal.clone().into()],
    )
    .unwrap();
    assert!(call.op().unwrap().same_as(&Expr::from(global)));
    assert_eq!(call.args().unwrap().len(), 1);
    assert!(call.attrs().unwrap().is_none());
    assert!(call.ty_args().unwrap().is_empty());

    let tuple_type = TupleType::new(vec![int_type.clone().into()]).unwrap();
    assert_eq!(tuple_type.fields().unwrap().len(), 1);
    assert!(Type::missing().unwrap().is_missing());

    let attrs = DictAttrs::from_dictionary(
        [(FfiString::from("answer"), Any::from(42i64))]
            .into_iter()
            .collect(),
    )
    .unwrap();
    assert_eq!(
        i64::try_from(
            attrs
                .dict()
                .unwrap()
                .get(&FfiString::from("answer"))
                .unwrap()
                .unwrap()
        )
        .unwrap(),
        42
    );

    let body = Evaluate::new(call).unwrap();
    let function =
        PrimFunc::with_metadata(vec![variable], body, int_type, attrs, Some(&span)).unwrap();
    let module = IRModule::from_expr(function.clone()).unwrap();
    assert_eq!(module.functions().unwrap().len(), 1);
    assert_eq!(module.global_var_map().unwrap().len(), 1);
    assert_eq!(module.source_map().unwrap().source_map().unwrap().len(), 0);
    assert!(module
        .functions()
        .unwrap()
        .iter()
        .next()
        .unwrap()
        .1
        .try_cast::<PrimFunc>()
        .unwrap()
        .body()
        .is_ok());
}

#[test]
fn tir_nodes_round_trip_through_reflection() {
    load_tvm_compiler();

    let x = Var::new("x", "int32").unwrap();
    let add = Add::with_span(x.clone(), int("int32", 1), None).unwrap();
    assert!(add.a().unwrap().same_as(&Expr::from(x.clone())));
    assert_eq!(
        add.b()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        1
    );

    let text = StringImm::new("failure").unwrap();
    assert_eq!(text.value().unwrap().as_str(), "failure");
    let assertion = AssertStmt::new(int("bool", 1), "ValueError", "failure").unwrap();
    assert_eq!(
        assertion.error_kind().unwrap().value().unwrap().as_str(),
        "ValueError"
    );
    assert_eq!(assertion.message_parts().unwrap().len(), 1);

    let conditional = IfThenElse::new(
        int("bool", 1),
        Evaluate::new(add.clone()).unwrap(),
        Some(assertion.clone().into()),
    )
    .unwrap();
    assert!(conditional.else_case().unwrap().is_some());
    let sequence = SeqStmt::new(vec![conditional.clone().into(), assertion.into()]).unwrap();
    assert_eq!(sequence.seq().unwrap().len(), 2);

    let loop_var = Var::new("i", "int32").unwrap();
    let loop_node =
        For::serial(loop_var.clone(), int("int32", 0), int("int32", 4), sequence).unwrap();
    assert_eq!(loop_node.kind().unwrap(), ForKind::Serial);
    assert!(loop_node.loop_var().unwrap().same_as(&loop_var));
    assert_eq!(
        loop_node
            .extent()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        4
    );
    assert!(loop_node.thread_binding().unwrap().is_none());
    assert!(loop_node.annotations().unwrap().is_empty());

    let function = PrimFunc::new(vec![x], loop_node.clone()).unwrap();
    assert_eq!(function.params().unwrap().len(), 1);
    assert!(function.body().unwrap().same_as(&Stmt::from(loop_node)));
    assert!(!function.ret_type().unwrap().is_missing());
}

#[test]
fn buffer_layout_and_block_nodes_use_native_semantics() {
    load_tvm_compiler();

    let producer: tvm::tirx::DataProducer = Function::get_global("te.Placeholder")
        .unwrap()
        .call_tuple((
            Array::new(vec![int("int64", 8)]),
            PrimType::new("float32").unwrap(),
            FfiString::from("input"),
        ))
        .unwrap()
        .try_into()
        .unwrap();
    assert_eq!(producer.shape().unwrap().len(), 1);
    assert_eq!(producer.data_type().unwrap().dtype().unwrap().bits, 32);
    assert_eq!(producer.name_hint().unwrap().as_str(), "input");

    let axis = Axis::get("m").unwrap();
    assert_eq!(axis.name().unwrap().as_str(), "m");
    assert!(axis.same_as(&Axis::get("m").unwrap()));
    let iter = Iter::new(int("int64", 8), int("int64", 1), axis.clone()).unwrap();
    assert_eq!(
        iter.extent()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        8
    );
    assert!(iter.axis().unwrap().same_as(&axis));
    let tile = TileLayout::new(vec![iter.clone()], Vec::new(), Map::new()).unwrap();
    assert_eq!(tile.shard().unwrap().len(), 1);
    assert!(tile.replica().unwrap().is_empty());
    assert!(Layout::from(tile.clone()).verify_well_formed().unwrap());

    let buffer_type = BufferType::new("global", "int32", vec![int("int64", 8)]).unwrap();
    assert_eq!(buffer_type.storage_scope().unwrap().as_str(), "global");
    assert_eq!(buffer_type.dtype().unwrap().dtype().unwrap().bits, 32);
    assert_eq!(buffer_type.shape().unwrap().len(), 1);
    assert!(buffer_type.data_alignment().unwrap() > 0);
    let buffer = buffer_type.new_var("A").unwrap();
    let index = Var::new("i", "int64").unwrap();
    let predicate = int("bool", 1);
    let load = BufferLoad::new(
        buffer.clone(),
        vec![index.clone().into()],
        Some(predicate.clone()),
    )
    .unwrap();
    let store = BufferStore::new(
        buffer.clone(),
        load.clone(),
        vec![index.clone().into()],
        Some(predicate.clone()),
    )
    .unwrap();
    assert!(load.buffer().unwrap().same_as(&buffer));
    assert_eq!(load.indices().unwrap().len(), 1);
    assert!(store.buffer().unwrap().same_as(&buffer));
    assert!(store.predicate().unwrap().is_some());

    let domain = Range::from_min_extent(int("int64", 0), int("int64", 8)).unwrap();
    let iter_var = IterVar::new(domain.clone(), index.clone(), IterVarType::DataParallel).unwrap();
    assert_eq!(iter_var.iter_type().unwrap(), IterVarType::DataParallel);
    assert!(iter_var.var().unwrap().same_as(&index));
    assert!(iter_var.span().unwrap().is_none());
    let converted = PrimExprConvertible::from(iter_var.clone())
        .to_prim_expr()
        .unwrap();
    assert!(converted.same_as(&Expr::from(index)));

    let region = BufferRegion::new(buffer.clone(), vec![domain]).unwrap();
    let match_buffer = MatchBufferRegion::new(buffer.clone(), region.clone()).unwrap();
    assert!(match_buffer.buffer().unwrap().same_as(&buffer));
    assert!(match_buffer.source().unwrap().same_as(&region));

    let block = SBlock::with_metadata(
        vec![iter_var],
        vec![region.clone()],
        vec![region],
        "copy",
        store.into(),
        None,
        Vec::new(),
        vec![match_buffer],
        Map::new(),
        None,
    )
    .unwrap();
    assert_eq!(block.name_hint().unwrap().as_str(), "copy");
    assert_eq!(block.iter_vars().unwrap().len(), 1);
    assert_eq!(block.reads().unwrap().len(), 1);
    assert_eq!(block.writes().unwrap().len(), 1);
    let realization = SBlockRealize::new(vec![int("int64", 0)], predicate, block.clone()).unwrap();
    assert!(realization.block().unwrap().same_as(&block));
}

#[test]
fn relax_nodes_round_trip_through_reflection() {
    load_tvm_compiler();

    let value = int("int32", 1);
    let tuple = RelaxTuple::new(vec![value.clone()]).unwrap();
    assert_eq!(tuple.fields().unwrap().len(), 1);

    let variable = Var::new("x", "int32").unwrap();
    let binding = VarBinding::new(variable.clone(), tuple.clone()).unwrap();
    assert!(binding.var().unwrap().same_as(&variable));
    assert!(binding.value().unwrap().same_as(&Expr::from(tuple.clone())));
    let block = BindingBlock::new(vec![Binding::from(binding)]).unwrap();
    assert_eq!(block.bindings().unwrap().len(), 1);
    let sequence = SeqExpr::new(vec![block], variable.clone()).unwrap();
    assert_eq!(sequence.blocks().unwrap().len(), 1);
    assert!(sequence
        .body()
        .unwrap()
        .same_as(&Expr::from(variable.clone())));

    let conditional = RelaxIf::new(int("bool", 1), sequence.clone(), value).unwrap();
    assert_eq!(
        conditional.true_branch().unwrap().blocks().unwrap().len(),
        1
    );
    assert!(conditional
        .false_branch()
        .unwrap()
        .blocks()
        .unwrap()
        .is_empty());

    let function = RelaxFunction::new(
        vec![variable],
        conditional,
        PrimType::new("int32").unwrap(),
        true,
    )
    .unwrap();
    assert_eq!(function.params().unwrap().len(), 1);
    assert!(function.is_pure().unwrap());
    assert!(function.attrs().unwrap().dict().unwrap().is_empty());
    assert!(function.body().unwrap().body().is_ok());
}

#[test]
fn invalid_constructor_inputs_are_native_errors() {
    load_tvm_compiler();

    let source_name = SourceName::get("invalid.tvm").unwrap();
    assert!(Span::new(source_name, i64::MAX, 1, 1, 1).is_err());

    let ordinary_var = Var::new("not_a_buffer", "int64").unwrap();
    assert!(BufferLoad::new(ordinary_var, vec![int("int64", 0)], None).is_err());

    assert!(Add::new(int("int32", 1), int("int64", 1)).is_err());
}

#[test]
fn rust_and_direct_native_construction_are_structurally_equal() {
    load_tvm_compiler();
    let lhs = int("int32", 1);
    let rhs = int("int32", 2);
    let rust = Add::new(lhs.clone(), rhs.clone()).unwrap();
    let native: Add = tvm::tvm_ffi::Function::get_global("tirx.Add")
        .unwrap()
        .call_tuple((lhs, rhs, Option::<Span>::None))
        .unwrap()
        .try_into()
        .unwrap();
    assert_ne!(object_pointer(&rust), object_pointer(&native));
    assert_structural_equal(&rust, &native);
}
