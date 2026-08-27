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

use tvm::analysis::{
    contains_int, expression_trace, first_int, loop_nesting, memory_access_statistics,
    node_statistics, side_effect, ExprTraceEvent,
};
use tvm::ir::{
    BaseFunc, Call, DictAttrs, Expr, GlobalVar, IRModule, IntImm, PrimType, SourceMap, Type, Var,
};
use tvm::relax::{Binding, BindingBlock, RelaxFunction, SeqExpr, VarBinding};
use tvm::tirx::{
    Add, AddObj, AssertStmt, BufferLoad, BufferStore, BufferType, Evaluate, For as TirFor,
    IfThenElse, Mul, PrimFunc, SeqStmt, Sub,
};
use tvm::transform;
use tvm::tvm_ffi::{
    structural_walk, Any, DefRegionKind, Map, ObjectIdentity, ObjectRefCast, ObjectRefCore,
    String as FfiString, WalkOrder, WalkResult,
};

mod common;
use common::{assert_structural_equal, load_tvm_compiler, object_pointer};

fn int(value: i64) -> Expr {
    IntImm::new("int32", value).unwrap().into()
}

fn int64(value: i64) -> Expr {
    IntImm::new("int64", value).unwrap().into()
}

fn sample_sum() -> Expr {
    Add::new(Add::new(int(1), int(2)).unwrap(), int(3))
        .unwrap()
        .into()
}

fn int_value(value: Expr) -> i64 {
    value.try_cast::<IntImm>().unwrap().value().unwrap()
}

fn module_function(module: &IRModule) -> PrimFunc {
    module
        .functions()
        .unwrap()
        .iter()
        .next()
        .unwrap()
        .1
        .try_cast::<PrimFunc>()
        .unwrap()
}

#[test]
fn structural_walk_covers_order_skip_and_interrupt() {
    load_tvm_compiler();
    let sum = sample_sum();

    assert_eq!(
        expression_trace(&sum, WalkOrder::PreOrder).unwrap(),
        vec![
            ExprTraceEvent::Add,
            ExprTraceEvent::Add,
            ExprTraceEvent::Int(1),
            ExprTraceEvent::Int(2),
            ExprTraceEvent::Int(3),
        ]
    );
    assert_eq!(
        expression_trace(&sum, WalkOrder::PostOrder).unwrap(),
        vec![
            ExprTraceEvent::Int(1),
            ExprTraceEvent::Int(2),
            ExprTraceEvent::Add,
            ExprTraceEvent::Int(3),
            ExprTraceEvent::Add,
        ]
    );
    assert!(contains_int(&sum, 2).unwrap());
    assert!(!contains_int(&sum, 9).unwrap());
    assert_eq!(first_int(&sum).unwrap(), Some(1));

    let left = sum
        .clone()
        .try_cast::<Add>()
        .unwrap()
        .a()
        .unwrap()
        .try_cast::<Add>()
        .unwrap();
    let left_pointer = object_pointer(&left);
    let mut seen = Vec::new();
    structural_walk(
        &sum,
        (
            |node: &AddObj| {
                if std::ptr::from_ref(node).cast::<()>() == left_pointer {
                    WalkResult::Skip
                } else {
                    WalkResult::Advance
                }
            },
            |node: &tvm::ir::IntImmObj| -> tvm::tvm_ffi::Result<WalkResult> {
                seen.push(node.value()?);
                Ok(WalkResult::Advance)
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap();
    assert_eq!(seen, vec![3]);
}

#[test]
fn structural_map_passes_rewrite_post_order_values() {
    load_tvm_compiler();

    let x: Expr = Var::new("x", "int32").unwrap().into();
    let neutral: Expr = Mul::new(Add::new(x.clone(), int(0)).unwrap(), int(1))
        .unwrap()
        .into();
    let simplified = transform::simplify_neutral_elements_expr(neutral).unwrap();
    assert!(simplified.same_as(&x));

    let constants: Expr = Sub::new(
        Mul::new(Add::new(int(2), int(3)).unwrap(), int(4)).unwrap(),
        int(1),
    )
    .unwrap()
    .into();
    assert_eq!(
        int_value(transform::fold_integer_constants_expr(constants).unwrap()),
        19
    );
    assert_eq!(
        int_value(transform::increment_int_immediates(int(41)).unwrap()),
        42
    );
    assert!(transform::increment_int_immediates(int64(i64::MAX)).is_err());
}

#[test]
fn callback_controlled_visit_and_mutate_handle_loop_scope() {
    load_tvm_compiler();

    let outer_var = Var::new("i", "int32").unwrap();
    let inner_var = Var::new("j", "int32").unwrap();
    let body_expr = Add::new(inner_var.clone(), int(0)).unwrap();
    let inner = TirFor::serial(
        inner_var,
        Add::new(int(0), int(0)).unwrap(),
        Add::new(int(4), int(0)).unwrap(),
        Evaluate::new(body_expr).unwrap(),
    )
    .unwrap();
    let outer =
        TirFor::serial(outer_var, Add::new(int(0), int(0)).unwrap(), int(2), inner).unwrap();

    let nesting = loop_nesting(&outer).unwrap();
    assert_eq!(nesting.loops, 2);
    assert_eq!(nesting.maximum_depth, 2);

    let mapped = transform::simplify_neutral_elements_in_loop_bodies(outer.clone().into()).unwrap();
    let mapped_outer = mapped.try_cast::<TirFor>().unwrap();
    assert!(mapped_outer.min().unwrap().try_cast::<Add>().is_ok());
    let mapped_inner = mapped_outer.body().unwrap().try_cast::<TirFor>().unwrap();
    assert_eq!(int_value(mapped_inner.min().unwrap()), 0);
    assert_eq!(int_value(mapped_inner.extent().unwrap()), 4);
    let mapped_body = mapped_inner.body().unwrap().try_cast::<Evaluate>().unwrap();
    assert!(mapped_body.value().unwrap().try_cast::<Var>().is_ok());
}

#[test]
fn mutation_remaps_unit_loop_variables_in_buffer_indices() {
    load_tvm_compiler();

    let buffer_type = BufferType::new("global", "int32", vec![int64(16)]).unwrap();
    let buffer = buffer_type.new_var("A").unwrap();
    let unit_var = Var::new("j", "int64").unwrap();
    let index: Expr = Add::new(unit_var.clone(), int64(1)).unwrap().into();
    let load: Expr = BufferLoad::new(buffer.clone(), vec![index.clone()], None)
        .unwrap()
        .into();
    let store = BufferStore::new(buffer.clone(), load, vec![index], None).unwrap();
    let loop_node = TirFor::serial(unit_var, int64(2), int64(1), store).unwrap();
    let function = PrimFunc::new(vec![buffer], loop_node).unwrap();

    let mapped = transform::eliminate_unit_loops_prim_func(function).unwrap();
    assert_eq!(node_statistics(&mapped).unwrap().loops, 0);
    let store = mapped.body().unwrap().try_cast::<BufferStore>().unwrap();
    let index = store
        .indices()
        .unwrap()
        .get(0)
        .unwrap()
        .try_cast::<Add>()
        .unwrap();
    assert_eq!(int_value(index.a().unwrap()), 2);
}

#[test]
fn typed_walk_collects_node_and_memory_access_statistics() {
    load_tvm_compiler();

    let buffer_type = BufferType::new("global", "int32", vec![int64(8)]).unwrap();
    let buffer = buffer_type.new_var("A").unwrap();
    let index = Var::new("i", "int64").unwrap();
    let predicate = IntImm::new("bool", 1).unwrap();
    let load = BufferLoad::new(
        buffer.clone(),
        vec![index.clone().into()],
        Some(predicate.clone().into()),
    )
    .unwrap();
    let store = BufferStore::new(
        buffer.clone(),
        load,
        vec![index.clone().into()],
        Some(predicate.into()),
    )
    .unwrap();
    let loop_node = TirFor::serial(index, int64(0), int64(8), store).unwrap();
    let function = PrimFunc::new(vec![buffer], loop_node).unwrap();

    let nodes = node_statistics(&function).unwrap();
    assert_eq!(nodes.loops, 1);
    assert_eq!(nodes.buffer_loads, 1);
    assert_eq!(nodes.buffer_stores, 1);
    assert!(nodes.variable_definitions >= 1);
    let accesses = memory_access_statistics(&function).unwrap();
    assert_eq!(accesses.loads, 1);
    assert_eq!(accesses.stores, 1);
    assert_eq!(accesses.predicated_loads, 1);
    assert_eq!(accesses.predicated_stores, 1);
    assert_eq!(accesses.maximum_load_rank, 1);
    assert_eq!(accesses.maximum_store_rank, 1);
}

#[test]
fn rust_prim_func_and_module_passes_run_through_tvm() {
    load_tvm_compiler();

    let x = Var::new("x", "int32").unwrap();
    let expression = Add::new(x.clone(), int(0)).unwrap();
    let assertion = AssertStmt::new(IntImm::new("bool", 1).unwrap(), "ValueError", "bad").unwrap();
    let body = SeqStmt::new(vec![
        Evaluate::new(expression).unwrap().into(),
        assertion.into(),
    ])
    .unwrap();
    let module = IRModule::from_expr(PrimFunc::new(vec![x], body).unwrap()).unwrap();

    let simplified = transform::simplify_add_zero()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let simplified_body = module_function(&simplified)
        .body()
        .unwrap()
        .try_cast::<SeqStmt>()
        .unwrap();
    assert!(simplified_body
        .seq()
        .unwrap()
        .get(0)
        .unwrap()
        .try_cast::<Evaluate>()
        .unwrap()
        .value()
        .unwrap()
        .try_cast::<Var>()
        .is_ok());

    let skipped = transform::skip_assert()
        .unwrap()
        .run(module.clone())
        .unwrap();
    assert_eq!(node_statistics(&skipped).unwrap().assertions, 0);

    let module_mapped = transform::simplify_add_zero_module_pass()
        .unwrap()
        .run(module)
        .unwrap();
    assert_structural_equal(&simplified, &module_mapped);
}

#[test]
fn known_control_flow_and_constant_folding_passes_run_through_tvm() {
    load_tvm_compiler();

    assert!(!side_effect(&int(1)).unwrap().may_update_state());
    let opaque_call: Expr = Call::new(
        PrimType::new("int32").unwrap(),
        GlobalVar::new("unknown_callee").unwrap(),
        Vec::new(),
    )
    .unwrap()
    .into();
    assert!(side_effect(&opaque_call).unwrap().may_update_state());

    let constants = IRModule::from_expr(
        PrimFunc::from_body(Evaluate::new(Add::new(int(2), int(3)).unwrap()).unwrap()).unwrap(),
    )
    .unwrap();
    let folded = transform::fold_integer_constants()
        .unwrap()
        .run(constants)
        .unwrap();
    let folded_evaluation = module_function(&folded)
        .body()
        .unwrap()
        .try_cast::<Evaluate>()
        .unwrap();
    assert_eq!(int_value(folded_evaluation.value().unwrap()), 5);

    let conditional = IfThenElse::new(
        Add::new(int(1), int(0)).unwrap(),
        Evaluate::new(int(7)).unwrap(),
        Some(Evaluate::new(int(9)).unwrap().into()),
    )
    .unwrap();
    let module = IRModule::from_expr(PrimFunc::from_body(conditional).unwrap()).unwrap();

    let controlled = transform::simplify_known_control_flow()
        .unwrap()
        .run(module)
        .unwrap();
    assert_eq!(node_statistics(&controlled).unwrap().conditionals, 0);
    let evaluation = module_function(&controlled)
        .body()
        .unwrap()
        .try_cast::<Evaluate>()
        .unwrap();
    assert_eq!(int_value(evaluation.value().unwrap()), 0);
}

#[test]
fn relax_renaming_updates_definitions_and_their_uses() {
    load_tvm_compiler();

    let parameter = Var::new("x", "int32").unwrap();
    let bound = Var::new("y", "int32").unwrap();
    let binding = VarBinding::new(bound.clone(), parameter.clone()).unwrap();
    let block = BindingBlock::new(vec![Binding::from(binding)]).unwrap();
    let body = SeqExpr::new(vec![block], bound.clone()).unwrap();
    let function =
        RelaxFunction::new(vec![parameter], body, PrimType::new("int32").unwrap(), true).unwrap();

    let renamed = transform::rename_bound_variables_function(function, "_r").unwrap();
    assert_eq!(
        renamed
            .params()
            .unwrap()
            .get(0)
            .unwrap()
            .name()
            .unwrap()
            .as_str(),
        "x_r"
    );
    let body = renamed.body().unwrap();
    let binding = body
        .blocks()
        .unwrap()
        .get(0)
        .unwrap()
        .bindings()
        .unwrap()
        .get(0)
        .unwrap()
        .try_cast::<VarBinding>()
        .unwrap();
    let definition = binding.var().unwrap();
    let use_site = body.body().unwrap().try_cast::<Var>().unwrap();
    assert_eq!(definition.name().unwrap().as_str(), "y_r");
    assert_eq!(
        ObjectIdentity::of(&definition),
        ObjectIdentity::of(&use_site)
    );
}

#[test]
fn module_pruning_keeps_reachable_and_externally_visible_functions() {
    load_tvm_compiler();

    let result_type: Type = PrimType::new("int32").unwrap().into();
    let main = GlobalVar::new("main").unwrap();
    let helper = GlobalVar::new("helper").unwrap();
    let exported = GlobalVar::new("exported").unwrap();
    let dead = GlobalVar::new("dead").unwrap();
    let helper_call = Call::new(result_type, helper.clone(), Vec::new()).unwrap();

    let main_function: BaseFunc = PrimFunc::from_body(Evaluate::new(helper_call).unwrap())
        .unwrap()
        .into();
    let helper_function: BaseFunc = PrimFunc::from_body(Evaluate::from_i64(1).unwrap())
        .unwrap()
        .into();
    let exported_attrs = DictAttrs::from_dictionary(
        [(
            FfiString::from("global_symbol"),
            Any::from(FfiString::from("exported_symbol")),
        )]
        .into_iter()
        .collect(),
    )
    .unwrap();
    let exported_function: BaseFunc = PrimFunc::with_metadata(
        Vec::new(),
        Evaluate::from_i64(2).unwrap(),
        Type::missing().unwrap(),
        exported_attrs,
        None,
    )
    .unwrap()
    .into();
    let dead_function: BaseFunc = PrimFunc::from_body(Evaluate::from_i64(3).unwrap())
        .unwrap()
        .into();
    let module = IRModule::with_metadata(
        [
            (main, main_function),
            (helper, helper_function),
            (exported, exported_function),
            (dead, dead_function),
        ]
        .into_iter()
        .collect(),
        SourceMap::new().unwrap(),
        DictAttrs::empty().unwrap(),
        Map::new(),
    )
    .unwrap();

    let direct = transform::prune_unreachable_functions_from_main(module.clone()).unwrap();
    let mut names = direct
        .functions()
        .unwrap()
        .iter()
        .map(|(global, _)| global.name_hint().unwrap().as_str().to_owned())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["exported", "helper", "main"]);

    let through_pass = transform::prune_unreachable_functions_pass(vec!["main".to_owned()])
        .unwrap()
        .run(module)
        .unwrap();
    assert_structural_equal(&direct, &through_pass);
}

#[test]
fn definition_regions_are_visible_to_typed_walk_callbacks() {
    load_tvm_compiler();
    let variable = Var::new("x", "int32").unwrap();
    let function = PrimFunc::new(
        vec![variable.clone()],
        Evaluate::new(variable.clone()).unwrap(),
    )
    .unwrap();
    let mut definitions = 0;
    let mut uses = 0;
    structural_walk(
        &function,
        |_: &tvm::ir::VarObj, region: DefRegionKind| {
            if region == DefRegionKind::None {
                uses += 1;
            } else {
                definitions += 1;
            }
            WalkResult::Advance
        },
        WalkOrder::PreOrder,
    )
    .unwrap();
    assert_eq!((definitions, uses), (1, 1));
}
