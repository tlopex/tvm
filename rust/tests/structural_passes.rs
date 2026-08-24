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

use std::path::PathBuf;
use std::sync::OnceLock;

use tvm::analysis::{
    contains_int, expression_trace, first_int, loop_nesting, memory_access_statistics,
    node_statistics, ExprTraceEvent,
};
use tvm::ir::{
    Call, DictAttrs, DummyGlobalInfo, Expr, GlobalVar, IRModule, IntImm, PrimType, Range,
    SourceMap, SourceName, Span, Type, Var,
};
use tvm::relax::{
    BindingBlock, If as RelaxIf, RelaxFunction, SeqExpr, Tuple as RelaxTuple, VarBinding,
};
use tvm::tirx::{
    Add, AddObj, AssertStmt, AssertStmtObj, Axis, BufferLoad, BufferRegion, BufferStore,
    BufferType, Evaluate, EvaluateObj, For as TirFor, IfThenElse, Iter, IterVar, IterVarType,
    Layout, Mul, PrimFunc, SBlock, SBlockRealize, SeqStmt, Stmt, Sub, TileLayout,
};
use tvm::transform;
use tvm::tvm_ffi::{
    dispatch, structural_map, structural_walk, Any, AnyCompatible, AnyMap, AnyView, DefRegionKind,
    Function, Map, Module, ObjectArc, ObjectRefCast, ObjectRefCore, Result, WalkOrder, WalkResult,
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

fn node_counts<R>(root: &R) -> (usize, usize)
where
    for<'a> AnyView<'a>: From<&'a R>,
{
    let mut asserts = 0;
    let mut evaluates = 0;
    structural_walk(
        root,
        (
            |_: &AssertStmtObj| {
                asserts += 1;
                WalkResult::Advance
            },
            |_: &EvaluateObj| {
                evaluates += 1;
                WalkResult::Advance
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap();
    (asserts, evaluates)
}

fn assert_structural_equal<L: AnyCompatible, R: AnyCompatible>(lhs: &L, rhs: &R) {
    let equal = Function::get_global("ffi.StructuralEqual")
        .unwrap()
        .call_packed(&[
            AnyView::from(lhs),
            AnyView::from(rhs),
            AnyView::from(&false),
            AnyView::from(&false),
        ])
        .unwrap();
    if !bool::try_from(equal).unwrap() {
        let repr = Function::get_global("ffi.ReprPrint").unwrap();
        let lhs = tvm::tvm_ffi::String::try_from(repr.call_packed(&[AnyView::from(lhs)]).unwrap())
            .unwrap();
        let rhs = tvm::tvm_ffi::String::try_from(repr.call_packed(&[AnyView::from(rhs)]).unwrap())
            .unwrap();
        panic!("structural mismatch:\nleft:  {lhs}\nright: {rhs}");
    }
}

fn object_pointer<O: ObjectRefCore>(value: &O) -> *const () {
    unsafe { ObjectArc::as_raw(O::data(value)).cast() }
}

fn int_expression(value: i64) -> Expr {
    Expr::int("int32", value).unwrap()
}

fn sample_sum() -> Expr {
    let one = int_expression(1);
    let two = int_expression(2);
    let three = int_expression(3);
    Expr::add(&Expr::add(&one, &two).unwrap(), &three).unwrap()
}

fn cpp_pass(name: &str) -> transform::Pass {
    Function::get_global(name)
        .unwrap()
        .call_packed(&[])
        .unwrap()
        .try_into()
        .unwrap()
}

#[test]
fn structural_walk_honors_pre_and_post_order() {
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
}

#[test]
fn structural_walk_interrupts_with_the_requested_payload() {
    load_tvm_compiler();
    let sum = sample_sum();

    assert!(contains_int(&sum, 2).unwrap());
    assert!(!contains_int(&sum, 9).unwrap());
    assert_eq!(first_int(&sum).unwrap(), Some(1));
    assert_eq!(
        first_int(&GlobalVar::new("without_literals").unwrap()).unwrap(),
        None
    );
}

#[test]
fn structural_walk_skip_prunes_only_the_selected_subtree() {
    load_tvm_compiler();
    let left = Expr::add(&int_expression(1), &int_expression(2)).unwrap();
    let root = Expr::add(&left, &int_expression(3)).unwrap();
    let left_pointer = object_pointer(&left);
    let mut seen_additions = 0;
    let mut seen_integers = Vec::new();

    structural_walk(
        &root,
        (
            |node: &AddObj| {
                seen_additions += 1;
                if node as *const AddObj as *const () == left_pointer {
                    WalkResult::Skip
                } else {
                    WalkResult::Advance
                }
            },
            |node: &tvm::ir::IntImmObj| -> Result<WalkResult> {
                seen_integers.push(node.value()?);
                Ok(WalkResult::Advance)
            },
        ),
        WalkOrder::PreOrder,
    )
    .unwrap();

    assert_eq!(seen_additions, 2);
    assert_eq!(seen_integers, vec![3]);
}

#[test]
fn reflected_getters_read_cpp_owned_nodes() {
    load_tvm_compiler();
    let one = IntImm::new("int32", 1).unwrap();
    let two = IntImm::new("int32", 2).unwrap();
    let add = Add::new(&one.clone().into(), &two.clone().into()).unwrap();
    let assertion = AssertStmt::new(&Expr::int("bool", 1).unwrap(), "ValueError", "bad").unwrap();
    let leaf_conditional = IfThenElse::new(
        &Expr::int("bool", 1).unwrap(),
        &Evaluate::from_i64(1).unwrap().into(),
        None,
    )
    .unwrap();

    assert_eq!(one.value().unwrap(), 1);
    assert_eq!(
        one.ty()
            .unwrap()
            .try_cast::<tvm::ir::PrimType>()
            .unwrap()
            .dtype()
            .unwrap()
            .bits,
        32
    );
    assert_eq!(
        add.lhs()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        1
    );
    assert_eq!(
        add.rhs()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        2
    );
    assert_eq!(
        assertion.error_kind().unwrap().value().unwrap().as_str(),
        "ValueError"
    );
    assert_eq!(
        assertion
            .message_parts()
            .unwrap()
            .get(0)
            .unwrap()
            .value()
            .unwrap()
            .as_str(),
        "bad"
    );
    assert_eq!(
        leaf_conditional
            .condition()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        1
    );
    assert!(leaf_conditional
        .then_case()
        .unwrap()
        .try_cast::<Evaluate>()
        .is_ok());
    assert!(leaf_conditional.else_case().unwrap().is_none());
}

#[test]
fn source_and_module_metadata_round_trip_cpp_objects() {
    load_tvm_compiler();
    let source_name = SourceName::get("contract-test.tvm").unwrap();
    let same_source_name = SourceName::get("contract-test.tvm").unwrap();
    let span = Span::new(&source_name, 2, 3, 4, 5).unwrap();

    assert_eq!(source_name.name().unwrap().as_str(), "contract-test.tvm");
    assert_eq!(
        object_pointer(&source_name),
        object_pointer(&same_source_name)
    );
    assert_eq!(
        object_pointer(&span.source_name().unwrap()),
        object_pointer(&source_name)
    );
    assert_eq!(span.line().unwrap(), 2);
    assert_eq!(span.column().unwrap(), 3);
    assert_eq!(span.end_line().unwrap(), 4);
    assert_eq!(span.end_column().unwrap(), 5);

    let int_type = PrimType::new("int32").unwrap();
    assert!(int_type.span().unwrap().is_none());
    let function = PrimFunc::from_body(&Evaluate::from_i64(0).unwrap()).unwrap();
    let module = IRModule::from_expr(&function).unwrap();
    assert_eq!(module.functions().unwrap().len(), 1);
    assert_eq!(module.global_var_map().unwrap().len(), 1);
    assert_eq!(module.source_map().unwrap().source_map().unwrap().len(), 0);

    // SourceMap itself is Rust-allocated; SourceMapAdd is a C++ semantic
    // operation mutating its ABI-compatible map field.
    let source_map = SourceMap::new();
    let source_name = source_map
        .add("module.tvm", "first line\nsecond line")
        .unwrap();
    let sources = source_map.source_map().unwrap();
    let source = sources.get(&source_name).unwrap().unwrap();
    assert_eq!(
        source.source_name().unwrap().name().unwrap().as_str(),
        "module.tvm"
    );
    assert_eq!(source.source().unwrap().as_str(), "first line\nsecond line");

    let dictionary: AnyMap<tvm::tvm_ffi::String> = [
        (tvm::tvm_ffi::String::from("number"), Any::from(7i64)),
        (
            tvm::tvm_ffi::String::from("text"),
            Any::from(tvm::tvm_ffi::String::from("value")),
        ),
    ]
    .into_iter()
    .collect();
    let attrs = DictAttrs::from_dictionary(&dictionary).unwrap();
    let dictionary = attrs.dictionary().unwrap();
    assert_eq!(
        i64::try_from(
            dictionary
                .get(&tvm::tvm_ffi::String::from("number"))
                .unwrap()
                .unwrap()
        )
        .unwrap(),
        7
    );
    assert_eq!(
        tvm::tvm_ffi::String::try_from(
            dictionary
                .get(&tvm::tvm_ffi::String::from("text"))
                .unwrap()
                .unwrap()
        )
        .unwrap()
        .as_str(),
        "value"
    );

    assert!(module.global_infos().unwrap().is_empty());
    let dummy = DummyGlobalInfo::new().unwrap();
    let updated = module
        .with_updated_global_info("dummy", vec![dummy.clone().into()])
        .unwrap();
    assert!(module.global_infos().unwrap().is_empty());
    let group = updated
        .global_infos()
        .unwrap()
        .get(&tvm::tvm_ffi::String::from("dummy"))
        .unwrap()
        .unwrap();
    assert_eq!(group.len(), 1);
    assert_eq!(
        object_pointer(&group.get(0).unwrap()),
        object_pointer(&dummy)
    );
}

#[test]
fn rust_skip_assert_matches_the_cpp_pass() {
    load_tvm_compiler();
    let condition = Expr::int("bool", 1).unwrap();
    let assertion: Stmt = AssertStmt::new(&condition, "RuntimeError", "failed")
        .unwrap()
        .into();
    let evaluation: Stmt = Evaluate::new(&sample_sum()).unwrap().into();
    let body = SeqStmt::new(vec![assertion, evaluation]).unwrap();
    let function = PrimFunc::from_body(&body).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result = transform::skip_assert()
        .unwrap()
        .run(module.clone())
        .unwrap();
    assert_eq!(node_counts(&module), (1, 1));
    let cpp_pass = cpp_pass("tirx.transform.SkipAssert");
    let cpp_result = tvm::transform::Pass::run(&cpp_pass, module).unwrap();

    assert_eq!(node_counts(&rust_result), (0, 1));
    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_skip_assert_rebuilds_conditional_branches_like_cpp() {
    load_tvm_compiler();
    let condition = Expr::int("bool", 1).unwrap();
    let assertion = || -> Stmt {
        AssertStmt::new(&condition, "RuntimeError", "failed")
            .unwrap()
            .into()
    };
    let then_case: Stmt = SeqStmt::new(vec![
        assertion(),
        Evaluate::new(&int_expression(1)).unwrap().into(),
    ])
    .unwrap()
    .into();
    let else_case: Stmt = SeqStmt::new(vec![
        assertion(),
        Evaluate::new(&int_expression(2)).unwrap().into(),
    ])
    .unwrap()
    .into();
    let conditional = IfThenElse::new(&condition, &then_case, Some(&else_case)).unwrap();
    assert!(conditional.else_case().unwrap().is_some());

    let function = PrimFunc::from_body(&conditional).unwrap();
    let module = IRModule::from_expr(&function).unwrap();
    let rust_result = transform::skip_assert()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.SkipAssert").run(module).unwrap();

    let statistics = node_statistics(&rust_result).unwrap();
    assert_eq!(statistics.assertions, 0);
    assert_eq!(statistics.conditionals, 1);
    assert_eq!(statistics.evaluations, 2);
    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn structural_map_preserves_map_keys_and_maps_only_values() {
    load_tvm_compiler();
    let key = GlobalVar::new("main").unwrap();
    let key_pointer = object_pointer(&key);
    let input = Map::from_iter([(key, int_expression(7))]);
    let mapped = structural_map(
        input,
        |value: IntImm| -> Result<Any> { Ok(Any::from(IntImm::new("int32", value.value()? + 1)?)) },
        WalkOrder::PostOrder,
    )
    .and_then(Map::<GlobalVar, Expr>::try_from)
    .unwrap();

    let (mapped_key, mapped_value) = mapped.iter().next().unwrap();
    assert_eq!(object_pointer(&mapped_key), key_pointer);
    assert_eq!(mapped_key.name_hint().unwrap().as_str(), "main");
    assert_eq!(
        mapped_value.try_cast::<IntImm>().unwrap().value().unwrap(),
        8
    );
}

#[test]
fn structural_map_reuses_a_uniquely_owned_array_container() {
    load_tvm_compiler();
    let input = tvm::tvm_ffi::Array::new(vec![int_expression(1), int_expression(2)]);
    let input_pointer = object_pointer(&input);
    let mapped = structural_map(
        input,
        |value: IntImm| -> Result<Any> { Ok(Any::from(IntImm::new("int32", value.value()? + 1)?)) },
        WalkOrder::PostOrder,
    )
    .and_then(tvm::tvm_ffi::Array::<Expr>::try_from)
    .unwrap();

    assert_eq!(object_pointer(&mapped), input_pointer);
    assert_eq!(
        mapped
            .iter()
            .map(|value| value.try_cast::<IntImm>().unwrap().value().unwrap())
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
}

#[derive(Default)]
struct DagProbe {
    calls: usize,
}

#[dispatch(map)]
impl DagProbe {
    fn map_conditional(&mut self, value: RelaxIf) -> Any {
        self.calls += 1;
        Any::from(value)
    }
}

#[test]
fn structural_map_memoizes_shared_relax_dag_nodes() {
    load_tvm_compiler();
    let shared = RelaxIf::new(
        &Expr::int("bool", 1).unwrap(),
        &int_expression(1),
        &int_expression(2),
    )
    .unwrap();
    assert_eq!(
        shared
            .condition()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        1
    );
    assert_eq!(
        shared
            .true_branch()
            .unwrap()
            .body()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        1
    );
    assert_eq!(
        shared
            .false_branch()
            .unwrap()
            .body()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        2
    );
    let shared_pointer = object_pointer(&shared);
    let root = RelaxTuple::new(vec![shared.clone().into(), shared.into()]).unwrap();
    let mut mapper = DagProbe::default();
    let mapped = structural_map(root, &mut mapper, WalkOrder::PostOrder)
        .and_then(RelaxTuple::try_from)
        .unwrap();

    let fields = mapped.fields().unwrap();
    assert_eq!(mapper.calls, 1);
    assert_eq!(object_pointer(&fields.get(0).unwrap()), shared_pointer);
    assert_eq!(
        object_pointer(&fields.get(0).unwrap()),
        object_pointer(&fields.get(1).unwrap())
    );
}

#[test]
fn rust_add_zero_pass_matches_cpp_stmt_simplify() {
    load_tvm_compiler();
    let one = int_expression(1);
    let zero = int_expression(0);
    let inner = Expr::add(&zero, &one).unwrap();
    let expression = Expr::add(&inner, &int_expression(0)).unwrap();
    let expected = int_expression(1);
    assert_structural_equal(
        &transform::simplify_add_zero_expr(expression.clone()).unwrap(),
        &expected,
    );

    let function = PrimFunc::from_body(&Evaluate::new(&expression).unwrap()).unwrap();
    let module = IRModule::from_expr(&function).unwrap();
    let rust_result = transform::simplify_add_zero()
        .unwrap()
        .run(module.clone())
        .unwrap();
    assert_eq!(node_statistics(&module).unwrap().additions, 2);
    assert_eq!(node_statistics(&rust_result).unwrap().additions, 0);
    let cpp_result = cpp_pass("tirx.transform.StmtSimplify").run(module).unwrap();
    assert_structural_equal(&rust_result, &cpp_result);
}

#[test]
fn rust_module_pass_maps_function_values_like_cpp() {
    load_tvm_compiler();
    let expression = Expr::add(
        &Expr::add(&int_expression(0), &int_expression(4)).unwrap(),
        &int_expression(0),
    )
    .unwrap();
    let function = PrimFunc::from_body(&Evaluate::new(&expression).unwrap()).unwrap();
    let main_module = IRModule::from_expr(&function).unwrap();
    let helper_var = GlobalVar::new("helper").unwrap();
    let helper_function: tvm::ir::BaseFunc = PrimFunc::from_body(
        &Evaluate::new(&Expr::add(&int_expression(0), &int_expression(9)).unwrap()).unwrap(),
    )
    .unwrap()
    .into();
    let module = main_module
        .with_updated_function(&helper_var, &helper_function)
        .unwrap();
    assert_eq!(main_module.functions().unwrap().len(), 1);
    assert_eq!(module.functions().unwrap().len(), 2);

    let rust_result = transform::simplify_add_zero_module_pass()
        .unwrap()
        .run(module.clone())
        .unwrap();
    assert_eq!(node_statistics(&module).unwrap().additions, 3);
    assert_eq!(node_statistics(&rust_result).unwrap().additions, 0);
    let cpp_result = cpp_pass("tirx.transform.StmtSimplify").run(module).unwrap();

    assert_eq!(rust_result.functions().unwrap().len(), 2);
    assert_structural_equal(&rust_result, &cpp_result);
}

#[derive(Default)]
struct RenameVariables {
    callback_calls: usize,
    regions: Vec<DefRegionKind>,
}

#[dispatch(map)]
impl RenameVariables {
    fn map_variable(&mut self, variable: Var, kind: DefRegionKind) -> Result<Any> {
        self.callback_calls += 1;
        self.regions.push(kind);
        let name = format!("{}_mapped", variable.name()?.as_str());
        Ok(Any::from(Var::with_type(&name, &variable.ty()?)?))
    }
}

#[test]
fn structural_map_memoizes_free_var_identity() {
    load_tvm_compiler();
    let variable = Var::new("x", "int32").unwrap();
    let body = Evaluate::new(&Expr::from(variable.clone())).unwrap();
    let function = PrimFunc::new(vec![variable.clone()], &body).unwrap();
    let mut mapper = RenameVariables::default();
    let mapped = structural_map(function.clone(), &mut mapper, WalkOrder::PostOrder)
        .and_then(PrimFunc::try_from)
        .unwrap();

    assert_eq!(mapper.callback_calls, 1);
    assert_eq!(mapper.regions, vec![DefRegionKind::Recursive]);
    let renamed = Var::new("x_mapped", "int32").unwrap();
    let expected_body = Evaluate::new(&Expr::from(renamed.clone())).unwrap();
    let expected = PrimFunc::new(vec![renamed], &expected_body).unwrap();
    assert_structural_equal(&mapped, &expected);
    assert_eq!(variable.name().unwrap().as_str(), "x");
}

#[derive(Default)]
struct ReplaceAddProbe {
    additions: usize,
    integers: usize,
}

#[dispatch(map)]
impl ReplaceAddProbe {
    fn map_add(&mut self, _value: Add) -> Result<Any> {
        self.additions += 1;
        Ok(Any::from(IntImm::new("int32", 0)?))
    }

    fn map_integer(&mut self, value: IntImm) -> Any {
        self.integers += 1;
        Any::from(value)
    }
}

#[test]
fn structural_map_preorder_replacement_prunes_original_children() {
    load_tvm_compiler();
    let expression = Expr::add(&int_expression(1), &int_expression(2)).unwrap();

    let mut pre = ReplaceAddProbe::default();
    let pre_result = structural_map(expression.clone(), &mut pre, WalkOrder::PreOrder)
        .and_then(Expr::try_from)
        .unwrap();
    let mut post = ReplaceAddProbe::default();
    let post_result = structural_map(expression, &mut post, WalkOrder::PostOrder)
        .and_then(Expr::try_from)
        .unwrap();

    assert_eq!((pre.additions, pre.integers), (1, 0));
    assert_eq!((post.additions, post.integers), (1, 2));
    assert_structural_equal(&pre_result, &int_expression(0));
    assert_structural_equal(&post_result, &int_expression(0));
}

fn nested_loop_statement() -> Stmt {
    let outer = Var::new("i", "int32").unwrap();
    let inner = Var::new("j", "int32").unwrap();
    let sum = Add::new(&outer.clone().into(), &inner.clone().into()).unwrap();
    let inner_body: Stmt = Evaluate::new(&Expr::from(sum)).unwrap().into();
    let inner_loop: Stmt =
        TirFor::serial(&inner, &int_expression(1), &int_expression(3), &inner_body)
            .unwrap()
            .into();
    TirFor::serial(&outer, &int_expression(0), &int_expression(4), &inner_loop)
        .unwrap()
        .into()
}

#[test]
fn walk_and_visit_handle_real_tir_loop_scopes() {
    load_tvm_compiler();
    let statement = nested_loop_statement();

    assert!(statement
        .clone()
        .try_cast::<TirFor>()
        .unwrap()
        .thread_binding()
        .unwrap()
        .is_none());

    let statistics = node_statistics(&statement).unwrap();
    assert_eq!(statistics.loops, 2);
    assert_eq!(statistics.statements, 3);
    assert_eq!(statistics.additions, 1);
    assert_eq!(statistics.variable_definitions, 2);
    assert_eq!(statistics.variable_uses, 2);
    let nesting = loop_nesting(&statement).unwrap();
    assert_eq!(nesting.loops, 2);
    assert_eq!(nesting.maximum_depth, 2);
}

#[test]
fn neutral_element_map_matches_cpp_stmt_simplify() {
    load_tvm_compiler();
    let variable = Var::new("x", "int32").unwrap();
    let add = Add::new(&variable.clone().into(), &int_expression(0)).unwrap();
    let multiply = Mul::new(&add.into(), &int_expression(1)).unwrap();
    let expression: Expr = Sub::new(&multiply.into(), &int_expression(0))
        .unwrap()
        .into();

    let mapped = transform::simplify_neutral_elements_expr(expression.clone()).unwrap();
    assert_structural_equal(&mapped, &Expr::from(variable.clone()));

    let function = PrimFunc::new(vec![variable], &Evaluate::new(&expression).unwrap()).unwrap();
    let rust_function = transform::simplify_neutral_elements_prim_func(function.clone()).unwrap();
    let cpp_module = cpp_pass("tirx.transform.StmtSimplify")
        .run(IRModule::from_expr(&function).unwrap())
        .unwrap();
    let rust_module = IRModule::from_expr(&rust_function).unwrap();
    assert_structural_equal(&rust_module, &cpp_module);
}

#[test]
fn map_renames_relax_definitions_and_preserves_identity_links() {
    load_tvm_compiler();
    let int_type: Type = PrimType::new("int32").unwrap().into();
    let parameter = Var::with_type("x", &int_type).unwrap();
    let bound = Var::with_type("result", &int_type).unwrap();
    let callee: Expr = GlobalVar::new("callee").unwrap().into();
    let call: Expr = Call::new(&int_type, &callee, vec![parameter.clone().into()])
        .unwrap()
        .into();
    let binding = VarBinding::new(&bound, &call).unwrap();
    assert!(binding.span().unwrap().is_none());
    let block = BindingBlock::new(vec![binding.into()]).unwrap();
    assert!(block.span().unwrap().is_none());
    let sequence = SeqExpr::new(vec![block], &bound.clone().into()).unwrap();
    let function = RelaxFunction::new(vec![parameter], &sequence.into(), &int_type, true).unwrap();

    let statistics = node_statistics(&function).unwrap();
    assert_eq!(statistics.relax_functions, 1);
    assert_eq!(statistics.sequence_expressions, 1);
    assert_eq!(statistics.binding_blocks, 1);
    assert_eq!(statistics.bindings, 1);
    assert_eq!(statistics.calls, 1);
    assert_eq!(statistics.variable_definitions, 2);
    assert_eq!(statistics.variable_uses, 2);

    let module = IRModule::from_expr(&function).unwrap();
    let mapped = transform::rename_bound_variables(function.clone().into(), "_rust")
        .and_then(|expr| expr.try_cast::<RelaxFunction>())
        .unwrap();
    let mapped_parameter = mapped.params().unwrap().get(0).unwrap();
    assert_eq!(mapped_parameter.name().unwrap().as_str(), "x_rust");

    let mapped_sequence = mapped.body().unwrap();
    let mapped_binding = mapped_sequence
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
    let mapped_bound = mapped_binding.variable().unwrap();
    assert_eq!(mapped_bound.name().unwrap().as_str(), "result_rust");

    let mapped_call = mapped_binding.value().unwrap().try_cast::<Call>().unwrap();
    let mapped_argument = mapped_call.arguments().unwrap().get(0).unwrap();
    assert_eq!(
        object_pointer(&mapped_argument),
        object_pointer(&mapped_parameter)
    );
    assert_eq!(
        object_pointer(&mapped_sequence.body().unwrap()),
        object_pointer(&mapped_bound)
    );

    let pass_result = transform::rename_bound_variables_pass("_pass")
        .unwrap()
        .run(module)
        .unwrap();
    let pass_function = pass_result
        .functions()
        .unwrap()
        .iter()
        .next()
        .unwrap()
        .1
        .try_cast::<RelaxFunction>()
        .unwrap();
    assert_eq!(
        pass_function
            .params()
            .unwrap()
            .get(0)
            .unwrap()
            .name()
            .unwrap()
            .as_str(),
        "x_pass"
    );
}

#[test]
fn mutate_can_limit_a_rewrite_to_loop_bodies() {
    load_tvm_compiler();
    let outer_var = Var::new("i", "int32").unwrap();
    let inner_var = Var::new("j", "int32").unwrap();
    let sum: Expr = Add::new(&outer_var.clone().into(), &inner_var.clone().into())
        .unwrap()
        .into();
    let inner_value: Expr = Sub::new(&sum, &int_expression(0)).unwrap().into();
    let inner_body: Stmt = Evaluate::new(&inner_value).unwrap().into();
    let inner_loop: Stmt = TirFor::serial(
        &inner_var,
        &Sub::new(&int_expression(1), &int_expression(0))
            .unwrap()
            .into(),
        &Mul::new(&int_expression(3), &int_expression(1))
            .unwrap()
            .into(),
        &inner_body,
    )
    .unwrap()
    .into();
    let statement: Stmt = TirFor::serial(
        &outer_var,
        &Add::new(&int_expression(0), &int_expression(0))
            .unwrap()
            .into(),
        &Add::new(&int_expression(4), &int_expression(0))
            .unwrap()
            .into(),
        &inner_loop,
    )
    .unwrap()
    .into();
    let original = statement.clone();
    let mapped = transform::simplify_neutral_elements_in_loop_bodies(statement)
        .unwrap()
        .try_cast::<TirFor>()
        .unwrap();

    assert!(mapped.minimum().unwrap().try_cast::<Add>().is_ok());
    assert!(mapped.extent().unwrap().try_cast::<Add>().is_ok());
    let inner = mapped.body().unwrap().try_cast::<TirFor>().unwrap();
    assert_eq!(
        inner
            .minimum()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        1
    );
    assert_eq!(
        inner
            .extent()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        3
    );

    let mapped_outer_var = mapped.loop_var().unwrap();
    let mapped_inner_var = inner.loop_var().unwrap();
    let inner_value = inner
        .body()
        .unwrap()
        .try_cast::<Evaluate>()
        .unwrap()
        .value()
        .unwrap()
        .try_cast::<Add>()
        .unwrap();
    assert_eq!(
        object_pointer(&inner_value.lhs().unwrap()),
        object_pointer(&mapped_outer_var)
    );
    assert_eq!(
        object_pointer(&inner_value.rhs().unwrap()),
        object_pointer(&mapped_inner_var)
    );

    let original_outer = original.try_cast::<TirFor>().unwrap();
    assert!(original_outer.minimum().unwrap().try_cast::<Add>().is_ok());
    let original_inner = original_outer.body().unwrap();
    let original_inner = original_inner.try_cast::<TirFor>().unwrap();
    assert!(original_inner.minimum().unwrap().try_cast::<Sub>().is_ok());
}

#[test]
fn buffer_and_block_bindings_round_trip_cpp_objects() {
    load_tvm_compiler();
    let extent = Expr::int("int64", 8).unwrap();
    let stride = Expr::int("int64", 1).unwrap();
    let axis_name = Axis::get("m").unwrap();
    let layout_iter = Iter::new(&extent, &stride, &axis_name).unwrap();
    let tile_layout = TileLayout::new(vec![layout_iter.clone()], Vec::new(), Map::new()).unwrap();
    let layout = Layout::from(tile_layout.clone());
    let buffer_type = BufferType::with_metadata(
        "global",
        &PrimType::new("int32").unwrap(),
        vec![extent.clone()],
        Vec::new(),
        &Expr::int("int64", 0).unwrap(),
        64,
        1,
        Some(&layout),
        Vec::new(),
        None,
    )
    .unwrap();
    let reflected_layout = buffer_type
        .layout()
        .unwrap()
        .unwrap()
        .try_cast::<TileLayout>()
        .unwrap();
    assert_eq!(reflected_layout.shard().unwrap().len(), 1);
    assert!(reflected_layout.replica().unwrap().is_empty());
    assert!(reflected_layout.offset().unwrap().is_empty());
    let reflected_iter = reflected_layout.shard().unwrap().get(0).unwrap();
    assert_eq!(
        reflected_iter
            .extent()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        8
    );
    assert_eq!(
        reflected_iter
            .stride()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        1
    );
    assert_eq!(reflected_iter.axis().unwrap().name().unwrap().as_str(), "m");
    let buffer = buffer_type.new_var("A").unwrap();
    let axis = Var::new("vi", "int64").unwrap();
    let axis_domain = Range::from_min_extent(
        &Expr::int("int64", 0).unwrap(),
        &Expr::int("int64", 8).unwrap(),
    )
    .unwrap();
    let iter_var = IterVar::new(&axis_domain, &axis, IterVarType::DataParallel).unwrap();
    assert!(BufferLoad::new(&axis, vec![Expr::int("int64", 0).unwrap()], None).is_err());
    let predicate = Expr::int("bool", 1).unwrap();
    let load = BufferLoad::new(&buffer, vec![axis.clone().into()], Some(&predicate)).unwrap();
    let store = BufferStore::new(
        &buffer,
        &load.clone().into(),
        vec![axis.clone().into()],
        Some(&predicate),
    )
    .unwrap();
    let region = BufferRegion::new(&buffer, vec![axis_domain.clone()]).unwrap();
    let annotations: AnyMap<tvm::tvm_ffi::String> = [
        (tvm::tvm_ffi::String::from("pipeline"), Any::from(2i64)),
        (
            tvm::tvm_ffi::String::from("tag"),
            Any::from(tvm::tvm_ffi::String::from("copy")),
        ),
    ]
    .into_iter()
    .collect();
    let block = SBlock::with_metadata(
        vec![iter_var],
        vec![region.clone()],
        vec![region],
        "copy",
        &store.clone().into(),
        None,
        Vec::new(),
        Vec::new(),
        &annotations,
        None,
    )
    .unwrap();
    assert_eq!(block.annotations().unwrap().len(), 2);
    let realization = SBlockRealize::new(
        vec![Expr::int("int64", 0).unwrap()],
        &Expr::int("bool", 1).unwrap(),
        &block,
    )
    .unwrap();
    let function = PrimFunc::new(vec![buffer.clone()], &realization).unwrap();

    assert_eq!(buffer_type.dtype().unwrap().dtype().unwrap().bits, 32);
    assert_eq!(buffer_type.storage_scope().unwrap().as_str(), "global");
    assert_eq!(buffer_type.shape().unwrap().len(), 1);
    assert!(buffer_type.strides().unwrap().is_empty());
    assert_eq!(buffer_type.data_alignment().unwrap(), 64);
    assert_eq!(buffer_type.offset_factor().unwrap(), 1);
    assert!(buffer_type.allocated_addresses().unwrap().is_empty());
    assert_eq!(
        buffer
            .ty()
            .unwrap()
            .try_cast::<BufferType>()
            .unwrap()
            .shape()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        object_pointer(&load.buffer().unwrap()),
        object_pointer(&buffer)
    );
    assert_eq!(load.indices().unwrap().len(), 1);
    assert!(load.predicate().unwrap().is_some());
    assert_eq!(
        object_pointer(&store.buffer().unwrap()),
        object_pointer(&buffer)
    );
    assert!(store.predicate().unwrap().is_some());
    assert_eq!(block.name_hint().unwrap().as_str(), "copy");
    assert_eq!(block.iter_vars().unwrap().len(), 1);
    let reflected_axis = block.iter_vars().unwrap().get(0).unwrap();
    assert_eq!(
        reflected_axis.iter_type().unwrap(),
        IterVarType::DataParallel
    );
    assert_eq!(
        object_pointer(&reflected_axis.variable().unwrap()),
        object_pointer(&axis)
    );
    assert_eq!(
        reflected_axis
            .domain()
            .unwrap()
            .unwrap()
            .extent()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        8
    );
    assert_eq!(block.reads().unwrap().len(), 1);
    assert_eq!(block.writes().unwrap().len(), 1);
    assert_eq!(
        realization.block().unwrap().name_hint().unwrap().as_str(),
        "copy"
    );

    let statistics = node_statistics(&function).unwrap();
    assert_eq!(statistics.blocks, 1);
    assert_eq!(statistics.block_realizations, 1);
    assert_eq!(statistics.buffer_loads, 1);
    assert_eq!(statistics.buffer_stores, 1);
    assert_eq!(
        memory_access_statistics(&function).unwrap(),
        tvm::analysis::MemoryAccessStatistics {
            loads: 1,
            stores: 1,
            predicated_loads: 1,
            predicated_stores: 1,
            maximum_load_rank: 1,
            maximum_store_rank: 1,
        }
    );
}

#[test]
fn rust_unit_loop_elimination_matches_cpp_on_buffer_indices() {
    load_tvm_compiler();
    let buffer_type =
        BufferType::new("global", "int32", vec![Expr::int("int64", 16).unwrap()]).unwrap();
    let buffer = buffer_type.new_var("A").unwrap();
    let outer_var = Var::new("i", "int64").unwrap();
    let unit_var = Var::new("j", "int64").unwrap();
    let index: Expr = Add::new(&outer_var.clone().into(), &unit_var.clone().into())
        .unwrap()
        .into();
    let load: Expr = BufferLoad::new(&buffer, vec![index.clone()], None)
        .unwrap()
        .into();
    let store: Stmt = BufferStore::new(&buffer, &load, vec![index], None)
        .unwrap()
        .into();
    let unit_loop: Stmt = TirFor::serial(
        &unit_var,
        &Expr::int("int64", 2).unwrap(),
        &Expr::int("int64", 1).unwrap(),
        &store,
    )
    .unwrap()
    .into();
    let outer_loop = TirFor::serial(
        &outer_var,
        &Expr::int("int64", 0).unwrap(),
        &Expr::int("int64", 4).unwrap(),
        &unit_loop,
    )
    .unwrap();
    let function = PrimFunc::new(vec![buffer], &outer_loop).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result = transform::eliminate_unit_loops()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerTIRxOpaque")
        .run(module.clone())
        .unwrap();

    assert_eq!(node_statistics(&module).unwrap().loops, 2);
    assert_eq!(node_statistics(&rust_result).unwrap().loops, 1);
    assert_structural_equal(&rust_result, &cpp_result);

    let mapped_function = rust_result
        .functions()
        .unwrap()
        .iter()
        .next()
        .unwrap()
        .1
        .try_cast::<PrimFunc>()
        .unwrap();
    let mapped_outer = mapped_function
        .body()
        .unwrap()
        .try_cast::<TirFor>()
        .unwrap();
    let mapped_store = mapped_outer
        .body()
        .unwrap()
        .try_cast::<BufferStore>()
        .unwrap();
    let mapped_index = mapped_store
        .indices()
        .unwrap()
        .get(0)
        .unwrap()
        .try_cast::<Add>()
        .unwrap();
    assert_eq!(
        mapped_index
            .rhs()
            .unwrap()
            .try_cast::<IntImm>()
            .unwrap()
            .value()
            .unwrap(),
        2
    );
    assert_eq!(memory_access_statistics(&rust_result).unwrap().loads, 1);
    assert_eq!(memory_access_statistics(&rust_result).unwrap().stores, 1);
}

#[test]
fn unit_loop_elimination_preserves_annotated_loops() {
    load_tvm_compiler();
    let loop_var = Var::new("i", "int64").unwrap();
    let body: Stmt = Evaluate::new(&Expr::from(loop_var.clone())).unwrap().into();
    let annotations: AnyMap<tvm::tvm_ffi::String> = [(
        tvm::tvm_ffi::String::from("keep_unit_loop"),
        Any::from(1i64),
    )]
    .into_iter()
    .collect();
    let loop_statement = TirFor::with_metadata(
        &loop_var,
        &Expr::int("int64", 7).unwrap(),
        &Expr::int("int64", 1).unwrap(),
        tvm::tirx::ForKind::Serial,
        &body,
        None,
        &annotations,
        None,
        None,
    )
    .unwrap();
    let function = PrimFunc::from_body(&loop_statement).unwrap();
    let module = IRModule::from_expr(&function).unwrap();

    let rust_result = transform::eliminate_unit_loops()
        .unwrap()
        .run(module.clone())
        .unwrap();
    let cpp_result = cpp_pass("tirx.transform.LowerTIRxOpaque")
        .run(module)
        .unwrap();

    assert_eq!(node_statistics(&rust_result).unwrap().loops, 1);
    assert_structural_equal(&rust_result, &cpp_result);
}
