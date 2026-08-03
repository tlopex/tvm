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

//! Demo: drive the stubgen-generated TIRx/IR bindings against this repo's
//! `build/lib/libtvm_compiler.so`.
//!
//! `src/generated/` is the output of `tvm-ffi-stubgen --target rust` with a
//! single hand patch (see `ffi_compat`; regeneration recipe in rust/README.md).
//! Three checks:
//!   1. scalar field read            (generated::ir::IntImm)
//!   2. Optional<ObjectRef> field    (generated::tirx::IfThenElse.else_case)
//!   3. padded layout w/ hidden C++  (generated::ir::Op.num_inputs, `_pad0`)
mod ffi_compat;
mod generated;

use generated::ir::{Expr, IntImm, IntImmObj, Op, SourceName, Span};
use generated::tirx::{Evaluate, IfThenElse, SeqStmt, Stmt};
use tvm_ffi::{Any, AnyView, Array, DLDataType, DLDataTypeExt, Function, Module, Result};

fn lib_path() -> String {
    format!(
        "{}/../build/lib/libtvm_compiler.so",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn main() -> Result<()> {
    let _lib = Module::load_from_file(lib_path())?;
    let int_imm = Function::get_global("ir.IntImm")?;
    let evaluate = Function::get_global("tirx.Evaluate")?;
    let if_then_else = Function::get_global("tirx.IfThenElse")?;
    let null = Any::default();

    let mk_int = |dt: &str, v: i64| -> Result<Any> {
        let dt = DLDataType::try_from_str(dt)?;
        int_imm.call_packed(&[AnyView::from(&dt), AnyView::from(&v), AnyView::from(&null)])
    };

    println!("=========== Example 1: generated::ir::IntImm (scalar field) ===========");
    let imm: IntImm = mk_int("int64", 42)?.try_into()?;
    println!("IntImm.value = {}", imm.value);
    assert_eq!(imm.value, 42);

    println!("\n=========== Example 2: generated::tirx::IfThenElse (Optional<Stmt> field) ===========");
    let cond = mk_int("bool", 1)?;
    let then_case: Any =
        evaluate.call_packed(&[AnyView::from(&mk_int("int32", 5)?), AnyView::from(&null)])?;
    let else_body: Any =
        evaluate.call_packed(&[AnyView::from(&mk_int("int32", 9)?), AnyView::from(&null)])?;

    let ite_none: IfThenElse = if_then_else
        .call_packed(&[
            AnyView::from(&cond),
            AnyView::from(&then_case),
            AnyView::from(&null),
            AnyView::from(&null),
        ])?
        .try_into()?;
    let ite_some: IfThenElse = if_then_else
        .call_packed(&[
            AnyView::from(&cond),
            AnyView::from(&then_case),
            AnyView::from(&else_body),
            AnyView::from(&null),
        ])?
        .try_into()?;

    println!("no-else  IfThenElse.else_case.has_value() = {}", ite_none.else_case.has_value());
    println!("with-else IfThenElse.else_case.has_value() = {}", ite_some.else_case.has_value());
    assert!(!ite_none.else_case.has_value());
    assert!(ite_some.else_case.has_value());

    println!("\n=========== Example 3: generated::ir::Op (padded layout) ===========");
    // OpNode has UNREGISTERED fields (`attrs_type_index` @96, private `index_`
    // @108); the generated `_pad0: [u8; 4]` keeps `num_inputs` at its real C++
    // offset 100.
    let get_op = Function::get_global("ir.GetOp")?;
    let fetch = |name: &str| -> Result<Op> {
        let name = tvm_ffi::String::from(name);
        Ok(get_op.call_packed(&[AnyView::from(&name)])?.try_into()?)
    };
    let variadic: Op = fetch("tirx.isfinite")?;
    let binary: Op = fetch("tirx.webgpu.subgroup_shuffle")?;
    println!("tirx.isfinite:                num_inputs = {}, support_level = {}",
             variadic.num_inputs, variadic.support_level);
    println!("tirx.webgpu.subgroup_shuffle: num_inputs = {}, support_level = {}",
             binary.num_inputs, binary.support_level);
    assert_eq!(variadic.num_inputs, -1);
    assert_eq!(binary.num_inputs, 2);

    println!("\n=========== Example 4: explicit new(arg1, ...) ctors (pure Rust construction) ===========");
    // Flat ctors take every field explicitly, base-chain fields first:
    // Evaluate::new(span, value) fills StmtObj.span through the base chain.
    let sn = SourceName::new(tvm_ffi::String::from("<demo>"));
    let span = Span::new(sn, 1, 0, 1, 8);
    let value: Expr = mk_int("int32", 7)?.try_into()?;
    let ev = Evaluate::new(span.clone(), value);
    let got = ev.value.downcast::<IntImmObj>().map(|n| n.value);
    let ev_stmt: Stmt = Any::from(AnyView::from(&ev)).try_into()?;
    let seq = SeqStmt::new(span, Array::new(vec![ev_stmt.clone()]));
    println!("Evaluate.value as IntImm = {:?}, .span.line = {}", got, ev_stmt.span.line);
    println!("SeqStmt.seq.len() = {}", seq.seq.len());
    assert_eq!(got, Some(7));
    assert_eq!(ev_stmt.span.line, 1);
    assert_eq!(seq.seq.len(), 1);

    println!("\nOK: the stubgen output compiles and mirrors the C++ objects exactly.");
    Ok(())
}
