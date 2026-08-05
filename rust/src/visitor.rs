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

//! Typed, fail-closed traversal over the stubgen-generated TIRx bindings.
//!
//! Generated object structs are opaque C++ object-layout prefixes.  This
//! visitor therefore operates only on owning generated reference wrappers and
//! reads children through their fallible reflection-backed getters.

use crate::generated::ir::{Call, Expr, FloatImm, IntImm, PrimType, Range};
use crate::generated::tirx::{
    Add, AllocBuffer, And, AssertStmt, AttrStmt, Bind, Break, Broadcast, Buffer, BufferLoad,
    BufferRegion, BufferStore, Cast, Continue, DeclBuffer, Div, Evaluate, FloorDiv, FloorMod, For,
    IfThenElse, IterVar, Let, Max, Min, Mod, Mul, Not, Or, ProducerLoad, Ramp, Reduce, SBlock,
    SBlockRealize, ScopeIdDefStmt, Select, SeqStmt, Shuffle, Stmt, StringImm, Sub,
    TilePrimitiveCall, Var, While, EQ, GE, GT, LE, LT, NE,
};
use crate::PrimExpr;
use tvm_ffi::{
    AnyCompatible, AnyValue, Array, Error, ObjectArc, ObjectRefCast, ObjectRefCore, Result,
    TypeIndex,
};

/// Borrowing, subtype-aware downcast using the runtime's canonical object cast.
pub fn try_downcast<R, N>(value: &R) -> Option<N>
where
    R: ObjectRefCore + AnyCompatible,
    N: AnyCompatible,
{
    value.downcast().ok()
}

/// Return whether an `Expr` satisfies TVM's `PrimExpr` type refinement.
///
/// `PrimExpr` is not a runtime object subtype: C++ defines it as an `Expr`
/// whose `ty` is `PrimType`. Heterogeneous `AnyValue` entries do not carry the
/// static [`PrimExpr`] wrapper, so their refinement must be checked here.
pub fn is_prim_expr(expr: &Expr) -> Result<bool> {
    Ok(try_downcast::<_, PrimType>(&expr.ty()?).is_some())
}

fn runtime_type_index<R: ObjectRefCore>(value: &R) -> Option<i32> {
    unsafe {
        let ptr = ObjectArc::as_raw(<R as ObjectRefCore>::data(value))
            as *const tvm_ffi::tvm_ffi_sys::TVMFFIObject;
        (!ptr.is_null()).then(|| (*ptr).type_index)
    }
}

fn type_error(message: impl AsRef<str>) -> Error {
    Error::new(tvm_ffi::error::TYPE_ERROR, message.as_ref(), "")
}

fn null_node(family: &str) -> Error {
    type_error(format!("cannot visit a null {family} ObjectRef"))
}

fn unsupported_node(family: &str, type_index: i32) -> Error {
    type_error(format!(
        "generated Rust {family} visitor has no dispatch entry for runtime type index {type_index}; regenerate or extend visitor.rs"
    ))
}

fn required<T>(value: Option<T>, path: &str) -> Result<T> {
    value.ok_or_else(|| type_error(format!("required IR field `{path}` is null")))
}

fn required_array_item<T>(values: &Array<Option<T>>, index: usize, path: &str) -> Result<T>
where
    T: AnyCompatible + Clone,
{
    let item = values.get(index).map_err(|error| {
        type_error(format!(
            "cannot decode required IR array element `{path}[{index}]`: {error}"
        ))
    })?;
    required(item, &format!("{path}[{index}]"))
}

fn any_array_item(values: &Array<AnyValue>, index: usize, path: &str) -> Result<AnyValue> {
    values.get(index).map_err(|error| {
        type_error(format!(
            "cannot decode heterogeneous IR array element `{path}[{index}]`: {error}"
        ))
    })
}

macro_rules! dispatch {
    ($visitor:expr, $value:expr, $( $node:ty => $method:ident ),+ $(,)?) => {
        $(
            if let Some(node) = $crate::visitor::try_downcast::<_, $node>($value) {
                return $visitor.$method(&node);
            }
        )+
    };
}

/// Dispatch an expression to a typed [`StmtExprVisitor`] method.
///
/// Unknown node kinds are errors instead of silently truncating traversal.
pub fn walk_expr<V: StmtExprVisitor + ?Sized>(visitor: &mut V, expr: &Expr) -> Result<()> {
    dispatch!(
        visitor,
        expr,
        Var => visit_var,
        BufferLoad => visit_buffer_load,
        ProducerLoad => visit_producer_load,
        Let => visit_let,
        Call => visit_call,
        Add => visit_add,
        Sub => visit_sub,
        Mul => visit_mul,
        Div => visit_div,
        Mod => visit_mod,
        FloorDiv => visit_floor_div,
        FloorMod => visit_floor_mod,
        Min => visit_min,
        Max => visit_max,
        EQ => visit_eq,
        NE => visit_ne,
        LT => visit_lt,
        LE => visit_le,
        GT => visit_gt,
        GE => visit_ge,
        And => visit_and,
        Or => visit_or,
        Reduce => visit_reduce,
        Cast => visit_cast,
        Not => visit_not,
        Select => visit_select,
        Ramp => visit_ramp,
        Broadcast => visit_broadcast,
        Shuffle => visit_shuffle,
        IntImm => visit_int_imm,
        FloatImm => visit_float_imm,
        StringImm => visit_string_imm,
    );
    let type_index = runtime_type_index(expr).ok_or_else(|| null_node("expression"))?;
    Err(unsupported_node("expression", type_index))
}

/// Dispatch a statement to a typed [`StmtExprVisitor`] method.
pub fn walk_stmt<V: StmtExprVisitor + ?Sized>(visitor: &mut V, stmt: &Stmt) -> Result<()> {
    dispatch!(
        visitor,
        stmt,
        Bind => visit_bind,
        AttrStmt => visit_attr_stmt,
        IfThenElse => visit_if_then_else,
        For => visit_for,
        While => visit_while,
        Break => visit_break,
        Continue => visit_continue,
        AllocBuffer => visit_alloc_buffer,
        DeclBuffer => visit_decl_buffer,
        BufferStore => visit_buffer_store,
        AssertStmt => visit_assert_stmt,
        SeqStmt => visit_seq_stmt,
        Evaluate => visit_evaluate,
        SBlock => visit_sblock,
        SBlockRealize => visit_sblock_realize,
        ScopeIdDefStmt => visit_scope_id_def_stmt,
        TilePrimitiveCall => visit_tile_primitive_call,
    );
    let type_index = runtime_type_index(stmt).ok_or_else(|| null_node("statement"))?;
    Err(unsupported_node("statement", type_index))
}

macro_rules! binary_visit_methods {
    ($( $method:ident($node:ty, $name:literal) ),+ $(,)?) => {
        $(
            fn $method(&mut self, node: &$node) -> Result<()> {
                let a = required(node.a()?, concat!($name, "::a"))?;
                let b = required(node.b()?, concat!($name, "::b"))?;
                self.visit_prim_expr(&a)?;
                self.visit_prim_expr(&b)
            }
        )+
    };
}

/// Recursive visitor for TIRx statements and expressions.
///
/// Override only the typed methods a pass cares about.  An override controls
/// recursion for that node, while the defaults preserve the child order of
/// TVM's C++ `StmtExprVisitor`.
pub trait StmtExprVisitor {
    fn visit_expr(&mut self, expr: &Expr) -> Result<()> {
        walk_expr(self, expr)
    }

    fn visit_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        walk_stmt(self, stmt)
    }

    /// Visit a checked primitive expression while preserving its refinement
    /// at the visitor API boundary.
    fn visit_prim_expr(&mut self, expr: &PrimExpr) -> Result<()> {
        self.visit_expr(expr.as_base())
    }

    fn visit_expr_array(&mut self, values: &Array<Option<Expr>>, path: &str) -> Result<()> {
        for index in 0..values.len() {
            let value = required_array_item(values, index, path)?;
            self.visit_expr(&value)?;
        }
        Ok(())
    }

    fn visit_prim_expr_array(
        &mut self,
        values: &Array<Option<PrimExpr>>,
        path: &str,
    ) -> Result<()> {
        for index in 0..values.len() {
            let value = required_array_item(values, index, path)?;
            self.visit_prim_expr(&value)?;
        }
        Ok(())
    }

    fn visit_buffer_def(&mut self, buffer: &Buffer, _alloc_data: bool) -> Result<()> {
        let shape = buffer.shape()?;
        self.visit_prim_expr_array(&shape, "Buffer::shape")?;

        let strides = buffer.strides()?;
        self.visit_prim_expr_array(&strides, "Buffer::strides")?;

        let elem_offset = required(buffer.elem_offset()?, "Buffer::elem_offset")?;
        self.visit_prim_expr(&elem_offset)
    }

    fn visit_buffer_use(&mut self, _buffer: &Buffer) -> Result<()> {
        Ok(())
    }

    fn visit_buffer_region(&mut self, region: &BufferRegion) -> Result<()> {
        let buffer = required(region.buffer()?, "BufferRegion::buffer")?;
        self.visit_buffer_use(&buffer)?;

        let ranges = region.region()?;
        for index in 0..ranges.len() {
            let range = required_array_item(&ranges, index, "BufferRegion::region")?;
            self.visit_range(&range, &format!("BufferRegion::region[{index}]"))?;
        }
        Ok(())
    }

    fn visit_range(&mut self, range: &Range, path: &str) -> Result<()> {
        let min = required(range.min()?, &format!("{path}.min"))?;
        let extent = required(range.extent()?, &format!("{path}.extent"))?;
        self.visit_prim_expr(&min)?;
        self.visit_prim_expr(&extent)
    }

    fn visit_iter_var_domain(&mut self, iter_var: &IterVar, path: &str) -> Result<()> {
        let domain = required(iter_var.dom()?, path)?;
        self.visit_range(&domain, path)
    }

    fn visit_var(&mut self, _node: &Var) -> Result<()> {
        Ok(())
    }

    fn visit_buffer_load(&mut self, node: &BufferLoad) -> Result<()> {
        let buffer = required(node.buffer()?, "BufferLoad::buffer")?;
        self.visit_buffer_use(&buffer)?;

        let indices = node.indices()?;
        self.visit_prim_expr_array(&indices, "BufferLoad::indices")
    }

    fn visit_producer_load(&mut self, node: &ProducerLoad) -> Result<()> {
        let indices = node.indices()?;
        self.visit_prim_expr_array(&indices, "ProducerLoad::indices")
    }

    fn visit_let(&mut self, node: &Let) -> Result<()> {
        let value = required(node.value()?, "Let::value")?;
        let body = required(node.body()?, "Let::body")?;
        self.visit_prim_expr(&value)?;
        self.visit_prim_expr(&body)
    }

    fn visit_call(&mut self, node: &Call) -> Result<()> {
        let args = node.args()?;
        self.visit_expr_array(&args, "Call::args")
    }

    binary_visit_methods!(
        visit_add(Add, "Add"),
        visit_sub(Sub, "Sub"),
        visit_mul(Mul, "Mul"),
        visit_div(Div, "Div"),
        visit_mod(Mod, "Mod"),
        visit_floor_div(FloorDiv, "FloorDiv"),
        visit_floor_mod(FloorMod, "FloorMod"),
        visit_min(Min, "Min"),
        visit_max(Max, "Max"),
        visit_eq(EQ, "EQ"),
        visit_ne(NE, "NE"),
        visit_lt(LT, "LT"),
        visit_le(LE, "LE"),
        visit_gt(GT, "GT"),
        visit_ge(GE, "GE"),
        visit_and(And, "And"),
        visit_or(Or, "Or"),
    );

    fn visit_reduce(&mut self, node: &Reduce) -> Result<()> {
        let axis = node.axis()?;
        for index in 0..axis.len() {
            let iter_var = required_array_item(&axis, index, "Reduce::axis")?;
            self.visit_iter_var_domain(&iter_var, &format!("Reduce::axis[{index}].dom"))?;
        }

        let source = node.source()?;
        self.visit_prim_expr_array(&source, "Reduce::source")?;

        let init = node.init()?;
        self.visit_prim_expr_array(&init, "Reduce::init")?;

        let condition = required(node.condition()?, "Reduce::condition")?;
        self.visit_prim_expr(&condition)
    }

    fn visit_cast(&mut self, node: &Cast) -> Result<()> {
        let value = required(node.value()?, "Cast::value")?;
        self.visit_prim_expr(&value)
    }

    fn visit_not(&mut self, node: &Not) -> Result<()> {
        let a = required(node.a()?, "Not::a")?;
        self.visit_prim_expr(&a)
    }

    fn visit_select(&mut self, node: &Select) -> Result<()> {
        let condition = required(node.condition()?, "Select::condition")?;
        let true_value = required(node.true_value()?, "Select::true_value")?;
        let false_value = required(node.false_value()?, "Select::false_value")?;
        self.visit_prim_expr(&condition)?;
        self.visit_prim_expr(&true_value)?;
        self.visit_prim_expr(&false_value)
    }

    fn visit_ramp(&mut self, node: &Ramp) -> Result<()> {
        let base = required(node.base()?, "Ramp::base")?;
        let stride = required(node.stride()?, "Ramp::stride")?;
        self.visit_prim_expr(&base)?;
        self.visit_prim_expr(&stride)
    }

    fn visit_broadcast(&mut self, node: &Broadcast) -> Result<()> {
        let value = required(node.value()?, "Broadcast::value")?;
        self.visit_prim_expr(&value)
    }

    fn visit_shuffle(&mut self, node: &Shuffle) -> Result<()> {
        let indices = node.indices()?;
        self.visit_prim_expr_array(&indices, "Shuffle::indices")?;

        let vectors = node.vectors()?;
        self.visit_prim_expr_array(&vectors, "Shuffle::vectors")
    }

    fn visit_int_imm(&mut self, _node: &IntImm) -> Result<()> {
        Ok(())
    }

    fn visit_float_imm(&mut self, _node: &FloatImm) -> Result<()> {
        Ok(())
    }

    fn visit_string_imm(&mut self, _node: &StringImm) -> Result<()> {
        Ok(())
    }

    fn visit_bind(&mut self, node: &Bind) -> Result<()> {
        let value = required(node.value()?, "Bind::value")?;
        self.visit_expr(&value)
    }

    fn visit_attr_stmt(&mut self, node: &AttrStmt) -> Result<()> {
        let value = required(node.value()?, "AttrStmt::value")?;
        let body = required(node.body()?, "AttrStmt::body")?;
        self.visit_prim_expr(&value)?;
        self.visit_stmt(&body)
    }

    fn visit_if_then_else(&mut self, node: &IfThenElse) -> Result<()> {
        let condition = required(node.condition()?, "IfThenElse::condition")?;
        let then_case = required(node.then_case()?, "IfThenElse::then_case")?;
        let else_case = node.else_case()?;
        self.visit_prim_expr(&condition)?;
        self.visit_stmt(&then_case)?;
        if let Some(else_case) = else_case {
            self.visit_stmt(&else_case)?;
        }
        Ok(())
    }

    fn visit_for(&mut self, node: &For) -> Result<()> {
        let min = required(node.min()?, "For::min")?;
        let extent = required(node.extent()?, "For::extent")?;
        let step = node.step()?;
        let body = required(node.body()?, "For::body")?;
        self.visit_prim_expr(&min)?;
        self.visit_prim_expr(&extent)?;
        if let Some(step) = step {
            self.visit_prim_expr(&step)?;
        }
        self.visit_stmt(&body)
    }

    fn visit_while(&mut self, node: &While) -> Result<()> {
        let condition = required(node.condition()?, "While::condition")?;
        let body = required(node.body()?, "While::body")?;
        self.visit_prim_expr(&condition)?;
        self.visit_stmt(&body)
    }

    fn visit_break(&mut self, _node: &Break) -> Result<()> {
        Ok(())
    }

    fn visit_continue(&mut self, _node: &Continue) -> Result<()> {
        Ok(())
    }

    fn visit_alloc_buffer(&mut self, node: &AllocBuffer) -> Result<()> {
        let buffer = required(node.buffer()?, "AllocBuffer::buffer")?;
        self.visit_buffer_def(&buffer, true)
    }

    fn visit_decl_buffer(&mut self, node: &DeclBuffer) -> Result<()> {
        let buffer = required(node.buffer()?, "DeclBuffer::buffer")?;
        self.visit_buffer_def(&buffer, false)
    }

    fn visit_buffer_store(&mut self, node: &BufferStore) -> Result<()> {
        let buffer = required(node.buffer()?, "BufferStore::buffer")?;
        let value = required(node.value()?, "BufferStore::value")?;
        let indices = node.indices()?;
        self.visit_buffer_use(&buffer)?;
        self.visit_prim_expr(&value)?;
        self.visit_prim_expr_array(&indices, "BufferStore::indices")
    }

    fn visit_assert_stmt(&mut self, node: &AssertStmt) -> Result<()> {
        let condition = required(node.condition()?, "AssertStmt::condition")?;
        let error_kind = required(node.error_kind()?, "AssertStmt::error_kind")?;
        let message_parts = node.message_parts()?;

        self.visit_prim_expr(&condition)?;
        let error_kind: Expr = error_kind.into();
        self.visit_expr(&error_kind)?;
        for index in 0..message_parts.len() {
            let part = required_array_item(&message_parts, index, "AssertStmt::message_parts")?;
            let part: Expr = part.into();
            self.visit_expr(&part)?;
        }
        Ok(())
    }

    fn visit_seq_stmt(&mut self, node: &SeqStmt) -> Result<()> {
        let seq = node.seq()?;
        for index in 0..seq.len() {
            let stmt = required_array_item(&seq, index, "SeqStmt::seq")?;
            self.visit_stmt(&stmt)?;
        }
        Ok(())
    }

    fn visit_evaluate(&mut self, node: &Evaluate) -> Result<()> {
        let value = required(node.value()?, "Evaluate::value")?;
        self.visit_expr(&value)
    }

    fn visit_sblock(&mut self, node: &SBlock) -> Result<()> {
        let iter_vars = node.iter_vars()?;
        for index in 0..iter_vars.len() {
            let iter_var = required_array_item(&iter_vars, index, "SBlock::iter_vars")?;
            self.visit_iter_var_domain(&iter_var, &format!("SBlock::iter_vars[{index}].dom"))?;
        }

        let alloc_buffers = node.alloc_buffers()?;
        for index in 0..alloc_buffers.len() {
            let buffer = required_array_item(&alloc_buffers, index, "SBlock::alloc_buffers")?;
            self.visit_buffer_def(&buffer, true)?;
        }

        let reads = node.reads()?;
        for index in 0..reads.len() {
            let region = required_array_item(&reads, index, "SBlock::reads")?;
            self.visit_buffer_region(&region)?;
        }

        let writes = node.writes()?;
        for index in 0..writes.len() {
            let region = required_array_item(&writes, index, "SBlock::writes")?;
            self.visit_buffer_region(&region)?;
        }

        let match_buffers = node.match_buffers()?;
        for index in 0..match_buffers.len() {
            let match_buffer = required_array_item(&match_buffers, index, "SBlock::match_buffers")?;
            let buffer = required(
                match_buffer.buffer()?,
                &format!("SBlock::match_buffers[{index}].buffer"),
            )?;
            let source = required(
                match_buffer.source()?,
                &format!("SBlock::match_buffers[{index}].source"),
            )?;
            self.visit_buffer_def(&buffer, true)?;
            self.visit_buffer_region(&source)?;
        }

        let init = node.init()?;
        if let Some(init) = init {
            self.visit_stmt(&init)?;
        }

        let body = required(node.body()?, "SBlock::body")?;
        self.visit_stmt(&body)
    }

    fn visit_sblock_realize(&mut self, node: &SBlockRealize) -> Result<()> {
        let iter_values = node.iter_values()?;
        let predicate = required(node.predicate()?, "SBlockRealize::predicate")?;
        let block = required(node.block()?, "SBlockRealize::block")?;
        self.visit_prim_expr_array(&iter_values, "SBlockRealize::iter_values")?;
        self.visit_prim_expr(&predicate)?;
        let block: Stmt = block.into();
        self.visit_stmt(&block)
    }

    fn visit_scope_id_def_stmt(&mut self, node: &ScopeIdDefStmt) -> Result<()> {
        let definition = required(node.def()?, "ScopeIdDefStmt::def")?;
        let extents = definition.extents()?;
        if let Some(extents) = extents {
            self.visit_prim_expr_array(&extents, "ScopeIdDefStmt::def.extents")?;
        }

        let preferred_extents = definition.preferred_extents()?;
        if let Some(preferred_extents) = preferred_extents {
            self.visit_prim_expr_array(
                &preferred_extents,
                "ScopeIdDefStmt::def.preferred_extents",
            )?;
        }
        Ok(())
    }

    fn visit_tile_primitive_any(&mut self, value: &AnyValue) -> Result<()> {
        if value.as_any().type_index() == TypeIndex::kTVMFFINone as i32 {
            return Ok(());
        }

        // BufferRegion is also PrimExprConvertible, but C++ deliberately
        // recognizes and skips it before attempting the conversion.
        let object = value.try_as::<tvm_ffi::object::ObjectRef>();
        if object
            .as_ref()
            .is_some_and(|object| try_downcast::<_, BufferRegion>(object).is_some())
        {
            return Ok(());
        }

        if let Some(object) = object {
            // C++ uses Any::as<PrimExpr>(), which is a strict reinterpretation:
            // it does not apply scalar or PrimExprConvertible fallbacks.
            if let Some(expr) = try_downcast::<_, PrimExpr>(&object) {
                return self.visit_prim_expr(&expr);
            }
            if let Some(stmt) = try_downcast::<_, Stmt>(&object) {
                return self.visit_stmt(&stmt);
            }
        }

        // Other values are metadata in the C++ visitor, not IR children.
        // Expression/statement subtypes still fail closed in walk_expr/
        // walk_stmt when their concrete node kind is unknown.
        Ok(())
    }

    fn visit_tile_primitive_call(&mut self, node: &TilePrimitiveCall) -> Result<()> {
        let args = node.args()?;
        for index in 0..args.len() {
            let value = any_array_item(&args, index, "TilePrimitiveCall::args")?;
            self.visit_tile_primitive_any(&value)?;
        }

        let config = node.config()?;
        for (_key, value) in &config {
            self.visit_tile_primitive_any(&value)?;
        }
        Ok(())
    }
}
