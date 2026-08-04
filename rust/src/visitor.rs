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
//! This module is deliberately hand-written on top of `generated`: it is a
//! prototype for code that the Rust stub generator should eventually emit from
//! the reflected inheritance graph.  Keeping it here lets real passes exercise
//! the generated layouts while making the missing generator functionality
//! concrete and testable.

use crate::generated::ir::{CallObj, Expr, FloatImmObj, IntImmObj, Range};
use crate::generated::tirx::{
    AddObj, AllocBufferObj, AndObj, AssertStmtObj, AttrStmtObj, BindObj, BreakObj, BroadcastObj,
    Buffer, BufferLoadObj, BufferRegion, BufferStoreObj, CastObj, ContinueObj, DeclBufferObj,
    DivObj, EQObj, EvaluateObj, FloorDivObj, FloorModObj, ForObj, GEObj, GTObj, IfThenElseObj,
    IterVar, LEObj, LTObj, LetObj, MaxObj, MinObj, ModObj, MulObj, NEObj, NotObj, OrObj,
    ProducerLoadObj, RampObj, ReduceObj, SBlockObj, SBlockRealizeObj, ScopeIdDefStmtObj, SelectObj,
    SeqStmtObj, ShuffleObj, Stmt, StringImmObj, SubObj, TilePrimitiveCallObj, VarObj, WhileObj,
};
use tvm_ffi::{Error, ObjectArc, ObjectRefCore, Result};

fn lookup_type_index(type_key: &str) -> Option<i32> {
    unsafe {
        let type_key = tvm_ffi::tvm_ffi_sys::TVMFFIByteArray::from_str(type_key);
        let mut type_index = 0;
        let status = tvm_ffi::tvm_ffi_sys::TVMFFITypeKeyToIndex(&type_key, &mut type_index);
        if status == 0 {
            Some(type_index)
        } else {
            // A failed safe call owns an Error in the thread-local raised slot.
            // Consume it before trying the next generated type key.
            drop(Error::from_raised());
            None
        }
    }
}

fn is_instance_index(runtime_type_index: i32, target_type_index: i32) -> bool {
    if runtime_type_index == target_type_index {
        return true;
    }
    unsafe {
        let runtime = tvm_ffi::tvm_ffi_sys::TVMFFIGetTypeInfo(runtime_type_index);
        let target = tvm_ffi::tvm_ffi_sys::TVMFFIGetTypeInfo(target_type_index);
        if runtime.is_null() || target.is_null() || (*runtime).type_depth <= (*target).type_depth {
            return false;
        }
        let ancestors = (*runtime).type_acenstors;
        if ancestors.is_null() {
            return false;
        }
        let ancestor = *ancestors.add((*target).type_depth as usize);
        !ancestor.is_null() && (*ancestor).type_index == target_type_index
    }
}

/// Borrowed, subtype-aware downcast that tolerates a generated type absent
/// from the loaded runtime.
///
/// Generated `downcast` currently calls `N::type_index()`, whose cached lookup
/// panics forever when bindings and the loaded `.so` differ.  Visitor dispatch
/// must instead fail closed and continue checking the runtime's available
/// types; a complete ABI guard can then report a useful mismatch separately.
pub fn try_downcast<R, N>(value: &R) -> Option<&N>
where
    R: ObjectRefCore,
    N: tvm_ffi::ObjectCore,
{
    unsafe {
        let raw = ObjectArc::as_raw(<R as ObjectRefCore>::data(value));
        if raw.is_null() {
            return None;
        }
        let header = raw as *const tvm_ffi::tvm_ffi_sys::TVMFFIObject;
        let target_type_index = lookup_type_index(N::TYPE_KEY)?;
        if is_instance_index((*header).type_index, target_type_index) {
            Some(&*(raw as *const N))
        } else {
            None
        }
    }
}

fn runtime_type_index<R: ObjectRefCore>(value: &R) -> Option<i32> {
    unsafe {
        let ptr = ObjectArc::as_raw(<R as ObjectRefCore>::data(value))
            as *const tvm_ffi::tvm_ffi_sys::TVMFFIObject;
        (!ptr.is_null()).then(|| (*ptr).type_index)
    }
}

fn null_node(family: &str) -> Error {
    Error::new(
        tvm_ffi::error::TYPE_ERROR,
        &format!("cannot visit a null {family} ObjectRef"),
        "",
    )
}

fn unsupported_node(family: &str, type_index: i32) -> Error {
    Error::new(
        tvm_ffi::error::TYPE_ERROR,
        &format!(
            "generated Rust {family} visitor has no dispatch entry for runtime type index {type_index}; regenerate or extend visitor.rs"
        ),
        "",
    )
}

macro_rules! dispatch {
    ($visitor:expr, $value:expr, $( $node:ty => $method:ident ),+ $(,)?) => {
        $(
            if let Some(node) = $crate::visitor::try_downcast::<_, $node>($value) {
                return $visitor.$method(node);
            }
        )+
    };
}

/// Dispatch an expression to a typed [`StmtExprVisitor`] method.
///
/// Unknown node kinds are errors instead of silently truncating traversal.
/// This makes a newly added C++ IR node visible immediately to Rust pass
/// authors, but the list currently has to be maintained by hand.
pub fn walk_expr<V: StmtExprVisitor + ?Sized>(visitor: &mut V, expr: &Expr) -> Result<()> {
    dispatch!(
        visitor,
        expr,
        VarObj => visit_var,
        BufferLoadObj => visit_buffer_load,
        ProducerLoadObj => visit_producer_load,
        LetObj => visit_let,
        CallObj => visit_call,
        AddObj => visit_add,
        SubObj => visit_sub,
        MulObj => visit_mul,
        DivObj => visit_div,
        ModObj => visit_mod,
        FloorDivObj => visit_floor_div,
        FloorModObj => visit_floor_mod,
        MinObj => visit_min,
        MaxObj => visit_max,
        EQObj => visit_eq,
        NEObj => visit_ne,
        LTObj => visit_lt,
        LEObj => visit_le,
        GTObj => visit_gt,
        GEObj => visit_ge,
        AndObj => visit_and,
        OrObj => visit_or,
        ReduceObj => visit_reduce,
        CastObj => visit_cast,
        NotObj => visit_not,
        SelectObj => visit_select,
        RampObj => visit_ramp,
        BroadcastObj => visit_broadcast,
        ShuffleObj => visit_shuffle,
        IntImmObj => visit_int_imm,
        FloatImmObj => visit_float_imm,
        StringImmObj => visit_string_imm,
    );
    let type_index = runtime_type_index(expr).ok_or_else(|| null_node("expression"))?;
    Err(unsupported_node("expression", type_index))
}

/// Dispatch a statement to a typed [`StmtExprVisitor`] method.
pub fn walk_stmt<V: StmtExprVisitor + ?Sized>(visitor: &mut V, stmt: &Stmt) -> Result<()> {
    dispatch!(
        visitor,
        stmt,
        BindObj => visit_bind,
        AttrStmtObj => visit_attr_stmt,
        IfThenElseObj => visit_if_then_else,
        ForObj => visit_for,
        WhileObj => visit_while,
        BreakObj => visit_break,
        ContinueObj => visit_continue,
        AllocBufferObj => visit_alloc_buffer,
        DeclBufferObj => visit_decl_buffer,
        BufferStoreObj => visit_buffer_store,
        AssertStmtObj => visit_assert_stmt,
        SeqStmtObj => visit_seq_stmt,
        EvaluateObj => visit_evaluate,
        SBlockObj => visit_sblock,
        SBlockRealizeObj => visit_sblock_realize,
        ScopeIdDefStmtObj => visit_scope_id_def_stmt,
        TilePrimitiveCallObj => visit_tile_primitive_call,
    );
    let type_index = runtime_type_index(stmt).ok_or_else(|| null_node("statement"))?;
    Err(unsupported_node("statement", type_index))
}

macro_rules! binary_visit_methods {
    ($( $method:ident($node:ty) ),+ $(,)?) => {
        $(
            fn $method(&mut self, node: &$node) -> Result<()> {
                self.visit_expr(&node.a)?;
                self.visit_expr(&node.b)
            }
        )+
    };
}

/// Recursive visitor for TIRx statements and expressions.
///
/// Override only the typed methods a pass cares about.  An override controls
/// recursion for that node, while the default implementation follows the same
/// semantic children and ordering as TVM's C++ `StmtExprVisitor`.
pub trait StmtExprVisitor {
    fn visit_expr(&mut self, expr: &Expr) -> Result<()> {
        walk_expr(self, expr)
    }

    fn visit_stmt(&mut self, stmt: &Stmt) -> Result<()> {
        walk_stmt(self, stmt)
    }

    fn visit_expr_array(&mut self, values: &tvm_ffi::Array<Expr>) -> Result<()> {
        for value in values {
            self.visit_expr(&value)?;
        }
        Ok(())
    }

    fn visit_buffer_def(&mut self, buffer: &Buffer, _alloc_data: bool) -> Result<()> {
        self.visit_expr_array(&buffer.shape)?;
        if buffer.strides.is_defined() {
            self.visit_expr_array(&buffer.strides)?;
        }
        self.visit_expr(&buffer.elem_offset)
    }

    fn visit_buffer_use(&mut self, _buffer: &Buffer) -> Result<()> {
        Ok(())
    }

    fn visit_buffer_region(&mut self, region: &BufferRegion) -> Result<()> {
        self.visit_buffer_use(&region.buffer)?;
        for range in &region.region {
            self.visit_range(&range, "BufferRegion::region")?;
        }
        Ok(())
    }

    /// Visit a range while diagnosing C++'s nullable `ObjectRef` state.
    ///
    /// `Range` is not an `Option` in generated layouts, but some C++ fields
    /// (notably `IterVar::dom`) may contain an undefined handle.  Checking here
    /// keeps the visitor's `Result` contract instead of triggering a surprising
    /// null-dereference panic through `Deref`.
    fn visit_range(&mut self, range: &Range, context: &str) -> Result<()> {
        if range.is_null() {
            return Err(Error::new(
                tvm_ffi::error::TYPE_ERROR,
                &format!("cannot visit {context}: Range ObjectRef is undefined"),
                "",
            ));
        }
        self.visit_expr(&range.min)?;
        self.visit_expr(&range.extent)
    }

    fn visit_iter_var_domain(&mut self, iter_var: &IterVar, context: &str) -> Result<()> {
        self.visit_range(&iter_var.dom, context)
    }

    fn visit_var(&mut self, _node: &VarObj) -> Result<()> {
        Ok(())
    }

    fn visit_buffer_load(&mut self, node: &BufferLoadObj) -> Result<()> {
        self.visit_buffer_use(&node.buffer)?;
        self.visit_expr_array(&node.indices)
    }

    fn visit_producer_load(&mut self, node: &ProducerLoadObj) -> Result<()> {
        self.visit_expr_array(&node.indices)
    }

    fn visit_let(&mut self, node: &LetObj) -> Result<()> {
        self.visit_expr(&node.value)?;
        self.visit_expr(&node.body)
    }

    fn visit_call(&mut self, node: &CallObj) -> Result<()> {
        self.visit_expr_array(&node.args)
    }

    binary_visit_methods!(
        visit_add(AddObj),
        visit_sub(SubObj),
        visit_mul(MulObj),
        visit_div(DivObj),
        visit_mod(ModObj),
        visit_floor_div(FloorDivObj),
        visit_floor_mod(FloorModObj),
        visit_min(MinObj),
        visit_max(MaxObj),
        visit_eq(EQObj),
        visit_ne(NEObj),
        visit_lt(LTObj),
        visit_le(LEObj),
        visit_gt(GTObj),
        visit_ge(GEObj),
        visit_and(AndObj),
        visit_or(OrObj),
    );

    fn visit_reduce(&mut self, node: &ReduceObj) -> Result<()> {
        for axis in &node.axis {
            self.visit_iter_var_domain(&axis, "Reduce::axis.dom")?;
        }
        self.visit_expr_array(&node.source)?;
        self.visit_expr_array(&node.init)?;
        self.visit_expr(&node.condition)
    }

    fn visit_cast(&mut self, node: &CastObj) -> Result<()> {
        self.visit_expr(&node.value)
    }

    fn visit_not(&mut self, node: &NotObj) -> Result<()> {
        self.visit_expr(&node.a)
    }

    fn visit_select(&mut self, node: &SelectObj) -> Result<()> {
        self.visit_expr(&node.condition)?;
        self.visit_expr(&node.true_value)?;
        self.visit_expr(&node.false_value)
    }

    fn visit_ramp(&mut self, node: &RampObj) -> Result<()> {
        self.visit_expr(&node.base)?;
        self.visit_expr(&node.stride)
    }

    fn visit_broadcast(&mut self, node: &BroadcastObj) -> Result<()> {
        self.visit_expr(&node.value)
    }

    fn visit_shuffle(&mut self, node: &ShuffleObj) -> Result<()> {
        self.visit_expr_array(&node.indices)?;
        self.visit_expr_array(&node.vectors)
    }

    fn visit_int_imm(&mut self, _node: &IntImmObj) -> Result<()> {
        Ok(())
    }

    fn visit_float_imm(&mut self, _node: &FloatImmObj) -> Result<()> {
        Ok(())
    }

    fn visit_string_imm(&mut self, _node: &StringImmObj) -> Result<()> {
        Ok(())
    }

    fn visit_bind(&mut self, node: &BindObj) -> Result<()> {
        self.visit_expr(&node.value)
    }

    fn visit_attr_stmt(&mut self, node: &AttrStmtObj) -> Result<()> {
        self.visit_expr(&node.value)?;
        self.visit_stmt(&node.body)
    }

    fn visit_if_then_else(&mut self, node: &IfThenElseObj) -> Result<()> {
        self.visit_expr(&node.condition)?;
        self.visit_stmt(&node.then_case)?;
        if let Some(else_case) = node.else_case.get() {
            self.visit_stmt(&else_case)?;
        }
        Ok(())
    }

    fn visit_for(&mut self, node: &ForObj) -> Result<()> {
        self.visit_expr(&node.min)?;
        self.visit_expr(&node.extent)?;
        if let Some(step) = node.step.get() {
            self.visit_expr(&step)?;
        }
        self.visit_stmt(&node.body)
    }

    fn visit_while(&mut self, node: &WhileObj) -> Result<()> {
        self.visit_expr(&node.condition)?;
        self.visit_stmt(&node.body)
    }

    fn visit_break(&mut self, _node: &BreakObj) -> Result<()> {
        Ok(())
    }

    fn visit_continue(&mut self, _node: &ContinueObj) -> Result<()> {
        Ok(())
    }

    fn visit_alloc_buffer(&mut self, node: &AllocBufferObj) -> Result<()> {
        self.visit_buffer_def(&node.buffer, true)
    }

    fn visit_decl_buffer(&mut self, node: &DeclBufferObj) -> Result<()> {
        self.visit_buffer_def(&node.buffer, false)
    }

    fn visit_buffer_store(&mut self, node: &BufferStoreObj) -> Result<()> {
        self.visit_buffer_use(&node.buffer)?;
        self.visit_expr(&node.value)?;
        self.visit_expr_array(&node.indices)
    }

    fn visit_assert_stmt(&mut self, node: &AssertStmtObj) -> Result<()> {
        self.visit_expr(&node.condition)?;
        let error_kind: Expr = node.error_kind.clone().into();
        self.visit_expr(&error_kind)?;
        for part in &node.message_parts {
            let part: Expr = part.into();
            self.visit_expr(&part)?;
        }
        Ok(())
    }

    fn visit_seq_stmt(&mut self, node: &SeqStmtObj) -> Result<()> {
        for stmt in &node.seq {
            self.visit_stmt(&stmt)?;
        }
        Ok(())
    }

    fn visit_evaluate(&mut self, node: &EvaluateObj) -> Result<()> {
        self.visit_expr(&node.value)
    }

    fn visit_sblock(&mut self, node: &SBlockObj) -> Result<()> {
        for iter_var in &node.iter_vars {
            self.visit_iter_var_domain(&iter_var, "SBlock::iter_vars.dom")?;
        }
        for buffer in &node.alloc_buffers {
            self.visit_buffer_def(&buffer, true)?;
        }
        for region in &node.reads {
            self.visit_buffer_region(&region)?;
        }
        for region in &node.writes {
            self.visit_buffer_region(&region)?;
        }
        for match_buffer in &node.match_buffers {
            self.visit_buffer_def(&match_buffer.buffer, true)?;
            self.visit_buffer_region(&match_buffer.source)?;
        }
        if let Some(init) = node.init.get() {
            self.visit_stmt(&init)?;
        }
        self.visit_stmt(&node.body)
    }

    fn visit_sblock_realize(&mut self, node: &SBlockRealizeObj) -> Result<()> {
        self.visit_expr_array(&node.iter_values)?;
        self.visit_expr(&node.predicate)?;
        let block: Stmt = node.block.clone().into();
        self.visit_stmt(&block)
    }

    fn visit_scope_id_def_stmt(&mut self, node: &ScopeIdDefStmtObj) -> Result<()> {
        if let Some(extents) = node.def.extents.get() {
            self.visit_expr_array(&extents)?;
        }
        if let Some(extents) = node.def.preferred_extents.get() {
            self.visit_expr_array(&extents)?;
        }
        Ok(())
    }

    fn visit_tile_primitive_call(&mut self, _node: &TilePrimitiveCallObj) -> Result<()> {
        // C++ declares args/config as heterogeneous Any containers, but the
        // generated markers incorrectly narrow them to ObjectRef.  Iterating
        // those markers can silently stop on a scalar Array item or panic on a
        // scalar Map value.  Refuse the node until stubgen emits real Any
        // container support; an explicit pass override may handle it through a
        // dedicated FFI helper in the meantime.
        Err(Error::new(
            tvm_ffi::error::TYPE_ERROR,
            "TilePrimitiveCall traversal requires Array<Any>/Map<String, Any>; generated bindings currently narrow these fields to ObjectRef",
            "",
        ))
    }
}
