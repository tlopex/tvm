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

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{AnyView, Array, ObjectArc, ObjectRefCast, Result};

use crate::ir::{BaseFunc, BaseFuncObj, DictAttrs, Expr, ExprObj, Type, Var};
/// Opaque Rust view of Relax's tuple expression.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.Tuple"]
#[type_final]
pub struct TupleObj {
    base: ExprObj,
}

/// Reference-counted handle to a Relax tuple expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Tuple {
    data: ObjectArc<TupleObj>,
}

impl std::ops::Deref for Tuple {
    type Target = TupleObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for TupleObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl TupleObj {
    /// Return the tuple fields.
    pub fn fields(&self) -> Result<Array<Expr>> {
        crate::reflected_field!(self, "fields")?.try_into()
    }
}

impl Tuple {
    /// Construct a Relax tuple.
    pub fn new(fields: Vec<Expr>) -> Result<Self> {
        let fields = Array::new(fields);
        let none = ();
        crate::global_function!("relax.Tuple")?
            .call_packed(&[AnyView::from(&fields), AnyView::from(&none)])?
            .try_into()
    }
}

impl From<Tuple> for Expr {
    fn from(value: Tuple) -> Self {
        value
            .try_cast()
            .expect("relax.expr.Tuple must be a subtype of ir.Expr")
    }
}

/// Opaque Rust view of Relax's DAG-valued conditional expression.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.If"]
#[type_final]
pub struct IfObj {
    base: ExprObj,
}

/// Reference-counted handle to a Relax conditional expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct If {
    data: ObjectArc<IfObj>,
}

impl std::ops::Deref for If {
    type Target = IfObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for IfObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl If {
    /// Construct a Relax conditional.  TVM wraps each branch in `SeqExpr`.
    pub fn new(condition: &Expr, true_branch: &Expr, false_branch: &Expr) -> Result<Self> {
        let none = ();
        crate::global_function!("relax.If")?
            .call_packed(&[
                AnyView::from(condition),
                AnyView::from(true_branch),
                AnyView::from(false_branch),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

impl From<If> for Expr {
    fn from(value: If) -> Self {
        value
            .try_cast()
            .expect("relax.expr.If must be a subtype of ir.Expr")
    }
}

/// Opaque Rust view of Relax's `BindingNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.Binding"]
pub struct BindingObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to any Relax binding.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Binding {
    data: ObjectArc<BindingObj>,
}

impl std::ops::Deref for Binding {
    type Target = BindingObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl BindingObj {
    /// Return the variable defined by this binding.
    pub fn variable(&self) -> Result<Var> {
        crate::reflected_field!(self, "var")?.try_into()
    }
}

/// Opaque Rust view of a Relax variable binding.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.VarBinding"]
#[type_final]
pub struct VarBindingObj {
    base: BindingObj,
}

/// Reference-counted handle to `var = value` in Relax.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct VarBinding {
    data: ObjectArc<VarBindingObj>,
}

impl std::ops::Deref for VarBinding {
    type Target = VarBindingObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for VarBindingObj {
    type Target = BindingObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl VarBindingObj {
    /// Return the expression assigned to the binding variable.
    pub fn value(&self) -> Result<Expr> {
        crate::reflected_field!(self, "value")?.try_into()
    }
}

impl VarBinding {
    /// Construct `variable = value`.
    pub fn new(variable: &Var, value: &Expr) -> Result<Self> {
        let none = ();
        crate::global_function!("relax.VarBinding")?
            .call_packed(&[
                AnyView::from(variable),
                AnyView::from(value),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

impl From<VarBinding> for Binding {
    fn from(value: VarBinding) -> Self {
        value
            .try_cast()
            .expect("relax.expr.VarBinding must be a subtype of relax.expr.Binding")
    }
}

/// Opaque Rust view of a Relax binding block.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.BindingBlock"]
pub struct BindingBlockObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to an ordered Relax binding block.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct BindingBlock {
    data: ObjectArc<BindingBlockObj>,
}

impl std::ops::Deref for BindingBlock {
    type Target = BindingBlockObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl BindingBlockObj {
    /// Return bindings in evaluation order.
    pub fn bindings(&self) -> Result<Array<Binding>> {
        crate::reflected_field!(self, "bindings")?.try_into()
    }
}

impl BindingBlock {
    /// Construct an ordinary (non-dataflow) binding block.
    pub fn new(bindings: Vec<Binding>) -> Result<Self> {
        let bindings = Array::new(bindings);
        let none = ();
        crate::global_function!("relax.BindingBlock")?
            .call_packed(&[AnyView::from(&bindings), AnyView::from(&none)])?
            .try_into()
    }
}

/// Opaque Rust view of a Relax sequence expression.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.SeqExpr"]
#[type_final]
pub struct SeqExprObj {
    base: ExprObj,
}

/// Reference-counted handle to ordered bindings followed by a body expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct SeqExpr {
    data: ObjectArc<SeqExprObj>,
}

impl std::ops::Deref for SeqExpr {
    type Target = SeqExprObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for SeqExprObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl SeqExprObj {
    /// Return binding blocks in evaluation order.
    pub fn blocks(&self) -> Result<Array<BindingBlock>> {
        crate::reflected_field!(self, "blocks")?.try_into()
    }

    /// Return the value produced after all binding blocks execute.
    pub fn body(&self) -> Result<Expr> {
        crate::reflected_field!(self, "body")?.try_into()
    }
}

impl SeqExpr {
    /// Construct ordered binding blocks followed by `body`.
    pub fn new(blocks: Vec<BindingBlock>, body: &Expr) -> Result<Self> {
        let blocks = Array::new(blocks);
        let none = ();
        crate::global_function!("relax.SeqExpr")?
            .call_packed(&[
                AnyView::from(&blocks),
                AnyView::from(body),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

impl From<SeqExpr> for Expr {
    fn from(value: SeqExpr) -> Self {
        value
            .try_cast()
            .expect("relax.expr.SeqExpr must be a subtype of ir.Expr")
    }
}

/// Opaque Rust view of a Relax function.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.Function"]
#[type_final]
pub struct RelaxFunctionObj {
    base: BaseFuncObj,
}

/// Reference-counted handle to a Relax function.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct RelaxFunction {
    data: ObjectArc<RelaxFunctionObj>,
}

impl std::ops::Deref for RelaxFunction {
    type Target = RelaxFunctionObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for RelaxFunctionObj {
    type Target = BaseFuncObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl RelaxFunctionObj {
    /// Return recursively defined function parameters.
    pub fn params(&self) -> Result<Array<Var>> {
        crate::reflected_field!(self, "params")?.try_into()
    }

    /// Return the function's sequence-expression body.
    pub fn body(&self) -> Result<SeqExpr> {
        crate::reflected_field!(self, "body")?.try_into()
    }

    /// Return the declared result type.
    pub fn return_type(&self) -> Result<Type> {
        crate::reflected_field!(self, "ret_ty")?.try_into()
    }

    /// Return whether the function is declared pure.
    pub fn is_pure(&self) -> Result<bool> {
        crate::reflected_field!(self, "is_pure")?.try_into()
    }
}

impl RelaxFunction {
    /// Construct a typed Relax function.
    pub fn new(params: Vec<Var>, body: &Expr, return_type: &Type, is_pure: bool) -> Result<Self> {
        let params = Array::new(params);
        let return_type = Some(return_type.clone());
        let attrs = DictAttrs::empty()?;
        let none = ();
        crate::global_function!("relax.Function")?
            .call_packed(&[
                AnyView::from(&params),
                AnyView::from(body),
                AnyView::from(&return_type),
                AnyView::from(&is_pure),
                AnyView::from(&attrs),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

impl From<RelaxFunction> for BaseFunc {
    fn from(value: RelaxFunction) -> Self {
        value
            .try_cast()
            .expect("relax.expr.Function must be a subtype of ir.BaseFunc")
    }
}

impl From<RelaxFunction> for Expr {
    fn from(value: RelaxFunction) -> Self {
        let function: BaseFunc = value.into();
        function
            .try_cast()
            .expect("ir.BaseFunc must be a subtype of ir.Expr")
    }
}
