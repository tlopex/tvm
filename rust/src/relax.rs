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

use crate::ir::{BaseFunc, BaseFuncObj, DictAttrs, Expr, ExprObj, Span, Type, Var};
/// ABI-complete Rust representation of Relax's tuple expression.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.Tuple"]
#[type_final]
pub struct TupleObj {
    base: ExprObj,
    fields: Array<Expr>,
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
        Ok(self.fields.clone())
    }
}

impl Tuple {
    /// Construct a Relax tuple through C++ so its derived tuple type is preserved.
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

/// ABI-complete Rust representation of Relax's DAG-valued conditional expression.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.If"]
#[type_final]
pub struct IfObj {
    base: ExprObj,
    cond: Expr,
    true_branch: SeqExpr,
    false_branch: SeqExpr,
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

impl IfObj {
    /// Return the conditional expression.
    pub fn condition(&self) -> Result<Expr> {
        Ok(self.cond.clone())
    }

    /// Return the sequence evaluated when the condition is true.
    pub fn true_branch(&self) -> Result<SeqExpr> {
        Ok(self.true_branch.clone())
    }

    /// Return the sequence evaluated when the condition is false.
    pub fn false_branch(&self) -> Result<SeqExpr> {
        Ok(self.false_branch.clone())
    }
}

impl If {
    /// Construct a Relax conditional directly in Rust, wrapping each branch in `SeqExpr`.
    pub fn new(condition: &Expr, true_branch: &Expr, false_branch: &Expr) -> Result<Self> {
        Ok(Self {
            data: ObjectArc::new(IfObj {
                base: ExprObj::new(Type::missing()?, None),
                cond: condition.clone(),
                true_branch: SeqExpr::from_expr(true_branch)?,
                false_branch: SeqExpr::from_expr(false_branch)?,
            }),
        })
    }
}

impl From<If> for Expr {
    fn from(value: If) -> Self {
        value
            .try_cast()
            .expect("relax.expr.If must be a subtype of ir.Expr")
    }
}

/// ABI-complete Rust representation of Relax's `BindingNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.Binding"]
pub struct BindingObj {
    base: tvm_ffi::Object,
    span: Option<Span>,
    var: Var,
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
    /// Return optional source metadata carried by this binding.
    pub fn span(&self) -> Result<Option<crate::ir::Span>> {
        Ok(self.span.clone())
    }

    /// Return the variable defined by this binding.
    pub fn variable(&self) -> Result<Var> {
        Ok(self.var.clone())
    }
}

/// ABI-complete Rust representation of a Relax variable binding.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.VarBinding"]
#[type_final]
pub struct VarBindingObj {
    base: BindingObj,
    value: Expr,
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
        Ok(self.value.clone())
    }
}

impl VarBinding {
    /// Construct `variable = value` directly in Rust.
    pub fn new(variable: &Var, value: &Expr) -> Result<Self> {
        Ok(Self {
            data: ObjectArc::new(VarBindingObj {
                base: BindingObj {
                    base: tvm_ffi::Object::new(),
                    span: None,
                    var: variable.clone(),
                },
                value: value.clone(),
            }),
        })
    }
}

impl From<VarBinding> for Binding {
    fn from(value: VarBinding) -> Self {
        value
            .try_cast()
            .expect("relax.expr.VarBinding must be a subtype of relax.expr.Binding")
    }
}

/// ABI-complete Rust representation of a Relax binding block.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.BindingBlock"]
pub struct BindingBlockObj {
    base: tvm_ffi::Object,
    bindings: Array<Binding>,
    span: Option<Span>,
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
        Ok(self.bindings.clone())
    }

    /// Return optional source metadata carried by this block.
    pub fn span(&self) -> Result<Option<crate::ir::Span>> {
        Ok(self.span.clone())
    }
}

impl BindingBlock {
    /// Construct an ordinary (non-dataflow) binding block directly in Rust.
    pub fn new(bindings: Vec<Binding>) -> Result<Self> {
        Ok(Self {
            data: ObjectArc::new(BindingBlockObj {
                base: tvm_ffi::Object::new(),
                bindings: Array::new(bindings),
                span: None,
            }),
        })
    }
}

/// ABI-complete Rust representation of a Relax sequence expression.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.SeqExpr"]
#[type_final]
pub struct SeqExprObj {
    base: ExprObj,
    blocks: Array<BindingBlock>,
    body: Expr,
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
        Ok(self.blocks.clone())
    }

    /// Return the value produced after all binding blocks execute.
    pub fn body(&self) -> Result<Expr> {
        Ok(self.body.clone())
    }
}

impl SeqExpr {
    /// Construct ordered binding blocks followed by `body` directly in Rust.
    pub fn new(blocks: Vec<BindingBlock>, body: &Expr) -> Result<Self> {
        Ok(Self {
            data: ObjectArc::new(SeqExprObj {
                base: ExprObj::new(Type::missing()?, None),
                blocks: Array::new(blocks),
                body: body.clone(),
            }),
        })
    }

    fn from_expr(body: &Expr) -> Result<Self> {
        match body.clone().try_cast::<Self>() {
            Ok(sequence) => Ok(sequence),
            Err(_) => Self::new(Vec::new(), body),
        }
    }
}

impl From<SeqExpr> for Expr {
    fn from(value: SeqExpr) -> Self {
        value
            .try_cast()
            .expect("relax.expr.SeqExpr must be a subtype of ir.Expr")
    }
}

/// ABI-complete Rust representation of a Relax function.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.Function"]
#[type_final]
pub struct RelaxFunctionObj {
    base: BaseFuncObj,
    params: Array<Var>,
    body: SeqExpr,
    ret_ty: Type,
    is_pure: bool,
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
        Ok(self.params.clone())
    }

    /// Return the function's sequence-expression body.
    pub fn body(&self) -> Result<SeqExpr> {
        Ok(self.body.clone())
    }

    /// Return the declared result type.
    pub fn return_type(&self) -> Result<Type> {
        Ok(self.ret_ty.clone())
    }

    /// Return whether the function is declared pure.
    pub fn is_pure(&self) -> Result<bool> {
        Ok(self.is_pure)
    }
}

impl RelaxFunction {
    /// Construct a typed Relax function through C++ type inference and validation.
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
