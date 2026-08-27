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
use tvm_ffi::{Array, ObjectArc, Result};

use crate::ir::{BaseFunc, BaseFuncObj, DictAttrs, Expr, ExprObj, Span, Type, Var};
/// Opaque Rust view of the common IR tuple used by Relax.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Tuple"]
#[type_final]
pub struct TupleObj {
    base: ExprObj,
}
crate::abi::reflected_fields!(TupleObj {
    fields => "fields": Array<Expr>,
});

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

impl Tuple {
    /// Construct a Relax tuple through TVM's native constructor.
    pub fn new(fields: Vec<Expr>) -> Result<Self> {
        Self::with_span(fields, None)
    }

    /// Construct a Relax tuple with optional source metadata.
    pub fn with_span(fields: Vec<Expr>, span: Option<&Span>) -> Result<Self> {
        tvm_ffi::cached_global_func!("relax.Tuple")
            .call_tuple((Array::new(fields), span.cloned()))?
            .try_into()
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
crate::abi::reflected_fields!(IfObj {
    cond => "cond": Expr,
    true_branch => "true_branch": SeqExpr,
    false_branch => "false_branch": SeqExpr,
});

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
    /// Construct a Relax conditional through TVM's native constructor.
    pub fn new<C, T, F>(condition: C, true_branch: T, false_branch: F) -> Result<Self>
    where
        C: Into<Expr>,
        T: Into<Expr>,
        F: Into<Expr>,
    {
        Self::with_span(condition, true_branch, false_branch, None)
    }

    /// Construct a Relax conditional with optional source metadata.
    pub fn with_span<C, T, F>(
        condition: C,
        true_branch: T,
        false_branch: F,
        span: Option<&Span>,
    ) -> Result<Self>
    where
        C: Into<Expr>,
        T: Into<Expr>,
        F: Into<Expr>,
    {
        tvm_ffi::cached_global_func!("relax.If")
            .call_tuple((
                condition.into(),
                true_branch.into(),
                false_branch.into(),
                span.cloned(),
            ))?
            .try_into()
    }
}

/// Opaque Rust view of Relax's `BindingNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.Binding"]
pub struct BindingObj {
    base: tvm_ffi::Object,
}
crate::abi::reflected_fields!(BindingObj {
    span => "span": Option<Span>,
    var => "var": Var,
});

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

/// Opaque Rust view of a Relax variable binding.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.VarBinding"]
#[type_final]
pub struct VarBindingObj {
    base: BindingObj,
}
crate::abi::reflected_fields!(VarBindingObj {
    value => "value": Expr,
});

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

impl VarBinding {
    /// Construct `variable = value` through TVM's native constructor.
    pub fn new<V, E>(variable: V, value: E) -> Result<Self>
    where
        V: Into<Var>,
        E: Into<Expr>,
    {
        Self::with_span(variable, value, None)
    }

    /// Construct `variable = value` with optional source metadata.
    pub fn with_span<V, E>(variable: V, value: E, span: Option<&Span>) -> Result<Self>
    where
        V: Into<Var>,
        E: Into<Expr>,
    {
        tvm_ffi::cached_global_func!("relax.VarBinding")
            .call_tuple((variable.into(), value.into(), span.cloned()))?
            .try_into()
    }
}

/// Opaque Rust view of a Relax binding block.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.BindingBlock"]
pub struct BindingBlockObj {
    base: tvm_ffi::Object,
}
crate::abi::reflected_fields!(BindingBlockObj {
    bindings => "bindings": Array<Binding>,
    span => "span": Option<Span>,
});

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

impl BindingBlock {
    /// Construct an ordinary binding block through TVM's native constructor.
    pub fn new(bindings: Vec<Binding>) -> Result<Self> {
        Self::with_span(bindings, None)
    }

    /// Construct an ordinary binding block with optional source metadata.
    pub fn with_span(bindings: Vec<Binding>, span: Option<&Span>) -> Result<Self> {
        tvm_ffi::cached_global_func!("relax.BindingBlock")
            .call_tuple((Array::new(bindings), span.cloned()))?
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
crate::abi::reflected_fields!(SeqExprObj {
    blocks => "blocks": Array<BindingBlock>,
    body => "body": Expr,
});

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

impl SeqExpr {
    /// Construct ordered binding blocks followed by `body` through TVM's native constructor.
    pub fn new<E>(blocks: Vec<BindingBlock>, body: E) -> Result<Self>
    where
        E: Into<Expr>,
    {
        Self::with_span(blocks, body, None)
    }

    /// Construct a sequence expression with optional source metadata.
    pub fn with_span<E>(blocks: Vec<BindingBlock>, body: E, span: Option<&Span>) -> Result<Self>
    where
        E: Into<Expr>,
    {
        tvm_ffi::cached_global_func!("relax.SeqExpr")
            .call_tuple((Array::new(blocks), body.into(), span.cloned()))?
            .try_into()
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
crate::abi::reflected_fields!(RelaxFunctionObj {
    params => "params": Array<Var>,
    body => "body": SeqExpr,
    ret_ty => "ret_ty": Type,
    is_pure => "is_pure": bool,
});

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

impl RelaxFunction {
    /// Construct a typed Relax function through TVM's native constructor.
    pub fn new<B, T>(params: Vec<Var>, body: B, return_type: T, is_pure: bool) -> Result<Self>
    where
        B: Into<Expr>,
        T: Into<Type>,
    {
        Self::with_metadata(
            params,
            body.into(),
            Some(return_type.into()),
            is_pure,
            DictAttrs::empty()?,
            None,
        )
    }

    /// Construct a Relax function through TVM's native constructor.
    pub fn with_metadata(
        params: Vec<Var>,
        body: Expr,
        return_type: Option<Type>,
        is_pure: bool,
        attrs: DictAttrs,
        span: Option<&Span>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("relax.Function")
            .call_tuple((
                Array::new(params),
                body,
                return_type,
                is_pure,
                attrs,
                span.cloned(),
            ))?
            .try_into()
    }
}

tvm_ffi::impl_object_upcast!(
    Tuple => Expr,
    If => Expr,
    VarBinding => Binding,
    SeqExpr => Expr,
    RelaxFunction => BaseFunc,
    RelaxFunction => Expr,
);
