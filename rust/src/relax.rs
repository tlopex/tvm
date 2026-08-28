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
use tvm_ffi::{Array, Error, Map, ObjectArc, ObjectRefCast, Result, TYPE_ERROR};

use crate::ir::{BaseFunc, BaseFuncObj, DictAttrs, Expr, ExprObj, Span, TupleType, Type, Var};
/// ABI-complete Rust representation of the common IR tuple used by Relax.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Tuple"]
#[type_final]
pub struct TupleObj {
    base: ExprObj,
    pub fields: Array<Expr>,
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

impl Tuple {
    /// Construct a Relax tuple directly in Rust and derive its tuple type when possible.
    pub fn new(fields: Vec<Expr>) -> Self {
        Self::with_span(fields, None)
    }

    /// Construct a Relax tuple with optional source metadata.
    pub fn with_span(fields: Vec<Expr>, span: Option<&Span>) -> Self {
        let tuple_type = fields
            .iter()
            .map(|field| field.ty.clone())
            .collect::<Vec<_>>();
        let tuple_type = if tuple_type.iter().any(Type::is_missing) {
            Type::missing()
        } else {
            TupleType::new(tuple_type).into()
        };
        Self::from_complete_fields(span.cloned(), tuple_type, Array::new(fields))
    }

    /// Construct a tuple from every physical field without re-deriving its type.
    pub fn from_complete_fields(span: Option<Span>, ty: Type, fields: Array<Expr>) -> Self {
        Self {
            data: ObjectArc::new(TupleObj {
                base: ExprObj::new(span, ty),
                fields,
            }),
        }
    }
}

/// ABI-complete Rust representation of Relax's DAG-valued conditional expression.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.If"]
#[type_final]
pub struct IfObj {
    base: ExprObj,
    pub cond: Expr,
    pub true_branch: SeqExpr,
    pub false_branch: SeqExpr,
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
    /// Construct a Relax conditional directly in Rust, wrapping each branch in `SeqExpr`.
    pub fn new<C, T, F>(condition: C, true_branch: T, false_branch: F) -> Self
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
    ) -> Self
    where
        C: Into<Expr>,
        T: Into<Expr>,
        F: Into<Expr>,
    {
        let condition = condition.into();
        let true_branch = SeqExpr::from_expr(true_branch);
        let false_branch = SeqExpr::from_expr(false_branch);
        Self::from_complete_fields(
            span.cloned(),
            Type::missing(),
            condition,
            true_branch,
            false_branch,
        )
    }

    /// Construct a conditional from every physical field without normalizing its branches.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: Type,
        cond: Expr,
        true_branch: SeqExpr,
        false_branch: SeqExpr,
    ) -> Self {
        Self {
            data: ObjectArc::new(IfObj {
                base: ExprObj::new(span, ty),
                cond,
                true_branch,
                false_branch,
            }),
        }
    }
}

/// ABI-complete Rust representation of Relax's `BindingNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.Binding"]
pub struct BindingObj {
    base: tvm_ffi::Object,
    pub span: Option<Span>,
    pub var: Var,
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

/// ABI-complete Rust representation of a Relax variable binding.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.VarBinding"]
#[type_final]
pub struct VarBindingObj {
    base: BindingObj,
    pub value: Expr,
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

impl VarBinding {
    /// Construct `variable = value` directly in Rust.
    pub fn new<V, E>(variable: V, value: E) -> Self
    where
        V: Into<Var>,
        E: Into<Expr>,
    {
        Self::with_span(variable, value, None)
    }

    /// Construct `variable = value` with optional source metadata.
    pub fn with_span<V, E>(variable: V, value: E, span: Option<&Span>) -> Self
    where
        V: Into<Var>,
        E: Into<Expr>,
    {
        Self::from_complete_fields(span.cloned(), variable.into(), value.into())
    }

    /// Construct a variable binding from every physical field.
    pub fn from_complete_fields(span: Option<Span>, var: Var, value: Expr) -> Self {
        Self {
            data: ObjectArc::new(VarBindingObj {
                base: BindingObj {
                    base: tvm_ffi::Object::new(),
                    span,
                    var,
                },
                value,
            }),
        }
    }
}

/// ABI-complete Rust representation of a Relax binding block.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.BindingBlock"]
pub struct BindingBlockObj {
    base: tvm_ffi::Object,
    pub bindings: Array<Binding>,
    pub span: Option<Span>,
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

impl BindingBlock {
    /// Construct an ordinary (non-dataflow) binding block directly in Rust.
    pub fn new(bindings: Vec<Binding>) -> Self {
        Self::with_span(bindings, None)
    }

    /// Construct an ordinary binding block with optional source metadata.
    pub fn with_span(bindings: Vec<Binding>, span: Option<&Span>) -> Self {
        Self::from_complete_fields(Array::new(bindings), span.cloned())
    }

    /// Construct a binding block from every physical field.
    pub fn from_complete_fields(bindings: Array<Binding>, span: Option<Span>) -> Self {
        Self {
            data: ObjectArc::new(BindingBlockObj {
                base: tvm_ffi::Object::new(),
                bindings,
                span,
            }),
        }
    }
}

/// ABI-complete Rust representation of a Relax sequence expression.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.SeqExpr"]
#[type_final]
pub struct SeqExprObj {
    base: ExprObj,
    pub blocks: Array<BindingBlock>,
    pub body: Expr,
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

impl SeqExpr {
    /// Construct ordered binding blocks followed by `body` directly in Rust.
    pub fn new<E>(blocks: Vec<BindingBlock>, body: E) -> Self
    where
        E: Into<Expr>,
    {
        Self::with_span(blocks, body, None)
    }

    /// Construct a sequence expression with optional source metadata.
    pub fn with_span<E>(blocks: Vec<BindingBlock>, body: E, span: Option<&Span>) -> Self
    where
        E: Into<Expr>,
    {
        Self::from_complete_fields(
            span.cloned(),
            Type::missing(),
            Array::new(blocks),
            body.into(),
        )
    }

    /// Construct a sequence expression from every physical field without re-deriving its type.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: Type,
        blocks: Array<BindingBlock>,
        body: Expr,
    ) -> Self {
        Self {
            data: ObjectArc::new(SeqExprObj {
                base: ExprObj::new(span, ty),
                blocks,
                body,
            }),
        }
    }

    fn from_expr<E>(body: E) -> Self
    where
        E: Into<Expr>,
    {
        let body = body.into();
        match body.clone().try_cast::<Self>() {
            Ok(sequence) => sequence,
            Err(_) => Self::new(Vec::new(), body),
        }
    }
}

/// ABI-complete Rust representation of a Relax function.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.Function"]
#[type_final]
pub struct RelaxFunctionObj {
    base: BaseFuncObj,
    pub params: Array<Var>,
    pub body: SeqExpr,
    pub ret_ty: Type,
    pub is_pure: bool,
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

impl RelaxFunction {
    /// Construct a typed Relax function using Rust-controlled normalization.
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
            DictAttrs::empty(),
            None,
        )
    }

    /// Construct a Relax function using the same native analysis services as C++.
    pub fn with_metadata(
        params: Vec<Var>,
        body: Expr,
        return_type: Option<Type>,
        is_pure: bool,
        attrs: DictAttrs,
        span: Option<&Span>,
    ) -> Result<Self> {
        let params = Array::new(params);
        let (body, return_type, function_type) =
            derive_function_fields(&params, body, return_type, is_pure)?;
        Ok(Self::from_complete_fields(
            span.cloned(),
            function_type,
            attrs,
            params,
            body,
            return_type,
            is_pure,
        ))
    }

    /// Construct a Relax function allocation entirely in Rust from its complete state.
    ///
    /// The caller supplies the already-derived function type and normalized
    /// `SeqExpr` body.  This is the lossless constructor shape stubgen can emit
    /// without embedding Relax's type-analysis algorithm in generated code.
    #[allow(clippy::too_many_arguments)]
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: Type,
        attrs: DictAttrs,
        params: Array<Var>,
        body: SeqExpr,
        ret_ty: Type,
        is_pure: bool,
    ) -> Self {
        Self {
            data: ObjectArc::new(RelaxFunctionObj {
                base: BaseFuncObj::new(span, ty, attrs),
                params,
                body,
                ret_ty,
                is_pure,
            }),
        }
    }
}

fn derive_function_fields(
    params: &Array<Var>,
    body: Expr,
    mut return_type: Option<Type>,
    is_pure: bool,
) -> Result<(SeqExpr, Type, Type)> {
    let mut parameter_types = Vec::with_capacity(params.len());
    for parameter in params.iter() {
        if parameter.ty.is_missing() {
            return Err(Error::new(
                TYPE_ERROR,
                "relax.Function requires every parameter to have a type",
                "",
            ));
        }
        parameter_types.push(parameter.ty.clone());
    }
    let parameter_types = Array::new(parameter_types);
    let body_type = (!body.ty.is_missing()).then(|| body.ty.clone());
    if body_type.is_none() && return_type.is_none() {
        return Err(Error::new(
            TYPE_ERROR,
            "Relax function needs an explicit return type or a typed body",
            "",
        ));
    }

    let use_body_type = match (&return_type, &body_type) {
        (None, Some(_)) => true,
        (Some(explicit), Some(inferred)) => type_is_base_of(explicit, inferred)?,
        _ => false,
    };
    if use_body_type {
        let body_type = body_type.expect("use_body_type requires a typed body");
        let parameter_tuple: Type = TupleType::new(parameter_types.iter().collect()).into();
        let definable: Array<Var> =
            tvm_ffi::cached_global_func!("relax.analysis.DefinableTIRVarsInType")
                .call_tuple((&parameter_tuple,))?
                .try_into()?;
        let shape_variables = Map::from_iter(
            definable
                .iter()
                .map(|variable| (variable.clone(), Expr::from(variable))),
        );
        let relax_variables = Map::<Var, Expr>::new();
        return_type = Some(
            tvm_ffi::cached_global_func!("relax.analysis.EraseToWellDefined")
                .call_tuple((&body_type, &shape_variables, &relax_variables))?
                .try_into()?,
        );
    }

    let return_type = return_type.expect("a return type was established above");
    let function_type = make_func_type(parameter_types, return_type.clone(), is_pure)?;
    Ok((SeqExpr::from_expr(body), return_type, function_type))
}

fn type_is_base_of(base: &Type, derived: &Type) -> Result<bool> {
    tvm_ffi::cached_global_func!("relax.TypeIsBaseOf")
        .call_tuple((base, derived))?
        .try_into()
}

pub(crate) fn make_func_type(params: Array<Type>, ret: Type, purity: bool) -> Result<Type> {
    tvm_ffi::cached_global_func!("relax.FuncType")
        .call_tuple((params, ret, purity, Option::<Span>::None))?
        .try_into()
}

pub(crate) fn make_any_type() -> Result<Type> {
    tvm_ffi::cached_global_func!("relax.AnyType")
        .call_tuple((Option::<Span>::None,))?
        .try_into()
}

pub(crate) fn make_shape_expr(values: Array<Expr>) -> Result<Expr> {
    tvm_ffi::cached_global_func!("relax.ShapeExpr")
        .call_tuple((values, Option::<Span>::None))?
        .try_into()
}

pub(crate) fn make_tensor_type(shape: Expr, dtype: crate::ir::PrimType) -> Result<Type> {
    tvm_ffi::cached_global_func!("relax.TensorType")
        .call_tuple((Some(shape), Some(dtype), -1_i32, (), Option::<Span>::None))?
        .try_into()
}

tvm_ffi::impl_object_upcast!(
    Tuple => Expr,
    If => Expr,
    VarBinding => Binding,
    SeqExpr => Expr,
    RelaxFunction => BaseFunc,
    RelaxFunction => Expr,
);
