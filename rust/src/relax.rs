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
use tvm_ffi::{AnyView, Array, ObjectArc, ObjectCore, ObjectRefCast, Result};

use crate::ir::{BaseFunc, BaseFuncObj, DictAttrs, Expr, ExprObj, Span, TupleType, Type, Var};
/// ABI-complete Rust representation of Relax's tuple expression.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.Tuple"]
#[type_final]
pub struct TupleObj {
    base: ExprObj,
    pub fields: Array<Expr>,
}
crate::abi::impl_object_layout!(TupleObj {
    "fields" => fields: Array<Expr>,
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
            data: crate::abi::allocate_object(TupleObj {
                base: ExprObj::new(span, ty),
                fields,
            }),
        }
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
    pub cond: Expr,
    pub true_branch: SeqExpr,
    pub false_branch: SeqExpr,
}
crate::abi::impl_object_layout!(IfObj {
    "cond" => cond: Expr,
    "true_branch" => true_branch: SeqExpr,
    "false_branch" => false_branch: SeqExpr,
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
    /// Construct a Relax conditional directly in Rust, wrapping each branch in `SeqExpr`.
    pub fn new(condition: &Expr, true_branch: &Expr, false_branch: &Expr) -> Self {
        Self::with_span(condition, true_branch, false_branch, None)
    }

    /// Construct a Relax conditional with optional source metadata.
    pub fn with_span(
        condition: &Expr,
        true_branch: &Expr,
        false_branch: &Expr,
        span: Option<&Span>,
    ) -> Self {
        let true_branch = SeqExpr::from_expr(true_branch);
        let false_branch = SeqExpr::from_expr(false_branch);
        Self::from_complete_fields(
            span.cloned(),
            Type::missing(),
            condition.clone(),
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
            data: crate::abi::allocate_object(IfObj {
                base: ExprObj::new(span, ty),
                cond,
                true_branch,
                false_branch,
            }),
        }
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
    pub span: Option<Span>,
    pub var: Var,
}
crate::abi::impl_object_layout!(BindingObj {
    "span" => span: Option<Span>,
    "var" => var: Var,
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

/// ABI-complete Rust representation of a Relax variable binding.
#[repr(C)]
#[derive(Object)]
#[type_key = "relax.expr.VarBinding"]
#[type_final]
pub struct VarBindingObj {
    base: BindingObj,
    pub value: Expr,
}
crate::abi::impl_object_layout!(VarBindingObj {
    "value" => value: Expr,
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
    /// Construct `variable = value` directly in Rust.
    pub fn new(variable: &Var, value: &Expr) -> Self {
        Self::with_span(variable, value, None)
    }

    /// Construct `variable = value` with optional source metadata.
    pub fn with_span(variable: &Var, value: &Expr, span: Option<&Span>) -> Self {
        Self::from_complete_fields(span.cloned(), variable.clone(), value.clone())
    }

    /// Construct a variable binding from every physical field.
    pub fn from_complete_fields(span: Option<Span>, var: Var, value: Expr) -> Self {
        Self {
            data: crate::abi::allocate_object(VarBindingObj {
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
    pub bindings: Array<Binding>,
    pub span: Option<Span>,
}
crate::abi::impl_object_layout!(BindingBlockObj {
    "bindings" => bindings: Array<Binding>,
    "span" => span: Option<Span>,
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
            data: crate::abi::allocate_object(BindingBlockObj {
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
crate::abi::impl_object_layout!(SeqExprObj {
    "blocks" => blocks: Array<BindingBlock>,
    "body" => body: Expr,
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
    /// Construct ordered binding blocks followed by `body` directly in Rust.
    pub fn new(blocks: Vec<BindingBlock>, body: &Expr) -> Self {
        Self::with_span(blocks, body, None)
    }

    /// Construct a sequence expression with optional source metadata.
    pub fn with_span(blocks: Vec<BindingBlock>, body: &Expr, span: Option<&Span>) -> Self {
        Self::from_complete_fields(
            span.cloned(),
            Type::missing(),
            Array::new(blocks),
            body.clone(),
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
            data: crate::abi::allocate_object(SeqExprObj {
                base: ExprObj::new(span, ty),
                blocks,
                body,
            }),
        }
    }

    fn from_expr(body: &Expr) -> Self {
        match body.clone().try_cast::<Self>() {
            Ok(sequence) => sequence,
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
    pub params: Array<Var>,
    pub body: SeqExpr,
    pub ret_ty: Type,
    pub is_pure: bool,
}
crate::abi::impl_object_layout!(RelaxFunctionObj {
    "params" => params: Array<Var>,
    "body" => body: SeqExpr,
    "ret_ty" => ret_ty: Type,
    "is_pure" => is_pure: bool,
});

impl crate::abi::ConstructorRecipe for RelaxFunctionObj {
    const NUM_INPUTS: usize = 4;
    const DERIVED_FIELDS: &'static [&'static str] = &["body", "ret_ty", "ty"];
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
    /// Construct a typed Relax function in Rust using its C ABI preparation table.
    pub fn new(params: Vec<Var>, body: &Expr, return_type: &Type, is_pure: bool) -> Result<Self> {
        Self::with_metadata(
            params,
            body,
            Some(return_type),
            is_pure,
            &DictAttrs::empty(),
            None,
        )
    }

    /// Construct a Relax function in Rust using its C ABI preparation table.
    pub fn with_metadata(
        params: Vec<Var>,
        body: &Expr,
        return_type: Option<&Type>,
        is_pure: bool,
        attrs: &DictAttrs,
        span: Option<&Span>,
    ) -> Result<Self> {
        let params = Array::new(params);
        let return_type = return_type.cloned();
        let prepared = crate::abi::prepare_constructor::<RelaxFunctionObj>(&[
            AnyView::from(&params),
            AnyView::from(body),
            AnyView::from(&return_type),
            AnyView::from(&is_pure),
        ])?;
        let body = crate::abi::prepared_field(&prepared, RelaxFunctionObj::TYPE_KEY, "body")?;
        let return_type =
            crate::abi::prepared_field(&prepared, RelaxFunctionObj::TYPE_KEY, "ret_ty")?;
        let function_type =
            crate::abi::prepared_field(&prepared, RelaxFunctionObj::TYPE_KEY, "ty")?;
        Ok(Self::from_complete_fields(
            span.cloned(),
            function_type,
            attrs.clone(),
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
            data: crate::abi::allocate_object(RelaxFunctionObj {
                base: BaseFuncObj::new(span, ty, attrs),
                params,
                body,
                ret_ty,
                is_pure,
            }),
        }
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

crate::abi::impl_rust_allocatable!(
    TupleObj,
    IfObj,
    VarBindingObj,
    BindingBlockObj,
    SeqExprObj,
    RelaxFunctionObj,
);
