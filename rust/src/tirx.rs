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
use tvm_ffi::{
    Any, AnyView, Array, DLDataType, DLDataTypeCode, DLDataTypeExt, Error, Map, ObjectArc,
    ObjectCore, ObjectRefCast, Result, String, TYPE_ERROR, VALUE_ERROR,
};

use crate::ir::{BaseFuncObj, DictAttrs, Expr, ExprObj, IntImm, Span, Var};

mod block;
mod buffer;

pub use block::{
    IterVar, IterVarObj, IterVarType, SBlock, SBlockObj, SBlockRealize, SBlockRealizeObj,
};
pub use buffer::{
    Axis, AxisObj, BufferLoad, BufferLoadObj, BufferRegion, BufferRegionObj, BufferStore,
    BufferStoreObj, BufferType, BufferTypeObj, DataProducer, DataProducerObj, Iter, IterObj,
    Layout, LayoutObj, MatchBufferRegion, MatchBufferRegionObj, TileLayout, TileLayoutObj,
};
/// ABI-complete Rust representation of TVM's `AddNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Add"]
#[type_final]
pub struct AddObj {
    base: ExprObj,
    pub a: Expr,
    pub b: Expr,
}
crate::abi::impl_object_layout!(AddObj {
    "a" => a: Expr,
    "b" => b: Expr,
});

/// Reference-counted handle to an addition expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Add {
    data: ObjectArc<AddObj>,
}

impl std::ops::Deref for Add {
    type Target = AddObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for AddObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl Add {
    /// Construct an addition expression directly in Rust.
    pub fn new<L, R>(lhs: L, rhs: R) -> Result<Self>
    where
        L: Into<Expr>,
        R: Into<Expr>,
    {
        Self::with_span(lhs, rhs, None)
    }

    /// Construct an addition expression with optional source metadata.
    pub fn with_span<L, R>(lhs: L, rhs: R, span: Option<&Span>) -> Result<Self>
    where
        L: Into<Expr>,
        R: Into<Expr>,
    {
        let lhs = lhs.into();
        let rhs = rhs.into();
        let result_type = matching_binary_type(&lhs, &rhs)?;
        Ok(Self::from_complete_fields(
            span.cloned(),
            result_type,
            lhs,
            rhs,
        ))
    }

    /// Construct an addition from every physical field without re-deriving its result type.
    pub fn from_complete_fields(span: Option<Span>, ty: crate::ir::Type, a: Expr, b: Expr) -> Self {
        Self {
            data: crate::abi::allocate_object(AddObj {
                base: ExprObj::new(span, ty),
                a,
                b,
            }),
        }
    }
}

pub(crate) fn primitive_type(expr: &Expr, context: &str) -> Result<crate::ir::PrimType> {
    expr.ty
        .clone()
        .try_cast::<crate::ir::PrimType>()
        .map_err(|_| {
            Error::new(
                TYPE_ERROR,
                &format!("{context} must have a primitive type"),
                "",
            )
        })
}

fn matching_binary_type(lhs: &Expr, rhs: &Expr) -> Result<crate::ir::Type> {
    let lhs_type = lhs.ty.clone();
    let lhs_dtype = primitive_type(lhs, "left binary operand")?.dtype;
    let rhs_dtype = primitive_type(rhs, "right binary operand")?.dtype;
    if lhs_dtype != rhs_dtype {
        return Err(Error::new(
            TYPE_ERROR,
            &format!(
                "mismatched binary operand types: {} vs. {}",
                lhs_dtype.to_string(),
                rhs_dtype.to_string()
            ),
            "",
        ));
    }
    Ok(lhs_type)
}

macro_rules! define_binary_expression {
    ($object:ident, $reference:ident, $type_key:literal, $description:literal) => {
        #[doc = concat!("ABI-complete Rust representation of TVM's `", $type_key, "` node.")]
        #[repr(C)]
        #[derive(Object)]
        #[type_key = $type_key]
        #[type_final]
        pub struct $object {
            base: ExprObj,
            pub a: Expr,
            pub b: Expr,
        }
        crate::abi::impl_object_layout!($object {
            "a" => a: Expr,
            "b" => b: Expr,
        });

        #[doc = concat!("Reference-counted handle to ", $description, ".")]
        #[repr(C)]
        #[derive(ObjectRef, Clone)]
        pub struct $reference {
            data: ObjectArc<$object>,
        }

        impl std::ops::Deref for $reference {
            type Target = $object;

            fn deref(&self) -> &Self::Target {
                &self.data
            }
        }

        impl std::ops::Deref for $object {
            type Target = ExprObj;

            fn deref(&self) -> &Self::Target {
                &self.base
            }
        }

        impl $reference {
            /// Construct the binary expression directly in Rust.
            pub fn new<L, R>(lhs: L, rhs: R) -> Result<Self>
            where
                L: Into<Expr>,
                R: Into<Expr>,
            {
                Self::with_span(lhs, rhs, None)
            }

            /// Construct the binary expression with optional source metadata.
            pub fn with_span<L, R>(lhs: L, rhs: R, span: Option<&Span>) -> Result<Self>
            where
                L: Into<Expr>,
                R: Into<Expr>,
            {
                let lhs = lhs.into();
                let rhs = rhs.into();
                let result_type = matching_binary_type(&lhs, &rhs)?;
                Ok(Self::from_complete_fields(
                    span.cloned(),
                    result_type,
                    lhs,
                    rhs,
                ))
            }

            /// Construct the expression from every physical field without re-deriving its type.
            pub fn from_complete_fields(
                span: Option<Span>,
                ty: crate::ir::Type,
                a: Expr,
                b: Expr,
            ) -> Self {
                Self {
                    data: crate::abi::allocate_object($object {
                        base: ExprObj::new(span, ty),
                        a,
                        b,
                    }),
                }
            }
        }
    };
}

define_binary_expression!(SubObj, Sub, "tirx.Sub", "a subtraction expression");
define_binary_expression!(MulObj, Mul, "tirx.Mul", "a multiplication expression");

/// ABI-complete Rust representation of TVM's `StringImmNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.StringImm"]
#[type_final]
pub struct StringImmObj {
    base: ExprObj,
    pub value: String,
}
crate::abi::impl_object_layout!(StringImmObj {
    "value" => value: String,
});

/// Reference-counted handle to a TIR string literal.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct StringImm {
    data: ObjectArc<StringImmObj>,
}

impl std::ops::Deref for StringImm {
    type Target = StringImmObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for StringImmObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl StringImm {
    /// Construct a string literal directly in Rust.
    pub fn new(value: &str) -> Self {
        Self::with_span(value, None)
    }

    /// Construct a string literal with optional source metadata.
    pub fn with_span(value: &str, span: Option<&Span>) -> Self {
        let value_type: crate::ir::Type = crate::ir::PrimType::void().into();
        Self::from_complete_fields(span.cloned(), value_type, String::from(value))
    }

    /// Construct a string literal from every physical field without re-deriving its type.
    pub fn from_complete_fields(span: Option<Span>, ty: crate::ir::Type, value: String) -> Self {
        Self {
            data: crate::abi::allocate_object(StringImmObj {
                base: ExprObj::new(span, ty),
                value,
            }),
        }
    }
}

/// ABI-complete Rust representation of TVM's `StmtNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Stmt"]
pub struct StmtObj {
    base: tvm_ffi::Object,
    pub span: Option<Span>,
}
crate::abi::impl_object_layout!(StmtObj {
    "span" => span: Option<Span>,
});

/// Reference-counted handle to any TIR statement.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Stmt {
    data: ObjectArc<StmtObj>,
}

impl std::ops::Deref for Stmt {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl StmtObj {
    pub(crate) fn new(span: Option<Span>) -> Self {
        Self {
            base: tvm_ffi::Object::new(),
            span,
        }
    }
}

impl Stmt {
    /// Normalize statements into TVM's canonical sequence representation.
    ///
    /// Nested sequences are flattened and `Evaluate(0)` nodes are removed.
    /// An empty result becomes `Evaluate(0)`, one remaining statement is
    /// returned directly, and two or more statements become a [`SeqStmt`].
    pub fn sequence(statements: Vec<Stmt>) -> Result<Self> {
        Self::sequence_with_span(statements, None)
    }

    /// Normalize statements while retaining a span on a newly created sequence.
    pub fn sequence_with_span(statements: Vec<Stmt>, span: Option<&Span>) -> Result<Self> {
        let mut flattened = Vec::new();
        for statement in statements {
            flatten_statement(statement, &mut flattened);
        }
        match flattened.len() {
            0 => Ok(Evaluate::from_i64(0)?.into()),
            1 => Ok(flattened.pop().expect("one statement is present")),
            _ => Ok(SeqStmt::from_complete_fields(span.cloned(), Array::new(flattened)).into()),
        }
    }
}

/// ABI-complete Rust representation of TVM's `AssertStmtNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.AssertStmt"]
#[type_final]
pub struct AssertStmtObj {
    base: StmtObj,
    pub condition: Expr,
    pub error_kind: StringImm,
    pub message_parts: Array<StringImm>,
}
crate::abi::impl_object_layout!(AssertStmtObj {
    "condition" => condition: Expr,
    "error_kind" => error_kind: StringImm,
    "message_parts" => message_parts: Array<StringImm>,
});

/// Reference-counted handle to a TIR assertion.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct AssertStmt {
    data: ObjectArc<AssertStmtObj>,
}

impl std::ops::Deref for AssertStmt {
    type Target = AssertStmtObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for AssertStmtObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// ABI-complete Rust representation of TVM's `EvaluateNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Evaluate"]
#[type_final]
pub struct EvaluateObj {
    base: StmtObj,
    pub value: Expr,
}
crate::abi::impl_object_layout!(EvaluateObj {
    "value" => value: Expr,
});

/// Reference-counted handle to a TIR evaluate statement.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Evaluate {
    data: ObjectArc<EvaluateObj>,
}

impl std::ops::Deref for Evaluate {
    type Target = EvaluateObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for EvaluateObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// ABI-complete Rust representation of TVM's `SeqStmtNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.SeqStmt"]
#[type_final]
pub struct SeqStmtObj {
    base: StmtObj,
    pub seq: Array<Stmt>,
}
crate::abi::impl_object_layout!(SeqStmtObj {
    "seq" => seq: Array<Stmt>,
});

/// Reference-counted handle to a sequence of statements.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct SeqStmt {
    data: ObjectArc<SeqStmtObj>,
}

impl std::ops::Deref for SeqStmt {
    type Target = SeqStmtObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for SeqStmtObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl SeqStmt {
    /// Construct and recursively flatten a sequence directly in Rust.
    pub fn new(statements: Vec<Stmt>) -> Result<Self> {
        Self::with_span(statements, None)
    }

    /// Construct and recursively flatten a sequence with source metadata.
    pub fn with_span(statements: Vec<Stmt>, span: Option<&Span>) -> Result<Self> {
        let requires_flattening = statements
            .iter()
            .any(|statement| statement.clone().try_cast::<SeqStmt>().is_ok());
        let flattened = if requires_flattening {
            let mut flattened = Vec::new();
            for statement in statements {
                flatten_statement(statement, &mut flattened);
            }
            flattened
        } else {
            statements
        };
        if flattened.len() < 2 {
            return Err(Error::new(
                VALUE_ERROR,
                if flattened.is_empty() {
                    "an empty SeqStmt is prohibited; use Evaluate(0) for a no-op"
                } else {
                    "a SeqStmt of length one is prohibited; use its single statement directly"
                },
                "",
            ));
        }
        Ok(Self::from_complete_fields(
            span.cloned(),
            Array::new(flattened),
        ))
    }

    /// Construct a sequence from its already-normalized physical fields.
    pub fn from_complete_fields(span: Option<Span>, seq: Array<Stmt>) -> Self {
        Self {
            data: crate::abi::allocate_object(SeqStmtObj {
                base: StmtObj::new(span),
                seq,
            }),
        }
    }
}

fn flatten_statement(statement: Stmt, output: &mut Vec<Stmt>) {
    match statement.clone().try_cast::<SeqStmt>() {
        Ok(sequence) => {
            for child in sequence.seq.iter() {
                flatten_statement(child, output);
            }
        }
        Err(_) if !is_evaluate_zero(&statement) => output.push(statement),
        Err(_) => {}
    }
}

fn is_evaluate_zero(statement: &Stmt) -> bool {
    statement
        .clone()
        .try_cast::<Evaluate>()
        .ok()
        .and_then(|evaluate| evaluate.value.clone().try_cast::<IntImm>().ok())
        .is_some_and(|literal| literal.value == 0)
}

/// ABI-complete Rust representation of TVM's `IfThenElseNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.IfThenElse"]
#[type_final]
pub struct IfThenElseObj {
    base: StmtObj,
    pub condition: Expr,
    pub then_case: Stmt,
    pub else_case: Option<Stmt>,
}
crate::abi::impl_object_layout!(IfThenElseObj {
    "condition" => condition: Expr,
    "then_case" => then_case: Stmt,
    "else_case" => else_case: Option<Stmt>,
});

/// Reference-counted handle to a conditional statement.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct IfThenElse {
    data: ObjectArc<IfThenElseObj>,
}

impl std::ops::Deref for IfThenElse {
    type Target = IfThenElseObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for IfThenElseObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl IfThenElse {
    /// Construct a conditional statement directly in Rust.
    pub fn new<C, T>(condition: C, then_case: T, else_case: Option<Stmt>) -> Result<Self>
    where
        C: Into<Expr>,
        T: Into<Stmt>,
    {
        Self::with_span(condition, then_case, else_case, None)
    }

    /// Construct a conditional statement with optional source metadata.
    pub fn with_span<C, T>(
        condition: C,
        then_case: T,
        else_case: Option<Stmt>,
        span: Option<&Span>,
    ) -> Result<Self>
    where
        C: Into<Expr>,
        T: Into<Stmt>,
    {
        let condition = condition.into();
        primitive_type(&condition, "IfThenElse condition")?;
        Ok(Self::from_complete_fields(
            span.cloned(),
            condition,
            then_case.into(),
            else_case,
        ))
    }

    /// Construct a conditional statement from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        condition: Expr,
        then_case: Stmt,
        else_case: Option<Stmt>,
    ) -> Self {
        Self {
            data: crate::abi::allocate_object(IfThenElseObj {
                base: StmtObj::new(span),
                condition,
                then_case,
                else_case,
            }),
        }
    }
}

/// Execution policy attached to a TIR `For` loop.
///
/// This is an open integer newtype rather than a Rust enum: reading a newer
/// C++ enumerator through an older generated binding must remain memory-safe.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ForKind(i32);

#[allow(non_upper_case_globals)]
impl ForKind {
    pub const Serial: Self = Self(0);
    pub const Parallel: Self = Self(1);
    pub const Vectorized: Self = Self(2);
    pub const Unrolled: Self = Self(3);
    pub const ThreadBinding: Self = Self(4);

    /// Preserve an enumerator not yet known by this Rust binding.
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }

    pub const fn as_raw(self) -> i32 {
        self.0
    }
}

impl TryFrom<i64> for ForKind {
    type Error = Error;

    fn try_from(value: i64) -> Result<Self> {
        i32::try_from(value).map(Self).map_err(|_| {
            Error::new(
                VALUE_ERROR,
                &format!("tirx.ForKind value {value} does not fit its native i32 representation"),
                "",
            )
        })
    }
}

/// ABI-complete Rust representation of TVM's `ForNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.For"]
#[type_final]
pub struct ForObj {
    base: StmtObj,
    pub loop_var: Var,
    pub min: Expr,
    pub extent: Expr,
    pub kind: ForKind,
    pub body: Stmt,
    pub thread_binding: Option<IterVar>,
    pub annotations: Map<String, Any>,
    pub step: Option<Expr>,
}
crate::abi::impl_object_layout!(ForObj {
    "loop_var" => loop_var: Var,
    "min" => min: Expr,
    "extent" => extent: Expr,
    "kind" => kind: ForKind,
    "body" => body: Stmt,
    "thread_binding" => thread_binding: Option<IterVar>,
    "annotations" => annotations: Map<String, Any>,
    "step" => step: Option<Expr>,
});

/// Reference-counted handle to a TIR loop.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct For {
    data: ObjectArc<ForObj>,
}

impl std::ops::Deref for For {
    type Target = ForObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for ForObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl For {
    /// Construct a serial loop with no thread binding, annotations, or custom step.
    pub fn serial<V, M, E, B>(loop_var: V, minimum: M, extent: E, body: B) -> Result<Self>
    where
        V: Into<Var>,
        M: Into<Expr>,
        E: Into<Expr>,
        B: Into<Stmt>,
    {
        Self::with_metadata(
            loop_var.into(),
            minimum.into(),
            extent.into(),
            ForKind::Serial,
            body.into(),
            None,
            Map::new(),
            None,
            None,
        )
    }

    /// Construct a loop directly in Rust with TVM's bound normalization and validation.
    #[allow(clippy::too_many_arguments)]
    pub fn with_metadata(
        loop_var: Var,
        minimum: Expr,
        extent: Expr,
        kind: ForKind,
        body: Stmt,
        thread_binding: Option<IterVar>,
        annotations: Map<String, Any>,
        step: Option<Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        let loop_expr: Expr = loop_var.clone().into();
        let loop_dtype = require_scalar_integer(&loop_expr, "loop_var")?;
        require_scalar_integer(&minimum, "min")?;
        require_scalar_integer(&extent, "extent")?;
        let minimum = normalize_loop_bound(&minimum, loop_dtype, "min")?;
        let extent = normalize_loop_bound(&extent, loop_dtype, "extent")?;
        let step = step
            .as_ref()
            .map(|step| {
                require_scalar_integer(step, "step")?;
                normalize_loop_bound(step, loop_dtype, "step")
            })
            .transpose()?;
        Ok(Self::from_complete_fields(
            span.cloned(),
            loop_var,
            minimum,
            extent,
            kind,
            body,
            thread_binding,
            annotations,
            step,
        ))
    }

    /// Construct a loop from every physical field after external validation.
    #[allow(clippy::too_many_arguments)]
    pub fn from_complete_fields(
        span: Option<Span>,
        loop_var: Var,
        min: Expr,
        extent: Expr,
        kind: ForKind,
        body: Stmt,
        thread_binding: Option<IterVar>,
        annotations: Map<String, Any>,
        step: Option<Expr>,
    ) -> Self {
        Self {
            data: crate::abi::allocate_object(ForObj {
                base: StmtObj::new(span),
                loop_var,
                min,
                extent,
                kind,
                body,
                thread_binding,
                annotations,
                step,
            }),
        }
    }
}

fn require_scalar_integer(value: &Expr, field: &str) -> Result<DLDataType> {
    let dtype = primitive_type(value, field)?.dtype;
    let is_integer =
        dtype.code == DLDataTypeCode::kDLInt as u8 || dtype.code == DLDataTypeCode::kDLUInt as u8;
    if dtype.lanes != 1 || !is_integer {
        return Err(Error::new(
            TYPE_ERROR,
            &format!("TIR For nodes require a scalar integer {field}"),
            "",
        ));
    }
    Ok(dtype)
}

fn normalize_loop_bound(value: &Expr, loop_dtype: DLDataType, field: &str) -> Result<Expr> {
    let value_dtype = primitive_type(value, field)?.dtype;
    if value_dtype == loop_dtype {
        return Ok(value.clone());
    }
    if let Ok(literal) = value.clone().try_cast::<IntImm>() {
        return Ok(IntImm::from_dtype(loop_dtype, literal.value)?.into());
    }
    if value_dtype.bits > loop_dtype.bits {
        return Err(Error::new(
            TYPE_ERROR,
            &format!("loop variable dtype is narrower than {field}"),
            "",
        ));
    }
    Err(Error::new(
        TYPE_ERROR,
        &format!("loop variable and {field} must have the same dtype"),
        "",
    ))
}

/// ABI-complete Rust representation of TVM's `PrimFuncNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.PrimFunc"]
#[type_final]
pub struct PrimFuncObj {
    base: BaseFuncObj,
    pub params: Array<Var>,
    pub ret_type: crate::ir::Type,
    pub body: Stmt,
}
crate::abi::impl_object_layout!(PrimFuncObj {
    "params" => params: Array<Var>,
    "ret_type" => ret_type: crate::ir::Type,
    "body" => body: Stmt,
});

impl crate::abi::ConstructorRecipe for PrimFuncObj {
    const NUM_INPUTS: usize = 3;
    const DERIVED_FIELDS: &'static [&'static str] = &["ret_type", "ty"];
}

/// Reference-counted handle to a TIR primitive function.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct PrimFunc {
    data: ObjectArc<PrimFuncObj>,
}

impl std::ops::Deref for PrimFunc {
    type Target = PrimFuncObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for PrimFuncObj {
    type Target = BaseFuncObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl AssertStmt {
    /// Construct a leaf assertion directly in Rust with one string message part.
    pub fn new<C>(condition: C, error_kind: &str, message: &str) -> Result<Self>
    where
        C: Into<Expr>,
    {
        let error_kind = StringImm::new(error_kind);
        let message = StringImm::new(message);
        Self::with_metadata(condition, error_kind, vec![message], None)
    }

    /// Construct an assertion from all fields accepted by the C++ constructor.
    pub fn with_metadata<C, K>(
        condition: C,
        error_kind: K,
        message_parts: Vec<StringImm>,
        span: Option<&Span>,
    ) -> Result<Self>
    where
        C: Into<Expr>,
        K: Into<StringImm>,
    {
        let condition = condition.into();
        let dtype = primitive_type(&condition, "AssertStmt condition")?.dtype;
        if dtype.code != DLDataTypeCode::kDLBool as u8 {
            return Err(Error::new(
                TYPE_ERROR,
                &format!(
                    "AssertStmt condition must have bool type, but received {}",
                    dtype.to_string()
                ),
                "",
            ));
        }
        Ok(Self::from_complete_fields(
            span.cloned(),
            condition,
            error_kind.into(),
            Array::new(message_parts),
        ))
    }

    /// Construct an assertion from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        condition: Expr,
        error_kind: StringImm,
        message_parts: Array<StringImm>,
    ) -> Self {
        Self {
            data: crate::abi::allocate_object(AssertStmtObj {
                base: StmtObj::new(span),
                condition,
                error_kind,
                message_parts,
            }),
        }
    }
}

impl Evaluate {
    /// Construct `Evaluate(value)` directly in Rust.
    pub fn new<E>(value: E) -> Result<Self>
    where
        E: Into<Expr>,
    {
        Self::with_span(value, None)
    }

    /// Construct `Evaluate(value)` with optional source metadata.
    pub fn with_span<E>(value: E, span: Option<&Span>) -> Result<Self>
    where
        E: Into<Expr>,
    {
        let value = value.into();
        if value.clone().try_cast::<Var>().is_ok()
            && value.ty.clone().try_cast::<buffer::BufferType>().is_ok()
        {
            return Err(Error::new(
                VALUE_ERROR,
                "a buffer variable cannot be used as a scalar Evaluate value",
                "",
            ));
        }
        Ok(Self::from_complete_fields(span.cloned(), value))
    }

    /// Construct an evaluation statement from every physical field after external validation.
    pub fn from_complete_fields(span: Option<Span>, value: Expr) -> Self {
        Self {
            data: crate::abi::allocate_object(EvaluateObj {
                base: StmtObj::new(span),
                value,
            }),
        }
    }

    /// Construct `Evaluate(IntImm("int32", value))`.
    pub fn from_i64(value: i64) -> Result<Self> {
        Self::new(IntImm::new("int32", value)?)
    }
}

impl PrimFunc {
    /// Construct a parameterless PrimFunc around `body`.
    pub fn from_body<S>(body: S) -> Result<Self>
    where
        S: Into<Stmt>,
    {
        Self::new(Vec::new(), body)
    }

    /// Construct a PrimFunc in Rust after deriving type metadata through its reflected method.
    pub fn new<S>(params: Vec<Var>, body: S) -> Result<Self>
    where
        S: Into<Stmt>,
    {
        let attrs = DictAttrs::empty();
        let ret_type = crate::ir::Type::missing();
        Self::with_metadata(params, body, ret_type, attrs, None)
    }

    /// Construct a PrimFunc in Rust after deriving type metadata through its reflected method.
    pub fn with_metadata<S, T, A>(
        params: Vec<Var>,
        body: S,
        ret_type: T,
        attrs: A,
        span: Option<&Span>,
    ) -> Result<Self>
    where
        S: Into<Stmt>,
        T: Into<crate::ir::Type>,
        A: Into<DictAttrs>,
    {
        let body: Stmt = body.into();
        let ret_type = ret_type.into();
        let attrs = attrs.into();
        let params = Array::new(params);
        let prepared = crate::abi::prepare_constructor::<PrimFuncObj>(&[
            AnyView::from(&params),
            AnyView::from(&body),
            AnyView::from(&ret_type),
        ])?;
        let ret_type = crate::abi::prepared_field(&prepared, PrimFuncObj::TYPE_KEY, "ret_type")?;
        let function_type = crate::abi::prepared_field(&prepared, PrimFuncObj::TYPE_KEY, "ty")?;
        Ok(Self::from_complete_fields(
            span.cloned(),
            function_type,
            attrs,
            params,
            ret_type,
            body,
        ))
    }

    /// Construct a PrimFunc allocation entirely in Rust from its complete state.
    ///
    /// `function_type` is the derived Relax-facing function type stored in the
    /// inherited `ExprObj::ty` field.  Supplying it explicitly keeps this raw
    /// constructor lossless; [`PrimFunc::new`] obtains exactly these derived
    /// fields from the language-independent reflected preparation method.
    #[allow(clippy::too_many_arguments)]
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: crate::ir::Type,
        attrs: DictAttrs,
        params: Array<Var>,
        ret_type: crate::ir::Type,
        body: Stmt,
    ) -> Self {
        Self {
            data: crate::abi::allocate_object(PrimFuncObj {
                base: BaseFuncObj::new(span, ty, attrs),
                params,
                ret_type,
                body,
            }),
        }
    }
}

tvm_ffi::impl_object_upcast!(
    Add => Expr,
    Sub => Expr,
    Mul => Expr,
    StringImm => Expr,
    For => Stmt,
    AssertStmt => Stmt,
    Evaluate => Stmt,
    SeqStmt => Stmt,
    IfThenElse => Stmt,
    PrimFunc => crate::ir::BaseFunc,
    PrimFunc => Expr,
);

crate::abi::impl_rust_allocatable!(
    AddObj,
    SubObj,
    MulObj,
    StringImmObj,
    AssertStmtObj,
    EvaluateObj,
    SeqStmtObj,
    IfThenElseObj,
    ForObj,
    PrimFuncObj,
);
