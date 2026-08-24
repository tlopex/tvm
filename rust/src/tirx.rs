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
    Any, AnyCompatible, AnyView, Array, Error, ObjectArc, ObjectRefCast, Result, String,
    VALUE_ERROR,
};

use crate::ir::{BaseFuncObj, DictAttrs, Expr, ExprObj, Span, Var};

mod block;
mod buffer;

pub use block::{
    IterVar, IterVarObj, IterVarType, SBlock, SBlockObj, SBlockRealize, SBlockRealizeObj,
};
pub use buffer::{
    BufferLoad, BufferLoadObj, BufferRegion, BufferRegionObj, BufferStore, BufferStoreObj,
    BufferType, BufferTypeObj, MatchBufferRegion, MatchBufferRegionObj,
};
/// Opaque Rust view of TVM's `AddNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Add"]
#[type_final]
pub struct AddObj {
    base: ExprObj,
}

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

impl AddObj {
    /// Return the left operand.
    pub fn lhs(&self) -> Result<Expr> {
        crate::reflected_field!(self, "a")?.try_into()
    }

    /// Return the right operand.
    pub fn rhs(&self) -> Result<Expr> {
        crate::reflected_field!(self, "b")?.try_into()
    }
}

impl Add {
    /// Construct an addition expression.
    pub fn new(lhs: &Expr, rhs: &Expr) -> Result<Self> {
        let none = ();
        crate::global_function!("tirx.Add")?
            .call_packed(&[AnyView::from(lhs), AnyView::from(rhs), AnyView::from(&none)])?
            .try_into()
    }
}

impl From<Add> for Expr {
    fn from(value: Add) -> Self {
        value
            .try_cast()
            .expect("tirx.Add must be a subtype of ir.Expr")
    }
}

macro_rules! define_binary_expression {
    ($object:ident, $reference:ident, $type_key:literal, $constructor:literal, $description:literal) => {
        #[doc = concat!("Opaque Rust view of TVM's `", $type_key, "` node.")]
        #[repr(C)]
        #[derive(Object)]
        #[type_key = $type_key]
        #[type_final]
        pub struct $object {
            base: ExprObj,
        }

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

        impl $object {
            /// Return the left operand.
            pub fn lhs(&self) -> Result<Expr> {
                crate::reflected_field!(self, "a")?.try_into()
            }

            /// Return the right operand.
            pub fn rhs(&self) -> Result<Expr> {
                crate::reflected_field!(self, "b")?.try_into()
            }
        }

        impl $reference {
            /// Construct the binary expression.
            pub fn new(lhs: &Expr, rhs: &Expr) -> Result<Self> {
                let none = ();
                crate::global_function!($constructor)?
                    .call_packed(&[AnyView::from(lhs), AnyView::from(rhs), AnyView::from(&none)])?
                    .try_into()
            }
        }

        impl From<$reference> for Expr {
            fn from(value: $reference) -> Self {
                value
                    .try_cast()
                    .expect(concat!($type_key, " must be a subtype of ir.Expr"))
            }
        }
    };
}

define_binary_expression!(
    SubObj,
    Sub,
    "tirx.Sub",
    "tirx.Sub",
    "a subtraction expression"
);
define_binary_expression!(
    MulObj,
    Mul,
    "tirx.Mul",
    "tirx.Mul",
    "a multiplication expression"
);

/// Opaque Rust view of TVM's `StringImmNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.StringImm"]
#[type_final]
pub struct StringImmObj {
    base: ExprObj,
}

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

impl StringImmObj {
    /// Return the string literal value.
    pub fn value(&self) -> Result<String> {
        crate::reflected_field!(self, "value")?.try_into()
    }
}

impl StringImm {
    /// Construct a string literal.
    pub fn new(value: &str) -> Result<Self> {
        let value = String::from(value);
        let none = ();
        crate::global_function!("tirx.StringImm")?
            .call_packed(&[AnyView::from(&value), AnyView::from(&none)])?
            .try_into()
    }
}

impl From<StringImm> for Expr {
    fn from(value: StringImm) -> Self {
        value
            .try_cast()
            .expect("tirx.StringImm must be a subtype of ir.Expr")
    }
}

/// Opaque Rust view of TVM's `StmtNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Stmt"]
pub struct StmtObj {
    base: tvm_ffi::Object,
}

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
    /// Return source-span metadata carried by this statement.
    pub fn span(&self) -> Result<Option<Span>> {
        crate::reflected_field!(self, "span")?.try_into()
    }
}

/// Opaque Rust view of TVM's `AssertStmtNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.AssertStmt"]
#[type_final]
pub struct AssertStmtObj {
    base: StmtObj,
}

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

impl AssertStmtObj {
    /// Return the assertion condition.
    pub fn condition(&self) -> Result<Expr> {
        crate::reflected_field!(self, "condition")?.try_into()
    }

    /// Return the exception kind.
    pub fn error_kind(&self) -> Result<StringImm> {
        crate::reflected_field!(self, "error_kind")?.try_into()
    }

    /// Return the message fragments.
    pub fn message_parts(&self) -> Result<Array<StringImm>> {
        crate::reflected_field!(self, "message_parts")?.try_into()
    }
}

/// Opaque Rust view of TVM's `EvaluateNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Evaluate"]
#[type_final]
pub struct EvaluateObj {
    base: StmtObj,
}

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

impl EvaluateObj {
    /// Return the evaluated expression.
    pub fn value(&self) -> Result<Expr> {
        crate::reflected_field!(self, "value")?.try_into()
    }
}

/// Opaque Rust view of TVM's `SeqStmtNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.SeqStmt"]
#[type_final]
pub struct SeqStmtObj {
    base: StmtObj,
}

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

impl SeqStmtObj {
    /// Return the statements in this sequence.
    pub fn statements(&self) -> Result<Array<Stmt>> {
        crate::reflected_field!(self, "seq")?.try_into()
    }
}

impl SeqStmt {
    /// Construct a sequence.  TVM requires at least two statements.
    pub fn new(statements: Vec<Stmt>) -> Result<Self> {
        let statements = Array::new(statements);
        let none = ();
        crate::global_function!("tirx.SeqStmt")?
            .call_packed(&[AnyView::from(&statements), AnyView::from(&none)])?
            .try_into()
    }
}

/// Opaque Rust view of TVM's `IfThenElseNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.IfThenElse"]
#[type_final]
pub struct IfThenElseObj {
    base: StmtObj,
}

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

impl IfThenElseObj {
    /// Return the condition.
    pub fn condition(&self) -> Result<Expr> {
        crate::reflected_field!(self, "condition")?.try_into()
    }

    /// Return the true branch.
    pub fn then_case(&self) -> Result<Stmt> {
        crate::reflected_field!(self, "then_case")?.try_into()
    }

    /// Return the optional false branch.
    pub fn else_case(&self) -> Result<Option<Stmt>> {
        crate::reflected_field!(self, "else_case")?.try_into()
    }
}

impl IfThenElse {
    /// Construct a conditional statement.
    pub fn new(condition: &Expr, then_case: &Stmt, else_case: Option<&Stmt>) -> Result<Self> {
        let else_case = else_case.cloned();
        let none = ();
        crate::global_function!("tirx.IfThenElse")?
            .call_packed(&[
                AnyView::from(condition),
                AnyView::from(then_case),
                AnyView::from(&else_case),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

/// Execution policy attached to a TIR `For` loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub enum ForKind {
    Serial = 0,
    Parallel = 1,
    Vectorized = 2,
    Unrolled = 3,
    ThreadBinding = 4,
}

impl TryFrom<i64> for ForKind {
    type Error = Error;

    fn try_from(value: i64) -> Result<Self> {
        match value {
            0 => Ok(Self::Serial),
            1 => Ok(Self::Parallel),
            2 => Ok(Self::Vectorized),
            3 => Ok(Self::Unrolled),
            4 => Ok(Self::ThreadBinding),
            _ => Err(Error::new(
                VALUE_ERROR,
                &format!("unknown tirx.ForKind value {value}"),
                "",
            )),
        }
    }
}

/// Opaque Rust view of TVM's `ForNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.For"]
#[type_final]
pub struct ForObj {
    base: StmtObj,
}

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

impl ForObj {
    /// Return the variable defined by this loop.
    pub fn loop_var(&self) -> Result<Var> {
        crate::reflected_field!(self, "loop_var")?.try_into()
    }

    /// Return the inclusive starting value.
    pub fn minimum(&self) -> Result<Expr> {
        crate::reflected_field!(self, "min")?.try_into()
    }

    /// Return the number of loop iterations.
    pub fn extent(&self) -> Result<Expr> {
        crate::reflected_field!(self, "extent")?.try_into()
    }

    /// Return the loop execution policy.
    pub fn kind(&self) -> Result<ForKind> {
        ForKind::try_from(i64::try_from(crate::reflected_field!(self, "kind")?)?)
    }

    /// Return the loop body.
    pub fn body(&self) -> Result<Stmt> {
        crate::reflected_field!(self, "body")?.try_into()
    }

    /// Return the optional thread axis bound by this loop.
    pub fn thread_binding(&self) -> Result<Option<IterVar>> {
        crate::reflected_field!(self, "thread_binding")?.try_into()
    }

    /// Return heterogeneous loop annotations as their complete FFI value.
    pub fn annotations(&self) -> Result<Any> {
        crate::reflected_field!(self, "annotations")
    }

    /// Return an optional non-unit loop step.
    pub fn step(&self) -> Result<Option<Expr>> {
        crate::reflected_field!(self, "step")?.try_into()
    }
}

impl For {
    /// Construct a serial loop with no thread binding, annotations, or custom step.
    pub fn serial(loop_var: &Var, minimum: &Expr, extent: &Expr, body: &Stmt) -> Result<Self> {
        Self::with_metadata(
            loop_var,
            minimum,
            extent,
            ForKind::Serial,
            body,
            None,
            &Any::from(()),
            None,
            None,
        )
    }

    /// Construct a loop while preserving all reflected fields used by a mutator.
    #[allow(clippy::too_many_arguments)]
    pub fn with_metadata(
        loop_var: &Var,
        minimum: &Expr,
        extent: &Expr,
        kind: ForKind,
        body: &Stmt,
        thread_binding: Option<&IterVar>,
        annotations: &Any,
        step: Option<&Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        let kind = kind as i64;
        let thread_binding = thread_binding.cloned();
        let step = step.cloned();
        let span = span.cloned();
        crate::global_function!("tirx.For")?
            .call_packed(&[
                AnyView::from(loop_var),
                AnyView::from(minimum),
                AnyView::from(extent),
                AnyView::from(&kind),
                AnyView::from(body),
                AnyView::from(&thread_binding),
                AnyView::from(annotations),
                AnyView::from(&step),
                AnyView::from(&span),
            ])?
            .try_into()
    }
}

impl From<For> for Stmt {
    fn from(value: For) -> Self {
        value
            .try_cast()
            .expect("tirx.For must be a subtype of tirx.Stmt")
    }
}

/// Opaque Rust view of TVM's `PrimFuncNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.PrimFunc"]
#[type_final]
pub struct PrimFuncObj {
    base: BaseFuncObj,
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

impl PrimFuncObj {
    /// Return the function parameters.
    pub fn params(&self) -> Result<Array<Var>> {
        crate::reflected_field!(self, "params")?.try_into()
    }

    /// Return the function return type.
    pub fn ret_type(&self) -> Result<crate::ir::Type> {
        crate::reflected_field!(self, "ret_type")?.try_into()
    }

    /// Return the function body.
    pub fn body(&self) -> Result<Stmt> {
        crate::reflected_field!(self, "body")?.try_into()
    }
}

impl AssertStmt {
    /// Construct a leaf assertion with one string message part.
    pub fn new(condition: &Expr, error_kind: &str, message: &str) -> Result<Self> {
        let error_kind = StringImm::new(error_kind)?;
        let message = StringImm::new(message)?;
        let message_parts = Array::new(vec![message]);
        let none = ();
        crate::global_function!("tirx.AssertStmt")?
            .call_packed(&[
                AnyView::from(condition),
                AnyView::from(&error_kind),
                AnyView::from(&message_parts),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

impl From<AssertStmt> for Stmt {
    fn from(value: AssertStmt) -> Self {
        value
            .try_cast()
            .expect("tirx.AssertStmt must be a subtype of tirx.Stmt")
    }
}

impl Evaluate {
    /// Construct `Evaluate(value)`.
    pub fn new<E: AnyCompatible>(value: &E) -> Result<Self> {
        let none = ();
        crate::global_function!("tirx.Evaluate")?
            .call_packed(&[AnyView::from(value), AnyView::from(&none)])?
            .try_into()
    }

    /// Construct `Evaluate(IntImm("int32", value))`.
    pub fn from_i64(value: i64) -> Result<Self> {
        Self::new(&Expr::int("int32", value)?)
    }
}

impl From<Evaluate> for Stmt {
    fn from(value: Evaluate) -> Self {
        value
            .try_cast()
            .expect("tirx.Evaluate must be a subtype of tirx.Stmt")
    }
}

impl From<SeqStmt> for Stmt {
    fn from(value: SeqStmt) -> Self {
        value
            .try_cast()
            .expect("tirx.SeqStmt must be a subtype of tirx.Stmt")
    }
}

impl From<IfThenElse> for Stmt {
    fn from(value: IfThenElse) -> Self {
        value
            .try_cast()
            .expect("tirx.IfThenElse must be a subtype of tirx.Stmt")
    }
}

impl PrimFunc {
    /// Construct a parameterless PrimFunc around `body`.
    pub fn from_body<S: AnyCompatible>(body: &S) -> Result<Self> {
        Self::new(Vec::new(), body)
    }

    /// Construct a PrimFunc with explicit parameters and a body.
    pub fn new<S: AnyCompatible>(params: Vec<Var>, body: &S) -> Result<Self> {
        let params = Array::new(params);
        let attrs = DictAttrs::empty()?;
        let ret_type = crate::global_function!("ir.TypeMissing")?.call_packed(&[])?;
        let none = ();
        crate::global_function!("tirx.PrimFunc")?
            .call_packed(&[
                AnyView::from(&params),
                AnyView::from(body),
                AnyView::from(&ret_type),
                AnyView::from(&attrs),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

impl From<PrimFunc> for crate::ir::BaseFunc {
    fn from(value: PrimFunc) -> Self {
        value
            .try_cast()
            .expect("tirx.PrimFunc must be a subtype of ir.BaseFunc")
    }
}

impl From<PrimFunc> for Expr {
    fn from(value: PrimFunc) -> Self {
        let base: crate::ir::BaseFunc = value.into();
        base.try_cast()
            .expect("ir.BaseFunc must be a subtype of ir.Expr")
    }
}
