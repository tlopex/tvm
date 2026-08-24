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
    AnyCompatible, AnyMap, AnyView, Array, DLDataTypeCode, DLDataTypeExt, Error, ObjectArc,
    ObjectRefCast, Result, String, TYPE_ERROR, VALUE_ERROR,
};

use crate::ir::{BaseFuncObj, DictAttrs, Expr, ExprObj, Span, Var};

mod block;
mod buffer;

pub use block::{
    IterVar, IterVarObj, IterVarType, SBlock, SBlockObj, SBlockRealize, SBlockRealizeObj,
};
pub use buffer::{
    Axis, AxisObj, BufferLoad, BufferLoadObj, BufferRegion, BufferRegionObj, BufferStore,
    BufferStoreObj, BufferType, BufferTypeObj, Iter, IterObj, Layout, LayoutObj, MatchBufferRegion,
    MatchBufferRegionObj, TileLayout, TileLayoutObj,
};
/// ABI-complete Rust representation of TVM's `AddNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Add"]
#[type_final]
pub struct AddObj {
    base: ExprObj,
    a: Expr,
    b: Expr,
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
        Ok(self.a.clone())
    }

    /// Return the right operand.
    pub fn rhs(&self) -> Result<Expr> {
        Ok(self.b.clone())
    }
}

impl Add {
    /// Construct an addition expression directly in Rust.
    pub fn new(lhs: &Expr, rhs: &Expr) -> Result<Self> {
        let result_type = matching_binary_type(lhs, rhs)?;
        Ok(Self {
            data: ObjectArc::new(AddObj {
                base: ExprObj::new(result_type, None),
                a: lhs.clone(),
                b: rhs.clone(),
            }),
        })
    }
}

impl From<Add> for Expr {
    fn from(value: Add) -> Self {
        value
            .try_cast()
            .expect("tirx.Add must be a subtype of ir.Expr")
    }
}

pub(crate) fn primitive_type(expr: &Expr, context: &str) -> Result<crate::ir::PrimType> {
    expr.ty()?.try_cast::<crate::ir::PrimType>().map_err(|_| {
        Error::new(
            TYPE_ERROR,
            &format!("{context} must have a primitive type"),
            "",
        )
    })
}

fn matching_binary_type(lhs: &Expr, rhs: &Expr) -> Result<crate::ir::Type> {
    let lhs_type = lhs.ty()?;
    let lhs_dtype = primitive_type(lhs, "left binary operand")?.dtype()?;
    let rhs_dtype = primitive_type(rhs, "right binary operand")?.dtype()?;
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
            a: Expr,
            b: Expr,
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
                Ok(self.a.clone())
            }

            /// Return the right operand.
            pub fn rhs(&self) -> Result<Expr> {
                Ok(self.b.clone())
            }
        }

        impl $reference {
            /// Construct the binary expression directly in Rust.
            pub fn new(lhs: &Expr, rhs: &Expr) -> Result<Self> {
                let result_type = matching_binary_type(lhs, rhs)?;
                Ok(Self {
                    data: ObjectArc::new($object {
                        base: ExprObj::new(result_type, None),
                        a: lhs.clone(),
                        b: rhs.clone(),
                    }),
                })
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

define_binary_expression!(SubObj, Sub, "tirx.Sub", "a subtraction expression");
define_binary_expression!(MulObj, Mul, "tirx.Mul", "a multiplication expression");

/// ABI-complete Rust representation of TVM's `StringImmNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.StringImm"]
#[type_final]
pub struct StringImmObj {
    base: ExprObj,
    value: String,
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
        Ok(self.value.clone())
    }
}

impl StringImm {
    /// Construct a string literal directly in Rust.
    pub fn new(value: &str) -> Result<Self> {
        Ok(Self {
            data: ObjectArc::new(StringImmObj {
                base: ExprObj::new(crate::ir::PrimType::new("void")?.into(), None),
                value: String::from(value),
            }),
        })
    }
}

impl From<StringImm> for Expr {
    fn from(value: StringImm) -> Self {
        value
            .try_cast()
            .expect("tirx.StringImm must be a subtype of ir.Expr")
    }
}

/// ABI-complete Rust representation of TVM's `StmtNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Stmt"]
pub struct StmtObj {
    base: tvm_ffi::Object,
    span: Option<Span>,
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
    pub(crate) fn new(span: Option<Span>) -> Self {
        Self {
            base: tvm_ffi::Object::new(),
            span,
        }
    }

    /// Return source-span metadata carried by this statement.
    pub fn span(&self) -> Result<Option<Span>> {
        Ok(self.span.clone())
    }
}

/// ABI-complete Rust representation of TVM's `AssertStmtNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.AssertStmt"]
#[type_final]
pub struct AssertStmtObj {
    base: StmtObj,
    condition: Expr,
    error_kind: StringImm,
    message_parts: Array<StringImm>,
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
        Ok(self.condition.clone())
    }

    /// Return the exception kind.
    pub fn error_kind(&self) -> Result<StringImm> {
        Ok(self.error_kind.clone())
    }

    /// Return the message fragments.
    pub fn message_parts(&self) -> Result<Array<StringImm>> {
        Ok(self.message_parts.clone())
    }
}

/// ABI-complete Rust representation of TVM's `EvaluateNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Evaluate"]
#[type_final]
pub struct EvaluateObj {
    base: StmtObj,
    value: Expr,
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
        Ok(self.value.clone())
    }
}

/// ABI-complete Rust representation of TVM's `SeqStmtNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.SeqStmt"]
#[type_final]
pub struct SeqStmtObj {
    base: StmtObj,
    seq: Array<Stmt>,
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
        Ok(self.seq.clone())
    }
}

impl SeqStmt {
    /// Construct and recursively flatten a sequence directly in Rust.
    pub fn new(statements: Vec<Stmt>) -> Result<Self> {
        let mut flattened = Vec::new();
        for statement in statements {
            flatten_statement(statement, &mut flattened)?;
        }
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
        Ok(Self {
            data: ObjectArc::new(SeqStmtObj {
                base: StmtObj::new(None),
                seq: Array::new(flattened),
            }),
        })
    }
}

fn flatten_statement(statement: Stmt, output: &mut Vec<Stmt>) -> Result<()> {
    match statement.clone().try_cast::<SeqStmt>() {
        Ok(sequence) => {
            for child in sequence.statements()?.iter() {
                flatten_statement(child, output)?;
            }
        }
        Err(_) => output.push(statement),
    }
    Ok(())
}

/// ABI-complete Rust representation of TVM's `IfThenElseNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.IfThenElse"]
#[type_final]
pub struct IfThenElseObj {
    base: StmtObj,
    condition: Expr,
    then_case: Stmt,
    else_case: Option<Stmt>,
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
        Ok(self.condition.clone())
    }

    /// Return the true branch.
    pub fn then_case(&self) -> Result<Stmt> {
        Ok(self.then_case.clone())
    }

    /// Return the optional false branch.
    pub fn else_case(&self) -> Result<Option<Stmt>> {
        Ok(self.else_case.clone())
    }
}

impl IfThenElse {
    /// Construct a conditional statement directly in Rust.
    pub fn new(condition: &Expr, then_case: &Stmt, else_case: Option<&Stmt>) -> Result<Self> {
        primitive_type(condition, "IfThenElse condition")?;
        Ok(Self {
            data: ObjectArc::new(IfThenElseObj {
                base: StmtObj::new(None),
                condition: condition.clone(),
                then_case: then_case.clone(),
                else_case: else_case.cloned(),
            }),
        })
    }
}

/// Execution policy attached to a TIR `For` loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
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

/// ABI-complete Rust representation of TVM's `ForNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.For"]
#[type_final]
pub struct ForObj {
    base: StmtObj,
    loop_var: Var,
    min: Expr,
    extent: Expr,
    kind: ForKind,
    body: Stmt,
    thread_binding: Option<IterVar>,
    annotations: AnyMap<String>,
    step: Option<Expr>,
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
        Ok(self.loop_var.clone())
    }

    /// Return the inclusive starting value.
    pub fn minimum(&self) -> Result<Expr> {
        Ok(self.min.clone())
    }

    /// Return the number of loop iterations.
    pub fn extent(&self) -> Result<Expr> {
        Ok(self.extent.clone())
    }

    /// Return the loop execution policy.
    pub fn kind(&self) -> Result<ForKind> {
        Ok(self.kind)
    }

    /// Return the loop body.
    pub fn body(&self) -> Result<Stmt> {
        Ok(self.body.clone())
    }

    /// Return the optional thread axis bound by this loop.
    pub fn thread_binding(&self) -> Result<Option<IterVar>> {
        Ok(self.thread_binding.clone())
    }

    /// Return heterogeneous loop annotations.
    pub fn annotations(&self) -> Result<AnyMap<String>> {
        Ok(self.annotations.clone())
    }

    /// Return an optional non-unit loop step.
    pub fn step(&self) -> Result<Option<Expr>> {
        Ok(self.step.clone())
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
            &AnyMap::new(),
            None,
            None,
        )
    }

    /// Construct a loop through C++ so bound dtypes and loop invariants are validated.
    #[allow(clippy::too_many_arguments)]
    pub fn with_metadata(
        loop_var: &Var,
        minimum: &Expr,
        extent: &Expr,
        kind: ForKind,
        body: &Stmt,
        thread_binding: Option<&IterVar>,
        annotations: &AnyMap<String>,
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

/// ABI-complete Rust representation of TVM's `PrimFuncNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.PrimFunc"]
#[type_final]
pub struct PrimFuncObj {
    base: BaseFuncObj,
    params: Array<Var>,
    ret_type: crate::ir::Type,
    body: Stmt,
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
        Ok(self.params.clone())
    }

    /// Return the function return type.
    pub fn ret_type(&self) -> Result<crate::ir::Type> {
        Ok(self.ret_type.clone())
    }

    /// Return the function body.
    pub fn body(&self) -> Result<Stmt> {
        Ok(self.body.clone())
    }
}

impl AssertStmt {
    /// Construct a leaf assertion directly in Rust with one string message part.
    pub fn new(condition: &Expr, error_kind: &str, message: &str) -> Result<Self> {
        let dtype = primitive_type(condition, "AssertStmt condition")?.dtype()?;
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
        let error_kind = StringImm::new(error_kind)?;
        let message = StringImm::new(message)?;
        let message_parts = Array::new(vec![message]);
        Ok(Self {
            data: ObjectArc::new(AssertStmtObj {
                base: StmtObj::new(None),
                condition: condition.clone(),
                error_kind,
                message_parts,
            }),
        })
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
    /// Construct `Evaluate(value)` directly in Rust.
    pub fn new(value: &Expr) -> Result<Self> {
        if value.clone().try_cast::<Var>().is_ok()
            && value.ty()?.try_cast::<buffer::BufferType>().is_ok()
        {
            return Err(Error::new(
                VALUE_ERROR,
                "a buffer variable cannot be used as a scalar Evaluate value",
                "",
            ));
        }
        Ok(Self {
            data: ObjectArc::new(EvaluateObj {
                base: StmtObj::new(None),
                value: value.clone(),
            }),
        })
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

    /// Construct a PrimFunc through C++ so its return and function types are derived.
    pub fn new<S: AnyCompatible>(params: Vec<Var>, body: &S) -> Result<Self> {
        let params = Array::new(params);
        let attrs = DictAttrs::empty()?;
        let ret_type = crate::ir::Type::missing()?;
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
