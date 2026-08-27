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
use tvm_ffi::{Any, Array, Error, Map, ObjectArc, ObjectRefCast, Result, String, VALUE_ERROR};

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
/// Opaque Rust view of TVM's `AddNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Add"]
#[type_final]
pub struct AddObj {
    base: ExprObj,
}
crate::abi::reflected_fields!(AddObj {
    a => "a": Expr,
    b => "b": Expr,
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
    /// Construct an addition expression through TVM's native constructor.
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
        tvm_ffi::cached_global_func!("tirx.Add")
            .call_tuple((lhs.into(), rhs.into(), span.cloned()))?
            .try_into()
    }
}

macro_rules! define_binary_expression {
    ($object:ident, $reference:ident, $type_key:literal, $description:literal) => {
        #[doc = concat!("Opaque Rust view of TVM's `", $type_key, "` node.")]
        #[repr(C)]
        #[derive(Object)]
        #[type_key = $type_key]
        #[type_final]
        pub struct $object {
            base: ExprObj,
        }
        crate::abi::reflected_fields!($object {
            a => "a": Expr,
            b => "b": Expr,
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
            /// Construct the binary expression through TVM's native constructor.
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
                tvm_ffi::cached_global_func!($type_key)
                    .call_tuple((lhs.into(), rhs.into(), span.cloned()))?
                    .try_into()
            }
        }
    };
}

define_binary_expression!(SubObj, Sub, "tirx.Sub", "a subtraction expression");
define_binary_expression!(MulObj, Mul, "tirx.Mul", "a multiplication expression");

/// Opaque Rust view of TVM's `StringImmNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.StringImm"]
#[type_final]
pub struct StringImmObj {
    base: ExprObj,
}
crate::abi::reflected_fields!(StringImmObj {
    value => "value": String,
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
    /// Construct a string literal through TVM's native constructor.
    pub fn new(value: &str) -> Result<Self> {
        Self::with_span(value, None)
    }

    /// Construct a string literal with optional source metadata.
    pub fn with_span(value: &str, span: Option<&Span>) -> Result<Self> {
        let value = String::from(value);
        tvm_ffi::cached_global_func!("tirx.StringImm")
            .call_tuple((&value, span.cloned()))?
            .try_into()
    }
}

/// Opaque Rust view of TVM's `StmtNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Stmt"]
pub struct StmtObj {
    base: tvm_ffi::Object,
}
crate::abi::reflected_fields!(StmtObj {
    span => "span": Option<Span>,
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
            flatten_statement(statement, &mut flattened)?;
        }
        match flattened.len() {
            0 => Ok(Evaluate::from_i64(0)?.into()),
            1 => Ok(flattened.pop().expect("one statement is present")),
            _ => Ok(SeqStmt::with_span(flattened, span)?.into()),
        }
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
crate::abi::reflected_fields!(AssertStmtObj {
    condition => "condition": Expr,
    error_kind => "error_kind": StringImm,
    message_parts => "message_parts": Array<StringImm>,
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

/// Opaque Rust view of TVM's `EvaluateNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Evaluate"]
#[type_final]
pub struct EvaluateObj {
    base: StmtObj,
}
crate::abi::reflected_fields!(EvaluateObj {
    value => "value": Expr,
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

/// Opaque Rust view of TVM's `SeqStmtNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.SeqStmt"]
#[type_final]
pub struct SeqStmtObj {
    base: StmtObj,
}
crate::abi::reflected_fields!(SeqStmtObj {
    seq => "seq": Array<Stmt>,
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
    /// Construct and recursively flatten a sequence through TVM's native constructor.
    pub fn new(statements: Vec<Stmt>) -> Result<Self> {
        Self::with_span(statements, None)
    }

    /// Construct and recursively flatten a sequence with source metadata.
    pub fn with_span(statements: Vec<Stmt>, span: Option<&Span>) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.SeqStmt")
            .call_tuple((Array::new(statements), span.cloned()))?
            .try_into()
    }
}

fn flatten_statement(statement: Stmt, output: &mut Vec<Stmt>) -> Result<()> {
    match statement.clone().try_cast::<SeqStmt>() {
        Ok(sequence) => {
            for child in sequence.seq()?.iter() {
                flatten_statement(child, output)?;
            }
        }
        Err(_) if !is_evaluate_zero(&statement)? => output.push(statement),
        Err(_) => {}
    }
    Ok(())
}

fn is_evaluate_zero(statement: &Stmt) -> Result<bool> {
    let Ok(evaluate) = statement.clone().try_cast::<Evaluate>() else {
        return Ok(false);
    };
    let Ok(literal) = evaluate.value()?.try_cast::<IntImm>() else {
        return Ok(false);
    };
    Ok(literal.value()? == 0)
}

/// Opaque Rust view of TVM's `IfThenElseNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.IfThenElse"]
#[type_final]
pub struct IfThenElseObj {
    base: StmtObj,
}
crate::abi::reflected_fields!(IfThenElseObj {
    condition => "condition": Expr,
    then_case => "then_case": Stmt,
    else_case => "else_case": Option<Stmt>,
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
    /// Construct a conditional statement through TVM's native constructor.
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
        tvm_ffi::cached_global_func!("tirx.IfThenElse")
            .call_tuple((condition.into(), then_case.into(), else_case, span.cloned()))?
            .try_into()
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

/// Opaque Rust view of TVM's `ForNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.For"]
#[type_final]
pub struct ForObj {
    base: StmtObj,
}
crate::abi::reflected_fields!(ForObj {
    loop_var => "loop_var": Var,
    min => "min": Expr,
    extent => "extent": Expr,
    kind_raw => "kind": i64,
    body => "body": Stmt,
    thread_binding => "thread_binding": Option<IterVar>,
    annotations => "annotations": Map<String, Any>,
    step => "step": Option<Expr>,
});

impl ForObj {
    pub fn kind(&self) -> Result<ForKind> {
        ForKind::try_from(self.kind_raw()?)
    }
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

    /// Construct a loop through TVM's native normalization and validation.
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
        tvm_ffi::cached_global_func!("tirx.For")
            .call_tuple((
                loop_var,
                minimum,
                extent,
                kind.as_raw(),
                body,
                thread_binding,
                annotations,
                step,
                span.cloned(),
            ))?
            .try_into()
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
crate::abi::reflected_fields!(PrimFuncObj {
    params => "params": Array<Var>,
    ret_type => "ret_type": crate::ir::Type,
    body => "body": Stmt,
});

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
    /// Construct a leaf assertion through TVM's native constructor.
    pub fn new<C>(condition: C, error_kind: &str, message: &str) -> Result<Self>
    where
        C: Into<Expr>,
    {
        let error_kind = StringImm::new(error_kind)?;
        let message = StringImm::new(message)?;
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
        tvm_ffi::cached_global_func!("tirx.AssertStmt")
            .call_tuple((
                condition.into(),
                error_kind.into(),
                Array::new(message_parts),
                span.cloned(),
            ))?
            .try_into()
    }
}

impl Evaluate {
    /// Construct `Evaluate(value)` through TVM's native constructor.
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
        tvm_ffi::cached_global_func!("tirx.Evaluate")
            .call_tuple((value.into(), span.cloned()))?
            .try_into()
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

    /// Construct a PrimFunc through TVM's native constructor.
    pub fn new<S>(params: Vec<Var>, body: S) -> Result<Self>
    where
        S: Into<Stmt>,
    {
        let attrs = DictAttrs::empty()?;
        let ret_type = crate::ir::Type::missing()?;
        Self::with_metadata(params, body, ret_type, attrs, None)
    }

    /// Construct a PrimFunc with metadata through TVM's native constructor.
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
        tvm_ffi::cached_global_func!("tirx.PrimFunc")
            .call_tuple((
                Array::new(params),
                body.into(),
                ret_type.into(),
                attrs.into(),
                span.cloned(),
            ))?
            .try_into()
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
