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
    AnyMap, AnyView, Array, DLDataTypeCode, Error, ObjectArc, ObjectRefCast, Result, String,
    TYPE_ERROR, VALUE_ERROR,
};

use super::{primitive_type, BufferRegion, MatchBufferRegion, Stmt, StmtObj};
use crate::ir::{Expr, PrimExprConvertible, PrimExprConvertibleObj, Range, Span, Var};

/// Scheduling role attached to a TIR block iteration variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum IterVarType {
    DataParallel = 0,
    ThreadIndex = 1,
    CommutativeReduction = 2,
    Ordered = 3,
    Opaque = 4,
    Unrolled = 5,
    Vectorized = 6,
    Parallelized = 7,
    Tensorized = 8,
}

impl TryFrom<i64> for IterVarType {
    type Error = Error;

    fn try_from(value: i64) -> Result<Self> {
        match value {
            0 => Ok(Self::DataParallel),
            1 => Ok(Self::ThreadIndex),
            2 => Ok(Self::CommutativeReduction),
            3 => Ok(Self::Ordered),
            4 => Ok(Self::Opaque),
            5 => Ok(Self::Unrolled),
            6 => Ok(Self::Vectorized),
            7 => Ok(Self::Parallelized),
            8 => Ok(Self::Tensorized),
            _ => Err(Error::new(
                VALUE_ERROR,
                &format!("unknown tirx.IterVarType value {value}"),
                "",
            )),
        }
    }
}

/// Opaque Rust view of a TIR block iteration variable.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.IterVar"]
#[type_final]
pub struct IterVarObj {
    base: PrimExprConvertibleObj,
}

/// Reference-counted handle to a block iteration variable.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct IterVar {
    data: ObjectArc<IterVarObj>,
}

impl std::ops::Deref for IterVar {
    type Target = IterVarObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for IterVarObj {
    type Target = PrimExprConvertibleObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl IterVarObj {
    /// Return the optional iteration domain.
    pub fn domain(&self) -> Result<Option<Range>> {
        crate::reflected_field!(self, "dom")?.try_into()
    }

    /// Return the variable defined by this axis.
    pub fn variable(&self) -> Result<Var> {
        crate::reflected_field!(self, "var")?.try_into()
    }

    /// Return the scheduling role of this axis.
    pub fn iter_type(&self) -> Result<IterVarType> {
        IterVarType::try_from(i64::try_from(crate::reflected_field!(self, "iter_type")?)?)
    }

    /// Return the runtime thread tag, or an empty string when no tag is set.
    pub fn thread_tag(&self) -> Result<String> {
        crate::reflected_field!(self, "thread_tag")?.try_into()
    }
}

impl IterVar {
    /// Construct an untagged iteration variable through its polymorphic C++ base.
    pub fn new(domain: &Range, variable: &Var, iter_type: IterVarType) -> Result<Self> {
        let iter_type = iter_type as i64;
        let thread_tag = String::from("");
        let none = ();
        crate::global_function!("tirx.IterVar")?
            .call_packed(&[
                AnyView::from(domain),
                AnyView::from(variable),
                AnyView::from(&iter_type),
                AnyView::from(&thread_tag),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

impl From<IterVar> for PrimExprConvertible {
    fn from(value: IterVar) -> Self {
        value
            .try_cast()
            .expect("tirx.IterVar must be a subtype of ir.PrimExprConvertible")
    }
}

/// ABI-complete Rust representation of TVM's current TIR block node.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.SBlock"]
#[type_final]
pub struct SBlockObj {
    base: StmtObj,
    iter_vars: Array<IterVar>,
    reads: Array<BufferRegion>,
    writes: Array<BufferRegion>,
    name_hint: String,
    alloc_buffers: Array<Var>,
    match_buffers: Array<MatchBufferRegion>,
    annotations: AnyMap<String>,
    init: Option<Stmt>,
    body: Stmt,
}

/// Reference-counted handle to a TIR scheduling block.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct SBlock {
    data: ObjectArc<SBlockObj>,
}

impl std::ops::Deref for SBlock {
    type Target = SBlockObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for SBlockObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl SBlockObj {
    /// Return the block axes defined recursively for its fields and body.
    pub fn iter_vars(&self) -> Result<Array<IterVar>> {
        Ok(self.iter_vars.clone())
    }

    /// Return declared read regions.
    pub fn reads(&self) -> Result<Array<BufferRegion>> {
        Ok(self.reads.clone())
    }

    /// Return declared write regions.
    pub fn writes(&self) -> Result<Array<BufferRegion>> {
        Ok(self.writes.clone())
    }

    /// Return the diagnostic block name.
    pub fn name_hint(&self) -> Result<String> {
        Ok(self.name_hint.clone())
    }

    /// Return buffers allocated within this block.
    pub fn allocated_buffers(&self) -> Result<Array<Var>> {
        Ok(self.alloc_buffers.clone())
    }

    /// Return match-buffer declarations.
    pub fn match_buffers(&self) -> Result<Array<MatchBufferRegion>> {
        Ok(self.match_buffers.clone())
    }

    /// Return heterogeneous scheduling annotations.
    pub fn annotations(&self) -> Result<AnyMap<String>> {
        Ok(self.annotations.clone())
    }

    /// Return an optional reduction initializer.
    pub fn init(&self) -> Result<Option<Stmt>> {
        Ok(self.init.clone())
    }

    /// Return the main block body.
    pub fn body(&self) -> Result<Stmt> {
        Ok(self.body.clone())
    }
}

impl SBlock {
    /// Construct a block with no axes, declared regions, or local buffers.
    pub fn new(name_hint: &str, body: &Stmt) -> Result<Self> {
        Self::with_metadata(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            name_hint,
            body,
            None,
            Vec::new(),
            Vec::new(),
            &AnyMap::new(),
            None,
        )
    }

    /// Construct a scheduling block directly in Rust with all structural fields.
    #[allow(clippy::too_many_arguments)]
    pub fn with_metadata(
        iter_vars: Vec<IterVar>,
        reads: Vec<BufferRegion>,
        writes: Vec<BufferRegion>,
        name_hint: &str,
        body: &Stmt,
        init: Option<&Stmt>,
        allocated_buffers: Vec<Var>,
        match_buffers: Vec<MatchBufferRegion>,
        annotations: &AnyMap<String>,
        span: Option<&Span>,
    ) -> Result<Self> {
        let iter_vars = Array::new(iter_vars);
        let reads = Array::new(reads);
        let writes = Array::new(writes);
        let name_hint = String::from(name_hint);
        let init = init.cloned();
        let allocated_buffers = Array::new(allocated_buffers);
        let match_buffers = Array::new(match_buffers);
        Ok(Self {
            data: ObjectArc::new(SBlockObj {
                base: StmtObj::new(span.cloned()),
                iter_vars,
                reads,
                writes,
                name_hint,
                alloc_buffers: allocated_buffers,
                match_buffers,
                annotations: annotations.clone(),
                init,
                body: body.clone(),
            }),
        })
    }
}

impl From<SBlock> for Stmt {
    fn from(value: SBlock) -> Self {
        value
            .try_cast()
            .expect("tirx.SBlock must be a subtype of tirx.Stmt")
    }
}

/// ABI-complete Rust representation of a block realization.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.SBlockRealize"]
#[type_final]
pub struct SBlockRealizeObj {
    base: StmtObj,
    iter_values: Array<Expr>,
    predicate: Expr,
    block: SBlock,
}

/// Reference-counted handle to one realized scheduling block.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct SBlockRealize {
    data: ObjectArc<SBlockRealizeObj>,
}

impl std::ops::Deref for SBlockRealize {
    type Target = SBlockRealizeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for SBlockRealizeObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl SBlockRealizeObj {
    /// Return the values bound to the block's axes.
    pub fn iter_values(&self) -> Result<Array<Expr>> {
        Ok(self.iter_values.clone())
    }

    /// Return the execution predicate.
    pub fn predicate(&self) -> Result<Expr> {
        Ok(self.predicate.clone())
    }

    /// Return the scheduling block being realized.
    pub fn block(&self) -> Result<SBlock> {
        Ok(self.block.clone())
    }
}

impl SBlockRealize {
    /// Construct one realization of `block` directly in Rust.
    pub fn new(iter_values: Vec<Expr>, predicate: &Expr, block: &SBlock) -> Result<Self> {
        if block.iter_vars()?.len() != iter_values.len() {
            return Err(Error::new(
                VALUE_ERROR,
                "SBlockRealize needs the same number of iter_vars and binding values",
                "",
            ));
        }
        for value in &iter_values {
            primitive_type(value, "SBlockRealize binding value")?;
        }
        let predicate_dtype = primitive_type(predicate, "SBlockRealize predicate")?.dtype()?;
        if predicate_dtype.code != DLDataTypeCode::kDLBool as u8 {
            return Err(Error::new(
                TYPE_ERROR,
                "SBlockRealize predicate must have bool type",
                "",
            ));
        }
        Ok(Self {
            data: ObjectArc::new(SBlockRealizeObj {
                base: StmtObj::new(None),
                iter_values: Array::new(iter_values),
                predicate: predicate.clone(),
                block: block.clone(),
            }),
        })
    }
}

impl From<SBlockRealize> for Stmt {
    fn from(value: SBlockRealize) -> Self {
        value
            .try_cast()
            .expect("tirx.SBlockRealize must be a subtype of tirx.Stmt")
    }
}
