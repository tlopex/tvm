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
use tvm_ffi::{Any, Array, Error, Map, ObjectArc, Result, String, VALUE_ERROR};

use super::{BufferRegion, MatchBufferRegion, Stmt, StmtObj};
use crate::ir::{Expr, PrimExprConvertible, PrimExprConvertibleObj, Range, Span, Var};

/// Scheduling role attached to a TIR block iteration variable.
///
/// Keep unknown future native enumerators representable instead of creating an
/// invalid Rust enum discriminant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct IterVarType(i32);

#[allow(non_upper_case_globals)]
impl IterVarType {
    pub const DataParallel: Self = Self(0);
    pub const ThreadIndex: Self = Self(1);
    pub const CommutativeReduction: Self = Self(2);
    pub const Ordered: Self = Self(3);
    pub const Opaque: Self = Self(4);
    pub const Unrolled: Self = Self(5);
    pub const Vectorized: Self = Self(6);
    pub const Parallelized: Self = Self(7);
    pub const Tensorized: Self = Self(8);

    /// Preserve an enumerator not yet known by this Rust binding.
    pub const fn from_raw(value: i32) -> Self {
        Self(value)
    }

    pub const fn as_raw(self) -> i32 {
        self.0
    }
}

impl TryFrom<i64> for IterVarType {
    type Error = Error;

    fn try_from(value: i64) -> Result<Self> {
        i32::try_from(value).map(Self).map_err(|_| {
            Error::new(
                VALUE_ERROR,
                &format!(
                    "tirx.IterVarType value {value} does not fit its native i32 representation"
                ),
                "",
            )
        })
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
crate::abi::reflected_fields!(IterVarObj {
    dom => "dom": Option<Range>,
    var => "var": Var,
    iter_type_raw => "iter_type": i64,
    thread_tag => "thread_tag": String,
    span => "span": Option<Span>,
});

impl IterVarObj {
    pub fn iter_type(&self) -> Result<IterVarType> {
        IterVarType::try_from(self.iter_type_raw()?)
    }
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

impl IterVar {
    /// Construct an untagged iteration variable through TVM's native constructor.
    pub fn new<D, V>(domain: D, variable: V, iter_type: IterVarType) -> Result<Self>
    where
        D: Into<Range>,
        V: Into<Var>,
    {
        Self::with_metadata(Some(domain.into()), variable.into(), iter_type, "", None)
    }

    /// Validate and construct an iteration variable, allowing a missing domain
    /// for thread axes just like the native constructor.
    pub fn with_metadata(
        domain: Option<Range>,
        variable: Var,
        iter_type: IterVarType,
        thread_tag: &str,
        span: Option<&Span>,
    ) -> Result<Self> {
        let thread_tag = String::from(thread_tag);
        tvm_ffi::cached_global_func!("tirx.IterVar")
            .call_tuple((
                domain,
                variable,
                iter_type.as_raw(),
                thread_tag,
                span.cloned(),
            ))?
            .try_into()
    }
}

/// Opaque Rust view of TVM's current TIR block node.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.SBlock"]
#[type_final]
pub struct SBlockObj {
    base: StmtObj,
}
crate::abi::reflected_fields!(SBlockObj {
    iter_vars => "iter_vars": Array<IterVar>,
    reads => "reads": Array<BufferRegion>,
    writes => "writes": Array<BufferRegion>,
    name_hint => "name_hint": String,
    alloc_buffers => "alloc_buffers": Array<Var>,
    match_buffers => "match_buffers": Array<MatchBufferRegion>,
    annotations => "annotations": Map<String, Any>,
    init => "init": Option<Stmt>,
    body => "body": Stmt,
});

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

impl SBlock {
    /// Construct a block with no axes, declared regions, or local buffers.
    pub fn new<B>(name_hint: &str, body: B) -> Result<Self>
    where
        B: Into<Stmt>,
    {
        Self::with_metadata(
            Vec::new(),
            Vec::new(),
            Vec::new(),
            name_hint,
            body.into(),
            None,
            Vec::new(),
            Vec::new(),
            Map::new(),
            None,
        )
    }

    /// Construct a scheduling block through TVM's native constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn with_metadata(
        iter_vars: Vec<IterVar>,
        reads: Vec<BufferRegion>,
        writes: Vec<BufferRegion>,
        name_hint: &str,
        body: Stmt,
        init: Option<Stmt>,
        allocated_buffers: Vec<Var>,
        match_buffers: Vec<MatchBufferRegion>,
        annotations: Map<String, Any>,
        span: Option<&Span>,
    ) -> Result<Self> {
        let name_hint = String::from(name_hint);
        tvm_ffi::cached_global_func!("tirx.SBlock")
            .call_tuple((
                Array::new(iter_vars),
                Array::new(reads),
                Array::new(writes),
                name_hint,
                body,
                init,
                Array::new(allocated_buffers),
                Array::new(match_buffers),
                annotations,
                span.cloned(),
            ))?
            .try_into()
    }
}

/// Opaque Rust view of a block realization.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.SBlockRealize"]
#[type_final]
pub struct SBlockRealizeObj {
    base: StmtObj,
}
crate::abi::reflected_fields!(SBlockRealizeObj {
    iter_values => "iter_values": Array<Expr>,
    predicate => "predicate": Expr,
    block => "block": SBlock,
});

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

impl SBlockRealize {
    /// Construct one realization of `block` through TVM's native constructor.
    pub fn new<P, B>(iter_values: Vec<Expr>, predicate: P, block: B) -> Result<Self>
    where
        P: Into<Expr>,
        B: Into<SBlock>,
    {
        Self::with_span(iter_values, predicate, block, None)
    }

    /// Construct one realization with optional source metadata.
    pub fn with_span<P, B>(
        iter_values: Vec<Expr>,
        predicate: P,
        block: B,
        span: Option<&Span>,
    ) -> Result<Self>
    where
        P: Into<Expr>,
        B: Into<SBlock>,
    {
        tvm_ffi::cached_global_func!("tirx.SBlockRealize")
            .call_tuple((
                Array::new(iter_values),
                predicate.into(),
                block.into(),
                span.cloned(),
            ))?
            .try_into()
    }
}

tvm_ffi::impl_object_upcast!(
    IterVar => PrimExprConvertible,
    SBlock => Stmt,
    SBlockRealize => Stmt,
);
