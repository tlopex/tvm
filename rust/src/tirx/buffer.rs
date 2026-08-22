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
use tvm_ffi::{Any, AnyView, Array, ObjectArc, ObjectRefCast, Result, String};

use super::{Stmt, StmtObj};
use crate::ir::{
    Expr, ExprObj, PrimExprConvertible, PrimExprConvertibleObj, PrimType, Range, Span, Type,
    TypeObj, Var,
};

/// Opaque Rust view of TVM's immutable buffer access contract.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.BufferType"]
#[type_final]
pub struct BufferTypeObj {
    base: TypeObj,
}

/// Reference-counted buffer type carried by an ordinary `ir.Var`.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct BufferType {
    data: ObjectArc<BufferTypeObj>,
}

impl std::ops::Deref for BufferType {
    type Target = BufferTypeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for BufferTypeObj {
    type Target = TypeObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl BufferTypeObj {
    /// Return the primitive element type.
    pub fn dtype(&self) -> Result<PrimType> {
        crate::reflected_field!(self, "dtype")?.try_into()
    }

    /// Return the logical address space.
    pub fn storage_scope(&self) -> Result<String> {
        crate::reflected_field!(self, "storage_scope")?.try_into()
    }

    /// Return logical extents for every accessed dimension.
    pub fn shape(&self) -> Result<Array<Expr>> {
        crate::reflected_field!(self, "shape")?.try_into()
    }

    /// Return explicit strides, or an empty array for compact storage.
    pub fn strides(&self) -> Result<Array<Expr>> {
        crate::reflected_field!(self, "strides")?.try_into()
    }

    /// Return the element offset from the physical base pointer.
    pub fn element_offset(&self) -> Result<Expr> {
        crate::reflected_field!(self, "elem_offset")?.try_into()
    }

    /// Return the required data-pointer alignment in bytes.
    pub fn data_alignment(&self) -> Result<i64> {
        crate::reflected_field!(self, "data_alignment")?.try_into()
    }

    /// Return the divisibility guarantee for the element offset.
    pub fn offset_factor(&self) -> Result<i64> {
        crate::reflected_field!(self, "offset_factor")?.try_into()
    }

    /// Return the optional layout without requiring a handwritten layout binding.
    pub fn layout(&self) -> Result<Any> {
        crate::reflected_field!(self, "layout")
    }

    /// Return any explicitly allocated multidimensional address.
    pub fn allocated_addresses(&self) -> Result<Array<Expr>> {
        crate::reflected_field!(self, "allocated_addr")?.try_into()
    }
}

impl BufferType {
    /// Construct a compact buffer type with TVM's standard alignment metadata.
    pub fn new(storage_scope: &str, dtype: &str, shape: Vec<Expr>) -> Result<Self> {
        Self::with_metadata(
            storage_scope,
            &PrimType::new(dtype)?,
            shape,
            Vec::new(),
            &Expr::int("int64", 0)?,
            64,
            1,
            &Any::from(()),
            Vec::new(),
            None,
        )
    }

    /// Construct a buffer type with all currently reflected access metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn with_metadata(
        storage_scope: &str,
        dtype: &PrimType,
        shape: Vec<Expr>,
        strides: Vec<Expr>,
        element_offset: &Expr,
        data_alignment: i64,
        offset_factor: i64,
        layout: &Any,
        allocated_addresses: Vec<Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        let storage_scope = String::from(storage_scope);
        let shape = Array::new(shape);
        let strides = Array::new(strides);
        let allocated_addresses = Array::new(allocated_addresses);
        let span = span.cloned();
        crate::global_function!("tirx.BufferType")?
            .call_packed(&[
                AnyView::from(&storage_scope),
                AnyView::from(dtype),
                AnyView::from(&shape),
                AnyView::from(&strides),
                AnyView::from(element_offset),
                AnyView::from(&data_alignment),
                AnyView::from(&offset_factor),
                AnyView::from(layout),
                AnyView::from(&allocated_addresses),
                AnyView::from(&span),
            ])?
            .try_into()
    }

    /// Construct a buffer variable.  Its runtime identity is an ordinary `ir.Var`.
    pub fn new_var(&self, name: &str) -> Result<Var> {
        let name = String::from(name);
        let none = ();
        crate::global_function!("tirx.BufferVar")?
            .call_packed(&[
                AnyView::from(&name),
                AnyView::from(self),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

impl From<BufferType> for Type {
    fn from(value: BufferType) -> Self {
        value
            .try_cast()
            .expect("tirx.BufferType must be a subtype of ir.Type")
    }
}

/// Opaque Rust view of a buffer read.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.BufferLoad"]
#[type_final]
pub struct BufferLoadObj {
    base: ExprObj,
}

/// Reference-counted handle to a TIR buffer read.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct BufferLoad {
    data: ObjectArc<BufferLoadObj>,
}

impl std::ops::Deref for BufferLoad {
    type Target = BufferLoadObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for BufferLoadObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl BufferLoadObj {
    /// Return the ordinary variable carrying this access's `BufferType`.
    pub fn buffer(&self) -> Result<Var> {
        crate::reflected_field!(self, "buffer")?.try_into()
    }

    /// Return one index per accessed dimension.
    pub fn indices(&self) -> Result<Array<Expr>> {
        crate::reflected_field!(self, "indices")?.try_into()
    }

    /// Return an optional vector access predicate.
    pub fn predicate(&self) -> Result<Option<Expr>> {
        crate::reflected_field!(self, "predicate")?.try_into()
    }
}

impl BufferLoad {
    /// Construct a buffer read.
    pub fn new(buffer: &Var, indices: Vec<Expr>, predicate: Option<&Expr>) -> Result<Self> {
        let indices = Array::new(indices);
        let predicate = predicate.cloned();
        let none = ();
        crate::global_function!("tirx.BufferLoad")?
            .call_packed(&[
                AnyView::from(buffer),
                AnyView::from(&indices),
                AnyView::from(&predicate),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

impl From<BufferLoad> for Expr {
    fn from(value: BufferLoad) -> Self {
        value
            .try_cast()
            .expect("tirx.BufferLoad must be a subtype of ir.Expr")
    }
}

/// Opaque Rust view of a buffer write.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.BufferStore"]
#[type_final]
pub struct BufferStoreObj {
    base: StmtObj,
}

/// Reference-counted handle to a TIR buffer write.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct BufferStore {
    data: ObjectArc<BufferStoreObj>,
}

impl std::ops::Deref for BufferStore {
    type Target = BufferStoreObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for BufferStoreObj {
    type Target = StmtObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl BufferStoreObj {
    /// Return the ordinary variable carrying this access's `BufferType`.
    pub fn buffer(&self) -> Result<Var> {
        crate::reflected_field!(self, "buffer")?.try_into()
    }

    /// Return the value written by this store.
    pub fn value(&self) -> Result<Expr> {
        crate::reflected_field!(self, "value")?.try_into()
    }

    /// Return one index per accessed dimension.
    pub fn indices(&self) -> Result<Array<Expr>> {
        crate::reflected_field!(self, "indices")?.try_into()
    }

    /// Return an optional vector access predicate.
    pub fn predicate(&self) -> Result<Option<Expr>> {
        crate::reflected_field!(self, "predicate")?.try_into()
    }
}

impl BufferStore {
    /// Construct a buffer write.
    pub fn new(
        buffer: &Var,
        value: &Expr,
        indices: Vec<Expr>,
        predicate: Option<&Expr>,
    ) -> Result<Self> {
        let indices = Array::new(indices);
        let predicate = predicate.cloned();
        let none = ();
        crate::global_function!("tirx.BufferStore")?
            .call_packed(&[
                AnyView::from(buffer),
                AnyView::from(value),
                AnyView::from(&indices),
                AnyView::from(&predicate),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

impl From<BufferStore> for Stmt {
    fn from(value: BufferStore) -> Self {
        value
            .try_cast()
            .expect("tirx.BufferStore must be a subtype of tirx.Stmt")
    }
}

/// Opaque Rust view of one declared buffer region.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.BufferRegion"]
#[type_final]
pub struct BufferRegionObj {
    base: PrimExprConvertibleObj,
}

/// Reference-counted handle to a multidimensional buffer region.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct BufferRegion {
    data: ObjectArc<BufferRegionObj>,
}

impl std::ops::Deref for BufferRegion {
    type Target = BufferRegionObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for BufferRegionObj {
    type Target = PrimExprConvertibleObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl BufferRegionObj {
    /// Return the ordinary variable carrying the region's `BufferType`.
    pub fn buffer(&self) -> Result<Var> {
        crate::reflected_field!(self, "buffer")?.try_into()
    }

    /// Return one range per declared dimension.
    pub fn region(&self) -> Result<Array<Range>> {
        crate::reflected_field!(self, "region")?.try_into()
    }
}

impl BufferRegion {
    /// Construct a declared region of `buffer`.
    pub fn new(buffer: &Var, region: Vec<Range>) -> Result<Self> {
        let region = Array::new(region);
        crate::global_function!("tirx.BufferRegion")?
            .call_packed(&[AnyView::from(buffer), AnyView::from(&region)])?
            .try_into()
    }
}

impl From<BufferRegion> for PrimExprConvertible {
    fn from(value: BufferRegion) -> Self {
        value
            .try_cast()
            .expect("tirx.BufferRegion must be a subtype of ir.PrimExprConvertible")
    }
}

/// Opaque Rust view of a match-buffer declaration.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.MatchBufferRegion"]
#[type_final]
pub struct MatchBufferRegionObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to a match-buffer declaration.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct MatchBufferRegion {
    data: ObjectArc<MatchBufferRegionObj>,
}

impl std::ops::Deref for MatchBufferRegion {
    type Target = MatchBufferRegionObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl MatchBufferRegionObj {
    /// Return the target buffer variable introduced by this declaration.
    pub fn buffer(&self) -> Result<Var> {
        crate::reflected_field!(self, "buffer")?.try_into()
    }

    /// Return the source region matched by the target buffer.
    pub fn source(&self) -> Result<BufferRegion> {
        crate::reflected_field!(self, "source")?.try_into()
    }
}

impl MatchBufferRegion {
    /// Construct a match-buffer declaration.
    pub fn new(buffer: &Var, source: &BufferRegion) -> Result<Self> {
        crate::global_function!("tirx.MatchBufferRegion")?
            .call_packed(&[AnyView::from(buffer), AnyView::from(source)])?
            .try_into()
    }
}
