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
use tvm_ffi::{AnyView, Array, Error, Map, ObjectArc, ObjectRefCast, Result, String, VALUE_ERROR};

use super::{primitive_type, Stmt, StmtObj};
use crate::ir::{
    Expr, ExprObj, PrimExprConvertible, PrimExprConvertibleObj, PrimType, Range, Span, Type,
    TypeObj, Var,
};

/// Opaque Rust view of TVM's abstract layout base class.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Layout"]
pub struct LayoutObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to a TIRx buffer layout.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Layout {
    data: ObjectArc<LayoutObj>,
}

impl std::ops::Deref for Layout {
    type Target = LayoutObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Opaque Rust view of one interned TIRx layout axis.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Axis"]
#[type_final]
pub struct AxisObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to an interned layout axis.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Axis {
    data: ObjectArc<AxisObj>,
}

impl std::ops::Deref for Axis {
    type Target = AxisObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl AxisObj {
    /// Return the registered axis name.
    pub fn name(&self) -> Result<String> {
        crate::reflected_field!(self, "name")?.try_into()
    }
}

impl Axis {
    /// Return TVM's process-wide interned axis for `name`.
    pub fn get(name: &str) -> Result<Self> {
        let name = String::from(name);
        crate::global_function!("tirx.AxisGet")?
            .call_packed(&[AnyView::from(&name)])?
            .try_into()
    }
}

/// ABI-complete Rust representation of one layout extent/stride/axis component.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Iter"]
#[type_final]
pub struct IterObj {
    base: tvm_ffi::Object,
    extent: Expr,
    stride: Expr,
    axis: Axis,
}

/// Reference-counted handle to one layout iterator.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Iter {
    data: ObjectArc<IterObj>,
}

impl std::ops::Deref for Iter {
    type Target = IterObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl IterObj {
    /// Return the number of logical positions.
    pub fn extent(&self) -> Result<Expr> {
        Ok(self.extent.clone())
    }

    /// Return the physical stride.
    pub fn stride(&self) -> Result<Expr> {
        Ok(self.stride.clone())
    }

    /// Return the target layout axis.
    pub fn axis(&self) -> Result<Axis> {
        Ok(self.axis.clone())
    }
}

impl Iter {
    /// Construct one layout iterator directly in Rust.
    pub fn new(extent: &Expr, stride: &Expr, axis: &Axis) -> Result<Self> {
        primitive_type(extent, "layout iterator extent")?;
        primitive_type(stride, "layout iterator stride")?;
        Ok(Self {
            data: ObjectArc::new(IterObj {
                base: tvm_ffi::Object::new(),
                extent: extent.clone(),
                stride: stride.clone(),
                axis: axis.clone(),
            }),
        })
    }
}

/// Opaque Rust view of TVM's concrete tiled layout.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.TileLayout"]
#[type_final]
pub struct TileLayoutObj {
    base: LayoutObj,
}

/// Reference-counted handle to a tiled layout.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct TileLayout {
    data: ObjectArc<TileLayoutObj>,
}

impl std::ops::Deref for TileLayout {
    type Target = TileLayoutObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for TileLayoutObj {
    type Target = LayoutObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl TileLayoutObj {
    /// Return iterators that partition the logical tile.
    pub fn shard(&self) -> Result<Array<Iter>> {
        crate::reflected_field!(self, "shard")?.try_into()
    }

    /// Return replicated layout iterators.
    pub fn replica(&self) -> Result<Array<Iter>> {
        crate::reflected_field!(self, "replica")?.try_into()
    }

    /// Return per-axis physical offsets.
    pub fn offset(&self) -> Result<Map<Axis, Expr>> {
        crate::reflected_field!(self, "offset")?.try_into()
    }
}

impl TileLayout {
    /// Construct a tile layout through C++ because `LayoutNode` is polymorphic.
    pub fn new(shard: Vec<Iter>, replica: Vec<Iter>, offset: Map<Axis, Expr>) -> Result<Self> {
        let shard = Array::new(shard);
        let replica = Array::new(replica);
        crate::global_function!("tirx.TileLayout")?
            .call_packed(&[
                AnyView::from(&shard),
                AnyView::from(&replica),
                AnyView::from(&offset),
            ])?
            .try_into()
    }
}

impl From<TileLayout> for Layout {
    fn from(value: TileLayout) -> Self {
        value
            .try_cast()
            .expect("tirx.TileLayout must be a subtype of tirx.Layout")
    }
}

/// ABI-complete Rust representation of TVM's immutable buffer access contract.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.BufferType"]
#[type_final]
pub struct BufferTypeObj {
    base: TypeObj,
    dtype: PrimType,
    storage_scope: String,
    shape: Array<Expr>,
    strides: Array<Expr>,
    elem_offset: Expr,
    data_alignment: i32,
    offset_factor: i32,
    layout: Option<Layout>,
    allocated_addr: Array<Expr>,
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
        Ok(self.dtype.clone())
    }

    /// Return the logical address space.
    pub fn storage_scope(&self) -> Result<String> {
        Ok(self.storage_scope.clone())
    }

    /// Return logical extents for every accessed dimension.
    pub fn shape(&self) -> Result<Array<Expr>> {
        Ok(self.shape.clone())
    }

    /// Return explicit strides, or an empty array for compact storage.
    pub fn strides(&self) -> Result<Array<Expr>> {
        Ok(self.strides.clone())
    }

    /// Return the element offset from the physical base pointer.
    pub fn element_offset(&self) -> Result<Expr> {
        Ok(self.elem_offset.clone())
    }

    /// Return the required data-pointer alignment in bytes.
    pub fn data_alignment(&self) -> Result<i64> {
        Ok(i64::from(self.data_alignment))
    }

    /// Return the divisibility guarantee for the element offset.
    pub fn offset_factor(&self) -> Result<i64> {
        Ok(i64::from(self.offset_factor))
    }

    /// Return the optional physical layout.
    pub fn layout(&self) -> Result<Option<Layout>> {
        Ok(self.layout.clone())
    }

    /// Return any explicitly allocated multidimensional address.
    pub fn allocated_addresses(&self) -> Result<Array<Expr>> {
        Ok(self.allocated_addr.clone())
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
            None,
            Vec::new(),
            None,
        )
    }

    /// Construct a buffer type directly in Rust with all reflected access metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn with_metadata(
        storage_scope: &str,
        dtype: &PrimType,
        shape: Vec<Expr>,
        strides: Vec<Expr>,
        element_offset: &Expr,
        data_alignment: i64,
        offset_factor: i64,
        layout: Option<&Layout>,
        allocated_addresses: Vec<Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        for extent in &shape {
            primitive_type(extent, "buffer shape extent")?;
        }
        for stride in &strides {
            primitive_type(stride, "buffer stride")?;
        }
        primitive_type(element_offset, "buffer element offset")?;
        for address in &allocated_addresses {
            primitive_type(address, "buffer allocated address")?;
        }
        let storage_scope = String::from(if storage_scope.is_empty() {
            "global"
        } else {
            storage_scope
        });
        let shape = Array::new(shape);
        let strides = Array::new(strides);
        let allocated_addresses = Array::new(allocated_addresses);
        let layout = layout.cloned();
        let data_alignment = if data_alignment <= 0 {
            64
        } else {
            i32::try_from(data_alignment).map_err(|_| integer_overflow("data_alignment"))?
        };
        let offset_factor = if offset_factor == 0 {
            1
        } else {
            i32::try_from(offset_factor).map_err(|_| integer_overflow("offset_factor"))?
        };
        Ok(Self {
            data: ObjectArc::new(BufferTypeObj {
                base: TypeObj::new(span.cloned()),
                dtype: dtype.clone(),
                storage_scope,
                shape,
                strides,
                elem_offset: element_offset.clone(),
                data_alignment,
                offset_factor,
                layout,
                allocated_addr: allocated_addresses,
            }),
        })
    }

    /// Construct a buffer variable.  Its runtime identity is an ordinary `ir.Var`.
    pub fn new_var(&self, name: &str) -> Result<Var> {
        Var::with_type(name, &self.clone().into())
    }
}

fn integer_overflow(field: &str) -> Error {
    Error::new(
        VALUE_ERROR,
        &format!("{field} does not fit TVM's 32-bit integer field"),
        "",
    )
}

impl From<BufferType> for Type {
    fn from(value: BufferType) -> Self {
        value
            .try_cast()
            .expect("tirx.BufferType must be a subtype of ir.Type")
    }
}

/// ABI-complete Rust representation of a buffer read.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.BufferLoad"]
#[type_final]
pub struct BufferLoadObj {
    base: ExprObj,
    buffer: Var,
    indices: Array<Expr>,
    predicate: Option<Expr>,
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
        Ok(self.buffer.clone())
    }

    /// Return one index per accessed dimension.
    pub fn indices(&self) -> Result<Array<Expr>> {
        Ok(self.indices.clone())
    }

    /// Return an optional vector access predicate.
    pub fn predicate(&self) -> Result<Option<Expr>> {
        Ok(self.predicate.clone())
    }
}

impl BufferLoad {
    /// Construct a buffer read through C++ index/predicate validation and dtype legalization.
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

/// ABI-complete Rust representation of a buffer write.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.BufferStore"]
#[type_final]
pub struct BufferStoreObj {
    base: StmtObj,
    buffer: Var,
    value: Expr,
    indices: Array<Expr>,
    predicate: Option<Expr>,
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
        Ok(self.buffer.clone())
    }

    /// Return the value written by this store.
    pub fn value(&self) -> Result<Expr> {
        Ok(self.value.clone())
    }

    /// Return one index per accessed dimension.
    pub fn indices(&self) -> Result<Array<Expr>> {
        Ok(self.indices.clone())
    }

    /// Return an optional vector access predicate.
    pub fn predicate(&self) -> Result<Option<Expr>> {
        Ok(self.predicate.clone())
    }
}

impl BufferStore {
    /// Construct a buffer write through C++ shape, lane, and predicate validation.
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
    /// Construct a declared region through its polymorphic C++ base.
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

/// ABI-complete Rust representation of a match-buffer declaration.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.MatchBufferRegion"]
#[type_final]
pub struct MatchBufferRegionObj {
    base: tvm_ffi::Object,
    buffer: Var,
    source: BufferRegion,
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
        Ok(self.buffer.clone())
    }

    /// Return the source region matched by the target buffer.
    pub fn source(&self) -> Result<BufferRegion> {
        Ok(self.source.clone())
    }
}

impl MatchBufferRegion {
    /// Construct through C++ scope, dtype, alignment, and region validation.
    pub fn new(buffer: &Var, source: &BufferRegion) -> Result<Self> {
        crate::global_function!("tirx.MatchBufferRegion")?
            .call_packed(&[AnyView::from(buffer), AnyView::from(source)])?
            .try_into()
    }
}
