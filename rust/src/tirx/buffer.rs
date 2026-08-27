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
use tvm_ffi::{Any, Array, Error, Function, Map, ObjectArc, ObjectCore, Result, String};

use super::{Stmt, StmtObj};
use crate::ir::{
    Expr, ExprObj, PrimExprConvertible, PrimExprConvertibleObj, PrimType, Range, Span, Type,
    TypeObj, Var,
};

/// Opaque Rust view of objects that expose tensor-like producer behavior.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.DataProducer"]
pub struct DataProducerObj {
    base: PrimExprConvertibleObj,
}

/// Reference-counted handle to any concrete data producer.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct DataProducer {
    data: ObjectArc<DataProducerObj>,
}

impl std::ops::Deref for DataProducer {
    type Target = DataProducerObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for DataProducerObj {
    type Target = PrimExprConvertibleObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl DataProducer {
    fn call<T>(&self, name: &str) -> Result<T>
    where
        T: TryFrom<Any, Error = Error>,
    {
        Function::from_type_method(DataProducerObj::type_index(), name)?
            .call_tuple((self,))?
            .try_into()
    }

    /// Return the producer's logical result shape.
    pub fn shape(&self) -> Result<Array<Expr>> {
        self.call("get_shape")
    }

    /// Return the producer's primitive element type.
    pub fn data_type(&self) -> Result<PrimType> {
        self.call("get_data_type")
    }

    /// Return the producer's diagnostic name.
    pub fn name_hint(&self) -> Result<String> {
        self.call("get_name_hint")
    }
}

/// Opaque Rust view of TVM's layout base class.
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

impl Layout {
    /// Check whether this layout can describe the supplied logical shape.
    pub fn compatible_with_shape(&self, shape: &Array<Expr>) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.LayoutCompatibleWithShape")
            .call_tuple((self, shape))?
            .try_into()
    }

    /// Validate the concrete layout through TVM's registered layout service.
    pub fn verify_well_formed(&self) -> Result<bool> {
        tvm_ffi::cached_global_func!("tirx.LayoutVerifyWellFormed")
            .call_tuple((self,))?
            .try_into()
    }

    /// Return the logical size for all axes or one named axis.
    pub fn get_size(&self, axis_name: Option<&str>) -> Result<Expr> {
        let axis_name = axis_name.map(String::from);
        tvm_ffi::cached_global_func!("tirx.LayoutGetSize")
            .call_tuple((self, axis_name))?
            .try_into()
    }

    /// Return the physical span for all axes or one named axis.
    pub fn get_span(&self, axis_name: Option<&str>) -> Result<Expr> {
        let axis_name = axis_name.map(String::from);
        tvm_ffi::cached_global_func!("tirx.LayoutGetSpan")
            .call_tuple((self, axis_name))?
            .try_into()
    }

    /// Map one structured coordinate through this layout.
    pub fn apply(&self, coord: &Array<Expr>) -> Result<Map<String, Expr>> {
        tvm_ffi::cached_global_func!("tirx.LayoutApply")
            .call_tuple((self, coord))?
            .try_into()
    }

    /// Map one flattened coordinate through this layout.
    pub fn apply_linear(&self, coord: &Expr) -> Result<Map<String, Expr>> {
        tvm_ffi::cached_global_func!("tirx.LayoutApplyLinear")
            .call_tuple((self, coord))?
            .try_into()
    }

    /// Map a coordinate whose dimensions are grouped by `shape`.
    pub fn apply_with_shape(
        &self,
        coord: &Array<Expr>,
        shape: &Array<Expr>,
    ) -> Result<Map<String, Expr>> {
        tvm_ffi::cached_global_func!("tirx.LayoutApplyWithShape")
            .call_tuple((self, coord, shape))?
            .try_into()
    }

    /// Return the canonical form of this layout.
    pub fn canonicalize(&self) -> Result<Layout> {
        tvm_ffi::cached_global_func!("tirx.LayoutCanonicalize")
            .call_tuple((self,))?
            .try_into()
    }

    /// Tile this layout with an outer tile.
    pub fn tile(
        &self,
        outer: &TileLayout,
        outer_shape: &Array<Expr>,
        inner_shape: &Array<Expr>,
    ) -> Result<Layout> {
        tvm_ffi::cached_global_func!("tirx.LayoutTile")
            .call_tuple((self, outer, outer_shape, inner_shape))?
            .try_into()
    }

    /// Restrict this layout to one region.
    pub fn slice(&self, shape: &Array<Expr>, region: &Array<Range>) -> Result<Option<Layout>> {
        tvm_ffi::cached_global_func!("tirx.LayoutSlice")
            .call_tuple((self, shape, region))?
            .try_into()
    }

    /// Form the layout direct sum with a left tile.
    pub fn direct_sum(
        &self,
        left: &TileLayout,
        left_shape: &Array<Expr>,
        right_shape: &Array<Expr>,
    ) -> Result<Layout> {
        tvm_ffi::cached_global_func!("tirx.LayoutDirectSum")
            .call_tuple((self, left, left_shape, right_shape))?
            .try_into()
    }

    /// Recover an outer tile when this layout is the tiled inner component.
    pub fn is_tile_inner(
        &self,
        layout: &Layout,
        tiled_shape: &Array<Expr>,
        inner_shape: &Array<Expr>,
    ) -> Result<Option<TileLayout>> {
        tvm_ffi::cached_global_func!("tirx.LayoutIsTileInner")
            .call_tuple((self, layout, tiled_shape, inner_shape))?
            .try_into()
    }

    /// Recover an inner layout when this layout is the tiled outer component.
    pub fn is_tile_outer(
        &self,
        layout: &Layout,
        tiled_shape: &Array<Expr>,
        outer_shape: &Array<Expr>,
    ) -> Result<Option<Layout>> {
        tvm_ffi::cached_global_func!("tirx.LayoutIsTileOuter")
            .call_tuple((self, layout, tiled_shape, outer_shape))?
            .try_into()
    }

    /// Recover the left direct-sum component when this layout is the right one.
    pub fn is_direct_sum_right(
        &self,
        layout: &Layout,
        interleaved_shape: &Array<Expr>,
        right_shape: &Array<Expr>,
    ) -> Result<Option<TileLayout>> {
        tvm_ffi::cached_global_func!("tirx.LayoutIsDirectSumRight")
            .call_tuple((self, layout, interleaved_shape, right_shape))?
            .try_into()
    }

    /// Recover the right direct-sum component when this layout is the left one.
    pub fn is_direct_sum_left(
        &self,
        layout: &Layout,
        interleaved_shape: &Array<Expr>,
        left_shape: &Array<Expr>,
    ) -> Result<Option<Layout>> {
        tvm_ffi::cached_global_func!("tirx.LayoutIsDirectSumLeft")
            .call_tuple((self, layout, interleaved_shape, left_shape))?
            .try_into()
    }
}

/// Opaque Rust view of one registered TIRx layout axis.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Axis"]
#[type_final]
pub struct AxisObj {
    base: tvm_ffi::Object,
}
crate::abi::reflected_fields!(AxisObj {
    name => "name": String,
});

/// Reference-counted handle identified by the shared axis registry index.
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

impl Axis {
    /// Resolve the axis through TVM's shared native registry.
    pub fn get(name: &str) -> Result<Self> {
        let name = String::from(name);
        tvm_ffi::cached_global_func!("tirx.AxisGet")
            .call_tuple((&name,))?
            .try_into()
    }
}

/// Opaque Rust view of one layout extent/stride/axis component.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Iter"]
#[type_final]
pub struct IterObj {
    base: tvm_ffi::Object,
}
crate::abi::reflected_fields!(IterObj {
    extent => "extent": Expr,
    stride => "stride": Expr,
    axis => "axis": Axis,
});

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

impl Iter {
    /// Construct one layout iterator through TVM's native constructor.
    pub fn new<E, S, A>(extent: E, stride: S, axis: A) -> Result<Self>
    where
        E: Into<Expr>,
        S: Into<Expr>,
        A: Into<Axis>,
    {
        tvm_ffi::cached_global_func!("tirx.Iter")
            .call_tuple((extent.into(), stride.into(), axis.into()))?
            .try_into()
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
crate::abi::reflected_fields!(TileLayoutObj {
    shard => "shard": Array<Iter>,
    replica => "replica": Array<Iter>,
    offset => "offset": Map<Axis, Expr>,
});

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

impl TileLayout {
    /// Construct a tile layout through TVM's native constructor.
    pub fn new(shard: Vec<Iter>, replica: Vec<Iter>, offset: Map<Axis, Expr>) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.TileLayout")
            .call_tuple((Array::new(shard), Array::new(replica), offset))?
            .try_into()
    }
}

/// Opaque Rust view of TVM's immutable buffer access contract.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.BufferType"]
#[type_final]
pub struct BufferTypeObj {
    base: TypeObj,
}
crate::abi::reflected_fields!(BufferTypeObj {
    dtype => "dtype": PrimType,
    storage_scope => "storage_scope": String,
    shape => "shape": Array<Expr>,
    strides => "strides": Array<Expr>,
    elem_offset => "elem_offset": Expr,
    data_alignment => "data_alignment": i32,
    offset_factor => "offset_factor": i32,
    layout => "layout": Option<Layout>,
    allocated_addr => "allocated_addr": Array<Expr>,
});

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

impl BufferType {
    /// Construct a compact buffer type through TVM's native constructor.
    pub fn new(storage_scope: &str, dtype: &str, shape: Vec<Expr>) -> Result<Self> {
        let dtype = PrimType::new(dtype)?;
        Self::call_native_constructor(
            storage_scope,
            dtype,
            shape,
            Vec::new(),
            None,
            0,
            0,
            None,
            Vec::new(),
            None,
        )
    }

    /// Construct a buffer type through TVM's native constructor with full metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn with_metadata(
        storage_scope: &str,
        dtype: PrimType,
        shape: Vec<Expr>,
        strides: Vec<Expr>,
        element_offset: Expr,
        data_alignment: i32,
        offset_factor: i32,
        layout: Option<Layout>,
        allocated_addresses: Vec<Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        Self::call_native_constructor(
            storage_scope,
            dtype,
            shape,
            strides,
            Some(element_offset),
            data_alignment,
            offset_factor,
            layout,
            allocated_addresses,
            span,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn call_native_constructor(
        storage_scope: &str,
        dtype: PrimType,
        shape: Vec<Expr>,
        strides: Vec<Expr>,
        element_offset: Option<Expr>,
        data_alignment: i32,
        offset_factor: i32,
        layout: Option<Layout>,
        allocated_addresses: Vec<Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        let storage_scope = String::from(storage_scope);
        tvm_ffi::cached_global_func!("tirx.BufferType")
            .call_tuple((
                storage_scope,
                dtype,
                Array::new(shape),
                Array::new(strides),
                element_offset,
                data_alignment,
                offset_factor,
                layout,
                Array::new(allocated_addresses),
                span.cloned(),
            ))?
            .try_into()
    }

    /// Construct a buffer variable.  Its runtime identity is an ordinary `ir.Var`.
    pub fn new_var(&self, name: &str) -> Result<Var> {
        let name = String::from(name);
        tvm_ffi::cached_global_func!("tirx.BufferVar")
            .call_tuple((&name, self, Option::<Span>::None))?
            .try_into()
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
crate::abi::reflected_fields!(BufferLoadObj {
    buffer => "buffer": Var,
    indices => "indices": Array<Expr>,
    predicate => "predicate": Option<Expr>,
});

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

impl BufferLoad {
    /// Construct a buffer read through TVM's native validation and constructor.
    pub fn new<B>(buffer: B, indices: Vec<Expr>, predicate: Option<Expr>) -> Result<Self>
    where
        B: Into<Var>,
    {
        Self::with_span(buffer.into(), indices, predicate, None)
    }

    /// Construct a validated buffer read with optional source metadata.
    pub fn with_span(
        buffer: Var,
        indices: Vec<Expr>,
        predicate: Option<Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.BufferLoad")
            .call_tuple((buffer, Array::new(indices), predicate, span.cloned()))?
            .try_into()
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
crate::abi::reflected_fields!(BufferStoreObj {
    buffer => "buffer": Var,
    value => "value": Expr,
    indices => "indices": Array<Expr>,
    predicate => "predicate": Option<Expr>,
});

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

impl BufferStore {
    /// Construct a buffer write through TVM's native validation and constructor.
    pub fn new<B, V>(
        buffer: B,
        value: V,
        indices: Vec<Expr>,
        predicate: Option<Expr>,
    ) -> Result<Self>
    where
        B: Into<Var>,
        V: Into<Expr>,
    {
        Self::with_span(buffer.into(), value.into(), indices, predicate, None)
    }

    /// Construct a validated buffer write with optional source metadata.
    pub fn with_span(
        buffer: Var,
        value: Expr,
        indices: Vec<Expr>,
        predicate: Option<Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("tirx.BufferStore")
            .call_tuple((buffer, value, Array::new(indices), predicate, span.cloned()))?
            .try_into()
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
crate::abi::reflected_fields!(BufferRegionObj {
    buffer => "buffer": Var,
    region => "region": Array<Range>,
});

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

impl BufferRegion {
    /// Validate and construct a declared region through TVM's native constructor.
    pub fn new<B>(buffer: B, region: Vec<Range>) -> Result<Self>
    where
        B: Into<Var>,
    {
        tvm_ffi::cached_global_func!("tirx.BufferRegion")
            .call_tuple((buffer.into(), Array::new(region)))?
            .try_into()
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
crate::abi::reflected_fields!(MatchBufferRegionObj {
    buffer => "buffer": Var,
    source => "source": BufferRegion,
});

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

impl MatchBufferRegion {
    /// Validate and construct a match-buffer declaration through TVM's native constructor.
    pub fn new<B, S>(buffer: B, source: S) -> Result<Self>
    where
        B: Into<Var>,
        S: Into<BufferRegion>,
    {
        tvm_ffi::cached_global_func!("tirx.MatchBufferRegion")
            .call_tuple((buffer.into(), source.into()))?
            .try_into()
    }
}

tvm_ffi::impl_object_upcast!(
    DataProducer => PrimExprConvertible,
    TileLayout => Layout,
    BufferType => Type,
    BufferLoad => Expr,
    BufferStore => Stmt,
    BufferRegion => PrimExprConvertible,
);
