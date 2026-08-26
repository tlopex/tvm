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
    Any, AnyView, Array, DLDataType, DLDataTypeCode, DLDataTypeExt, Error, Function, Map,
    ObjectArc, ObjectCore, ObjectRefCast, Result, String, TYPE_ERROR, VALUE_ERROR,
};

use super::{primitive_type, Stmt, StmtObj};
use crate::ir::{
    Expr, ExprObj, PrimExprConvertible, PrimExprConvertibleObj, PrimType, Range, Span, Type,
    TypeObj, Var,
};

/// ABI-complete prefix for objects that expose tensor-like producer behavior.
///
/// This base is intentionally not Rust-allocatable on its own. Concrete
/// producers must register the shared reflected methods.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.DataProducer"]
pub struct DataProducerObj {
    base: PrimExprConvertibleObj,
}
crate::abi::impl_object_layout!(DataProducerObj {});

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
        Function::from_type_method(AnyView::from(self).type_index(), name)?
            .call_tuple_with_len::<1, _>((self,))?
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

impl From<DataProducer> for PrimExprConvertible {
    fn from(value: DataProducer) -> Self {
        value
            .try_cast()
            .expect("tirx.DataProducer must be a subtype of ir.PrimExprConvertible")
    }
}

/// ABI-complete Rust representation of TVM's layout base class.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Layout"]
pub struct LayoutObj {
    base: tvm_ffi::Object,
}
crate::abi::impl_object_layout!(LayoutObj {});

impl LayoutObj {
    fn new() -> Self {
        Self {
            base: tvm_ffi::Object::new(),
        }
    }
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
    fn method(&self, name: &str) -> Result<Function> {
        Function::from_type_method(AnyView::from(self).type_index(), name)
    }

    #[inline]
    fn call0<T>(&self, name: &str) -> Result<T>
    where
        T: TryFrom<Any, Error = Error>,
    {
        self.method(name)?
            .call_tuple_with_len::<1, _>((self,))?
            .try_into()
    }

    #[inline]
    fn call1<T>(&self, name: &str, arg0: AnyView<'_>) -> Result<T>
    where
        T: TryFrom<Any, Error = Error>,
    {
        self.method(name)?
            .call_packed(&[AnyView::from(self), arg0])?
            .try_into()
    }

    #[inline]
    fn call2<T>(&self, name: &str, arg0: AnyView<'_>, arg1: AnyView<'_>) -> Result<T>
    where
        T: TryFrom<Any, Error = Error>,
    {
        self.method(name)?
            .call_packed(&[AnyView::from(self), arg0, arg1])?
            .try_into()
    }

    #[inline]
    fn call3<T>(
        &self,
        name: &str,
        arg0: AnyView<'_>,
        arg1: AnyView<'_>,
        arg2: AnyView<'_>,
    ) -> Result<T>
    where
        T: TryFrom<Any, Error = Error>,
    {
        self.method(name)?
            .call_packed(&[AnyView::from(self), arg0, arg1, arg2])?
            .try_into()
    }

    /// Check whether this layout can describe the supplied logical shape.
    pub fn compatible_with_shape(&self, shape: &Array<Expr>) -> Result<bool> {
        self.call1("compatible_with_shape", AnyView::from(shape))
    }

    /// Validate the concrete layout through its reflected method.
    pub fn verify_well_formed(&self) -> Result<bool> {
        self.call0("verify_well_formed")
    }

    /// Return the logical size for all axes or one named axis.
    pub fn get_size(&self, axis_name: Option<&str>) -> Result<Expr> {
        let axis_name = axis_name.map(String::from);
        self.call1("get_size", AnyView::from(&axis_name))
    }

    /// Return the physical span for all axes or one named axis.
    pub fn get_span(&self, axis_name: Option<&str>) -> Result<Expr> {
        let axis_name = axis_name.map(String::from);
        self.call1("get_span", AnyView::from(&axis_name))
    }

    /// Map one structured coordinate through this layout.
    pub fn apply(&self, coord: &Array<Expr>) -> Result<Map<String, Expr>> {
        self.call1("apply", AnyView::from(coord))
    }

    /// Map one flattened coordinate through this layout.
    pub fn apply_linear(&self, coord: &Expr) -> Result<Map<String, Expr>> {
        self.call1("apply_linear", AnyView::from(coord))
    }

    /// Map a coordinate whose dimensions are grouped by `shape`.
    pub fn apply_with_shape(
        &self,
        coord: &Array<Expr>,
        shape: &Array<Expr>,
    ) -> Result<Map<String, Expr>> {
        self.call2(
            "apply_with_shape",
            AnyView::from(coord),
            AnyView::from(shape),
        )
    }

    /// Return the canonical form of this layout.
    pub fn canonicalize(&self) -> Result<Layout> {
        self.call0("canonicalize")
    }

    /// Tile this layout with an outer tile.
    pub fn tile(
        &self,
        outer: &TileLayout,
        outer_shape: &Array<Expr>,
        inner_shape: &Array<Expr>,
    ) -> Result<Layout> {
        self.call3(
            "tile",
            AnyView::from(outer),
            AnyView::from(outer_shape),
            AnyView::from(inner_shape),
        )
    }

    /// Restrict this layout to one region.
    pub fn slice(&self, shape: &Array<Expr>, region: &Array<Range>) -> Result<Option<Layout>> {
        self.call2("slice", AnyView::from(shape), AnyView::from(region))
    }

    /// Form the layout direct sum with a left tile.
    pub fn direct_sum(
        &self,
        left: &TileLayout,
        left_shape: &Array<Expr>,
        right_shape: &Array<Expr>,
    ) -> Result<Layout> {
        self.call3(
            "direct_sum",
            AnyView::from(left),
            AnyView::from(left_shape),
            AnyView::from(right_shape),
        )
    }

    /// Recover an outer tile when this layout is the tiled inner component.
    pub fn is_tile_inner(
        &self,
        layout: &Layout,
        tiled_shape: &Array<Expr>,
        inner_shape: &Array<Expr>,
    ) -> Result<Option<TileLayout>> {
        self.call3(
            "is_tile_inner",
            AnyView::from(layout),
            AnyView::from(tiled_shape),
            AnyView::from(inner_shape),
        )
    }

    /// Recover an inner layout when this layout is the tiled outer component.
    pub fn is_tile_outer(
        &self,
        layout: &Layout,
        tiled_shape: &Array<Expr>,
        outer_shape: &Array<Expr>,
    ) -> Result<Option<Layout>> {
        self.call3(
            "is_tile_outer",
            AnyView::from(layout),
            AnyView::from(tiled_shape),
            AnyView::from(outer_shape),
        )
    }

    /// Recover the left direct-sum component when this layout is the right one.
    pub fn is_direct_sum_right(
        &self,
        layout: &Layout,
        interleaved_shape: &Array<Expr>,
        right_shape: &Array<Expr>,
    ) -> Result<Option<TileLayout>> {
        self.call3(
            "is_direct_sum_right",
            AnyView::from(layout),
            AnyView::from(interleaved_shape),
            AnyView::from(right_shape),
        )
    }

    /// Recover the right direct-sum component when this layout is the left one.
    pub fn is_direct_sum_left(
        &self,
        layout: &Layout,
        interleaved_shape: &Array<Expr>,
        left_shape: &Array<Expr>,
    ) -> Result<Option<Layout>> {
        self.call3(
            "is_direct_sum_left",
            AnyView::from(layout),
            AnyView::from(interleaved_shape),
            AnyView::from(left_shape),
        )
    }
}

/// ABI-complete Rust representation of one registered TIRx layout axis.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Axis"]
#[type_final]
pub struct AxisObj {
    base: tvm_ffi::Object,
    pub name: String,
    pub registry_index: u32,
}
crate::abi::impl_object_layout!(AxisObj {
    "name" => name: String,
    "registry_index" => registry_index: u32,
});

impl crate::abi::ConstructorRecipe for AxisObj {
    const NUM_INPUTS: usize = 1;
    const DERIVED_FIELDS: &'static [&'static str] = &["registry_index"];
}

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
    /// Resolve registry metadata and allocate the axis handle in Rust.
    pub fn get(name: &str) -> Result<Self> {
        let name = String::from(name);
        let prepared = crate::abi::prepare_constructor::<AxisObj>(&[AnyView::from(&name)])?;
        let registry_index =
            crate::abi::prepared_field::<i64>(&prepared, AxisObj::TYPE_KEY, "registry_index")
                .and_then(|index| {
                    u32::try_from(index).map_err(|_| {
                        Error::new(VALUE_ERROR, "axis registry index does not fit u32", "")
                    })
                })?;
        Ok(Self::from_prepared_fields(name, registry_index))
    }

    // Keep this allocator private: an arbitrary name/index pair can alias the
    // wrong entry in C++ attribute tables.  `get` first asks the shared axis
    // registry for the index and is the only safe public construction path.
    fn from_prepared_fields(name: String, registry_index: u32) -> Self {
        Self {
            data: crate::abi::allocate_object(AxisObj {
                base: tvm_ffi::Object::new(),
                name,
                registry_index,
            }),
        }
    }
}

// Compile-check the generated owned allocator contract without making the raw
// registry identity constructor part of the public API.
const _: fn(String, u32) -> Axis = Axis::from_prepared_fields;

/// ABI-complete Rust representation of one layout extent/stride/axis component.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Iter"]
#[type_final]
pub struct IterObj {
    base: tvm_ffi::Object,
    pub extent: Expr,
    pub stride: Expr,
    pub axis: Axis,
}
crate::abi::impl_object_layout!(IterObj {
    "extent" => extent: Expr,
    "stride" => stride: Expr,
    "axis" => axis: Axis,
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
    /// Construct one layout iterator directly in Rust.
    pub fn new(extent: &Expr, stride: &Expr, axis: &Axis) -> Result<Self> {
        primitive_type(extent, "layout iterator extent")?;
        primitive_type(stride, "layout iterator stride")?;
        Ok(Self::from_complete_fields(
            extent.clone(),
            stride.clone(),
            axis.clone(),
        ))
    }

    /// Construct one layout iterator from every physical field after external validation.
    pub fn from_complete_fields(extent: Expr, stride: Expr, axis: Axis) -> Self {
        Self {
            data: crate::abi::allocate_object(IterObj {
                base: tvm_ffi::Object::new(),
                extent,
                stride,
                axis,
            }),
        }
    }
}

/// ABI-complete Rust representation of TVM's concrete tiled layout.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.TileLayout"]
#[type_final]
pub struct TileLayoutObj {
    base: LayoutObj,
    pub shard: Array<Iter>,
    pub replica: Array<Iter>,
    pub offset: Map<Axis, Expr>,
}
crate::abi::impl_object_layout!(TileLayoutObj {
    "shard" => shard: Array<Iter>,
    "replica" => replica: Array<Iter>,
    "offset" => offset: Map<Axis, Expr>,
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
    /// Construct a tile layout directly in Rust.
    pub fn new(shard: Vec<Iter>, replica: Vec<Iter>, offset: Map<Axis, Expr>) -> Self {
        Self::from_complete_fields(Array::new(shard), Array::new(replica), offset)
    }

    /// Construct a tile layout from every physical field.
    pub fn from_complete_fields(
        shard: Array<Iter>,
        replica: Array<Iter>,
        offset: Map<Axis, Expr>,
    ) -> Self {
        Self {
            data: crate::abi::allocate_object(TileLayoutObj {
                base: LayoutObj::new(),
                shard,
                replica,
                offset,
            }),
        }
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
    pub dtype: PrimType,
    pub storage_scope: String,
    pub shape: Array<Expr>,
    pub strides: Array<Expr>,
    pub elem_offset: Expr,
    pub data_alignment: i32,
    pub offset_factor: i32,
    pub layout: Option<Layout>,
    pub allocated_addr: Array<Expr>,
}
crate::abi::impl_object_layout!(BufferTypeObj {
    "dtype" => dtype: PrimType,
    "storage_scope" => storage_scope: String,
    "shape" => shape: Array<Expr>,
    "strides" => strides: Array<Expr>,
    "elem_offset" => elem_offset: Expr,
    "data_alignment" => data_alignment: i32,
    "offset_factor" => offset_factor: i32,
    "layout" => layout: Option<Layout>,
    "allocated_addr" => allocated_addr: Array<Expr>,
});

impl crate::abi::ConstructorRecipe for BufferTypeObj {
    const NUM_INPUTS: usize = 5;
    const DERIVED_FIELDS: &'static [&'static str] = &[
        "storage_scope",
        "elem_offset",
        "data_alignment",
        "offset_factor",
    ];
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

impl BufferType {
    /// Construct a compact buffer type directly in Rust.
    pub fn new(storage_scope: &str, dtype: &str, shape: Vec<Expr>) -> Result<Self> {
        let dtype = PrimType::new(dtype)?;
        Self::prepare_and_allocate(
            storage_scope,
            &dtype,
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

    /// Construct a buffer type directly in Rust with all reflected access metadata.
    #[allow(clippy::too_many_arguments)]
    pub fn with_metadata(
        storage_scope: &str,
        dtype: &PrimType,
        shape: Vec<Expr>,
        strides: Vec<Expr>,
        element_offset: &Expr,
        data_alignment: i32,
        offset_factor: i32,
        layout: Option<&Layout>,
        allocated_addresses: Vec<Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        Self::prepare_and_allocate(
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
    fn prepare_and_allocate(
        storage_scope: &str,
        dtype: &PrimType,
        shape: Vec<Expr>,
        strides: Vec<Expr>,
        element_offset: Option<&Expr>,
        data_alignment: i32,
        offset_factor: i32,
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
        if let Some(element_offset) = element_offset {
            primitive_type(element_offset, "buffer element offset")?;
        }
        for address in &allocated_addresses {
            primitive_type(address, "buffer allocated address")?;
        }
        let storage_scope = String::from(storage_scope);
        let shape = Array::new(shape);
        let strides = Array::new(strides);
        let allocated_addresses = Array::new(allocated_addresses);
        let layout = layout.cloned();
        let element_offset = element_offset.cloned();
        let prepared = crate::abi::prepare_constructor::<BufferTypeObj>(&[
            AnyView::from(&storage_scope),
            AnyView::from(&shape),
            AnyView::from(&element_offset),
            AnyView::from(&data_alignment),
            AnyView::from(&offset_factor),
        ])?;
        let storage_scope: String =
            crate::abi::prepared_field(&prepared, BufferTypeObj::TYPE_KEY, "storage_scope")?;
        let element_offset: Expr =
            crate::abi::prepared_field(&prepared, BufferTypeObj::TYPE_KEY, "elem_offset")?;
        let data_alignment: i64 =
            crate::abi::prepared_field(&prepared, BufferTypeObj::TYPE_KEY, "data_alignment")?;
        let offset_factor: i64 =
            crate::abi::prepared_field(&prepared, BufferTypeObj::TYPE_KEY, "offset_factor")?;
        let data_alignment =
            i32::try_from(data_alignment).map_err(|_| integer_overflow("data_alignment"))?;
        let offset_factor =
            i32::try_from(offset_factor).map_err(|_| integer_overflow("offset_factor"))?;
        Ok(Self::from_complete_fields(
            span.cloned(),
            dtype.clone(),
            storage_scope,
            shape,
            strides,
            element_offset,
            data_alignment,
            offset_factor,
            layout,
            allocated_addresses,
        ))
    }

    /// Construct a buffer type from every physical field without applying defaults.
    #[allow(clippy::too_many_arguments)]
    pub fn from_complete_fields(
        span: Option<Span>,
        dtype: PrimType,
        storage_scope: String,
        shape: Array<Expr>,
        strides: Array<Expr>,
        elem_offset: Expr,
        data_alignment: i32,
        offset_factor: i32,
        layout: Option<Layout>,
        allocated_addr: Array<Expr>,
    ) -> Self {
        Self {
            data: crate::abi::allocate_object(BufferTypeObj {
                base: TypeObj::new(span),
                dtype,
                storage_scope,
                shape,
                strides,
                elem_offset,
                data_alignment,
                offset_factor,
                layout,
                allocated_addr,
            }),
        }
    }

    /// Construct a buffer variable.  Its runtime identity is an ordinary `ir.Var`.
    pub fn new_var(&self, name: &str) -> Var {
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
    pub buffer: Var,
    pub indices: Array<Expr>,
    pub predicate: Option<Expr>,
}
crate::abi::impl_object_layout!(BufferLoadObj {
    "buffer" => buffer: Var,
    "indices" => indices: Array<Expr>,
    "predicate" => predicate: Option<Expr>,
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
    /// Construct a buffer read directly in Rust after validating its access types.
    pub fn new(buffer: &Var, indices: Vec<Expr>, predicate: Option<&Expr>) -> Result<Self> {
        Self::with_span(buffer, indices, predicate, None)
    }

    /// Construct a validated buffer read with optional source metadata.
    pub fn with_span(
        buffer: &Var,
        indices: Vec<Expr>,
        predicate: Option<&Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        let buffer_type = buffer_type(buffer)?;
        validate_access_dimensions(buffer, &buffer_type, &indices)?;
        validate_nonfinal_indices(&indices)?;

        let buffer_dtype = buffer_type.dtype.clone();
        let result_type = if let Some(index) = indices.last() {
            let index_type = primitive_type(index, "buffer load index")?;
            vectorized_buffer_type(&buffer_dtype, &index_type)?
        } else {
            buffer_dtype.clone()
        };
        if let Some(predicate) = predicate {
            validate_load_predicate(&buffer_dtype, indices.last(), predicate)?;
        }

        Ok(Self::from_complete_fields(
            span.cloned(),
            result_type.into(),
            buffer.clone(),
            Array::new(indices),
            predicate.cloned(),
        ))
    }

    /// Construct a buffer load from every physical field without re-deriving its result type.
    pub fn from_complete_fields(
        span: Option<Span>,
        ty: Type,
        buffer: Var,
        indices: Array<Expr>,
        predicate: Option<Expr>,
    ) -> Self {
        Self {
            data: crate::abi::allocate_object(BufferLoadObj {
                base: ExprObj::new(span, ty),
                buffer,
                indices,
                predicate,
            }),
        }
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
    pub buffer: Var,
    pub value: Expr,
    pub indices: Array<Expr>,
    pub predicate: Option<Expr>,
}
crate::abi::impl_object_layout!(BufferStoreObj {
    "buffer" => buffer: Var,
    "value" => value: Expr,
    "indices" => indices: Array<Expr>,
    "predicate" => predicate: Option<Expr>,
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
    /// Construct a buffer write directly in Rust after validating its access types.
    pub fn new(
        buffer: &Var,
        value: &Expr,
        indices: Vec<Expr>,
        predicate: Option<&Expr>,
    ) -> Result<Self> {
        Self::with_span(buffer, value, indices, predicate, None)
    }

    /// Construct a validated buffer write with optional source metadata.
    pub fn with_span(
        buffer: &Var,
        value: &Expr,
        indices: Vec<Expr>,
        predicate: Option<&Expr>,
        span: Option<&Span>,
    ) -> Result<Self> {
        let buffer_type = buffer_type(buffer)?;
        validate_access_dimensions(buffer, &buffer_type, &indices)?;
        validate_nonfinal_indices(&indices)?;

        let buffer_dtype = buffer_type.dtype.clone();
        let value_dtype = primitive_type(value, "buffer store value")?;
        let index_dtype = indices
            .last()
            .map(|index| primitive_type(index, "buffer store index"))
            .transpose()?;
        let expected_dtype = match &index_dtype {
            Some(index_dtype) => vectorized_buffer_type(&buffer_dtype, index_dtype)?,
            None => buffer_dtype.clone(),
        };
        if expected_dtype.dtype != value_dtype.dtype {
            return Err(Error::new(
                TYPE_ERROR,
                &format!(
                    "dtype mismatch on BufferStore: expected {}, got {}",
                    expected_dtype.dtype.to_string(),
                    value_dtype.dtype.to_string()
                ),
                "",
            ));
        }
        if let Some(predicate) = predicate {
            validate_store_predicate(&value_dtype, predicate)?;
        }

        Ok(Self::from_complete_fields(
            span.cloned(),
            buffer.clone(),
            value.clone(),
            Array::new(indices),
            predicate.cloned(),
        ))
    }

    /// Construct a buffer write from every physical field after external validation.
    pub fn from_complete_fields(
        span: Option<Span>,
        buffer: Var,
        value: Expr,
        indices: Array<Expr>,
        predicate: Option<Expr>,
    ) -> Self {
        Self {
            data: crate::abi::allocate_object(BufferStoreObj {
                base: StmtObj::new(span),
                buffer,
                value,
                indices,
                predicate,
            }),
        }
    }
}

fn buffer_type(buffer: &Var) -> Result<BufferType> {
    buffer.ty.clone().try_cast::<BufferType>().map_err(|_| {
        Error::new(
            TYPE_ERROR,
            "buffer access requires a variable whose type is tirx.BufferType",
            "",
        )
    })
}

fn validate_access_dimensions(
    buffer: &Var,
    buffer_type: &BufferType,
    indices: &[Expr],
) -> Result<()> {
    if buffer_type.shape.len() == indices.len() {
        Ok(())
    } else {
        Err(Error::new(
            VALUE_ERROR,
            &format!(
                "buffer {} is {}-dimensional but received {} indices",
                buffer.name.as_str(),
                buffer_type.shape.len(),
                indices.len()
            ),
            "",
        ))
    }
}

fn validate_nonfinal_indices(indices: &[Expr]) -> Result<()> {
    for index in indices.iter().take(indices.len().saturating_sub(1)) {
        let dtype = primitive_type(index, "buffer index")?.dtype;
        if encoded_lanes(dtype) != 1 {
            return Err(Error::new(
                TYPE_ERROR,
                "only the last index of a buffer access may be a vector type",
                "",
            ));
        }
    }
    Ok(())
}

fn vectorized_buffer_type(buffer: &PrimType, index: &PrimType) -> Result<PrimType> {
    let buffer_lanes = encoded_lanes(buffer.dtype);
    let index_lanes = encoded_lanes(index.dtype);
    let buffer_scalable = buffer_lanes < -1;
    let index_scalable = index_lanes < -1;
    if buffer_scalable && index_scalable {
        return Err(Error::new(
            TYPE_ERROR,
            "index dtype and buffer dtype cannot both be scalable",
            "",
        ));
    }
    let lanes = lane_factor(buffer.dtype)?
        .checked_mul(lane_factor(index.dtype)?)
        .ok_or_else(|| Error::new(VALUE_ERROR, "buffer access lane count overflow", ""))?;
    vector_type_like(buffer.dtype, lanes, buffer_scalable || index_scalable)
}

fn validate_load_predicate(
    buffer: &PrimType,
    index: Option<&Expr>,
    predicate: &Expr,
) -> Result<()> {
    let index = index
        .map(|index| primitive_type(index, "buffer load index"))
        .transpose()?;
    let predicate = primitive_type(predicate, "buffer load predicate")?;
    let index_scalable = index
        .as_ref()
        .is_some_and(|index| encoded_lanes(index.dtype) < -1);
    let predicate_scalable = encoded_lanes(predicate.dtype) < -1;
    if index_scalable != predicate_scalable {
        return Err(Error::new(
            TYPE_ERROR,
            "predicate mask dtype and load indices must both be scalable",
            "",
        ));
    }
    let index_lanes = index
        .as_ref()
        .map(|index| lane_factor(index.dtype))
        .transpose()?
        .unwrap_or(1);
    let expected_lanes = index_lanes
        .checked_mul(lane_factor(buffer.dtype)?)
        .ok_or_else(|| Error::new(VALUE_ERROR, "buffer load lane count overflow", ""))?;
    if lane_factor(predicate.dtype)? != expected_lanes {
        return Err(Error::new(
            TYPE_ERROR,
            "predicate mask lanes must match the loaded value lanes",
            "",
        ));
    }
    validate_predicate_element_type(&predicate)
}

fn validate_store_predicate(value: &PrimType, predicate: &Expr) -> Result<()> {
    let predicate = primitive_type(predicate, "buffer store predicate")?;
    if (encoded_lanes(value.dtype) < -1) != (encoded_lanes(predicate.dtype) < -1) {
        return Err(Error::new(
            TYPE_ERROR,
            "predicate mask dtype and value dtype must both be scalable",
            "",
        ));
    }
    if lane_factor(value.dtype)? != lane_factor(predicate.dtype)? {
        return Err(Error::new(
            TYPE_ERROR,
            "predicate mask lanes must match the stored value lanes",
            "",
        ));
    }
    validate_predicate_element_type(&predicate)
}

fn validate_predicate_element_type(predicate: &PrimType) -> Result<()> {
    let dtype = predicate.dtype;
    let is_boolean = dtype.code == DLDataTypeCode::kDLBool as u8;
    let is_uint1 = dtype.code == DLDataTypeCode::kDLUInt as u8 && dtype.bits == 1;
    if is_boolean || is_uint1 {
        Ok(())
    } else {
        Err(Error::new(
            TYPE_ERROR,
            "predicate mask elements must be boolean values",
            "",
        ))
    }
}

fn vector_type_like(element: DLDataType, lanes: i32, scalable: bool) -> Result<PrimType> {
    if lanes <= 0 || lanes > i32::from(i16::MAX) {
        return Err(Error::new(
            VALUE_ERROR,
            "buffer access lane count does not fit the DLPack encoding",
            "",
        ));
    }
    let encoded = if scalable {
        i16::try_from(-lanes)
    } else {
        i16::try_from(lanes)
    }
    .map_err(|_| Error::new(VALUE_ERROR, "invalid buffer access lane count", ""))?;
    PrimType::from_dtype(DLDataType {
        code: element.code,
        bits: element.bits,
        lanes: encoded as u16,
    })
}

fn lane_factor(dtype: DLDataType) -> Result<i32> {
    let encoded = encoded_lanes(dtype);
    if encoded < -1 {
        Ok(i32::from(-encoded))
    } else if encoded > 0 {
        Ok(i32::from(encoded))
    } else {
        Err(Error::new(TYPE_ERROR, "invalid vector lane encoding", ""))
    }
}

fn encoded_lanes(dtype: DLDataType) -> i16 {
    dtype.lanes as i16
}

impl From<BufferStore> for Stmt {
    fn from(value: BufferStore) -> Self {
        value
            .try_cast()
            .expect("tirx.BufferStore must be a subtype of tirx.Stmt")
    }
}

/// ABI-complete Rust representation of one declared buffer region.
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.BufferRegion"]
#[type_final]
pub struct BufferRegionObj {
    base: PrimExprConvertibleObj,
    pub buffer: Var,
    pub region: Array<Range>,
}
crate::abi::impl_object_layout!(BufferRegionObj {
    "buffer" => buffer: Var,
    "region" => region: Array<Range>,
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
    /// Validate and construct a declared region directly in Rust.
    pub fn new(buffer: &Var, region: Vec<Range>) -> Result<Self> {
        let region = Array::new(region);
        let dimensions = buffer_type(buffer)?.shape.len();
        if dimensions != region.len() {
            return Err(Error::new(
                VALUE_ERROR,
                &format!(
                    "BufferRegion dimension mismatch: buffer has {dimensions}, region has {}",
                    region.len()
                ),
                "",
            ));
        }
        Ok(Self::from_complete_fields(buffer.clone(), region))
    }

    /// Construct a buffer region from every physical field after external validation.
    pub fn from_complete_fields(buffer: Var, region: Array<Range>) -> Self {
        Self {
            data: crate::abi::allocate_object(BufferRegionObj {
                base: PrimExprConvertibleObj::new(),
                buffer,
                region,
            }),
        }
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
    pub buffer: Var,
    pub source: BufferRegion,
}
crate::abi::impl_object_layout!(MatchBufferRegionObj {
    "buffer" => buffer: Var,
    "source" => source: BufferRegion,
});

impl crate::abi::ConstructorRecipe for MatchBufferRegionObj {
    const NUM_INPUTS: usize = 2;
    const DERIVED_FIELDS: &'static [&'static str] = &[];
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

impl MatchBufferRegion {
    /// Validate the declaration and allocate the object directly in Rust.
    ///
    /// Analyzer-backed validation runs through the type's reflected static
    /// preparation method; it never allocates the final node.
    pub fn new(buffer: &Var, source: &BufferRegion) -> Result<Self> {
        let _ = crate::abi::prepare_constructor::<MatchBufferRegionObj>(&[
            AnyView::from(buffer),
            AnyView::from(source),
        ])?;
        Ok(Self::from_complete_fields(buffer.clone(), source.clone()))
    }

    /// Allocate a match-buffer declaration directly after its invariants have been validated.
    pub fn from_complete_fields(buffer: Var, source: BufferRegion) -> Self {
        Self {
            data: crate::abi::allocate_object(MatchBufferRegionObj {
                base: tvm_ffi::Object::new(),
                buffer,
                source,
            }),
        }
    }
}

crate::abi::impl_rust_allocatable!(
    AxisObj,
    IterObj,
    TileLayoutObj,
    BufferTypeObj,
    BufferLoadObj,
    BufferStoreObj,
    BufferRegionObj,
    MatchBufferRegionObj,
);
