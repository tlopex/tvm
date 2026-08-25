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
#include "../../../ir/c_abi_utils.h"
#include "utils.h"

namespace tvm {
namespace tirx {

/**************** Layout ****************/
namespace {

using ir_abi::FromABI;
using ir_abi::ReturnExpected;
using ir_abi::ToBorrowedABI;

template <typename RefType>
struct LayoutABI {
  static TVMFFIAny CompatibleWithShape(TVMFFIAny self, TVMFFIAny shape) noexcept {
    return ReturnExpected([&]() {
      return FromABI<RefType>(self)->CompatibleWithShape(FromABI<ffi::Array<PrimExpr>>(shape));
    });
  }

  static TVMFFIAny VerifyWellFormed(TVMFFIAny self) noexcept {
    return ReturnExpected([&]() { return FromABI<RefType>(self)->VerifyWellFormed(); });
  }

  static TVMFFIAny GetSize(TVMFFIAny self, TVMFFIAny axis_name) noexcept {
    return ReturnExpected([&]() {
      return FromABI<RefType>(self)->GetSize(FromABI<ffi::Optional<ffi::String>>(axis_name));
    });
  }

  static TVMFFIAny GetSpan(TVMFFIAny self, TVMFFIAny axis_name) noexcept {
    return ReturnExpected([&]() {
      return FromABI<RefType>(self)->GetSpan(FromABI<ffi::Optional<ffi::String>>(axis_name));
    });
  }

  static TVMFFIAny Apply(TVMFFIAny self, TVMFFIAny coord) noexcept {
    return ReturnExpected(
        [&]() { return FromABI<RefType>(self)->Apply(FromABI<ffi::Array<PrimExpr>>(coord)); });
  }

  static TVMFFIAny ApplyLinear(TVMFFIAny self, TVMFFIAny coord) noexcept {
    return ReturnExpected(
        [&]() { return FromABI<RefType>(self)->Apply(FromABI<PrimExpr>(coord)); });
  }

  static TVMFFIAny ApplyWithShape(TVMFFIAny self, TVMFFIAny coord, TVMFFIAny shape) noexcept {
    return ReturnExpected([&]() {
      return FromABI<RefType>(self)->Apply(FromABI<ffi::Array<PrimExpr>>(coord),
                                           FromABI<ffi::Array<PrimExpr>>(shape));
    });
  }

  static TVMFFIAny Canonicalize(TVMFFIAny self) noexcept {
    return ReturnExpected([&]() { return FromABI<RefType>(self)->Canonicalize(); });
  }

  static TVMFFIAny Tile(TVMFFIAny self, TVMFFIAny outer, TVMFFIAny outer_shape,
                        TVMFFIAny inner_shape) noexcept {
    return ReturnExpected([&]() {
      return FromABI<RefType>(self)->Tile(FromABI<TileLayout>(outer),
                                          FromABI<ffi::Array<PrimExpr>>(outer_shape),
                                          FromABI<ffi::Array<PrimExpr>>(inner_shape));
    });
  }

  static TVMFFIAny Slice(TVMFFIAny self, TVMFFIAny shape, TVMFFIAny region) noexcept {
    return ReturnExpected([&]() {
      return FromABI<RefType>(self)->Slice(FromABI<ffi::Array<PrimExpr>>(shape),
                                           FromABI<Region>(region));
    });
  }

  static TVMFFIAny DirectSum(TVMFFIAny self, TVMFFIAny left, TVMFFIAny left_shape,
                             TVMFFIAny right_shape) noexcept {
    return ReturnExpected([&]() {
      return FromABI<RefType>(self)->DirectSum(FromABI<TileLayout>(left),
                                               FromABI<ffi::Array<PrimExpr>>(left_shape),
                                               FromABI<ffi::Array<PrimExpr>>(right_shape));
    });
  }

  static TVMFFIAny IsTileInner(TVMFFIAny self, TVMFFIAny layout, TVMFFIAny tiled_shape,
                               TVMFFIAny inner_shape) noexcept {
    return ReturnExpected([&]() {
      return FromABI<RefType>(self)->IsTileInner(FromABI<Layout>(layout),
                                                 FromABI<ffi::Array<PrimExpr>>(tiled_shape),
                                                 FromABI<ffi::Array<PrimExpr>>(inner_shape));
    });
  }

  static TVMFFIAny IsTileOuter(TVMFFIAny self, TVMFFIAny layout, TVMFFIAny tiled_shape,
                               TVMFFIAny outer_shape) noexcept {
    return ReturnExpected([&]() {
      return FromABI<RefType>(self)->IsTileOuter(FromABI<Layout>(layout),
                                                 FromABI<ffi::Array<PrimExpr>>(tiled_shape),
                                                 FromABI<ffi::Array<PrimExpr>>(outer_shape));
    });
  }

  static TVMFFIAny IsDirectSumRight(TVMFFIAny self, TVMFFIAny layout, TVMFFIAny interleaved_shape,
                                    TVMFFIAny right_shape) noexcept {
    return ReturnExpected([&]() {
      return FromABI<RefType>(self)->IsDirectSumRight(
          FromABI<Layout>(layout), FromABI<ffi::Array<PrimExpr>>(interleaved_shape),
          FromABI<ffi::Array<PrimExpr>>(right_shape));
    });
  }

  static TVMFFIAny IsDirectSumLeft(TVMFFIAny self, TVMFFIAny layout, TVMFFIAny interleaved_shape,
                                   TVMFFIAny left_shape) noexcept {
    return ReturnExpected([&]() {
      return FromABI<RefType>(self)->IsDirectSumLeft(
          FromABI<Layout>(layout), FromABI<ffi::Array<PrimExpr>>(interleaved_shape),
          FromABI<ffi::Array<PrimExpr>>(left_shape));
    });
  }

  static const TVMTIRXLayoutVTable* VTable() {
    static const TVMTIRXLayoutVTable vtable{
        &CompatibleWithShape,
        &VerifyWellFormed,
        &GetSize,
        &GetSpan,
        &Apply,
        &ApplyLinear,
        &ApplyWithShape,
        &Canonicalize,
        &Tile,
        &Slice,
        &DirectSum,
        &IsTileInner,
        &IsTileOuter,
        &IsDirectSumRight,
        &IsDirectSumLeft,
    };
    return &vtable;
  }
};

const TVMTIRXLayoutVTable* GetLayoutVTable(const LayoutNode* self) {
  static ffi::reflection::TypeAttrColumn column(TVM_TIRX_LAYOUT_VTABLE_ATTR);
  ffi::AnyView attr = column[self->type_index()];
  TVM_FFI_CHECK(attr.type_index() == ffi::TypeIndex::kTVMFFIOpaquePtr, TypeError)
      << "Layout type " << self->GetTypeKey() << " does not register a C ABI layout vtable";
  auto* vtable = static_cast<const TVMTIRXLayoutVTable*>(attr.cast<void*>());
  TVM_FFI_CHECK(vtable != nullptr, TypeError)
      << "Layout type " << self->GetTypeKey() << " registers a null layout vtable";
  return vtable;
}

template <typename Result>
Result FromExpected(TVMFFIAny value) {
  return ffi::details::ExpectedUnsafe::MoveFromTVMFFIAny<Result>(value).value();
}

}  // namespace

bool LayoutNode::CompatibleWithShape(const ffi::Array<PrimExpr>& shape) const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->compatible_with_shape != nullptr, TypeError)
      << "Layout vtable is missing compatible_with_shape";
  return FromExpected<bool>(vtable->compatible_with_shape(ToBorrowedABI(ffi::GetRef<Layout>(this)),
                                                          ToBorrowedABI(shape)));
}

bool LayoutNode::VerifyWellFormed() const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->verify_well_formed != nullptr, TypeError)
      << "Layout vtable is missing verify_well_formed";
  return FromExpected<bool>(vtable->verify_well_formed(ToBorrowedABI(ffi::GetRef<Layout>(this))));
}

PrimExpr LayoutNode::GetSize(ffi::Optional<ffi::String> axis_name) const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->get_size != nullptr, TypeError) << "Layout vtable is missing get_size";
  return FromExpected<PrimExpr>(
      vtable->get_size(ToBorrowedABI(ffi::GetRef<Layout>(this)), ToBorrowedABI(axis_name)));
}

PrimExpr LayoutNode::GetSpan(ffi::Optional<ffi::String> axis_name) const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->get_span != nullptr, TypeError) << "Layout vtable is missing get_span";
  return FromExpected<PrimExpr>(
      vtable->get_span(ToBorrowedABI(ffi::GetRef<Layout>(this)), ToBorrowedABI(axis_name)));
}

ffi::Map<ffi::String, PrimExpr> LayoutNode::Apply(ffi::Array<PrimExpr> coord) const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->apply != nullptr, TypeError) << "Layout vtable is missing apply";
  return FromExpected<ffi::Map<ffi::String, PrimExpr>>(
      vtable->apply(ToBorrowedABI(ffi::GetRef<Layout>(this)), ToBorrowedABI(coord)));
}

ffi::Map<ffi::String, PrimExpr> LayoutNode::Apply(PrimExpr coord) const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->apply_linear != nullptr, TypeError)
      << "Layout vtable is missing apply_linear";
  return FromExpected<ffi::Map<ffi::String, PrimExpr>>(
      vtable->apply_linear(ToBorrowedABI(ffi::GetRef<Layout>(this)), ToBorrowedABI(coord)));
}

ffi::Map<ffi::String, PrimExpr> LayoutNode::Apply(const ffi::Array<PrimExpr>& coord,
                                                  const ffi::Array<PrimExpr>& shape) const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->apply_with_shape != nullptr, TypeError)
      << "Layout vtable is missing apply_with_shape";
  return FromExpected<ffi::Map<ffi::String, PrimExpr>>(vtable->apply_with_shape(
      ToBorrowedABI(ffi::GetRef<Layout>(this)), ToBorrowedABI(coord), ToBorrowedABI(shape)));
}

Layout LayoutNode::Canonicalize() const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->canonicalize != nullptr, TypeError)
      << "Layout vtable is missing canonicalize";
  return FromExpected<Layout>(vtable->canonicalize(ToBorrowedABI(ffi::GetRef<Layout>(this))));
}

Layout LayoutNode::Tile(const TileLayout& outer, const ffi::Array<PrimExpr>& outer_shape,
                        const ffi::Array<PrimExpr>& inner_shape) const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->tile != nullptr, TypeError) << "Layout vtable is missing tile";
  return FromExpected<Layout>(vtable->tile(ToBorrowedABI(ffi::GetRef<Layout>(this)),
                                           ToBorrowedABI(outer), ToBorrowedABI(outer_shape),
                                           ToBorrowedABI(inner_shape)));
}

ffi::Optional<Layout> LayoutNode::Slice(const ffi::Array<PrimExpr>& shape,
                                        const Region& region) const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->slice != nullptr, TypeError) << "Layout vtable is missing slice";
  return FromExpected<ffi::Optional<Layout>>(vtable->slice(
      ToBorrowedABI(ffi::GetRef<Layout>(this)), ToBorrowedABI(shape), ToBorrowedABI(region)));
}

Layout LayoutNode::DirectSum(const TileLayout& left, const ffi::Array<PrimExpr>& left_shape,
                             const ffi::Array<PrimExpr>& right_shape) const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->direct_sum != nullptr, TypeError) << "Layout vtable is missing direct_sum";
  return FromExpected<Layout>(vtable->direct_sum(ToBorrowedABI(ffi::GetRef<Layout>(this)),
                                                 ToBorrowedABI(left), ToBorrowedABI(left_shape),
                                                 ToBorrowedABI(right_shape)));
}

ffi::Optional<TileLayout> LayoutNode::IsTileInner(const Layout& tile_layout,
                                                  const ffi::Array<PrimExpr>& tiled_shape,
                                                  const ffi::Array<PrimExpr>& inner_shape) const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->is_tile_inner != nullptr, TypeError)
      << "Layout vtable is missing is_tile_inner";
  return FromExpected<ffi::Optional<TileLayout>>(
      vtable->is_tile_inner(ToBorrowedABI(ffi::GetRef<Layout>(this)), ToBorrowedABI(tile_layout),
                            ToBorrowedABI(tiled_shape), ToBorrowedABI(inner_shape)));
}

ffi::Optional<Layout> LayoutNode::IsTileOuter(const Layout& tile_layout,
                                              const ffi::Array<PrimExpr>& tiled_shape,
                                              const ffi::Array<PrimExpr>& outer_shape) const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->is_tile_outer != nullptr, TypeError)
      << "Layout vtable is missing is_tile_outer";
  return FromExpected<ffi::Optional<Layout>>(
      vtable->is_tile_outer(ToBorrowedABI(ffi::GetRef<Layout>(this)), ToBorrowedABI(tile_layout),
                            ToBorrowedABI(tiled_shape), ToBorrowedABI(outer_shape)));
}

ffi::Optional<TileLayout> LayoutNode::IsDirectSumRight(
    const Layout& sum_layout, const ffi::Array<PrimExpr>& interleaved_shape,
    const ffi::Array<PrimExpr>& right_shape) const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->is_direct_sum_right != nullptr, TypeError)
      << "Layout vtable is missing is_direct_sum_right";
  return FromExpected<ffi::Optional<TileLayout>>(vtable->is_direct_sum_right(
      ToBorrowedABI(ffi::GetRef<Layout>(this)), ToBorrowedABI(sum_layout),
      ToBorrowedABI(interleaved_shape), ToBorrowedABI(right_shape)));
}

ffi::Optional<Layout> LayoutNode::IsDirectSumLeft(const Layout& sum_layout,
                                                  const ffi::Array<PrimExpr>& interleaved_shape,
                                                  const ffi::Array<PrimExpr>& left_shape) const {
  const auto* vtable = GetLayoutVTable(this);
  TVM_FFI_CHECK(vtable->is_direct_sum_left != nullptr, TypeError)
      << "Layout vtable is missing is_direct_sum_left";
  return FromExpected<ffi::Optional<Layout>>(vtable->is_direct_sum_left(
      ToBorrowedABI(ffi::GetRef<Layout>(this)), ToBorrowedABI(sum_layout),
      ToBorrowedABI(interleaved_shape), ToBorrowedABI(left_shape)));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<LayoutNode>();
  refl::EnsureTypeAttrColumn(TVM_TIRX_LAYOUT_VTABLE_ATTR);
  refl::TypeAttrDef<TileLayoutNode>().attr(
      TVM_TIRX_LAYOUT_VTABLE_ATTR,
      reinterpret_cast<void*>(const_cast<TVMTIRXLayoutVTable*>(LayoutABI<TileLayout>::VTable())));
  refl::TypeAttrDef<ComposeLayoutNode>().attr(
      TVM_TIRX_LAYOUT_VTABLE_ATTR, reinterpret_cast<void*>(const_cast<TVMTIRXLayoutVTable*>(
                                       LayoutABI<ComposeLayout>::VTable())));
  auto def = refl::GlobalDef();
  def.def("tirx.LayoutCompatibleWithShape",
          [](Layout layout, Array<PrimExpr> shape) { return layout->CompatibleWithShape(shape); });
  def.def("tirx.LayoutVerifyWellFormed", [](Layout layout) { return layout->VerifyWellFormed(); });
  def.def("tirx.LayoutGetSize", [](Layout layout, ffi::Optional<ffi::String> axis_name) {
    return layout->GetSize(axis_name);
  });
  def.def("tirx.LayoutGetSpan", [](Layout layout, ffi::Optional<ffi::String> axis_name) {
    return layout->GetSpan(axis_name);
  });
  def.def("tirx.LayoutApplyWithShape",
          [](Layout layout, ffi::Array<PrimExpr> coord, ffi::Array<PrimExpr> shape) {
            return layout->Apply(coord, shape);
          });
  def.def("tirx.LayoutApply",
          [](Layout layout, ffi::Array<PrimExpr> coord) { return layout->Apply(coord); });
  def.def("tirx.LayoutApplyLinear",
          [](Layout layout, PrimExpr coord) { return layout->Apply(coord); });
  def.def("tirx.LayoutCanonicalize", [](Layout layout) { return layout->Canonicalize(); });
  def.def("tirx.LayoutTile", [](Layout layout, TileLayout outer, ffi::Array<PrimExpr> outer_shape,
                                ffi::Array<PrimExpr> inner_shape) {
    return layout->Tile(outer, outer_shape, inner_shape);
  });
  def.def("tirx.LayoutDirectSum",
          [](Layout layout, TileLayout left, ffi::Array<PrimExpr> left_shape,
             ffi::Array<PrimExpr> right_shape) {
            return layout->DirectSum(left, left_shape, right_shape);
          });
  def.def("tirx.LayoutIsTileInner",
          [](Layout layout, Layout tile_layout, ffi::Array<PrimExpr> tiled_shape,
             ffi::Array<PrimExpr> inner_shape) {
            return layout->IsTileInner(tile_layout, tiled_shape, inner_shape);
          });
  def.def("tirx.LayoutIsTileOuter",
          [](Layout layout, Layout tile_layout, ffi::Array<PrimExpr> tiled_shape,
             ffi::Array<PrimExpr> outer_shape) {
            return layout->IsTileOuter(tile_layout, tiled_shape, outer_shape);
          });
  def.def("tirx.LayoutIsDirectSumRight",
          [](Layout layout, Layout sum_layout, ffi::Array<PrimExpr> interleaved_shape,
             ffi::Array<PrimExpr> right_shape) {
            return layout->IsDirectSumRight(sum_layout, interleaved_shape, right_shape);
          });
  def.def("tirx.LayoutIsDirectSumLeft",
          [](Layout layout, Layout sum_layout, ffi::Array<PrimExpr> interleaved_shape,
             ffi::Array<PrimExpr> left_shape) {
            return layout->IsDirectSumLeft(sum_layout, interleaved_shape, left_shape);
          });
  def.def("tirx.LayoutSlice",
          [](Layout layout, ffi::Array<PrimExpr> shape, Region region) -> ffi::Optional<Layout> {
            return layout->Slice(shape, region);
          });
}

}  // namespace tirx
}  // namespace tvm
