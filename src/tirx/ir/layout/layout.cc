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
#include <tvm/ffi/reflection/accessor.h>

#include "utils.h"

namespace tvm {
namespace tirx {

/**************** Layout ****************/
namespace {

template <typename Result, typename... Args>
Result CallLayoutMethod(const LayoutNode* self, const char* name, Args&&... args) {
  return ffi::reflection::GetMethod(self->GetTypeKey(), name)
      .CallExpected<Result>(ffi::GetRef<Layout>(self), std::forward<Args>(args)...)
      .value();
}

}  // namespace

bool LayoutNode::CompatibleWithShape(const ffi::Array<PrimExpr>& shape) const {
  return CallLayoutMethod<bool>(this, "compatible_with_shape", shape);
}

bool LayoutNode::VerifyWellFormed() const {
  return CallLayoutMethod<bool>(this, "verify_well_formed");
}

PrimExpr LayoutNode::GetSize(ffi::Optional<ffi::String> axis_name) const {
  return CallLayoutMethod<PrimExpr>(this, "get_size", std::move(axis_name));
}

PrimExpr LayoutNode::GetSpan(ffi::Optional<ffi::String> axis_name) const {
  return CallLayoutMethod<PrimExpr>(this, "get_span", std::move(axis_name));
}

ffi::Map<ffi::String, PrimExpr> LayoutNode::Apply(ffi::Array<PrimExpr> coord) const {
  return CallLayoutMethod<ffi::Map<ffi::String, PrimExpr>>(this, "apply", std::move(coord));
}

ffi::Map<ffi::String, PrimExpr> LayoutNode::Apply(PrimExpr coord) const {
  return CallLayoutMethod<ffi::Map<ffi::String, PrimExpr>>(this, "apply_linear", std::move(coord));
}

ffi::Map<ffi::String, PrimExpr> LayoutNode::Apply(const ffi::Array<PrimExpr>& coord,
                                                  const ffi::Array<PrimExpr>& shape) const {
  return CallLayoutMethod<ffi::Map<ffi::String, PrimExpr>>(this, "apply_with_shape", coord, shape);
}

Layout LayoutNode::Canonicalize() const { return CallLayoutMethod<Layout>(this, "canonicalize"); }

Layout LayoutNode::Tile(const TileLayout& outer, const ffi::Array<PrimExpr>& outer_shape,
                        const ffi::Array<PrimExpr>& inner_shape) const {
  return CallLayoutMethod<Layout>(this, "tile", outer, outer_shape, inner_shape);
}

ffi::Optional<Layout> LayoutNode::Slice(const ffi::Array<PrimExpr>& shape,
                                        const Region& region) const {
  return CallLayoutMethod<ffi::Optional<Layout>>(this, "slice", shape, region);
}

Layout LayoutNode::DirectSum(const TileLayout& left, const ffi::Array<PrimExpr>& left_shape,
                             const ffi::Array<PrimExpr>& right_shape) const {
  return CallLayoutMethod<Layout>(this, "direct_sum", left, left_shape, right_shape);
}

ffi::Optional<TileLayout> LayoutNode::IsTileInner(const Layout& tile_layout,
                                                  const ffi::Array<PrimExpr>& tiled_shape,
                                                  const ffi::Array<PrimExpr>& inner_shape) const {
  return CallLayoutMethod<ffi::Optional<TileLayout>>(this, "is_tile_inner", tile_layout,
                                                     tiled_shape, inner_shape);
}

ffi::Optional<Layout> LayoutNode::IsTileOuter(const Layout& tile_layout,
                                              const ffi::Array<PrimExpr>& tiled_shape,
                                              const ffi::Array<PrimExpr>& outer_shape) const {
  return CallLayoutMethod<ffi::Optional<Layout>>(this, "is_tile_outer", tile_layout, tiled_shape,
                                                 outer_shape);
}

ffi::Optional<TileLayout> LayoutNode::IsDirectSumRight(
    const Layout& sum_layout, const ffi::Array<PrimExpr>& interleaved_shape,
    const ffi::Array<PrimExpr>& right_shape) const {
  return CallLayoutMethod<ffi::Optional<TileLayout>>(this, "is_direct_sum_right", sum_layout,
                                                     interleaved_shape, right_shape);
}

ffi::Optional<Layout> LayoutNode::IsDirectSumLeft(const Layout& sum_layout,
                                                  const ffi::Array<PrimExpr>& interleaved_shape,
                                                  const ffi::Array<PrimExpr>& left_shape) const {
  return CallLayoutMethod<ffi::Optional<Layout>>(this, "is_direct_sum_left", sum_layout,
                                                 interleaved_shape, left_shape);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::ObjectDef<LayoutNode>();
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
