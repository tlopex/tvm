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

#include <gtest/gtest.h>
#include <tvm/ffi/function.h>
#include <tvm/ir/base_expr.h>
#include <tvm/ir/constructor_c_api.h>
#include <tvm/ir/expr_c_api.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/data_producer_c_api.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/layout_c_api.h>

#include <cstring>
#include <string>
#include <string_view>
#include <unordered_set>

namespace {

const TVMFFITypeAttrColumn* GetColumn(const char* attr_name) {
  TVMFFIByteArray name{attr_name, std::strlen(attr_name)};
  return TVMFFIGetTypeAttrColumn(&name);
}

const TVMFFIAny* LookupAttr(const TVMFFITypeAttrColumn* column, int32_t type_index) {
  if (column == nullptr || column->data == nullptr || type_index < column->begin_index ||
      type_index >= column->begin_index + column->size) {
    return nullptr;
  }
  return &column->data[type_index - column->begin_index];
}

bool IsStrictSubtype(const TVMFFITypeInfo* info, int32_t base_type_index) {
  for (int32_t depth = 0; depth < info->type_depth; ++depth) {
    if (info->type_ancestors[depth]->type_index == base_type_index) {
      return true;
    }
  }
  return false;
}

template <typename VTable>
const VTable* RequireTable(const TVMFFIAny* attr, uint32_t expected_version,
                           std::string_view type_key, std::string_view attr_name) {
  EXPECT_NE(attr, nullptr) << "Type " << type_key << " does not register `" << attr_name << "`";
  if (attr == nullptr) return nullptr;
  tvm::ffi::AnyView value = tvm::ffi::AnyView::CopyFromTVMFFIAny(*attr);
  EXPECT_EQ(value.type_index(), tvm::ffi::TypeIndex::kTVMFFIOpaquePtr)
      << "Type " << type_key << " does not register `" << attr_name << "` as an opaque pointer";
  if (value.type_index() != tvm::ffi::TypeIndex::kTVMFFIOpaquePtr) return nullptr;
  const auto* table = static_cast<const VTable*>(value.cast<void*>());
  EXPECT_NE(table, nullptr) << "Type " << type_key << " registers a null `" << attr_name << "`";
  if (table == nullptr) return nullptr;
  EXPECT_EQ(table->abi_version, expected_version)
      << "Type " << type_key << " registers the wrong `" << attr_name << "` ABI version";
  EXPECT_GE(table->struct_size, sizeof(VTable))
      << "Type " << type_key << " registers a truncated `" << attr_name << "` table";
  if (table->abi_version != expected_version || table->struct_size < sizeof(VTable)) {
    return nullptr;
  }
  return table;
}

template <typename VTable, typename Checker>
void CheckConcreteSubtypes(int32_t base_type_index, const char* attr_name,
                           uint32_t expected_version,
                           const std::unordered_set<std::string_view>& abstract_type_keys,
                           Checker checker) {
  const auto column = GetColumn(attr_name);
  ASSERT_NE(column, nullptr) << "Missing type-attribute column `" << attr_name << "`";
  tvm::ffi::Array<tvm::ffi::String> type_keys =
      tvm::ffi::Function::GetGlobalRequired("ffi.GetRegisteredTypeKeys")()
          .cast<tvm::ffi::Array<tvm::ffi::String>>();
  size_t checked = 0;
  for (const tvm::ffi::String& key : type_keys) {
    std::string_view type_key(key.data(), key.size());
    int32_t type_index = tvm::ffi::TypeKeyToIndex(type_key);
    const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(type_index);
    ASSERT_NE(info, nullptr);
    if (!IsStrictSubtype(info, base_type_index) || abstract_type_keys.count(type_key) != 0) {
      continue;
    }
    SCOPED_TRACE(std::string(type_key));
    const VTable* table =
        RequireTable<VTable>(LookupAttr(column, type_index), expected_version, type_key, attr_name);
    if (table != nullptr) {
      checker(table);
    }
    ++checked;
  }
  EXPECT_GT(checked, 0U) << "No concrete subtypes found for `" << attr_name << "`";
}

}  // namespace

TEST(IRCABIVTable, EveryConcreteBehaviorSubtypeRegistersACompleteTable) {
  CheckConcreteSubtypes<TVMIRPrimExprConvertibleVTable>(
      tvm::PrimExprConvertibleNode::_GetOrAllocRuntimeTypeIndex(),
      TVM_IR_PRIM_EXPR_CONVERTIBLE_VTABLE_ATTR, TVM_IR_PRIM_EXPR_CONVERTIBLE_VTABLE_ABI_VERSION,
      {"tirx.DataProducer"},
      [](const TVMIRPrimExprConvertibleVTable* table) { EXPECT_NE(table->to_prim_expr, nullptr); });

  CheckConcreteSubtypes<TVMTIRXDataProducerVTable>(
      tvm::tirx::DataProducerNode::_GetOrAllocRuntimeTypeIndex(),
      TVM_TIRX_DATA_PRODUCER_VTABLE_ATTR, TVM_TIRX_DATA_PRODUCER_VTABLE_ABI_VERSION, {},
      [](const TVMTIRXDataProducerVTable* table) {
        EXPECT_NE(table->get_shape, nullptr);
        EXPECT_NE(table->get_data_type, nullptr);
        EXPECT_NE(table->get_name_hint, nullptr);
      });

  CheckConcreteSubtypes<TVMTIRXLayoutVTable>(
      tvm::tirx::LayoutNode::_GetOrAllocRuntimeTypeIndex(), TVM_TIRX_LAYOUT_VTABLE_ATTR,
      TVM_TIRX_LAYOUT_VTABLE_ABI_VERSION, {}, [](const TVMTIRXLayoutVTable* table) {
        EXPECT_NE(table->compatible_with_shape, nullptr);
        EXPECT_NE(table->verify_well_formed, nullptr);
        EXPECT_NE(table->get_size, nullptr);
        EXPECT_NE(table->get_span, nullptr);
        EXPECT_NE(table->apply, nullptr);
        EXPECT_NE(table->apply_linear, nullptr);
        EXPECT_NE(table->apply_with_shape, nullptr);
        EXPECT_NE(table->canonicalize, nullptr);
        EXPECT_NE(table->tile, nullptr);
        EXPECT_NE(table->slice, nullptr);
        EXPECT_NE(table->direct_sum, nullptr);
        EXPECT_NE(table->is_tile_inner, nullptr);
        EXPECT_NE(table->is_tile_outer, nullptr);
        EXPECT_NE(table->is_direct_sum_right, nullptr);
        EXPECT_NE(table->is_direct_sum_left, nullptr);
      });
}

TEST(IRCABIVTable, EveryRegisteredConstructorTableHasACompatibleHeader) {
  const auto* column = GetColumn(TVM_IR_CONSTRUCTOR_VTABLE_ATTR);
  ASSERT_NE(column, nullptr);
  tvm::ffi::Array<tvm::ffi::String> type_keys =
      tvm::ffi::Function::GetGlobalRequired("ffi.GetRegisteredTypeKeys")()
          .cast<tvm::ffi::Array<tvm::ffi::String>>();
  size_t checked = 0;
  for (const tvm::ffi::String& key : type_keys) {
    std::string_view type_key(key.data(), key.size());
    int32_t type_index = tvm::ffi::TypeKeyToIndex(type_key);
    const TVMFFIAny* attr = LookupAttr(column, type_index);
    if (attr == nullptr || attr->type_index == tvm::ffi::TypeIndex::kTVMFFINone) continue;
    SCOPED_TRACE(std::string(type_key));
    const auto* table = RequireTable<TVMIRConstructorVTable>(
        attr, TVM_IR_CONSTRUCTOR_VTABLE_ABI_VERSION, type_key, TVM_IR_CONSTRUCTOR_VTABLE_ATTR);
    if (table != nullptr) {
      EXPECT_GE(table->num_args, 0);
      EXPECT_NE(table->prepare, nullptr);
    }
    ++checked;
  }
  EXPECT_GT(checked, 0U);
}
