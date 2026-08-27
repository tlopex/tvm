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
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/layout.h>

#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

bool IsStrictSubtype(const TVMFFITypeInfo* info, int32_t base_type_index) {
  for (int32_t depth = 0; depth < info->type_depth; ++depth) {
    if (info->type_ancestors[depth]->type_index == base_type_index) {
      return true;
    }
  }
  return false;
}
const TVMFFIMethodInfo* FindMethod(const TVMFFITypeInfo* info, std::string_view name) {
  for (int32_t index = 0; index < info->num_methods; ++index) {
    const TVMFFIMethodInfo* method = &info->methods[index];
    if (std::string_view(method->name.data, method->name.size) == name) {
      return method;
    }
  }
  return nullptr;
}

void RequireMethod(const TVMFFITypeInfo* info, std::string_view name, bool is_static = false) {
  const TVMFFIMethodInfo* method = FindMethod(info, name);
  ASSERT_NE(method, nullptr) << "Type "
                             << std::string_view(info->type_key.data, info->type_key.size)
                             << " does not register `" << name << '`';
  EXPECT_EQ(method->method.type_index, tvm::ffi::TypeIndex::kTVMFFIFunction);
  EXPECT_EQ((method->flags & kTVMFFIFieldFlagBitMaskIsStaticMethod) != 0, is_static);
}

void CheckConcreteSubtypes(int32_t base_type_index,
                           const std::unordered_set<std::string_view>& abstract_type_keys,
                           const std::vector<std::string_view>& methods) {
  tvm::ffi::Array<tvm::ffi::String> type_keys =
      tvm::ffi::Function::GetGlobalRequired("ffi.GetRegisteredTypeKeys")()
          .cast<tvm::ffi::Array<tvm::ffi::String>>();
  size_t checked = 0;
  for (const tvm::ffi::String& key : type_keys) {
    std::string_view type_key(key.data(), key.size());
    const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(tvm::ffi::TypeKeyToIndex(type_key));
    ASSERT_NE(info, nullptr);
    if (!IsStrictSubtype(info, base_type_index) || abstract_type_keys.count(type_key) != 0) {
      continue;
    }
    SCOPED_TRACE(std::string(type_key));
    for (std::string_view method : methods) {
      RequireMethod(info, method);
    }
    ++checked;
  }
  EXPECT_GT(checked, 0U);
}

}  // namespace

TEST(IRReflectedMethod, EveryConcreteBehaviorSubtypeRegistersRequiredMethods) {
  CheckConcreteSubtypes(tvm::PrimExprConvertibleNode::_GetOrAllocRuntimeTypeIndex(),
                        {"tirx.DataProducer"}, {"to_prim_expr"});
  CheckConcreteSubtypes(tvm::tirx::DataProducerNode::_GetOrAllocRuntimeTypeIndex(), {},
                        {"get_shape", "get_data_type", "get_name_hint", "to_prim_expr"});
  CheckConcreteSubtypes(
      tvm::tirx::LayoutNode::_GetOrAllocRuntimeTypeIndex(), {},
      {"compatible_with_shape", "verify_well_formed", "get_size", "get_span", "apply",
       "apply_linear", "apply_with_shape", "canonicalize", "tile", "slice", "direct_sum",
       "is_tile_inner", "is_tile_outer", "is_direct_sum_right", "is_direct_sum_left"});
}

TEST(IRReflectedMethod, SemanticConstructorsRegisterStaticPreparationMethod) {
  for (std::string_view type_key : {"tirx.Axis", "tirx.BufferType", "tirx.PrimFunc",
                                    "relax.expr.Function", "tirx.MatchBufferRegion"}) {
    SCOPED_TRACE(std::string(type_key));
    const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(tvm::ffi::TypeKeyToIndex(type_key));
    ASSERT_NE(info, nullptr);
    RequireMethod(info, "__ffi_prepare__", true);
  }
}
