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
#include <tvm/ffi/reflection/accessor.h>

#include <string>
#include <string_view>

namespace {

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

}  // namespace

TEST(IRReflectedMethod, RustAllocatedSemanticConstructorsRegisterPreparationMethod) {
  for (std::string_view type_key :
       {"tirx.BufferType", "tirx.PrimFunc", "relax.expr.Function", "tirx.MatchBufferRegion"}) {
    SCOPED_TRACE(std::string(type_key));
    const TVMFFITypeInfo* info = TVMFFIGetTypeInfo(tvm::ffi::TypeKeyToIndex(type_key));
    ASSERT_NE(info, nullptr);
    RequireMethod(info, tvm::ffi::reflection::type_attr::kPrepare, true);
  }
}
