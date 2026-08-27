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

void RequireInstanceMethod(const TVMFFITypeInfo* info, std::string_view name) {
  ASSERT_NE(info, nullptr);
  const TVMFFIMethodInfo* method = FindMethod(info, name);
  ASSERT_NE(method, nullptr) << "Missing method `" << name << '`';
  EXPECT_EQ(method->method.type_index, tvm::ffi::TypeIndex::kTVMFFIFunction);
  EXPECT_EQ(method->flags & kTVMFFIFieldFlagBitMaskIsStaticMethod, 0);
}

}  // namespace

TEST(IRReflectedMethod, RegistersLanguageAgnosticBehaviorEntryPoints) {
  const TVMFFITypeInfo* convertible =
      TVMFFIGetTypeInfo(tvm::PrimExprConvertibleNode::_GetOrAllocRuntimeTypeIndex());
  RequireInstanceMethod(convertible, "to_prim_expr");

  const TVMFFITypeInfo* producer =
      TVMFFIGetTypeInfo(tvm::tirx::DataProducerNode::_GetOrAllocRuntimeTypeIndex());
  RequireInstanceMethod(producer, "get_shape");
  RequireInstanceMethod(producer, "get_data_type");
  RequireInstanceMethod(producer, "get_name_hint");

  EXPECT_TRUE(tvm::ffi::Function::GetGlobal("tirx.analysis.SideEffect").has_value());
}
