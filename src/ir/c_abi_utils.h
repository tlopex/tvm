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

#ifndef TVM_SRC_IR_C_ABI_UTILS_H_
#define TVM_SRC_IR_C_ABI_UTILS_H_

#include <tvm/ffi/expected.h>

#include <exception>
#include <type_traits>
#include <utility>

namespace tvm {
namespace ir_abi {

template <typename Callback>
TVMFFIAny ReturnExpected(Callback&& callback) noexcept {
  using Result = std::decay_t<decltype(callback())>;
  try {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(ffi::Expected<Result>(callback()));
  } catch (const ffi::Error& error) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(ffi::Expected<Result>(error));
  } catch (const std::exception& error) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
        ffi::Expected<Result>(ffi::Error("InternalError", error.what(), "")));
  } catch (...) {
    return ffi::details::ExpectedUnsafe::MoveToTVMFFIAny(
        ffi::Expected<Result>(ffi::Error("InternalError", "unknown C++ exception", "")));
  }
}

template <typename T>
T FromABI(TVMFFIAny value) {
  return ffi::AnyView::CopyFromTVMFFIAny(value).cast<T>();
}

template <typename T>
TVMFFIAny ToBorrowedABI(const T& value) {
  // AnyView::CopyToTVMFFIAny copies only the borrowed ABI cell.  It does not
  // increment the object reference count or transfer ownership to the callee.
  return ffi::AnyView(value).CopyToTVMFFIAny();
}

inline void CheckArity(int32_t actual, int32_t expected) {
  TVM_FFI_CHECK_EQ(actual, expected, TypeError)
      << "C ABI constructor preparation expected " << expected << " arguments, but received "
      << actual;
}

}  // namespace ir_abi
}  // namespace tvm

#endif  // TVM_SRC_IR_C_ABI_UTILS_H_
