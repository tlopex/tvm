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

/*!
 * \file lower_tirp.cc
 * \brief Compose the TIRp lowering pipeline from individual passes.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {
namespace transform {

Pass LowerTIRp() {
  std::vector<tvm::transform::Pass> passes = {LowerTIRpResolveScopeIds(), LowerTIRpScheduleOps()};
  if (const char* env = std::getenv("TVM_PRINT_AFTER_TIRP_SCHEDULE_OPS")) {
    passes.push_back(tvm::transform::PrintIR());
  }
  passes.push_back(LowerTIRpResolveScopeSlices());
  passes.push_back(LowerTIRpDedupCuTensorMaps());
  passes.push_back(LowerTIRpCleanup());
  return tvm::transform::Sequential(passes, "tir.LowerTIRp");
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tir.transform.LowerTIRpResolveScopeIds", LowerTIRpResolveScopeIds)
      .def("tir.transform.LowerTIRpScheduleOps", LowerTIRpScheduleOps)
      .def("tir.transform.LowerTIRpResolveScopeSlices", LowerTIRpResolveScopeSlices)
      .def("tir.transform.LowerTIRpDedupCuTensorMaps", LowerTIRpDedupCuTensorMaps)
      .def("tir.transform.LowerTIRpCleanup", LowerTIRpCleanup)
      .def("tir.transform.LowerTIRp", LowerTIRp);
}

}  // namespace transform
}  // namespace tir
}  // namespace tvm
