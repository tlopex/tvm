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

#ifndef TVM_IR_CONSTRUCTOR_C_API_H_
#define TVM_IR_CONSTRUCTOR_C_API_H_

#include <tvm/ffi/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Type attribute containing a TVMIRConstructorVTable pointer. */
#define TVM_IR_CONSTRUCTOR_VTABLE_ATTR "__constructor_vtable__"

/*!
 * \brief Validate constructor inputs and return derived object fields.
 *
 * Inputs are borrowed. The returned TVMFFIAny is owning and contains either
 * an ffi.Map<String, Any> or an ffi.Error. Map keys are physical reflected
 * field names, including inherited fields such as `ty`; an empty map means
 * validation succeeded without deriving fields. This hook never allocates the
 * final IR node. The calling language combines the original inputs, defaults,
 * and returned fields, then constructs the complete object itself.
 */
typedef TVMFFIAny (*TVMIRFConstructorPrepare)(const TVMFFIAny* args, int32_t num_args);

/*! \brief Language-neutral semantic-constructor preparation table. */
typedef struct {
  /*! \brief Number of borrowed arguments accepted by `prepare`. */
  int32_t num_args;
  /*! \brief Validation and derived-field callback. */
  TVMIRFConstructorPrepare prepare;
} TVMIRConstructorVTable;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_IR_CONSTRUCTOR_C_API_H_
