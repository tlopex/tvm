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

#ifndef TVM_IR_EXPR_C_API_H_
#define TVM_IR_EXPR_C_API_H_

#include <tvm/ffi/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Type attribute containing a TVMIRPrimExprConvertibleVTable pointer. */
#define TVM_IR_PRIM_EXPR_CONVERTIBLE_VTABLE_ATTR "__to_prim_expr__"

/*!
 * \brief Convert one borrowed object into an owning PrimExpr or Error.
 *
 * The returned TVMFFIAny follows the same Expected-value convention as the
 * structural visit/mutate ABI: a successful call returns the owning result,
 * while a failed call returns an owning ffi.Error.
 */
typedef TVMFFIAny (*TVMIRFToPrimExpr)(TVMFFIAny value);

/*! \brief Language-neutral behavior table for PrimExpr-convertible objects. */
typedef struct {
  TVMIRFToPrimExpr to_prim_expr;
} TVMIRPrimExprConvertibleVTable;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_IR_EXPR_C_API_H_
