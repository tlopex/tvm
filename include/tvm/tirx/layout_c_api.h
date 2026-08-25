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

#ifndef TVM_TIRX_LAYOUT_C_API_H_
#define TVM_TIRX_LAYOUT_C_API_H_

#include <tvm/ffi/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Type attribute containing a TVMTIRXLayoutVTable pointer. */
#define TVM_TIRX_LAYOUT_VTABLE_ATTR "__layout_vtable__"

/*! \brief ABI version of TVMTIRXLayoutVTable's required prefix. */
#define TVM_TIRX_LAYOUT_VTABLE_ABI_VERSION 1

/*!
 * \brief ABI calls with zero through three arguments in addition to the layout.
 *
 * Every argument is borrowed.  The returned TVMFFIAny is owning and contains
 * either the successful result or an ffi.Error, following the structural
 * visit/mutate Expected-value convention.
 */
typedef TVMFFIAny (*TVMTIRXLayoutCall0)(TVMFFIAny layout);
typedef TVMFFIAny (*TVMTIRXLayoutCall1)(TVMFFIAny layout, TVMFFIAny arg0);
typedef TVMFFIAny (*TVMTIRXLayoutCall2)(TVMFFIAny layout, TVMFFIAny arg0, TVMFFIAny arg1);
typedef TVMFFIAny (*TVMTIRXLayoutCall3)(TVMFFIAny layout, TVMFFIAny arg0, TVMFFIAny arg1,
                                        TVMFFIAny arg2);

/*! \brief Language-neutral behavior table for one concrete layout type. */
typedef struct {
  /*! \brief Version of the required table prefix. */
  uint32_t abi_version;
  /*! \brief Total number of bytes provided by this table. */
  uint32_t struct_size;
  TVMTIRXLayoutCall1 compatible_with_shape;
  TVMTIRXLayoutCall0 verify_well_formed;
  TVMTIRXLayoutCall1 get_size;
  TVMTIRXLayoutCall1 get_span;
  TVMTIRXLayoutCall1 apply;
  TVMTIRXLayoutCall1 apply_linear;
  TVMTIRXLayoutCall2 apply_with_shape;
  TVMTIRXLayoutCall0 canonicalize;
  TVMTIRXLayoutCall3 tile;
  TVMTIRXLayoutCall2 slice;
  TVMTIRXLayoutCall3 direct_sum;
  TVMTIRXLayoutCall3 is_tile_inner;
  TVMTIRXLayoutCall3 is_tile_outer;
  TVMTIRXLayoutCall3 is_direct_sum_right;
  TVMTIRXLayoutCall3 is_direct_sum_left;
} TVMTIRXLayoutVTable;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_TIRX_LAYOUT_C_API_H_
