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

#ifndef TVM_TIRX_DATA_PRODUCER_C_API_H_
#define TVM_TIRX_DATA_PRODUCER_C_API_H_

#include <tvm/ffi/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Type attribute containing a TVMTIRXDataProducerVTable pointer. */
#define TVM_TIRX_DATA_PRODUCER_VTABLE_ATTR "__data_producer_vtable__"

/*! \brief ABI version of TVMTIRXDataProducerVTable's required prefix. */
#define TVM_TIRX_DATA_PRODUCER_VTABLE_ABI_VERSION 1

/*!
 * \brief Inspect one borrowed data producer.
 *
 * The returned TVMFFIAny is owning and contains either the successful result
 * or an ffi.Error, following the structural visit/mutate Expected convention.
 */
typedef TVMFFIAny (*TVMTIRXDataProducerCall)(TVMFFIAny producer);

/*! \brief Language-neutral behavior table for a concrete data-producer type. */
typedef struct {
  /*! \brief Version of the required table prefix. */
  uint32_t abi_version;
  /*! \brief Total number of bytes provided by this table. */
  uint32_t struct_size;
  TVMTIRXDataProducerCall get_shape;
  TVMTIRXDataProducerCall get_data_type;
  TVMTIRXDataProducerCall get_name_hint;
} TVMTIRXDataProducerVTable;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TVM_TIRX_DATA_PRODUCER_C_API_H_
