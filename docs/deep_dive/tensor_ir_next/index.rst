..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. _tensor-ir-next-deep-dive:

TensorIRX
=========
TensorIRX, also called TensorIR Next, is TVM's programming model for explicit
tile-level accelerator kernels. The APIs live under the ``tvm.tirx`` namespace
and lower tile primitives to target-specific code. This section is not a
replacement for S-TIR, and it is not a reference for every class exported from
``tvm.tirx``.

TensorIR Next provides layout abstractions, hierarchical execution scopes,
asynchronous pipeline primitives, and a tile-primitive dispatch framework for
writing high-performance accelerator kernels. This guide covers the core
programming model and the target-specific CUDA and AWS Trainium paths, including
CUDA Ampere/Hopper copy and synchronization paths, Blackwell TMEM and tcgen05,
and Trainium backend extension points.

Users express kernels in terms of tile primitives (``Tx.gemm_async``,
``Tx.copy_async``, etc.) and layouts, and the TIRX compiler lowers these
to target-specific code through tile-primitive dispatch and scheduling.
Current backends include NVIDIA CUDA and AWS Trainium.

Start here if you already know basic TVMScript/TIR and want to understand how
TensorIR Next represents explicit tile operations, layouts, barriers, and
target-specific dispatch. Many snippets are schematic; complete runnable
coverage lives in ``tests/python/tirx``.

The sections are grouped around the programming model, hardware-specific
backend paths, compiler internals, backend authoring, and utility APIs used by
real kernels and tests.

.. toctree::
    :maxdepth: 2

    abstraction
    gpu_architecture
    gpu_async
    gpu_tensorcore
    layout
    execution_model
    operators
    trainium
    cuda_intrinsics
    architecture
    new_backend
    dsl_utilities
