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

.. _tirx-cuda-intrinsics:

CUDA Intrinsics Namespace
=========================
Most kernel authors should use tile primitives such as ``Tx.copy_async`` and
``Tx.gemm_async`` instead of calling CUDA intrinsics directly. The CUDA
intrinsic layer is the lower-level codegen surface used by tile-primitive
dispatch implementations. It lives in
``python/tvm/tirx/operator/intrinsics/cuda`` and maps ``tirx.*`` intrinsic
calls to CUDA helper functions, inline PTX, and required header snippets.

How Intrinsics Are Registered
-----------------------------
The helper ``device_intrinsic`` in
``python/tvm/tirx/operator/intrinsics/_schema.py`` registers one CUDA device
helper and its codegen entry. It:

- creates a ``__forceinline__ __device__`` helper function;
- registers a codegen callback under ``tirx.<op_name>``;
- marks the TVM op as opaque so it can be emitted through ``call_intrin``;
- optionally reports header dependency tags needed by the helper body.

For intrinsics whose concrete form depends on attributes, a module can instead
use ``@register_codegen`` and choose a helper dynamically. This pattern is used
for operations such as ``ptx_cp_async``, ``ptx_tcgen05_mma``, and reductions,
where the final PTX form depends on shape, dtype, memory scope, or modifier
arguments.

Module Map
----------
The CUDA intrinsic package is split by hardware feature:

.. list-table::
   :header-rows: 1
   :widths: 24 45 45

   * - Module
     - Feature area
     - Examples
   * - ``mma``
     - PTX MMA and matrix load/store helpers for older tensor-core paths.
     - ``ptx_mma``, ``ptx_ldmatrix``, ``ptx_stmatrix``.
   * - ``wgmma``
     - Hopper warpgroup matrix multiply.
     - ``ptx_wgmma_mma_async_ss``, ``ptx_wgmma_mma_async_rs``,
       ``ptx_wgmma_noop_barrier``.
   * - ``tcgen05``
     - Blackwell tensor memory and 5th-generation tensor-core operations.
     - ``ptx_tcgen05_alloc``, ``ptx_tcgen05_ld``, ``ptx_tcgen05_st``,
       ``ptx_tcgen05_mma``, ``ptx_tcgen05_commit``, ``ptx_tcgen05_cp``.
   * - ``cp_async``
     - Asynchronous copy instructions.
     - ``ptx_cp_async``, bulk shared/cluster copies, and TMA tensor copies.
   * - ``sync``
     - Synchronization primitives.
     - Mbarrier arrive/wait, cluster barriers, warp vote/elect/sync,
       fences, ``__syncthreads`` and ``__syncwarp`` helpers.
   * - ``math``
     - Device math helpers and reductions.
     - Packed f32x2 arithmetic, reciprocal/exponential helpers,
       warp reductions, and CTA reductions.
   * - ``memory``
     - Loads, stores, casts, atomics, and address helpers.
     - Typed byte copies, ``ldg``, ``ld.global.acquire``, atomics,
       shared-memory pointer casts, and TMEM address helpers.
   * - ``nvshmem``
     - NVSHMEM device operations.
     - PE queries, quiet/fence/barrier, get/put memory paths,
       signal/wait, and signaled put helpers.
   * - ``misc``
     - Utility intrinsics.
     - Register budget control, register fetch, profiler timing,
       ``printf``, trap, and sleep helpers.

Header Tags
-----------
Codegen callbacks can request header tags. The CUDA header generator validates
the tags and emits the corresponding includes or helper definitions. Current
tags include:

.. code:: text

    cuda
    cuda/barrier
    cooperative_groups
    fp16
    bf16
    fp8
    fp6
    fp4
    int8
    math_constants
    mma
    warp_shuffle
    cast_smem_ptr_to_int
    get_tmem_addr
    gmma_descriptor
    smem_descriptor
    instr_descriptor
    instr_descriptor_block_scaled
    get_time_stamp
    nvshmem
    elect_one_sync

For example, an NVSHMEM intrinsic requests ``nvshmem`` so the generated CUDA
source includes ``nvshmem.h`` and ``nvshmemx.h``. A tcgen05 descriptor helper
can request descriptor tags so the emitted source contains the required helper
struct and bitfield code.

Relationship to Tile Primitives
-------------------------------
Tile primitives choose schedules at the ``TilePrimitiveCall`` level. CUDA
intrinsics are usually the last step inside those schedules:

.. list-table::
   :header-rows: 1
   :widths: 34 44

   * - Tile primitive path
     - Intrinsic layer used
   * - ``tirx.copy_async`` with TMA
     - ``cp_async`` TMA and bulk-copy helpers.
   * - ``tirx.copy_async`` with SMEM/TMEM
     - ``tcgen05`` copy, load, store, and address helpers.
   * - ``tirx.gemm_async`` on Blackwell
     - ``tcgen05`` MMA, descriptor, commit, and wait helpers.
   * - CUDA reductions
     - ``math`` warp/CTA reductions plus ``sync`` when a scope barrier is
       required.
   * - CUDA elementwise schedules
     - ``math``, ``memory``, and scope synchronization helpers.
   * - Distributed persistent kernels
     - ``nvshmem`` helpers and scheduler-side rank queries.

This separation keeps user-facing kernels stable: most code stays at the
tile-primitive level, while dispatch authors can add or refine intrinsic
lowering without changing the DSL surface.

Adding a CUDA Intrinsic
-----------------------
Add a CUDA intrinsic when a dispatch implementation needs a target helper that
cannot be expressed with existing TIR or existing ``tirx.*`` intrinsics.

1. Pick the module that matches the hardware feature.
2. Register a fixed helper with ``device_intrinsic`` or a dynamic helper with
   ``@register_codegen``.
3. Return a ``cuda_func_call`` and any required header tags.
4. Use the new intrinsic from a tile-primitive dispatch implementation.
5. Add a focused lowering/codegen test that checks the emitted CUDA or PTX
   shape.

Prefer a tile primitive when the operation is meaningful to kernel authors.
Prefer a CUDA intrinsic when the operation is only an implementation detail of a
specific backend schedule.
