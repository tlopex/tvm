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

.. _tirx-abstraction:

TIRX Abstraction
================

From TensorIR to TIRX
----------------------
TensorIR represents tensor programs as loop nests with schedule blocks and axis
annotations. This loop-level abstraction is effective for expressing and
transforming individual tensor computations. TIRX builds on TVM's TIR
infrastructure, but asks kernel authors to work with explicit tile primitives
instead of relying only on loop-nest schedules. This programming model is
designed for modern GPU architectures (Hopper, Blackwell) that require reasoning
about:

- **Hardware-specific memory hierarchies**: Global memory, shared memory (SMEM),
  tensor memory (TMEM), and registers, each with different access patterns and
  swizzle requirements.
- **Asynchronous execution**: Tensor Memory Accelerator (TMA) for async data
  movement, asynchronous matrix multiply-accumulate (MMA) via tensor cores,
  and barrier-based synchronization between producers and consumers.
- **Hierarchical thread organization**: Clusters of CTAs, warpgroups within CTAs,
  and warp-level cooperative operations that go beyond simple thread bindings.
- **Software pipelining**: Multi-stage producer-consumer pipelines that overlap
  data loading with computation.

TIRX introduces abstractions that directly map to these hardware concepts,
allowing kernel authors to express tile-level intent while the compiler expands
tile primitives into target-specific implementation code.

A Simple Example
----------------
Consider how matrix multiplication looks in S-TIR versus TIRX.

**S-TIR** — loop-level abstraction:

.. code:: python

    from tvm.script import tirx as T

    @T.prim_func(s_tir=True)
    def matmul(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

In this branch, S-TIR examples use the ``tirx`` TVMScript dialect with
``s_tir=True``. This function still uses loop-level blocks rather than TIRX
tile primitives.

S-TIR spells out loops, spatial/reduce axes, and the accumulation pattern. The
compiler applies schedule transformations (tiling, vectorization, thread
binding) to optimize this for a target GPU.

**TIRX** — tile-primitive abstraction:

.. note::

   This is a schematic snippet meant to show the level of abstraction. A complete
   kernel also declares launch scopes, tile loops, pipeline bookkeeping, barriers,
   accumulator storage, and final stores. See ``tests/python/tirx`` for runnable
   examples.

.. code:: python

    from tvm.script import tirx as Tx

    @Tx.prim_func
    def matmul(
        A: Tx.Buffer((M, K), "float16"),
        B: Tx.Buffer((K, N), "float16"),
        C: Tx.Buffer((M, N), "float16"),
    ):
        A_smem = Tx.alloc_buffer((BLK_M, BLK_K), "float16", scope="shared")
        B_smem = Tx.alloc_buffer((BLK_K, BLK_N), "float16", scope="shared")

        Tx.copy_async(A_smem, A[m_start:m_start + BLK_M, k:k + BLK_K])
        Tx.copy_async(B_smem, B[k:k + BLK_K, n_start:n_start + BLK_N])

        Tx.gemm_async(C_acc, A_smem, B_smem)

TIRX expresses the same computation with high-level tile primitives. The
tile-primitive dispatch system selects the target implementation — TMA
descriptors, MMA instructions, barrier configurations, and memory layouts — but
kernel authors still choose scopes, tile shapes, pipeline structure, alignment,
and dispatch parameters.

Compilation Pipeline
--------------------
A TIRX kernel passes through these stages:

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - Stage
     - IR form
     - Main work
   * - Authoring
     - ``@Tx.prim_func`` Python DSL
     - Build explicit tile-level kernels with ``Tx.copy_async``,
       ``Tx.gemm_async``, layouts, scopes, and barriers.
   * - Parsing
     - TIR with ``TilePrimitiveCall`` nodes
     - Preserve each tile primitive as a first-class IR node.
   * - LowerTIRx
     - TIR with concrete hardware intrinsics
     - Dispatch each tile primitive to the best target-specific schedule.
   * - TIR lowering
     - Lowered TIR
     - Run standard lowering passes such as buffer flattening, simplification,
       and host/device splitting.
   * - Code generation
     - Target source or runtime module
     - Emit target-specific code, such as CUDA source with inline PTX or a
       Trainium runtime module, for barriers, tensor cores, and other hardware
       operations.

Key Concepts
------------
The following sections cover the building blocks of TIRX in detail:

- :ref:`tirx-gpu-architecture` — GPU memory hierarchy and thread execution model.
- :ref:`tirx-gpu-async` — TMA, asynchronous barriers, commit groups, and software
  pipelining.
- :ref:`tirx-gpu-tensorcore` — 5th generation tensor core (tcgen05) instructions
  on Blackwell.
- :ref:`tirx-layout` — The Axe Layout system for describing hardware-aware data
  layouts.
- :ref:`tirx-execution-model` — TIRX's execution scopes, barrier abstractions,
  ring state, pipeline endpoints, and tile scheduling.
- :ref:`tirx-operators` — The tile-primitive system, dispatch mechanism, and
  compilation pipeline.
- :ref:`tirx-trainium` — Trainium memory scopes, NKI instruction emission, and
  the Trainium-specific lowering pipeline.
- :ref:`tirx-cuda-intrinsics` — The low-level CUDA intrinsic namespace used by
  dispatch implementations.
- :ref:`tirx-dsl-utilities` — Inline macros, meta classes, hints, external
  kernels, and verification helpers.
