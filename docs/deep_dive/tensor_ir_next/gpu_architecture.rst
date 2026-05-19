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

.. _tirx-gpu-architecture:

GPU Architecture Background
===========================
This section provides background on the GPU hardware concepts that TIRX targets.

GPU Memory Hierarchy
--------------------

.. image:: /_static/img/tirx/memory_hierarchy.png
   :alt: GPU memory hierarchy showing global memory, shared memory, tensor memory, and registers
   :width: 80%
   :align: center

Global Memory (HBM)
~~~~~~~~~~~~~~~~~~~
Global memory is the largest and slowest memory space. Kernel inputs and outputs
reside here. On modern GPUs, global memory bandwidth is the primary bottleneck
for memory-bound kernels.

In TIRX, global memory buffers are declared with ``scope="global"`` (the default):

.. code:: python

    A: Tx.Buffer((M, K), "float16")  # implicitly scope="global"

Direct access to global memory from compute threads is inefficient.
High-performance tiled kernels often stage global data through ``Tx.copy_async``;
depending on scope and dispatch this may lower to TMA or another copy primitive
(see :ref:`tirx-gpu-async`).

Shared Memory (SMEM)
~~~~~~~~~~~~~~~~~~~~
Shared memory is a fast, per-SM memory space visible to all threads within a CTA.
SMEM stages data between global memory (loaded by async copy primitives such as
TMA) and compute (through registers or tensor memory).

In TIRX, shared-memory buffers are typically allocated from a shared-memory pool:

.. code:: python

    pool = Tx.SMEMPool()  # bump allocator for shared memory
    A_smem = pool.alloc((TILE_M, TILE_K), "float16", align=128)

``SMEMPool`` manages a contiguous region of dynamic shared memory
(``shared.dyn``), providing bump allocation with alignment control. Multiple
buffers, such as data tiles and barriers, share the same SMEM space.

**Bank Conflicts and Swizzling**

Shared memory is organized into 32 banks. When multiple threads in a warp access
the same bank, those accesses serialize — a *bank conflict*.

For a 2D tile stored row-major, consecutive columns map to consecutive banks.
Accessing a column slice, however, causes all threads to hit the same banks.

**Swizzling** remaps addresses to distribute accesses across banks:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Without swizzle
     - With 128B swizzle
   * - Row 0: bank 0, 1, 2, 3, ...
     - Row 0: bank 0, 1, 2, 3, ...
   * - Row 1: bank 0, 1, 2, 3, ...
     - Row 1: bank 4, 5, 6, 7, ... (XOR with row)
   * - Row 2: bank 0, 1, 2, 3, ...
     - Row 2: bank 0, 1, 2, 3, ... (pattern repeats)

TMA supports hardware swizzle modes (32B, 64B, 128B) that apply this remapping
automatically during the copy. When reading data back from SMEM, the same swizzle
pattern must be respected. In TIRX, this is expressed with ``SwizzleLayout``
(see :ref:`tirx-layout`).

TIRX supports the same swizzle modes as PTX. In practice, 128B is the default
choice for MMA operand tiles.

The full set of supported modes:

.. list-table::
   :header-rows: 1
   :widths: 15 15 40

   * - Mode
     - Atom Size
     - Use Case
   * - None
     - N/A
     - Epilogue buffers, small buffers without bank-conflict pressure
   * - 32B
     - 32 bytes
     - Rarely used in practice
   * - 64B
     - 64 bytes
     - Occasionally used for small tiles
   * - 128B
     - 128 bytes
     - Default for MMA operand tiles (best bank-conflict avoidance)

Tensor Memory (TMEM)
~~~~~~~~~~~~~~~~~~~~
Tensor Memory is a memory space introduced in Blackwell (SM100), directly
connected to the tensor core unit. It serves as the accumulator storage for 5th
generation tensor core (tcgen05) operations, and depending on the MMA variant,
may also source operand A.

Key properties:

- **Per CTA allocation**: TMEM is allocated per CTA, with warp and warpgroup
  access restrictions on individual load/store and MMA instructions.
- **Columnar layout**: TMEM is organized as rows (mapped to warp lanes) and
  columns. A TMEM allocation specifies the number of columns; the hardware
  assigns rows based on warp identity.
- **Requires explicit allocation**: Unlike registers or shared memory, TMEM must
  be explicitly allocated (``tcgen05.alloc``) and deallocated (``tcgen05.dealloc``)
  within the kernel.
- **Pool column count must be power of 2**: The underlying ``tcgen05.alloc``
  pool size uses 32, 64, 128, 256, or 512 columns. ``TMEMPool`` then carves
  logical buffers out of that pool.

In TIRX, TMEM buffers are declared over an allocated TMEM address. The
``TMEMPool`` helper returns ``scope="tmem"`` buffers; ``commit`` emits the
``tcgen05.alloc`` guard, and ``dealloc`` emits the ``tcgen05.dealloc`` guard:

.. code:: python

    smem_pool = Tx.SMEMPool()
    tmem_pool = Tx.TMEMPool(smem_pool, total_cols=512, cta_group=1)
    C_tmem = tmem_pool.alloc((128, BLK_N), "float32")
    tmem_pool.commit()
    # ... tcgen05 operations ...
    tmem_pool.dealloc()

Data movement involving TMEM uses dedicated tcgen05 instructions:

.. list-table::
   :header-rows: 1
   :widths: 25 50

   * - Instruction
     - Description
   * - ``tcgen05.alloc``
     - Allocate TMEM columns
   * - ``tcgen05.dealloc``
     - Free TMEM columns
   * - ``tcgen05.cp``
     - Copy SMEM → TMEM (with optional decompression)
   * - ``tcgen05.ld``
     - Load TMEM → registers (async; requires ``tcgen05.wait::ld``)
   * - ``tcgen05.st``
     - Store registers → TMEM (async; requires ``tcgen05.wait::st``)
   * - ``tcgen05.mma``
     - Matrix multiply-accumulate; reads A from TMEM or SMEM descriptor,
       B from SMEM descriptor, writes D to TMEM

Registers
~~~~~~~~~
Registers are the fastest storage, private to each thread. In TIRX:

.. code:: python

    val = Tx.local_scalar("float32")          # single scalar
    buf = Tx.alloc_buffer((4,), "float32", scope="local")  # small local array

Register pressure is a key constraint in high-performance kernels. The
tile-primitive dispatch system coordinates register use for MMA operands,
accumulator spills, and intermediate results.

Memory Scope Summary
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 20

   * - Scope
     - TIRX Scope
     - Capacity
     - Latency
     - Access Pattern
   * - Global (HBM)
     - ``"global"``
     - GBs
     - ~400 cycles
     - Direct or via async copy to SMEM
   * - Shared (SMEM)
     - ``"shared"``
     - 164–228KB/SM
     - ~30 cycles
     - Direct or via TMA; swizzle for bank conflicts
   * - Tensor (TMEM)
     - ``"tmem"``
     - Per CTA allocation
     - Low (direct tensor core connection)
     - tcgen05 instructions only (Blackwell)
   * - Registers
     - ``"local"``
     - 255/thread
     - Fastest
     - Direct thread access

GPU Execution Model
-------------------

Thread Hierarchy
~~~~~~~~~~~~~~~~

.. code:: text

    Grid (kernel launch)
     └── Cluster (portable up to 8 CTAs; larger nonportable sizes are hardware-specific)
          └── CTA / Thread Block (up to 1024 threads)
               └── Warpgroup (128 threads = 4 warps, Hopper+)
                    └── Warp (32 threads)
                         └── Thread (single lane)

A **thread** is the smallest execution unit, with its own registers and a unique
``threadIdx`` within its CTA.

A **warp** groups 32 threads that execute in lockstep (SIMT). Warp-level
primitives include shuffle (``__shfl_sync``), vote (``__ballot_sync``), and
elect (``elect_sync``).

A **warpgroup** consists of 4 consecutive warps (128 threads). Warpgroups are
the fundamental unit for tensor core operations: WGMMA on Hopper, tcgen05 MMA
on Blackwell.

A **CTA (Thread Block)** contains up to 1024 threads sharing the same shared
memory. Each CTA maps to a single SM.

A **cluster** (Hopper+) groups multiple CTAs that can directly access each
other's shared memory via Distributed Shared Memory (``DSMEM``). CUDA's portable
cluster size is up to 8 CTAs; larger nonportable cluster sizes may exist on some
hardware and should be queried before use. Clusters enable multicast TMA,
cross-CTA shared memory access, and cooperative MMA (``cta_group=2``).

TIRX maps this hierarchy to scope-relative identifiers such as ``Tx.cta_id``,
``Tx.cta_id_in_cluster``, ``Tx.warpgroup_id``, and ``Tx.thread_id``. Guarded
thread scopes are opened with ``Tx.thread(...)``. For the full TIRX API, see
:ref:`tirx-execution-model`.

Synchronization Primitives
~~~~~~~~~~~~~~~~~~~~~~~~~~
Each scope level has a corresponding synchronization mechanism:

.. list-table::
   :header-rows: 1
   :widths: 20 30 30

   * - Scope
     - Mechanism
     - PTX
   * - Warp
     - Implicit (SIMT) or ``__syncwarp``
     - ``__syncwarp`` / warp-level sync
   * - Warpgroup
     - Named barrier
     - ``bar.sync {id}, 128``
   * - CTA
     - ``__syncthreads``
     - ``bar.sync 0, {blockDim}``
   * - Cluster
     - Cluster barrier
     - ``barrier.cluster.arrive`` / ``barrier.cluster.wait``

For fine-grained producer-consumer synchronization, TIRX uses **mbarriers**,
covered in :ref:`tirx-gpu-async`.

Warp Specialization
~~~~~~~~~~~~~~~~~~~
High-performance kernels often assign different roles to different warps within
a CTA. The following illustrates one common Blackwell GEMM pattern; actual role
assignments vary by kernel:

- **Warp 3 (TMA producer)**: Issues TMA copy operations, manages barriers.
- **Warp 2 (Transform)**: Performs data transformations (e.g., scale factor
  transpose for block-scaled FP8).
- **Warp 0 (MMA consumer)**: Issues tcgen05.mma operations, performs epilogue.

Role assignment is expressed through conditional execution on warp ID:

.. code:: python

    warp_id = Tx.warp_id_in_wg([NUM_WARPS])

    if warp_id == 3:
        Tx.copy_async(smem_buf, global_tile)  # TMA producer
    elif warp_id == 0:
        Tx.gemm_async(acc, A_smem, B_smem)    # MMA consumer

**elect_sync**

Many tensor core and barrier operations must be issued by exactly one thread
per warp. ``Tx.ptx.elect_sync()`` selects the lowest active lane:

.. code:: python

    if Tx.ptx.elect_sync():
        bar.arrive(stage, cta_group=CTA_GROUP)

.. note::

    ``elect_sync`` selects **1 thread per warp**, not 1 per warpgroup. A
    warpgroup of 4 warps produces 4 elected threads. For operations requiring
    exactly 1 thread per warpgroup (e.g., TMA store), use
    ``Tx.thread_id_in_wg`` and guard ``Tx.thread`` with ``tid == 0`` instead.
