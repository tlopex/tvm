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

.. _tirx-execution-model:

Execution and Scheduling Abstractions
=====================================
TIRX provides Python-level abstractions that map to the hardware execution
hierarchy and asynchronous programming model described in the previous sections.
This section covers the TIRX-specific APIs.

Execution Scopes
----------------
TIRX models the GPU thread hierarchy as a chain of nested **execution scopes**:

.. code:: text

    world → kernel → cluster → cta → warpgroup → warp → thread

Each scope groups execution units with shared communication and synchronization
capabilities.

**ExecScope** represents a named level in this hierarchy:

.. code:: python

    from tvm.tirx import ExecScope
    scope = ExecScope("warpgroup")

**ScopeIdDef** defines scope identifiers — variables that represent a unit's
position within its parent scope:

.. code:: python

    from tvm.tirx import ScopeIdDef, Var

    # Define 4 warpgroups within a CTA
    wg_var = Var("wg_id", "int32")
    scope_id = ScopeIdDef(
        def_ids=[wg_var],
        extents=[4],
        parent="cta",
        cur="warpgroup"
    )

Scope IDs in the DSL
~~~~~~~~~~~~~~~~~~~~
In TIRX kernels, scope IDs are accessed via scope-specific helpers:

.. code:: python

    # CTA position within a cluster (2D grid)
    cta_m, cta_n = Tx.cta_id_in_cluster([CLUSTER_M, CLUSTER_N])

    # CTA position within the kernel launch grid
    cta_id = Tx.cta_id([NUM_CTAS])

    # Warpgroup and warp IDs within the CTA hierarchy
    wg_id = Tx.warpgroup_id([NUM_WARPGROUPS])
    warp_id = Tx.warp_id_in_wg([WARPS_PER_WARPGROUP])
    lane = Tx.lane_id([32])

Thread identifiers use ``Tx.thread_id`` or ``Tx.thread_id_in_wg``. Thread
predicates are passed to ``Tx.thread(...)``:

.. code:: python

    tid = Tx.thread_id_in_wg([THREADS_PER_WARPGROUP])

    # Execute with only the first thread in the warpgroup
    with Tx.thread(tid == 0):
        bar.init(count)

    # Execute with the first 16 threads
    with Tx.thread((0 <= tid) & (tid < 16)):
        bar.arrive(stage=s)

Active Thread Filters
~~~~~~~~~~~~~~~~~~~~~
``Tx.filter`` narrows the currently active thread set for a declared scope-id
variable. It is used when a predicate should be visible to TIRX's execution
context analysis instead of being treated as an ordinary opaque boolean.

There are two forms:

.. code:: python

    # True when warp_id is in [0, 1)
    if Tx.filter(warp_id, 0, 1):
        ...

    # True when tid satisfies the given predicate
    with Tx.thread(Tx.filter(tid, tid == 0)):
        ...

The first form expresses a half-open range. The second form attaches an
arbitrary predicate to the scope-id variable. In both cases, the variable must
come from a TIRX scope-id helper such as ``Tx.warp_id(...)`` or
``Tx.thread_id_in_wg(...)``. Dispatch predicates use this active-thread
metadata to distinguish single-thread, single-warp, warpgroup, and CTA-level
implementations.

Warp Roles and Register Budgets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SM100 kernels often assign different warps or warpgroups to TMA, MMA, and store
work. ``WarpRole`` and ``WarpgroupRole`` package the common guard + execution
scope + register-budget pattern:

.. code:: python

    from tvm.tirx.lang.warp_role import WarpRole, WarpgroupRole

    warp_id = Tx.warp_id([8])
    wg_id = Tx.warpgroup_id([2])

    with WarpRole(warp_id, 1, regs=48):
        # TMA or epilogue work for warp 1
        ...

    with WarpRole(warp_id, 0, regs=232, increase=True):
        # MMA work for warp 0, with a larger register budget
        ...

    with WarpgroupRole(wg_id, (0, 2), regs=200, increase=True):
        # Work assigned to a range of warpgroups
        ...

These helpers emit the corresponding ``if`` guard, enter ``Tx.warp()`` or
``Tx.warpgroup()``, and insert ``Tx.ptx.setmaxnreg`` when a register budget is
provided.

Barrier Abstractions
--------------------
TIRX wraps the hardware mbarrier mechanism in three Python classes, all allocated
from an ``SMEMPool`` (shared memory).

.. code:: python

    from tvm.tirx.lang.pipeline import MBarrier, Pipe, RingState, TCGen05Bar, TMABar
    from tvm.tirx.lang.tile_scheduler import (
        BaseTileScheduler,
        ClusterPersistentScheduler2D,
    )

MBarrier
~~~~~~~~
Basic barrier signaled by thread arrivals:

.. code:: python

    bar = MBarrier(pool, depth=PIPE_DEPTH)

    # Initialize (called by leader thread)
    bar.init(count=NUM_THREADS)

    # Producer: signal arrival
    bar.arrive(stage=s)

    # Consumer: wait for all arrivals
    bar.wait(stage=s, phase=p)

``MBarrier`` stores an array of ``depth`` mbarrier objects in shared memory,
one per pipeline stage. This enables multi-stage pipelining where each stage has
its own independent barrier. ``init`` emits a leader-thread guard by default, so
callers usually do not need to wrap it in a separate ``Tx.thread`` block.

TMABar
~~~~~~
Barrier that also tracks expected TMA transaction bytes:

.. code:: python

    bar = TMABar(pool, depth=PIPE_DEPTH)
    bar.init(count=1)

    # Producer: tell barrier to expect N bytes from TMA
    bar.arrive(stage=s, tx_count=TILE_BYTES)
    Tx.copy_async(dst_smem, src_global, dispatch="tma", mbar=bar.ptr_to([s]))

    # TMA hardware signals the barrier upon copy completion

    # Consumer: wait for data arrival
    bar.wait(stage=s, phase=p)

``TMABar.arrive`` calls ``mbarrier.arrive.expect_tx``, which registers the
expected byte count. The TMA copy must also receive the mbarrier pointer via
``mbar=bar.ptr_to([stage])`` so the hardware can signal completion when the
specified number of bytes have been written to shared memory.

TCGen05Bar
~~~~~~~~~~
Barrier signaled by tcgen05 MMA completion:

.. code:: python

    bar = TCGen05Bar(pool, depth=PIPE_DEPTH)
    bar.init(count=1)

    # MMA consumer: commit MMA results to barrier
    # CRITICAL: exactly the thread that issued the MMA should commit.
    tid = Tx.thread_id_in_wg([THREADS_PER_WARPGROUP])
    with Tx.thread(tid == 0):
        bar.arrive(stage=s, cta_group=CTA_GROUP, cta_mask=mask)

    # Next consumer: wait for MMA completion
    bar.wait(stage=s, phase=p)

``TCGen05Bar.arrive`` calls ``tcgen05.commit``, creating a commit group that
signals the mbarrier when the most recent MMA operation completes.

.. warning::

    ``TCGen05Bar.arrive()`` must be guarded so that exactly the intended issuing
    thread calls ``tcgen05.commit``. The class itself does NOT include this
    guard — callers are responsible. ``Tx.ptx.elect_sync()`` is sufficient in a
    single-warp scope; in a warpgroup scope, also guard by warp ID or use
    ``Tx.thread_id_in_wg(...) == 0``. See :ref:`tirx-gpu-async` for the detailed
    explanation.

RingState and Pipe
------------------
``RingState`` tracks the current stage and phase for software-pipelined loops:

.. code:: python

    state = RingState(depth=PIPE_DEPTH)

    # Initialize: producer starts at stage 0, phase 1 (ready to produce)
    state.init(phase=1)

    # Advance to next stage (wraps around, flips phase at boundary)
    state.advance()

    # Use in barrier operations:
    bar.wait(state.stage, state.phase)

``RingState`` manages two local scalars:

- **stage**: Current pipeline stage index (0 to ``pipe_depth - 1``).
- **phase**: Current phase (0 or 1), flips each time the stage wraps around.

The producer starts at phase 1 because it must skip the first wait — at the
start of the pipeline, no consumer has signaled yet, so the "empty" barrier is
already in the completed state at phase 0. Starting the producer at phase 1
ensures its first ``bar.wait(stage, phase)`` succeeds immediately.

A typical pipeline loop:

.. code:: python

    prod_state = RingState(PIPE_DEPTH)
    prod_state.init(phase=1)

    cons_state = RingState(PIPE_DEPTH)
    cons_state.init(phase=0)

    for k_iter in range(num_k_tiles + PIPE_DEPTH - 1):
        if warp_id == TMA_WARP:
            if k_iter < num_k_tiles:
                # Wait for consumer to release this stage
                bar_empty.wait(prod_state.stage, prod_state.phase)

                # Issue TMA copy into this stage's buffer
                bar_full.arrive(prod_state.stage, tx_count=TILE_BYTES)
                Tx.copy_async(
                    smem[prod_state.stage],
                    global_tile[k_iter],
                    dispatch="tma",
                    mbar=bar_full.ptr_to([prod_state.stage]),
                )

            prod_state.advance()

        elif warp_id == MMA_WARP:
            if k_iter >= PIPE_DEPTH - 1:
                # Wait for data to arrive in this stage
                bar_full.wait(cons_state.stage, cons_state.phase)

                # Compute
                Tx.gemm_async(acc, A_smem[cons_state.stage], B_smem[cons_state.stage])

                # Signal that this stage's buffer can be reused
                if Tx.ptx.elect_sync():
                    bar_empty.arrive(cons_state.stage)

                cons_state.advance()

For standard producer/consumer flows, ``Pipe`` packages the full/empty barrier
pair and creates endpoints with their own ``RingState``:

.. code:: python

    pipe = Pipe.tma(pool, PIPE_DEPTH, empty_count=1)
    producer = pipe.producer()
    consumer = pipe.consumer()

    producer.wait()
    producer.signal(tx_count=TILE_BYTES)
    producer.advance()

    consumer.wait()
    # ... consume the staged data ...
    tid = Tx.thread_id_in_wg([THREADS_PER_WARPGROUP])
    with Tx.thread(tid == 0):
        consumer.signal(cta_group=CTA_GROUP)
    consumer.advance()

Tile Scheduling
---------------
**Persistent kernels** process multiple output tiles by looping rather than
launching one CTA per tile. TIRX provides tile schedulers that map CTA/cluster
IDs to output tile coordinates.

BaseTileScheduler
~~~~~~~~~~~~~~~~~
Base class that maintains ``m_idx``, ``n_idx``, and ``linear_idx`` state. It is
intended for subclassing: subclasses implement ``update_current_m_n_idx`` to map
the current linear work index to tile coordinates.

.. code:: python

    from tvm.script import tirx as Tx
    from tvm.tirx.lang.tile_scheduler import BaseTileScheduler

    class RowMajorScheduler(BaseTileScheduler):
        @Tx.inline
        def update_current_m_n_idx(self, linear_idx):
            self.m_idx = linear_idx // NUM_N_TILES
            self.n_idx = linear_idx % NUM_N_TILES

    scheduler = RowMajorScheduler(prefix="sched")
    scheduler.init(linear_init=cluster_id)
    while scheduler.valid(total_tiles):
        m, n = scheduler.m_idx, scheduler.n_idx
        # ... process tile (m, n) ...
        scheduler.next_tile(step=num_clusters)

ClusterPersistentScheduler2D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A 2D scheduler with L2 cache locality optimization:

.. code:: python

    scheduler = ClusterPersistentScheduler2D(
        prefix="sched",
        num_m_tiles=M // BLOCK_M,
        num_n_tiles=N // BLOCK_N,
        num_clusters=NUM_CLUSTERS,
        l2_group_size=8,     # group this many M-rows for L2 locality
        cluster_m=1,         # cluster dimension in M
        cluster_n=1,         # cluster dimension in N
        serpentine=False,     # CUTLASS-style serpentine swizzle
    )

    scheduler.init(cluster_id=Tx.cluster_id([NUM_CLUSTERS]))
    while scheduler.valid():
        m, n = scheduler.m_idx, scheduler.n_idx
        # ... process tile ...
        scheduler.next_tile()

The scheduler supports two tile ordering modes:

- **Group-major** (default): Tiles are grouped into L2-friendly row groups of
  ``l2_group_size``. Within each group, tiles are traversed column-major to
  maximize SMEM reuse of B tiles.
- **Serpentine** (``serpentine=True``): CUTLASS-style 2D block swizzle with
  serpentine (zigzag) traversal within each group, further improving L2 hit
  rates by visiting nearby tiles consecutively.

Other Built-in Tile Schedulers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The helpers in ``tvm.tirx.lang.tile_scheduler`` are ``@Tx.meta_class`` Python
objects whose methods emit TIR through ``@Tx.inline``. They are meant to be
constructed in Python and then used inside ``@Tx.prim_func`` bodies.

.. list-table::
   :header-rows: 1
   :widths: 30 35 45

   * - Scheduler
     - Coordinates
     - Use case
   * - ``GroupMajor3D``
     - ``m_idx``, ``n_idx``, ``k_idx``
     - 3D group-major traversal over M/N/K tiles. It handles a dynamic or
       static M extent, full M groups, and a final partial M group without
       emitting divide-by-zero code for tail-free cases.
   * - ``RankAwareGroupMajorTileScheduler``
     - ``m_idx``, ``n_idx``
     - NVSHMEM-aware group-major scheduling. It queries
       ``Tx.nvshmem.my_pe()`` and visits remote M rows before local rows so
       distributed kernels can overlap remote communication with useful work.
   * - ``IndexedTripleTileScheduler``
     - ``b_idx``, ``h_idx``, ``q_idx``
     - Indirect work-list scheduling. The scheduler reads ``b_indices``,
       ``h_indices``, ``q_indices``, and ``tiles_indptr`` so each SM owns a
       contiguous slice of precomputed batch/head/query tiles.
   * - ``FlashAttentionLinearScheduler``
     - ``batch_idx``, ``head_idx``, ``m_block_idx``
     - Non-causal flash-attention traversal. It maps a linear CTA id to
       batch, head, and query-block coordinates, then advances by
       ``num_ctas`` for persistent kernels.
   * - ``FlashAttentionLPTScheduler``
     - ``batch_idx``, ``head_idx``, ``m_block_idx``
     - Causal flash-attention traversal for non-persistent kernels. It uses
       Longest Processing Time ordering by reversing the query-block index and
       applies an L2 swizzle over the batch/head dimension.

Use these classes as building blocks rather than as a closed scheduler set. A
kernel can subclass ``BaseTileScheduler`` or compose the same scalar-state
pattern when it needs a problem-specific work order.

SMEMPool
--------
Shared memory in TIRX is managed via a **bump allocator**:

.. code:: python

    pool = Tx.SMEMPool()

    # Allocate MMA operand buffers with inferred swizzled layouts
    A_smem = pool.alloc_mma((64, 64), "float16")
    B_smem = pool.alloc_mma((64, 64), "float16")
    bar_buf = pool.alloc((4,), "uint64", align=8)

    # Commit the total allocation size
    pool.commit()

``SMEMPool`` manages a contiguous region of dynamic shared memory. Each
``alloc`` call bumps an internal offset, respecting alignment requirements.
The ``commit()`` call emits an annotation recording the total allocation size,
which the CUDA runtime uses to set the dynamic shared memory size at launch.

Alignment requirements:

.. list-table::
   :header-rows: 1
   :widths: 35 20 30

   * - Buffer Type
     - Alignment
     - Reason
   * - Swizzled MMA operand buffers
     - ``pool.alloc_mma(..., align=1024)``
     - Infers the swizzled MMA layout and alignment
   * - Unswizzled data buffers
     - ``align=128``
     - 128-byte cache line alignment
   * - Barriers (``uint64``)
     - ``align=8``
     - 8-byte natural alignment

Place large, high-alignment buffers first to minimize padding waste. For
pipelined kernels, allocate ``PIPE_DEPTH`` copies of each data buffer
(e.g., ``pool.alloc((PIPE_DEPTH, TILE_M, TILE_K), ...)``).

TMEMPool
--------
Blackwell TMEM is allocated in columns through ``tcgen05.alloc``. TIRX exposes
this through ``TMEMPool``, which stores the hardware-allocated TMEM base address
in shared memory and carves logical TMEM buffers out of that allocation:

.. code:: python

    smem_pool = Tx.SMEMPool()
    tmem_pool = Tx.TMEMPool(
        smem_pool,
        total_cols=512,
        cta_group=1,
        alloc_warp=0,
    )

    acc = tmem_pool.alloc((128, 128), "float32")
    sf = tmem_pool.alloc_sf((128, 8), "float8_e8m0fnu", sf_per_mma=1)

    tmem_pool.commit()
    # ... tcgen05 operations using acc / sf ...
    tmem_pool.dealloc()

``TMEMPool.alloc`` returns a ``scope="tmem"`` buffer with either an explicit
layout or a default TMEM layout inferred from the requested shape and dtype.
``alloc_sf`` is a convenience helper for tcgen05 block-scale factor buffers.
``region`` creates staged views over an existing TMEM buffer for pipelined
kernels.

``commit`` emits the guarded ``tcgen05.alloc`` sequence, and ``dealloc`` emits
``tcgen05.relinquish_alloc_permit`` followed by ``tcgen05.dealloc``. The
``alloc_warp`` and ``dealloc_warp`` parameters select which warp issues those
warp-uniform instructions.
