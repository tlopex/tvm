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

.. _tirx-gpu-async:

Asynchronous Programming Model
===============================
TIRX kernels rely on asynchronous TMA copies, tensor core MMA, and barrier-based
synchronization to overlap data movement with compute.

This section focuses on the underlying hardware mechanisms. Higher-level TIRX
abstractions are described in :ref:`tirx-execution-model`.

Tensor Memory Accelerator (TMA)
-------------------------------
TMA is a hardware unit (introduced in Hopper, enhanced in Blackwell) for
asynchronous bulk copies between global memory and shared memory. A single thread
initiates a multi-dimensional tensor copy; the hardware handles address
generation, data movement, and completion signaling.

Key properties:

- **Descriptor-based**: TMA uses a *tensor map* descriptor encoding base address,
  shape, strides, and swizzle mode. The descriptor is created on the host and
  passed to the kernel.
- **Asynchronous**: Copies proceed in the background while compute threads
  execute other work. The TMA hardware signals completion via mbarrier.
- **Multi-dimensional**: TMA natively supports 1D–5D tensor copies, handling
  strided access patterns in hardware.
- **Multicast**: In cluster configurations, a single TMA operation can deliver
  data to shared memory of multiple CTAs simultaneously.

In TIRX, TMA is one ``Tx.copy_async`` dispatch path. Global-to-shared TMA copies
must be forced to the TMA variant and connected to an mbarrier:

.. code:: python

    bar.arrive(stage=s, tx_count=TILE_BYTES)
    Tx.copy_async(
        smem_buf,
        global_buf[row:row + TILE_M, col:col + TILE_K],
        dispatch="tma",
        mbar=bar.ptr_to([s]),
    )

The underlying PTX instruction family is ``cp.async.bulk.tensor``. A
representative form:

.. code:: text

    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
        [smem_addr], [tensormap, {x, y}], [mbar_addr];

The tile-primitive dispatch system handles low-level setup: descriptor
generation, expected transaction counts, and host-side tensor maps.

mbarrier (Asynchronous Barrier)
-------------------------------
``mbarrier`` is the synchronization primitive used for asynchronous operations
on Hopper and Blackwell. Unlike traditional barriers, it can track both thread
arrivals and expected async transactions.

Key capabilities:

- **Expected transaction counts**: An mbarrier can be told to expect a certain
  number of bytes from async operations (e.g., TMA). The barrier completes only
  when all expected bytes arrive AND all threads arrive.
- **Phase-based waiting**: Completion flips a parity bit (phase), distinguishing
  consecutive uses. This allows producer-consumer patterns without reinitializing
  the barrier between iterations.
- **Try-wait with timeout**: Threads can poll for completion with a configurable
  timeout, allowing software to avoid indefinite blocking when a signal is
  delayed or missing.

Lifecycle
~~~~~~~~~

1. **Initialize**: ``mbarrier.init(count)`` — set expected thread arrival count.
2. **Expect transactions** *(when TMA or other async ops contribute to
   completion)*: ``mbarrier.arrive.expect_tx(byte_count)`` — register expected
   bytes.
3. **Async operation completes**: TMA hardware signals the barrier when the copy
   finishes.
4. **Arrive**: ``mbarrier.arrive()`` — threads signal their arrival.
5. **Wait**: ``mbarrier.try_wait(phase)`` — consumer blocks until all arrivals
   and expected transactions complete. Phase flips after completion.

PTX Instruction Forms
~~~~~~~~~~~~~~~~~~~~~

.. code:: text

    // Initialize with expected arrival count
    mbarrier.init.shared.b64 [bar_addr], count;

    // Register expected TMA bytes
    mbarrier.arrive.expect_tx.shared.b64 _, [bar_addr], byte_count;

    // Thread arrives at barrier
    mbarrier.arrive.shared.b64 _, [bar_addr];

    // Remote arrive (to another CTA's barrier within the cluster)
    mbarrier.arrive.shared::cluster.b64 _, [remote_bar_addr];

    // Wait for barrier completion (with timeout loop)
    mbarrier.try_wait.parity.shared::cta.b64 complete, [bar_addr], phase, ticks;

Phase-Based Synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~
mbarriers use a parity bit (phase) to distinguish consecutive uses. After
initialization, the barrier starts at phase 0. When all arrivals and expected
transactions complete, the phase flips to 1. The next round uses phase 1, and
so on.

A producer-consumer pair can therefore operate without reinitializing the barrier
between iterations:

.. code:: text

    Phase 0: Producer loads data → arrives. Consumer waits(phase=0) → sees data.
    Phase 1: Producer loads next data → arrives. Consumer waits(phase=1) → sees data.
    Phase 0: (repeats)

In TIRX, phase tracking is managed by ``RingState`` or by ``Pipe`` endpoints
(see :ref:`tirx-execution-model`).

Memory Fences
-------------
Asynchronous operations may require proxy fences when data crosses between the
generic proxy used by normal SMEM accesses and the async proxy used by bulk async
operations. The required fence depends on direction and producer.

**fence.proxy.async**

For TMA loads, observing completion through the mbarrier includes the required
generic/async proxy ordering for the copied SMEM data. An explicit
``fence.proxy.async`` is needed in the opposite direction: before an async-proxy
operation consumes data that was produced by generic SMEM writes.

.. code:: text

    // Generic SMEM writes first.
    fence.proxy.async.shared::cta;  // make generic writes visible to async proxy
    cp.async.bulk.tensor...;        // async-proxy consumer

**tcgen05.fence::\***

Reading TMEM data written by tcgen05 operations (MMA, cp) requires a
three-step fence pattern around a synchronization point:

1. Issue ``tcgen05.fence::before_thread_sync`` before the synchronization.
2. Perform the synchronization (e.g., ``bar.sync``).
3. Issue ``tcgen05.fence::after_thread_sync`` immediately after.

.. code:: text

    tcgen05.fence::before_thread_sync;
    bar.sync ...;
    tcgen05.fence::after_thread_sync;

.. warning::

    ``tcgen05.fence::after_thread_sync`` references the **most recent** thread
    synchronization point. If additional waits or syncs are inserted between
    the TMEM producer's signal and the fence, the fence may reference the wrong
    sync. Read all needed TMEM data immediately after a single
    ``fence::after_thread_sync``, before any further synchronization.
    Violating this causes silent data corruption.

Commit Groups
-------------
``tcgen05.commit`` creates a completion group associated with previously issued
MMA work. The associated mbarrier fires when the MMA work completes, not when
commit executes.

Usage pattern:

.. code:: text

    tcgen05.mma ...;
    tcgen05.commit [bar_addr];

Key properties:

- **Per-thread**: ``tcgen05.commit`` operates per-thread. Only the thread that
  issued the MMA, for example an elected lane in a warp-scope dispatch, has a
  non-empty commit group. Other threads create empty groups that signal
  immediately.
- **In-order completion**: Commit group K must complete before commit group K+1.
  Waiting on a later group guarantees all prior groups have completed.
- **Deferred signaling**: The mbarrier is signaled upon MMA completion, not upon
  ``tcgen05.commit`` execution.

.. warning::

    Guard ``tcgen05.commit`` so exactly the intended issuing thread calls it.
    ``Tx.ptx.elect_sync()`` selects one lane per warp, so it is sufficient only
    inside a single-warp scope. In a warpgroup scope, additionally guard by
    ``warp_id`` or ``Tx.thread_id_in_wg(...) == 0``; otherwise multiple warps can
    commit and release the barrier too early.

    .. code:: python

        tid = Tx.thread_id_in_wg([THREADS_PER_WARPGROUP])
        with Tx.thread(tid == 0):
            bar.arrive(stage, cta_group=CTA_GROUP)

Software Pipelining
-------------------
Software pipelining overlaps asynchronous data movement with computation across
multiple stages. Each stage owns its own SMEM buffers and synchronization
objects.

The following timeline illustrates a 3-stage pipeline:

.. code:: text

    Stage 0:  [Load ██████]
    Stage 1:       [Load ██████]  [Compute ██████]
    Stage 2:            [Load ██████]  [Compute ██████]
    Stage 0:                 [Load ██████]  [Compute ██████]
    ...

The producer (TMA warp) fills stage ``s`` while the consumer (MMA warp)
processes stage ``s-1``. Two barrier types enforce ordering:

- **tma2mma** (``TMABar``): Producer → consumer handoff. Signals when data
  reaches SMEM; consumer waits before reading.
- **mma2tma** (``MBarrier`` or ``TCGen05Bar``): Consumer → producer handoff.
  Signals when SMEM can be reused; producer waits before overwriting.

Per-stage interaction between the two warps:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - TMA warp (producer)
     - MMA warp (consumer)
   * - Wait on ``mma2tma[s]`` (SMEM reusable)
     - Wait on ``tma2mma[s]`` (data ready)
   * - Set expected transaction count on ``tma2mma[s]``
     - Wait observes TMA completion; data is visible to generic SMEM loads
   * - Launch TMA copy into ``smem[s]``
     - Issue ``tcgen05.mma`` on ``smem[s]``
   * - —
     - Commit completion → signal ``mma2tma[s]``

.. note::

    The number of pipeline stages is a tradeoff: more stages enable better
    overlap but consume more SMEM.

Barrier Types Summary
---------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 25 25

   * - Type
     - Signaled By
     - PTX
     - Use Case
   * - ``MBarrier``
     - Thread arrive
     - ``mbarrier.arrive``
     - General thread-to-thread sync
   * - ``TMABar``
     - TMA + arrive
     - ``mbarrier.arrive.expect_tx``
     - TMA data-ready signaling
   * - ``TCGen05Bar``
     - MMA commit
     - ``tcgen05.commit``
     - MMA completion signaling (Blackwell)
