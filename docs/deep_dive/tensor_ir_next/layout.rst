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

.. _tirx-layout:

Axe Layout System
=================
The Axe Layout system describes how data is physically arranged in hardware
memory. It provides composable primitives for multi-axis layouts, tiling,
swizzling, and thread-to-data mappings.

Why a Layout System?
--------------------
The same logical tensor has different physical arrangements depending on where
it resides and how it is accessed:

- A **global memory** tile may be stored row-major with no special layout.
- A **shared memory** tile may use a 128B swizzle pattern to avoid bank conflicts.
- A **TMEM** buffer has a columnar layout where rows map to warp lanes.
- A **register** fragment distributes elements across threads in a pattern
  determined by the MMA instruction.

The Axe Layout system captures all of these patterns with a small set of
composable primitives: **Iter**, **Axis**, **TileLayout**, **SwizzleLayout**,
and **ComposeLayout**.

Core Concepts
-------------

Axis
~~~~
An **Axis** represents a named dimension in the physical address space. Axes
can be **thread axes** (representing thread-level parallelism) or **memory axes**
(representing memory organization).

Thread axes (pre-registered singletons):

.. list-table::
   :header-rows: 1
   :widths: 15 50

   * - Axis
     - Meaning
   * - ``laneid``
     - Lane within a warp (0–31)
   * - ``warpid``
     - Warp within a CTA
   * - ``wgid``
     - Warpgroup ID
   * - ``tid_in_wg``
     - Thread index within a warpgroup (0–127)
   * - ``wid_in_wg``
     - Warp index within a warpgroup (0–3)
   * - ``tx``
     - threadIdx.x
   * - ``bx``, ``by``, ``bz``
     - blockIdx.x/y/z
   * - ``cbx``, ``cby``, ``cbz``
     - Cluster-relative block index
   * - ``pid``
     - Persistent thread ID

Memory axes:

.. list-table::
   :header-rows: 1
   :widths: 15 50

   * - Axis
     - Meaning
   * - ``m``
     - Generic memory axis
   * - ``P``
     - Partition axis
   * - ``F``
     - Fragment axis
   * - ``Bank``
     - SMEM bank axis
   * - ``TCol``
     - TMEM column axis
   * - ``TLane``
     - TMEM lane (row) axis

Axes are accessed via ``Axis.get("name")`` or as attributes on the ``Axis``
class:

.. code:: python

    from tvm.tirx.layout import Axis
    lane = Axis.laneid     # pre-registered singleton
    custom = Axis.get("my_axis")  # creates if not exists

Iter
~~~~
An **Iter** (iterator) is the fundamental building block of a layout. It consists
of:

- **extent**: How many elements along this axis.
- **stride**: The stride between consecutive elements.
- **axis**: Which named axis this iterator belongs to.

.. code:: python

    from tvm.tirx.layout import Axis, Iter

    # 16 elements along laneid axis, stride 1
    Iter(extent=16, stride=1, axis=Axis.laneid)

    # 8 elements along a memory axis, stride 64
    Iter(extent=8, stride=64, axis=Axis.m)

TileLayout
~~~~~~~~~~
A **TileLayout** maps logical coordinates to physical positions via **shard**
and **replicate** iterators:

- **Shard iterators**: Define how data is partitioned across an axis. Each
  position along the shard axis holds a unique piece of data.
- **Replicate iterators**: Define how data is duplicated. Each position along
  the replicate axis holds a copy of the same data.

The builder syntax uses ``S[...]`` for shard and ``R[...]`` for replicate:

.. code:: python

    from tvm.tirx.layout import R, S, TileLayout, laneid

    # Simple: 8 elements, stride 1
    layout = TileLayout(S[8])

    # Multi-dimensional: 8×4 tile
    layout = TileLayout(S[(8, 4)])

    # With explicit strides: 8 elements at stride 2 along laneid axis
    layout = TileLayout(S[8:2 @ laneid])

    # Shard + replicate: 8 unique elements, each replicated 4 times
    layout = TileLayout(S[8] + R[4])

    # With offset: base address offset of 3
    layout = TileLayout(S[8] + R[2] + 3)

The ``@`` operator binds a stride to an axis:

.. code:: python

    from tvm.tirx.layout import S, TileLayout, laneid, warpid

    # stride 1 on laneid axis, stride 128 on warpid axis
    TileLayout(S[(16, 4):(1 @ laneid, 128 @ warpid)])

**from_iters** — alternative construction from explicit iterators:

.. code:: python

    from tvm.tirx.layout import Axis, Iter, TileLayout

    TileLayout.from_iters(
        shard=[Iter(16, 1, Axis.laneid), Iter(4, 128, Axis.warpid)],
        replica=[Iter(2, 1, Axis.m)]
    )

SwizzleLayout
~~~~~~~~~~~~~
A **SwizzleLayout** represents the XOR-based swizzle patterns used by TMA for
bank-conflict-free shared memory access. It is parameterized by three values:

- **per_element**: ``log2(16 bytes / sizeof(dtype))`` — the element count in
  the 128-bit granule used by the swizzle encoding.
- **swizzle_len**: Swizzle-mode exponent (typically 3 for 128B swizzle).
- **atom_len**: Atom length exponent (typically 3).

.. code:: python

    from tvm.tirx.layout import SwizzleLayout

    # 128B TMA swizzle for different data types:
    swizzle_fp8  = SwizzleLayout(4, 3, 3)  # FP8:  16 elements per atom
    swizzle_bf16 = SwizzleLayout(3, 3, 3)  # BF16: 8 elements per atom
    swizzle_f32  = SwizzleLayout(2, 3, 3)  # F32:  4 elements per atom

The formula for ``per_element`` is:

.. code:: text

    per_element = log2(16 / sizeof(dtype))

    FP8  (1 byte):  log2(16/1) = 4
    BF16 (2 bytes): log2(16/2) = 3
    F32  (4 bytes): log2(16/4) = 2

ComposeLayout
~~~~~~~~~~~~~
A **ComposeLayout** combines a ``SwizzleLayout`` with a ``TileLayout``, applying
the swizzle pattern on top of a tiled data arrangement. This is used for SMEM
buffers that are both swizzled (for bank-conflict avoidance) and tiled (for
MMA access patterns):

.. code:: python

    from tvm.tirx.layout import ComposeLayout, S, SwizzleLayout, TileLayout

    # SMEM buffer with 128B swizzle for BF16 MMA operand
    layout = ComposeLayout(
        SwizzleLayout(3, 3, 3),
        TileLayout(S[(8, 64)])
    )

Layout Operations
-----------------

.. list-table::
   :header-rows: 1
   :widths: 20 35 30

   * - Method
     - Signature
     - Description
   * - ``apply``
     - ``layout.apply(*coords, shape=None) → dict[str, PrimExpr]``
     - Map logical coordinates to physical axis positions. Returns a dict
       keyed by axis name.
   * - ``tile``
     - ``layout.tile(outer, outer_shape, inner_shape)``
     - Split into hierarchical inner/outer tiles. Returns a ``TileLayout``
       or ``ComposeLayout``.
   * - ``canonicalize``
     - ``layout.canonicalize() → Layout``
     - Simplify to canonical form by merging redundant iterators.
   * - ``slice``
     - ``layout.slice(shape, region) → Layout | None``
     - Extract a sub-region. ``region`` is a list of half-open ``(begin, end)``
       pairs per dimension.
   * - ``direct_sum``
     - ``layout.direct_sum(left, left_shape, right_shape)``
     - Interleave two layouts in the output space. Used for multi-CTA
       patterns where two CTAs contribute to different output regions.
   * - ``is_tile_inner``
     - ``base.is_tile_inner(composite, tiled_shape, inner_shape) → TileLayout | None``
     - Treat ``base`` as the inner component and recover the outer component.
       Returns ``None`` if decomposition is not possible.
   * - ``is_tile_outer``
     - ``base.is_tile_outer(composite, tiled_shape, outer_shape) → Layout | None``
     - Treat ``base`` as the outer component and recover the inner component.

Example — ``apply``:

.. code:: python

    from tvm.tirx.layout import S, TileLayout, laneid, m

    layout = TileLayout(S[(16, 8):(1 @ laneid, 32 @ m)])
    result = layout.apply(3, 5)
    # result: {"laneid": 3, "m": 160}

Example — ``slice``:

.. code:: python

    layout = TileLayout(S[(64, 64):(64, 1)])
    sub = layout.slice(shape=[64, 64], region=[(0, 32), (0, 64)])

Common Layout Patterns
----------------------

**SMEM tile for MMA operand (BF16, 128B swizzle):**

.. code:: python

    A_smem = pool.alloc_mma((BLOCK_M, BLOCK_K), "bfloat16")

**SMEM epilogue buffer (no swizzle):**

.. code:: python

    D_smem = pool.alloc(
        (EPI_TILE, MMA_N), "bfloat16",
        align=128  # no swizzle → 128-byte alignment is sufficient
    )

**TMEM layout (usually inferred):**

TMEM layout is determined by the hardware. For ordinary accumulator buffers,
``TMEMPool.alloc`` infers a default layout, and instructions such as
``tcgen05.st.16x256b`` assign physical rows based on warp identity. Specialized
buffers, such as block-scale factors, may still pass an explicit layout helper.

.. warning::

    Swizzled SMEM buffers accessed by MMA descriptors require **1024-byte**
    alignment. Allocating a swizzled buffer without ``align=1024`` causes
    the MMA to read scrambled data — the output norm may look correct but the
    actual values are wrong (cosine similarity near 0).

    Non-swizzled buffers (epilogue, barriers) only need ``align=128``.
    Allocation order matters: place small unswizzled buffers after large
    swizzled ones to avoid bumping alignment.
