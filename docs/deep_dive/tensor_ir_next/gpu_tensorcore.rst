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

.. _tirx-gpu-tensorcore:

5th Generation Tensor Cores (tcgen05)
=====================================
Blackwell introduces the 5th generation tensor core (tcgen05). Rather than
operating on register fragments, tcgen05 uses **TMEM** for accumulator storage
and, depending on the MMA variant, may source operand A from TMEM or SMEM.
Operand B is read from SMEM via **matrix descriptors**.

The async barriers and commit groups described in :ref:`tirx-gpu-async` are the
synchronization mechanism for the tensor core operations described here. For
TIRX's higher-level abstractions, see :ref:`tirx-execution-model` and
:ref:`tirx-operators`.

Architecture Overview
---------------------

.. image:: /_static/img/tirx/tcgen05_architecture.png
   :alt: Blackwell tcgen05 data path with shared memory, tensor memory, and tensor cores
   :width: 80%
   :align: center

Key differences from WGMMA (Hopper):

- **TMEM replaces register accumulators**: Results live in TMEM, not in registers
  distributed across warp lanes.
- **Fully asynchronous**: ``tcgen05.mma`` returns immediately; completion is
  tracked via commit groups and mbarriers.
- **Descriptor-based operands**: Operand B is an SMEM matrix descriptor. Operand A
  is either an SMEM matrix descriptor or a TMEM address, depending on
  ``use_a_tmem``.
- **CTA group support**: Two CTAs can cooperatively issue MMA
  (``cta_group=2``), doubling the effective M dimension.

CTA Groups
~~~~~~~~~~
The ``cta_group`` parameter controls how many CTAs cooperate on a single MMA
operation:

- ``cta_group=1``: A single CTA issues MMA independently. Each CTA has its own
  TMEM allocation, descriptors, and barriers.
- ``cta_group=2``: Two CTAs within the same cluster jointly issue MMA. The M
  dimension is split across the pair — each CTA contributes half the M rows.
  Barriers and commit operations must use ``cta_group=2`` and a ``cta_mask``
  identifying the partner CTA.

When ``cta_group=2``, the instruction descriptor, MMA/copy instruction where
applicable, and ``tcgen05.commit`` must all use the same CTA-group setting.
SMEM matrix descriptors do not carry ``cta_group``; they only describe the SMEM
tile layout. ``tcgen05.commit`` also needs the appropriate ``cta_mask``.

TMEM Allocation and Layout
--------------------------
TMEM is organized as a 2D array of rows and columns:

- **Rows**: Mapped to warp lanes. Each warp in a warpgroup occupies 16
  consecutive physical rows, with gaps between warps:

  .. list-table::
     :header-rows: 1
     :widths: 30 30

     * - Warp
       - Physical Rows
     * - Warp 0
       - 0–15
     * - Warp 1
       - 32–47
     * - Warp 2
       - 64–79
     * - Warp 3
       - 96–111

- **Columns**: Allocated explicitly. The underlying ``tcgen05.alloc`` pool
  column count must be a power of 2 (32, 64, 128, 256, or 512); TIRX sub-buffers
  are carved from that pool.

``tcgen05.alloc`` writes the allocated base address to shared memory, from which
all threads in the warpgroup can read it:

.. code:: text

    tcgen05.alloc [smem_result_addr], n_cols;

For a ``BLOCK_M=64`` MMA output, rows map to M indices sequentially: warp 0
covers M[0:16], warp 1 covers M[16:32], warp 2 covers M[32:48], warp 3 covers
M[48:64].

TIRX presents TMEM tiles through logical layouts. Epilogues typically copy the
logical TMEM region into a local view and then wait for the asynchronous load:

.. code:: python

    Tx.copy_async(C_view[:, :], C_tmem[0:128, 0:N])
    Tx.ptx.tcgen05.wait.ld()

Matrix Descriptors
------------------
Matrix descriptors tell the tensor core how to interpret data in shared memory.
A descriptor encodes:

- **Base address**: Starting SMEM address of the operand tile.
- **Leading dimension offset (LDO)**: Byte stride between matrix rows.
- **Stride byte offset (SBO/``sdo``)**: Byte stride between swizzle-aligned blocks.
- **Swizzle mode**: The SMEM swizzle pattern (none, 32B, 64B, 128B).

In TIRX:

.. code:: python

    Tx.ptx.tcgen05.encode_matrix_descriptor(desc, addr, ldo, sdo, swizzle)

.. note::

    For K-major B operand tiles with 128B swizzle, the SBO must be computed
    using the swizzled block size, not the full BLOCK_K:

    .. code:: text

        BLOCK_SWIZZLED_BK = 128 / sizeof(dtype)   # NOT full BLOCK_K
        sdo = 8 * BLOCK_SWIZZLED_BK * sizeof(dtype) / 16

    Using the full ``BLOCK_K`` instead of ``BLOCK_SWIZZLED_BK`` doubles the
    ``sdo`` and produces incorrect results.

Instruction Descriptors
-----------------------
The instruction descriptor configures the MMA operation itself:

- **Data types**: A, B, and D (accumulator) types. Supported combinations include
  FP16×FP16→FP32, BF16×BF16→FP32, FP8×FP8→FP32, and NvFP4×NvFP4→FP32.
- **Shape**: M×N×K tile dimensions for a single MMA operation.
- **Transpose flags**: Whether A and/or B are transposed.
- **CTA group**: 1 (single CTA) or 2 (pair of CTAs).
- **Negate/saturate flags**: Optional operand negation and result saturation.

.. code:: python

    # Standard instruction descriptor
    Tx.ptx.tcgen05.encode_instr_descriptor(
        desc,
        d_dtype=d_dtype,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        M=M,
        N=N,
        K=K,
        trans_a=trans_a,
        trans_b=trans_b,
        n_cta_groups=cta_group,
        neg_a=neg_a,
        neg_b=neg_b,
        sat_d=sat_d,
    )

    # Block-scaled instruction descriptor (for FP8/NvFP4 with scale factors)
    Tx.ptx.tcgen05.encode_instr_descriptor_block_scaled(
        desc,
        d_dtype=d_dtype,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sfa_dtype=sfa_dtype,
        sfb_dtype=sfb_dtype,
        sfa_tmem_addr=sfa_tmem_addr,
        sfb_tmem_addr=sfb_tmem_addr,
        M=M,
        N=N,
        K=K,
        trans_a=trans_a,
        trans_b=trans_b,
        n_cta_groups=cta_group,
        neg_a=neg_a,
        neg_b=neg_b,
    )

MMA Instruction
---------------
The core compute instruction is exposed as a TIRX wrapper:

.. code:: python

    Tx.ptx.tcgen05.mma(
        d_tmem_addr,
        a_operand,
        b_desc,
        i_desc,
        d_dtype=d_dtype,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        use_a_tmem=use_a_tmem,
        cta_group=CTA_GROUP,
        enable_input_d=accum,
        scale_input_d=0,
    )

Parameters:

- ``d_tmem_addr``: TMEM address for the accumulator/output.
- ``a_operand``: TMEM address or SMEM descriptor for the A operand.
- ``b_desc``: SMEM matrix descriptor for B.
- ``i_desc``: Instruction descriptor (types, shape, config).
- ``enable_input_d``: Whether to accumulate into the existing D buffer. Zero
  means ``D = A * B``; nonzero means ``D = A * B + D``.
- ``scale_input_d``: Optional power-of-two scaling of the input D accumulator.
- Positional ``disable_output_lane`` masks may be passed before the keyword
  arguments when only part of the output tile should be updated.

Variants:

.. list-table::
   :header-rows: 1
   :widths: 30 50

   * - Instruction
     - Description
   * - ``tcgen05.mma``
     - Standard dense MMA
   * - ``tcgen05.mma.sp``
     - Structured sparse MMA (2:4 sparsity)
   * - ``tcgen05.mma`` (block-scaled)
     - MMA with per-block scale factors (FP8/NvFP4)
   * - ``tcgen05.mma.sp`` (block-scaled)
     - Sparse + block-scaled MMA

Block-Scaled MMA
-----------------
For FP8 and NvFP4 data types, block scaling multiplies each block of elements by
a per-block scale factor, improving numerical range without increasing element
precision. Scale factors are stored in TMEM and referenced by the instruction
descriptor.

The pattern in TIRX:

.. code:: python

    # Scale factors loaded into TMEM every 4 K-iterations
    if stage % 4 == 0:
        Tx.ptx.tcgen05.cp(sf_tmem, sf_smem_desc, shape="128x128b")

    # Update instruction descriptor to reference current scale factors
    Tx.cuda.runtime_instr_desc(Tx.address_of(i_desc), stage % 4)

    # Issue block-scaled MMA
    Tx.ptx.tcgen05.mma.block_scale(
        d_tmem, a_tmem, b_desc, sfa_tmem_addr, sfb_tmem_addr, i_desc,
        d_dtype=d_dtype,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sfa_dtype=sfa_dtype,
        sfb_dtype=sfb_dtype,
        use_a_tmem=True,
        cta_group=CTA_GROUP,
        enable_input_d=accum,
    )

Other tcgen05 Operations
------------------------

**tcgen05.cp** — Copy SMEM → TMEM:

Loads scale factors, sparsity metadata, or input operands into TMEM. Supports
optional decompression (e.g., FP8 → FP16).

.. code:: text

    tcgen05.cp.cta_group::1.128x256b [tmem_addr], smem_desc;

**tcgen05.shift** — Shift TMEM allocation:

Moves a TMEM allocation to a different base address, used for double-buffering
TMEM resources.

.. code:: text

    tcgen05.shift.cta_group::1.down [tmem_addr];

**tcgen05.ld / tcgen05.st** — TMEM ↔ Register transfers:

Used in epilogue to read MMA results from TMEM into registers, or to store
register data back to TMEM.

.. code:: text

    // Load TMEM → registers (async, requires wait)
    tcgen05.ld.sync.aligned.16x256b.x1.b32 {regs}, [tmem_addr];
    tcgen05.wait::ld.sync.aligned;

    // Store registers → TMEM (async, requires wait)
    tcgen05.st.sync.aligned.16x256b.x1.b32 [tmem_addr], {regs};
    tcgen05.wait::st.sync.aligned;

.. note::

    ``tcgen05.ld`` and ``tcgen05.st`` are asynchronous. TIRX's current
    ``Tx.copy_async`` paths for TMEM/local transfers leave completion to the
    caller, so issue ``Tx.ptx.tcgen05.wait.ld()`` or
    ``Tx.ptx.tcgen05.wait.st()`` before consuming the destination.

Supported Data Types
--------------------
The table below describes the current high-level ``Tx.gemm_async(...,
dispatch="tcgen05")`` schedule. Raw tcgen05 intrinsics expose additional
hardware combinations, but the dispatcher currently narrows the supported GEMM
forms.

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - A Type
     - B Type
     - D Type
     - Notes
   * - FP16
     - FP16
     - FP32
     - Standard half-precision
   * - BF16
     - BF16
     - FP32
     - Brain floating point
   * - FP8 (E4M3)
     - FP8 (E4M3)
     - FP32
     - Block-scaled GEMM with scale factors
   * - NvFP4 (E2M1)
     - NvFP4 (E2M1)
     - FP32
     - Block-scaled GEMM with scale factors

Generation Comparison
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 25

   * - Feature
     - WGMMA (Hopper/SM90)
     - tcgen05 (Blackwell/SM100)
   * - Accumulator
     - Registers (per-thread)
     - TMEM (per CTA, accessed by warpgroup operations)
   * - A operand
     - Registers or SMEM descriptor
     - TMEM or SMEM descriptor
   * - B operand
     - SMEM descriptor
     - SMEM descriptor
   * - Synchronization
     - ``wgmma.commit_group`` + ``wgmma.wait_group``
     - ``tcgen05.commit`` → mbarrier
   * - Execution
     - Warp-cooperative (all 4 warps)
     - Single issuing thread selected by the schedule (for example a
       single-thread scope, or an elected lane in a warp scope)
   * - Block scaling
     - Not supported
     - FP8, NvFP4 with per-block scale factors
