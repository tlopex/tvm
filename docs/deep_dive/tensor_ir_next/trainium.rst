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

.. _tirx-trainium:

Trainium and NKI Backend
========================
TensorIRX also contains a Trainium backend under
``python/tvm/tirx/operator/tile_primitive/trn``. The CUDA backend is organized
around GPU scopes such as CTA, warpgroup, warp, shared memory, TMEM, and CUDA
intrinsics. The Trainium backend is organized around kernel-scope tile
operations, Trainium memory scopes, and ``Tx.nki`` instruction emission.

Most Trainium tile-primitive dispatch functions require:

- the target kind to be ``trn``;
- the current execution scope to be ``kernel``;
- buffers to use Trainium-compatible scopes and layouts.

Memory Scopes and Layouts
-------------------------
Trainium kernels mainly use global memory plus two explicit device-local scopes:

.. list-table::
   :header-rows: 1
   :widths: 24 40 40

   * - Scope
     - Meaning
     - Typical use
   * - ``global``
     - Kernel argument memory.
     - Inputs and outputs visible outside the kernel.
   * - ``trn.sbuf``
     - Trainium scratch buffer scope.
     - Intermediate tiles used by NKI tensor instructions.
   * - ``trn.psum``
     - Trainium partial-sum scope.
     - Accumulators and reduction or matmul partial sums.

When a buffer is declared in ``trn.sbuf`` or ``trn.psum`` with a string layout,
the TVMScript builder converts it through ``TileLayout.trainium``. For
``trn.psum``, the layout is additionally converted with ``to_psum()``. This is
where Trainium's P/F layout axes become part of the buffer type, rather than
being a convention in indexing code.

The Trainium allocator pass assigns ``allocated_addr`` metadata to constant-size
``trn.sbuf`` buffers that do not already have an address. Dispatch code also
uses explicit ``allocated_addr`` for special private buffers such as identity
tensors or PSUM accumulators.

Tile Primitive Families
-----------------------
The Trainium dispatch package covers these tile-primitive families:

.. list-table::
   :header-rows: 1
   :widths: 28 42 45

   * - Package
     - Registered operators
     - Lowered work
   * - ``trn/copy``
     - ``tirx.copy``
     - Global to ``trn.sbuf`` loads, ``trn.sbuf`` to global stores,
       ``trn.sbuf``/``trn.psum`` copies, transpose helper paths, and
       workspace-backed copies.
   * - ``trn/gemm``
     - ``tirx.gemm``
     - NKI matmul emission with PSUM accumulator workspaces.
   * - ``trn/unary``
     - ``tirx.reciprocal``, ``tirx.sqrt``, ``tirx.exp``, ``tirx.memset``
     - Unary tensor instructions and bias/scale variants when a constant
       workspace is required.
   * - ``trn/binary``
     - ``tirx.add``, ``tirx.sub``, ``tirx.mul``, ``tirx.maximum``,
       ``tirx.minimum``
     - Tensor-tensor and tensor-scalar binary instructions.
   * - ``trn/reduction``
     - ``tirx.sum``, ``tirx.max``, ``tirx.min``
     - NKI tensor reductions, including partial-reduce workspace handling.
   * - ``trn/select``
     - ``tirx.select``
     - Predicate-driven select using the TensorIRX predicate representation.
   * - ``trn/compose_op``
     - ``tirx.binary_chain``, ``tirx.binary_reduce``, ``tirx.compose_op``,
       ``tirx.reduce_negate``, ``tirx.unary_reduce``
     - Fused Trainium compositions such as tensor-scalar-reduce and
       activation-reduce patterns.

Instruction Emission
--------------------
Trainium dispatch functions emit concrete ``Tx.nki`` calls inside an attribute
region:

.. code:: python

    with Tx.attr(0, "tensorized_nki_instruction", 1):
        Tx.nki.matmul(...)

The attribute marks the region as an NKI tensorized instruction. Different
operator families use the corresponding NKI builders, such as ``Tx.nki.matmul``,
``Tx.nki.tensor_copy``, ``Tx.nki.tensortensor``,
``Tx.nki.tensorscalar``, ``Tx.nki.tensorreduce``, and
``Tx.nki.affine_select``.

Private Buffer Allocation
-------------------------
Some Trainium schedules need helper buffers that are not explicit user
arguments. Examples include:

- identity tensors used by transpose-like copy paths;
- PSUM accumulators for matmul;
- partial-reduction buffers;
- constant bias or scale buffers for unary variants.

``TrnPrivateBufferAlloc`` runs before ``LowerTIRx``. It walks
``TilePrimitiveCall`` nodes in an allocation-only dispatch context, asks each
operator for its private buffers, and inserts the required ``AllocBuffer`` and
initialization statements around the kernel body. Schedule implementations then
retrieve these buffers from the tile primitive workspace during normal
lowering.

Trainium Pipeline
-----------------
The Trainium build path is selected with:

.. code:: python

    mod = tvm.tirx.build(ir_module, target="trn", pipeline="trn")

The pipeline in ``python/tvm/tirx/compilation_pipeline.py`` is:

.. list-table::
   :header-rows: 1
   :widths: 30 55

   * - Pass
     - Purpose
   * - ``TrnPrivateBufferAlloc``
     - Materialize Trainium-private workspaces required by tile primitives.
   * - ``TrnNaiveAllocator``
     - Assign ``allocated_addr`` metadata for constant-size Trainium buffers.
   * - ``LowerTIRx``
     - Dispatch tile primitives to Trainium-specific implementations.
   * - ``DecorateDeviceScope``
     - Mark device scope for the S-TIR/TIR lowering path.
   * - ``Simplify`` and ``LowerTIRxOpaque``
     - Simplify expressions and lower remaining opaque TIRX constructs.
   * - ``LoopPartition`` and ``HoistIfThenElse``
     - Canonicalize control flow before host/device splitting.
   * - ``RemoveNoOp``
     - Remove empty statements introduced during lowering.
   * - ``AnnotateEntryFunc``, ``AnnotateDeviceRegions``, ``SplitHostDevice``
     - Prepare host/device functions.
   * - ``MakePackedAPI`` and ``LowerDeviceKernelLaunch``
     - Generate the packed entry point and launch path.

Trainium final device passes currently run ``Simplify``. This is intentionally
smaller than the CUDA finalization path because NKI emission is already encoded
by the Trainium dispatch functions.

When to Read This Page
----------------------
Kernel authors mostly need the memory scopes, layouts, and registered
operators. Backend authors should also read the private-buffer and pipeline
sections, then compare the dispatch functions in ``trn/`` with the backend
extension workflow in :ref:`tirx-new-backend`.
