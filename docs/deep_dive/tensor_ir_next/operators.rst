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

.. _tirx-operators:

Operators, Scheduling, and Compilation
======================================
TIRX kernels are built from **tile primitives** — high-level building blocks
like ``gemm_async``, ``copy_async``, and ``add``. Each primitive call in the DSL
becomes a ``TilePrimitiveCall`` node in the IR. During compilation, the
**tile-primitive dispatch** system selects a hardware-specific implementation
for each ``TilePrimitiveCall`` based on data types, memory scopes, execution
scope, layout, and target architecture.

Operator Hierarchy
------------------
All TIRX tile primitives inherit from ``TilePrimitiveCall`` and are organized
into a type hierarchy:

.. code:: text

    TilePrimitiveCall
    ├── UnaryOp (output, input)
    │   ├── Zero         — zero out all elements
    │   ├── Cast         — type conversion
    │   ├── Reciprocal   — 1/x
    │   ├── Fill         — fill with scalar value
    │   ├── Memset       — set all elements to value
    │   ├── SiLU         — x * sigmoid(x)
    │   └── UnaryOpWithBiasScale (output, input, bias, scale)
    │       ├── Sqrt     — square root with optional bias/scale
    │       ├── Exp      — exponential with optional bias/scale
    │       └── Exp2     — base-2 exponential with optional bias/scale
    │
    ├── BinaryOp (output, lhs, rhs)
    │   ├── Add          — element-wise addition
    │   ├── Sub          — element-wise subtraction
    │   ├── Mul          — element-wise multiplication
    │   ├── FDiv         — floating-point division
    │   ├── Maximum      — element-wise max
    │   ├── Minimum      — element-wise min
    │   └── Select       — conditional selection with predicate
    │
    ├── ReduceOp (output, input, reduce_axes, accum)
    │   ├── Sum          — sum reduction
    │   ├── Max          — max reduction
    │   ├── Min          — min reduction
    │   └── ReduceNegate — reduction followed by negation
    │
    ├── FMA              — fused multiply-add: dst = src * scale + bias
    ├── Copy             — synchronous copy between memory scopes
    ├── CopyAsync        — asynchronous copy (variants: tma, non-bulk-copy,
    │                      dsmem, smem->tmem, tmem<->local)
    ├── Gemm             — synchronous matrix multiply: D = αAB + βC
    ├── GemmAsync        — asynchronous MMA (tcgen05 on Blackwell)
    ├── PermuteDims      — tensor dimension permutation
    │
    ├── BinaryReduce     — fused binary + reduction
    ├── UnaryReduce      — fused unary + reduction
    ├── BinaryChain      — chain of two binary operations
    └── GenericOp        — dynamically-resolved custom operators

DSL Interface
-------------
In the TIRX DSL, operators are invoked as functions on the ``Tx`` namespace.
Many element-wise and data-movement operators follow a common signature pattern:

.. code:: python

    Tx.op_name(dst, src, ..., workspace=None, dispatch=None, **config)

Parameters:

- **dst** / **src**: ``Buffer`` or ``BufferRegion`` specifying the operand tiles.
  Buffer regions support slicing: ``A[m:m+64, k:k+128]``.
- **workspace**: Optional dict of named temporary buffers.
- **dispatch**: Optional string to force a specific schedule variant.
- **config** (``**kwargs``): Additional keyword arguments passed to the schedule.

Core Operators
--------------

**Data Movement:**

.. code:: python

    # Synchronous copy (layout-aware, scope-aware)
    Tx.copy(dst_smem, src_global[row:row+M, col:col+K])

    # Asynchronous copy (dispatch selects variant: tma, dsmem, etc.)
    Tx.copy_async(dst_smem, src_global[row:row+M, col:col+K])

    # Fill buffer with scalar
    Tx.fill(dst, 0.0)

**Matrix Operations:**

.. code:: python

    # Async GEMM (tcgen05 on Blackwell)
    Tx.gemm_async(C, A, B, accum=True)

    # Block-scaled async GEMM (FP8/NvFP4)
    Tx.gemm_async(C, A, B, SFA=scale_a, SFB=scale_b, accum=True)

    # Synchronous GEMM: D = alpha * A @ B + beta * C
    Tx.gemm(D, A, B, C, alpha=1.0, beta=0.0)

**Element-wise Operations:**

.. code:: python

    Tx.add(dst, src1, src2)
    Tx.sub(dst, src1, src2)
    Tx.mul(dst, src1, 0.5)        # scalar second operand OK

    Tx.sqrt(dst, src)
    Tx.exp2(dst, src)
    Tx.silu(dst, src)             # x * sigmoid(x)
    Tx.reciprocal(dst, src)
    Tx.cast(dst_f16, src_f32)     # type conversion
    Tx.fma(dst, src, scale, bias) # dst = src * scale + bias

    # Trainium-only today
    Tx.maximum(dst, src1, src2)

**Reductions:**

.. code:: python

    Tx.sum(dst, src, axes=-1)         # reduce last axis
    Tx.max(dst, src, axes=-1, accum=True)  # accumulate into dst

**Composite Operations:**

.. code:: python

    # Binary + Reduce: dst_bin = src1 * src2, dst_red = sum(dst_bin)
    Tx.binary_reduce(dst_bin, dst_red, src1, src2, "mul", "sum")

    # Unary + Reduce: dst_un = exp2(src), dst_red = sum(dst_un)
    Tx.unary_reduce(dst_un, dst_red, src, "exp2", "sum")

    # Binary chain: dst = (op0_src op0 data) op1 op1_src
    Tx.binary_chain(dst, data, op0_src, op1_src, "mul", "add")

Operator Scheduling
-------------------
The scheduling system maps abstract operators to concrete hardware
implementations via **predicate-based dispatch**.

Registration
~~~~~~~~~~~~
Schedule implementations are registered with the ``@register_dispatch``
decorator:

.. code:: python

    from tvm.tirx import PrimFunc
    from tvm.tirx.operator.tile_primitive.cuda.common import validate_copy_op
    from tvm.tirx.operator.tile_primitive.cuda.exec_scope_utils import single_thread
    from tvm.tirx.operator.tile_primitive import (
        DispatchContext,
        fail,
        predicate,
        register_dispatch,
    )
    from tvm.tirx.stmt import TilePrimitiveCall


    def _is_valid_copy(op, sctx):
        ok = validate_copy_op(op, sctx)
        return ok, None if ok else "not a valid copy op"


    def _is_single_thread(op, sctx):
        ok = single_thread(op, sctx)
        return ok, None if ok else f"unsupported exec_scope {sctx.scope_kind}"


    # From cuda/copy_async/tma.py
    @register_dispatch(
        "copy_async",
        "cuda",
        variant="tma",
        priority=10,
        when=[
            predicate("validate_copy_op", _is_valid_copy),
            predicate("single_thread", _is_single_thread),
        ],
    )
    def copy_async_dispatch_tma(
        op: TilePrimitiveCall,
        sctx: DispatchContext,
    ) -> PrimFunc:
        return copy_tma_impl(op, sctx)

Each registration specifies:

- **op_name**: The operator short name (e.g., ``"copy_async"``). The
  ``"tirx."`` prefix is added internally by ``get_tirx_op``.
- **target_kind**: The hardware target (``"cuda"`` or ``"trn"``).
- **variant**: A human-readable name for this implementation.
- **priority**: Higher priority variants are tried first (default: 0).
- **when**: A list of ``Predicate`` objects that must all pass.

Dispatch Predicates
~~~~~~~~~~~~~~~~~~~
Dispatch predicates are named conditions that determine whether a schedule
variant is applicable. They are separate from ``tvm.tirx.predicate.Predicate``,
which is used by expression-level operators such as ``Tx.select``.

.. code:: python

    from tvm.tirx.operator.tile_primitive.dispatcher import Predicate, fail, predicate
    from tvm.tirx.stmt import TilePrimitiveCall

    # Simple boolean predicate (downcast to access .dst, .src buffer regions)
    def is_fp16(op_call, sctx):
        del sctx
        op_call = TilePrimitiveCall.downcast(op_call)
        return op_call.src.buffer.dtype == "float16"

    pred = predicate("is_fp16", is_fp16)

    # Predicate with reason on failure
    def check_alignment(op_call, sctx):
        op_call = TilePrimitiveCall.downcast(op_call)
        last_extent = op_call.src.region[-1].extent
        if last_extent % 128 != 0:
            return False, "K dimension must be multiple of 128"
        return True
    pred = predicate("k_aligned", check_alignment)

    # Predicate that raises DispatchFail for detailed diagnostics
    def check_smem(op_call, sctx):
        op_call = TilePrimitiveCall.downcast(op_call)
        if not op_call.dst.buffer.scope().startswith("shared"):
            fail("destination must be in shared memory")
        return True
    pred = predicate("smem_dst", check_smem)

Dispatch Process
~~~~~~~~~~~~~~~~
When ``LowerTIRx`` encounters a ``TilePrimitiveCall``, it calls
``run_dispatch``:

1. Look up all registered variants for ``(op_call.op, target.kind.name)`` in
   ``_DISPATCH_TABLE``.
2. Sort by ``(-priority, variant_name)``.
3. If ``op_call.dispatch`` is set, filter to that variant only.
4. For each variant, evaluate all predicates:
   - If all pass → call the implementation → return the ``PrimFunc``.
   - If any fail → record the failure reason and try the next variant.
5. If no variant matches, raise ``RuntimeError`` with a diagnostic table
   showing each variant and why it was rejected.

DispatchContext
~~~~~~~~~~~~~~~
The ``DispatchContext`` provides dispatch implementations with information about
the compilation environment:

.. code:: text

    class DispatchContext:
        target: Target              # hardware target (cuda, trn)
        exec_scope: ExecScope       # current execution scope
        launch_params: dict         # kernel launch parameters
        var_range_map: dict         # loop variable ranges
        alloc_only: bool            # True during buffer-allocation-only pass
        callbacks: dict             # internal callback storage
        shared_state: dict          # state persisting across dispatch calls
        inter: dict[str, list]      # inter-scope layout bindings
        intra: dict[str, list]      # intra-scope layout bindings
        scope_kind: str             # kernel, cluster, cta, warpgroup, warp, thread

        def is_cuda(self) -> bool
        def is_trn(self) -> bool
        def add_alloc_buffer(self, buffer)              # request buffer allocation
        def add_init_stmt(self, stmt, host=False)       # add init statement
        def add_post_buffer_def_stmt(self, buffer, stmt)  # stmt after buffer def
        def cache_get(self, key) -> Object | None       # cross-dispatch cache lookup
        def cache_set(self, key, value)                 # cross-dispatch cache store
        @property
        def is_thread(self) -> bool
        @property
        def is_warp(self) -> bool
        @property
        def is_warpgroup(self) -> bool
        @property
        def is_cta(self) -> bool
        @property
        def is_cluster(self) -> bool

Available Schedules
~~~~~~~~~~~~~~~~~~~
The current tile-primitive registry contains the following schedule variants.
The left column is the public TIRX operator name; the CUDA and Trainium columns
show the variants registered for each target kind.

.. list-table::
   :header-rows: 1
   :widths: 25 38 37

   * - Operator
     - CUDA variants
     - Trainium variants
   * - ``tirx.copy``
     - ``local_view``, ``vec_load``, ``default``
     - ``default``
   * - ``tirx.copy_async``
     - ``non-bulk-copy``, ``dsmem``, ``smem->tmem``, ``tma``,
       ``tmem<->local``
     - none
   * - ``tirx.gemm_async``
     - ``tcgen05``
     - none
   * - ``tirx.gemm``
     - none
     - ``default``
   * - ``tirx.zero``
     - ``per_thread``, ``shared_distributed``, ``tile_local``
     - none
   * - ``tirx.fill``
     - ``per_thread``, ``shared_distributed``, ``tile_local``
     - none
   * - ``tirx.cast``
     - ``per_thread``, ``shared_distributed``, ``tile_local``
     - none
   * - ``tirx.reciprocal``
     - ``per_thread``, ``shared_distributed``, ``tile_local``
     - ``unary``
   * - ``tirx.sqrt``
     - ``per_thread``, ``shared_distributed``, ``tile_local``
     - ``unary_with_bias_scale``
   * - ``tirx.exp``
     - ``per_thread``, ``shared_distributed``, ``tile_local``
     - ``unary_with_bias_scale``
   * - ``tirx.exp2``
     - ``per_thread``, ``shared_distributed``, ``tile_local``
     - none
   * - ``tirx.silu``
     - ``per_thread``, ``shared_distributed``, ``tile_local``
     - none
   * - ``tirx.add``
     - ``per_thread``, ``shared_distributed``, ``tile_local``
     - ``binary``
   * - ``tirx.sub``
     - ``per_thread``, ``shared_distributed``, ``tile_local``
     - ``binary``
   * - ``tirx.mul``
     - ``per_thread``, ``shared_distributed``, ``tile_local``
     - ``binary``
   * - ``tirx.fdiv``
     - ``per_thread``, ``shared_distributed``, ``tile_local``
     - none
   * - ``tirx.fma``
     - ``per_thread``, ``shared_distributed``, ``tile_local``
     - none
   * - ``tirx.maximum``
     - none
     - ``binary``
   * - ``tirx.minimum``
     - none
     - ``binary``
   * - ``tirx.sum``
     - ``packed_add_sum``, ``local``, ``shared``
     - ``reduction``
   * - ``tirx.max``
     - ``3input_maxmin``, ``local``, ``shared``
     - ``reduction``
   * - ``tirx.min``
     - ``3input_maxmin``, ``local``, ``shared``
     - ``reduction``
   * - ``tirx.permute_dims``
     - ``vectorized_permute_dims_last_2d``
     - none
   * - ``tirx.memset``
     - none
     - ``unary``
   * - ``tirx.select``
     - none
     - ``default``
   * - ``tirx.binary_chain``
     - none
     - ``default``
   * - ``tirx.binary_reduce``
     - none
     - ``default``
   * - ``tirx.compose_op``
     - none
     - ``default``
   * - ``tirx.reduce_negate``
     - none
     - ``default``
   * - ``tirx.unary_reduce``
     - none
     - ``default``

Compilation Pipeline
--------------------
The main TIRX module pipeline (``tirx_pipeline()`` in
``python/tvm/tirx/compilation_pipeline.py``) processes a kernel through these
stages before host/device finalization:

.. code:: text

    @Tx.prim_func input
    │
    ├─ LowerTIRx             — TilePrimitiveDispatch + cleanup + ExecScope stripping
    ├─ UnifyThreadBinding    — normalize thread index expressions
    ├─ Simplify              — arithmetic simplification
    ├─ LowerTIRxOpaque       — handle remaining opaque blocks
    ├─ FlattenBuffer         — flatten multi-dimensional buffer access
    ├─ BF16ComputeLegalize   — legalize BF16 compute operations
    ├─ NarrowDataType        — narrow to 32-bit where possible
    ├─ VectorizeLoop         — vectorize inner loops
    ├─ UnrollLoop            — unroll loops
    ├─ Simplify              — arithmetic simplification (second pass)
    ├─ CommonSubexprElim     — eliminate common subexpressions
    │                          (unless tir.disable_cse_tir)
    ├─ FP8ComputeLegalize    — legalize FP8 compute
    ├─ VerifyMemory          — verify memory access patterns
    ├─ AnnotateEntryFunc     — mark kernel entry points
    ├─ AnnotateDeviceRegions — mark device code regions
    ├─ SplitHostDevice       — separate host and device code
    ├─ MakePackedAPI         — generate packed function interface
    ├─ FP8StorageLegalize    — legalize FP8 storage
    ├─ BF16StorageLegalize   — legalize BF16 storage
    └─ LowerDeviceKernelLaunch — generate kernel launch code

The pipeline is selected via:

.. code:: python

    import tvm
    from tvm import tirx

    # Build with TIRX pipeline
    mod = tirx.build(ir_module, target="cuda", pipeline="tirx")

Trainium Pipeline
-----------------
For AWS Trainium targets, a separate pipeline (``trn_pipeline()``) is available
with Trainium-specific passes:

.. code:: python

    mod = tirx.build(ir_module, target="trn", pipeline="trn")

It starts with Trainium-private buffer allocation and a naive allocator before
running ``LowerTIRx``. It then runs ``DecorateDeviceScope``, simplification,
``LowerTIRxOpaque``, loop partitioning, if-hoisting, no-op removal, host/device
splitting, packed-API generation, and Trainium-specific device finalization.

Introspection
-------------
To see all registered schedule variants:

.. code:: python

    from tvm.tirx.operator.tile_primitive import list_registered_schedules
    schedules = list_registered_schedules()
    # Returns (sorted by -priority, variant):
    # {"tirx.copy_async": {"cuda": ["non-bulk-copy", "dsmem", ...]}, ...}
