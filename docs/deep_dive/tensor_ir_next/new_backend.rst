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

.. _tirx-new-backend:

Adding a New Backend
====================
This guide walks through the steps to add support for a new accelerator or GPU
architecture in TIRX. Adding a new dispatch variant for an existing target can
often be done in Python. A completely new target kind (beyond the existing
``cuda`` and ``trn``) also needs TVM target registration, intrinsic exposure,
and a codegen backend.

The intended reader is a backend author or TensorIRX maintainer. Kernel authors
using an existing CUDA or Trainium backend can treat this page as background
material and start with the abstraction, operator, and GPU execution guides.

Overview
--------
Adding a new backend usually starts with **dispatch functions** that convert
abstract ``TilePrimitiveCall`` nodes into target-specific IR. If the target
already exists in TVM, this may be enough. If the target kind is new, the
dispatch layer is only one piece; the backend also needs target registration,
pipeline selection, target builtins, and final code generation. The existing C++
layers (IR nodes, FFI, lowering passes) are shared across all backends.

.. code:: text

    What you reuse (shared):
    ├── IR node definitions (TilePrimitiveCall, ExecScopeStmt, etc.)
    ├── TVMScript DSL (@Tx.prim_func, Tx.copy_async, Tx.gemm_async)
    ├── Lowering pipeline (TilePrimitiveDispatch → Cleanup → StripExecScope)
    └── Printer / debugging tools

    What you write (per-backend):
    ├── Dispatch implementations for each operator
    ├── Target-specific intrinsics / builtins
    └── Codegen / target.build integration for a new target kind

The CUDA and Trainium backends serve as reference implementations. CUDA is the
most complete; Trainium shows a simpler backend structure.

Step 1: Create the Dispatch Module
-----------------------------------
Create a new directory under the dispatch registry:

.. code:: text

    python/tvm/tirx/operator/tile_primitive/
    ├── cuda/          # existing CUDA backend
    ├── trn/           # existing Trainium backend
    └── my_gpu/        # your new backend
        ├── __init__.py
        ├── common.py
        ├── copy/
        │   ├── __init__.py
        │   ├── default.py
        │   └── utils.py
        ├── unary/
        │   ├── __init__.py
        │   └── default.py
        ├── binary/
        │   ├── __init__.py
        │   └── default.py
        └── gemm/
            ├── __init__.py
            └── default.py

Each tile-primitive category can get its own sub-package. Within each, implement
one or more dispatch variants.

Step 2: Register Dispatch Variants
-----------------------------------
Use ``@register_dispatch`` to register implementations. Start with low-priority
default variants, then add optimized ones:

.. code:: python

    # my_gpu/copy/default.py
    from tvm.tirx import PrimFunc
    from tvm.tirx.operator.tile_primitive import (
        DispatchContext,
        predicate,
        register_dispatch,
    )
    from tvm.tirx.stmt import TilePrimitiveCall
    from tvm.script import tirx as Tx


    def _is_valid_copy(op_call: TilePrimitiveCall, sctx: DispatchContext):
        """Check that this is a well-formed copy operation."""
        op_call = TilePrimitiveCall.downcast(op_call)
        dst_region, src_region = op_call.dst, op_call.src
        ok = dst_region is not None and src_region is not None
        return ok, "copy requires dst and src regions"


    @register_dispatch(
        "copy",
        "my_gpu",               # target_kind must match your TVM target
        variant="default",
        priority=0,
        when=[
            predicate("validate_copy", _is_valid_copy),
        ],
    )
    def copy_default(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
        """Scalar element-by-element copy fallback."""
        # Delegate to a shared utility that builds the PrimFunc
        return copy_default_impl(op_call, sctx)

The implementation function (for example, ``copy_default_impl`` in a local
``utils.py``) builds a ``PrimFunc`` using
``@Tx.prim_func(check_well_formed=False)``. The inner function calls helper
routines that use ``Tx.grid``, ``Tx.buffer_store``, etc. to construct IR. See
existing CUDA copy schedules for complete working references.

**Key parameters:**

- **op_name**: Must match the TIRX operator name (``"copy"``, ``"copy_async"``,
  ``"gemm_async"``, ``"add"``, ``"sum"``, etc.).
- **target_kind**: String that matches your ``tvm.target.Target.kind.name``.
- **variant**: Human-readable name. Users can force it via ``dispatch="default"``.
- **priority**: Higher priority variants are tried first. Use 0 for fallbacks,
  10+ for optimized paths.
- **when**: List of predicates. All must pass for this variant to be selected.

Step 3: Write Predicates
--------------------------
Predicates control when a variant is applicable. Write predicates that check
hardware capabilities, data types, memory scopes, and layout compatibility:

.. code:: python

    from tvm.tirx.operator.tile_primitive import predicate
    from tvm.tirx.stmt import TilePrimitiveCall

    # Check execution scope
    def exec_scope_ok(op_call, sctx, expected_scopes=None):
        scope_name = sctx.scope_kind
        if expected_scopes and scope_name not in expected_scopes:
            return False, f"exec scope is '{scope_name}', need {expected_scopes}"
        return True

    # Check data types
    def dtype_supported(op_call, sctx, supported_dtypes=None):
        op_call = TilePrimitiveCall.downcast(op_call)
        dtype = op_call.src.buffer.dtype
        if supported_dtypes and dtype not in supported_dtypes:
            return False, f"dtype '{dtype}' not in {supported_dtypes}"
        return True

    # Check memory scopes
    def is_smem_to_local(op_call, sctx):
        op_call = TilePrimitiveCall.downcast(op_call)
        src_scope = op_call.src.buffer.scope()
        dst_scope = op_call.dst.buffer.scope()
        ok = src_scope.startswith("shared") and dst_scope == "local"
        reason = None if ok else f"expected shared->local, got {src_scope}->{dst_scope}"
        return ok, reason

Use ``sctx.scope_kind`` or helpers such as ``sctx.is_cta`` and
``sctx.is_thread`` when checking where an operator appears. Predicates that
return ``(False, reason_string)`` provide diagnostic messages when dispatch
fails. Always include a reason — it makes debugging much easier.

Step 4: Implement Operator Schedules
--------------------------------------
Each dispatch function receives a ``TilePrimitiveCall`` and a ``DispatchContext``,
and returns a ``PrimFunc``. The returned function contains concrete IR — loops,
buffer accesses, intrinsic calls — with no remaining ``TilePrimitiveCall`` nodes.

**DispatchContext provides:**

.. code:: python

    sctx.target          # tvm.target.Target object
    sctx.exec_scope      # current ExecScope object
    sctx.scope_kind      # canonical scope string: kernel, cluster, cta, ...
    sctx.launch_params   # kernel launch parameters
    sctx.var_range_map   # loop variable ranges (for bounds analysis)
    sctx.shared_state    # persistent dict across dispatch calls in same kernel

**Methods for resource management:**

.. code:: python

    # Request a buffer allocation (hoisted to function scope)
    sctx.add_alloc_buffer(buffer)

    # Add a device-side initialization statement (runs before main loop)
    sctx.add_init_stmt(stmt, host=False)

    # Add a host-side initialization statement (runs before kernel launch)
    sctx.add_init_stmt(stmt, host=True)

    # Add a statement after a buffer's definition
    sctx.add_post_buffer_def_stmt(buffer, stmt)

    # Cross-dispatch caching; persists across tile primitive dispatches
    # in one kernel.
    sctx.cache_set(key, value)
    cached = sctx.cache_get(key)

**Schematic example: optimized copy with vectorization:**

The helper predicates and index functions below are placeholders for backend
code that derives loop extents and coordinates from the buffer regions.

.. code:: python

    @register_dispatch(
        "copy", "my_gpu",
        variant="vec_load",
        priority=5,
        when=[
            predicate("validate", _is_valid_copy),
            predicate("vectorizable", _can_vectorize),
        ],
    )
    def copy_vec_load(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
        op_call = TilePrimitiveCall.downcast(op_call)
        dst_region, src_region = op_call.dst, op_call.src
        dst, src = dst_region.buffer, src_region.buffer

        def do_copy(dst, src):
            with Tx.grid(*extents) as lvs:
                Tx.buffer_store(dst, src[src_coords(lvs)], dst_coords(lvs))

        @Tx.prim_func(check_well_formed=False)
        def impl():
            do_copy(dst, src)

        return impl

The standard pattern is:

1. ``TilePrimitiveCall.downcast(op_call)`` to access ``.dst``, ``.src`` regions.
2. Define a helper function that uses ``Tx.grid``, ``Tx.buffer_store``, etc.
3. Wrap it in ``@Tx.prim_func(check_well_formed=False)``.

Step 5: Register the Module
-----------------------------
Ensure your dispatch module is imported during TIRX initialization. Add it to
the dispatch package's ``__init__.py``:

The real ``__init__.py`` uses wildcard imports to trigger ``@register_dispatch``
decorators at import time:

.. code:: text

    # python/tvm/tirx/operator/tile_primitive/__init__.py  (schematic)
    from .dispatcher import fail, predicate, register_dispatch, ...
    from .registry import DispatchContext
    from .cuda.copy import *
    from .cuda.copy_async import *
    from .cuda.gemm_async import *
    from .trn import *
    from .my_gpu import *    # add your new backend here

Your backend package should also import its own submodules so their decorators
execute:

.. code:: text

    # python/tvm/tirx/operator/tile_primitive/my_gpu/__init__.py
    from .copy import *
    from .unary import *
    from .binary import *
    from .gemm import *

Step 6: Define Target Builtins (Optional)
------------------------------------------
If your GPU has special instructions (like NVIDIA's TMA or Blackwell's tcgen05),
define them as target builtins following the existing pattern:

**C++ side** — declare in ``include/tvm/tirx/target_builtin/``:

.. code:: cpp

    // include/tvm/tirx/target_builtin/my_gpu.h
    #ifndef TVM_TIRX_TARGET_BUILTIN_MY_GPU_H_
    #define TVM_TIRX_TARGET_BUILTIN_MY_GPU_H_

    #include <tvm/tirx/op.h>

    namespace tvm {
    namespace tirx {
    namespace builtin {

    TVM_DLL const Op& my_gpu_special_load();

    }  // namespace builtin
    }  // namespace tirx
    }  // namespace tvm

    #endif  // TVM_TIRX_TARGET_BUILTIN_MY_GPU_H_

Register in ``src/tirx/op/target_builtin/``:

.. code:: cpp

    // src/tirx/op/target_builtin/my_gpu.cc
    #include <tvm/tirx/target_builtin/my_gpu.h>
    #include <tvm/tirx/op.h>

    namespace tvm {
    namespace tirx {
    namespace builtin {

    const Op& my_gpu_special_load() {
      static const Op& op = Op::Get("tirx.my_gpu_special_load");
      return op;
    }

    TVM_TIRX_REGISTER_OP("my_gpu_special_load")
        .set_num_inputs(2);

    }  // namespace builtin
    }  // namespace tirx
    }  // namespace tvm

**Python side** — expose through ``tvm.tirx.op`` and use in dispatch
implementations via ``Tx.*`` namespace, similar to how existing CUDA builtins
are accessed as ``Tx.ptx.*``.

For a new target builtin to be usable from TVMScript, wire each layer:

- Add a wrapper in ``python/tvm/tirx/op.py`` that emits ``call_intrin`` for the
  registered op name.
- Add a namespace entry in ``python/tvm/tirx/script/builder/ir.py`` so users can
  call it as ``Tx.my_gpu.special_load(...)`` or another target-specific namespace.
- Teach the printer about the namespace form and add parser/printer tests.
- Add codegen handling: CUDA-style targets usually use ``register_codegen`` in
  ``python/tvm/tirx/operator/intrinsics/cuda/`` (or an analogous backend-specific
  registry); other targets may handle the op directly in their C++ ``CodeGen``
  class.

Step 7: Testing
---------------
Write tests that verify dispatch selection and numerical correctness:

.. code:: python

    # tests/python/tirx/codegen/test_codegen_my_gpu.py
    import tvm
    from tvm.script import tirx as Tx

    def test_copy_dispatch():
        @Tx.prim_func
        def kernel(
            A: Tx.Buffer((64, 64), "float16"),
            B: Tx.Buffer((64, 64), "float16"),
        ):
            with Tx.kernel():
                cta_m, cta_n = Tx.cta_id([64, 64])
                with Tx.cta():
                    Tx.copy(B, A)

        mod = tvm.IRModule({"main": kernel})
        target = tvm.target.Target("my_gpu")

        # Lowering-only test (requires target-kind registration and dispatch
        # imports, but not target.build.my_gpu):
        with target, tvm.transform.PassContext():
            lowered = tvm.tirx.transform.LowerTIRx()(mod)
        assert "tirx.copy" not in lowered.script()

        # Full build additionally requires target.build.my_gpu:
        # built = tvm.tirx.build(mod, target=target, pipeline="tirx")

``tvm.tirx.build`` eventually calls ``target.build.<kind>``. For a new target
kind, register that global function, for example
``refl::GlobalDef().def("target.build.my_gpu", BuildMyGpu)``. If the backend
needs Trainium-like prepasses, add a custom pipeline entry instead of using
``pipeline="tirx"`` unchanged.

Minimum Viable Backend
----------------------
To get a new backend working end-to-end, implement these operators in priority
order:

1. **copy** (default variant) — element-by-element copy between memory scopes.
2. **unary ops** (zero, cast, fill) — basic initialization and type conversion.
3. **binary ops** (add, mul) — element-wise arithmetic.
4. **reduction** (sum, max) — reductions along axes.
5. **gemm / gemm_async** — matrix multiplication (the performance-critical path).

Each can start as a simple scalar loop (priority=0), then be incrementally
optimized with vectorized, hardware-specific variants at higher priorities.

Reference: Existing Backends
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 40

   * - Backend
     - Path
     - Notes
   * - CUDA
     - ``tile_primitive/cuda/``
     - Full-featured: TMA, tcgen05, multicast, warp specialization.
       ~40 files. Reference for high-performance dispatch implementations.
   * - Trainium
     - ``tile_primitive/trn/``
     - Simpler structure: 20+ files. Good starting point for understanding
       the minimum required dispatch coverage. Includes custom instruction
       generator and workspace utilities.
