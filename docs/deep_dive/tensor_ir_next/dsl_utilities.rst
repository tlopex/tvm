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

.. _tirx-dsl-utilities:

DSL Utilities and Verification
==============================
The main TensorIRX DSL surface is ``@Tx.prim_func`` plus tile primitives,
execution scopes, layouts, and pipeline helpers. The implementation also
provides utilities that make larger kernels practical: inline macros,
metadata-backed helper classes, scalar state, hints, external kernel calls, and
verification helpers.

Inline Macros
-------------
``@Tx.inline`` defines a Python helper that expands into TIR at parse time. It
follows Python lexical scoping: an inline function can capture variables from
its definition scope, and an inline defined inside a ``@Tx.prim_func`` can see
the surrounding parsed variables at the point where it is defined.

.. code:: python

    from tvm.script import tirx as Tx

    @Tx.inline
    def store_two(A, i, value):
        A[i] = value
        A[i + 1] = value

    @Tx.prim_func(private=True)
    def kernel(A: Tx.Buffer((128,), "int32")):
        store_two(A, 0, Tx.int32(7))

Inline helpers are useful for repeated instruction sequences, small scheduler
steps, and pipeline endpoint actions. They are not runtime calls: after parsing,
the resulting function contains only the expanded TIR.

Meta Classes and Scalar State
-----------------------------
``@Tx.meta_class`` marks Python helper objects as parser metadata values.
Instances can own TensorIRX scalar state or buffers in ``__init__`` and expose
``@Tx.inline`` methods that emit TIR. The built-in tile schedulers, pipeline
objects, and allocation pools use this pattern.

.. code:: python

    @Tx.meta_class
    class Counter:
        def __init__(self):
            self.value = Tx.local_scalar("int32")

        @Tx.inline
        def init(self, start):
            self.value = start

        @Tx.inline
        def advance(self, step):
            self.value = self.value + step

Scalar helper APIs are:

.. list-table::
   :header-rows: 1
   :widths: 28 45

   * - Helper
     - Meaning
   * - ``Tx.alloc_scalar(dtype, scope)``
     - Allocate a one-element buffer and return its scalar load/store view.
   * - ``Tx.local_scalar(dtype)``
     - Allocate scalar state in local memory.
   * - ``Tx.shared_scalar(dtype)``
     - Allocate scalar state in shared memory.
   * - ``Tx.decl_scalar(dtype, data, ...)``
     - Declare a scalar view from an existing pointer.

Outside meta-class construction these helpers return a wrapper so regular
Python assignment syntax updates the underlying scalar. During meta-class
construction the raw scalar view is stored directly on the helper object.

Hints
-----
``Tx.hint`` attaches non-semantic guidance to the IR as an ``AttrStmt`` with
``attr_key="tirx_hint"``. It can be a standalone statement, a scoped context
manager, or a keyword stored on a tile primitive.

.. code:: python

    Tx.hint("persistent scheduler", mode="l2_swizzle")

    with Tx.hint("pipeline stage", depth="4"):
        Tx.copy_async(dst, src)

    Tx.add(B, A, Tx.float32(1), hint="use_fast_math")

Hints round-trip through the TVMScript printer/parser. They are intended for
compiler experiments, tests, and scheduling annotations. Dispatch code should
not rely on a free-form hint unless the expected key/value contract is also
documented by the schedule using it.

External Kernel Calls
---------------------
``Tx.call_kernel`` lets a TIRX function attach and launch an external CUDA
kernel. It supports two sources:

- a CUDA source string or path, compiled with NVCC through ``SourceKernel``;
- a Triton JIT function, compiled through ``TritonKernel``.

For CUDA source kernels, launch arguments are a pair of grid and block
dimensions. The compiled PTX or cubin is attached to the current IRModule as an
external runtime module, and the body emits a ``call_packed`` to the kernel.

.. code:: python

    cuda_src = r"""
    extern "C" __global__ void set_one(float* A, int n) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) A[i] = 1.0f;
    }
    """

    N = 1024

    @Tx.prim_func
    def kernel(A: Tx.Buffer((N,), "float32")):
        Tx.call_kernel(
            cuda_src,
            [[(N + 127) // 128], [128]],
            A.data,
            N,
            kernel_name="set_one",
        )

If the CUDA source includes ``nvshmem.h`` or ``nvshmemx.h``, the external
kernel path emits a cubin instead of PTX so the NVSHMEM device library can be
linked correctly. Triton kernels use Triton's compiled metadata for the number
of warps and dynamic shared memory launch parameter.

Analysis and Verification
-------------------------
``tvm.tirx.analysis`` wraps common validation utilities:

.. list-table::
   :header-rows: 1
   :widths: 34 50

   * - Utility
     - Purpose
   * - ``expr_deep_equal(lhs, rhs)``
     - Compare expressions without variable remapping. Prefer
       ``tvm.ir.structural_equal`` when remapping is desired.
   * - ``verify_ssa(func)``
     - Check SSA form.
   * - ``verify_memory(func)``
     - Check for illegal host-side direct memory access.
   * - ``undefined_vars(node, defs=None)``
     - Return variables used without definition.
   * - ``verify_well_formed(obj, assert_mode=True)``
     - Run general well-formedness checks on a TIRX function or module.
   * - ``verify_tirx_well_formed(obj, assert_mode=True, device_func=False)``
     - Run TIRX-specific checks, including scope/layout consistency.

Typical usage:

.. code:: python

    from tvm.tirx.analysis import verify_tirx_well_formed

    verify_tirx_well_formed(func)

The parser runs well-formedness checks by default for ``@Tx.prim_func``. Backend
dispatch implementations often use ``@Tx.prim_func(check_well_formed=False)``
for generated helper functions and rely on the surrounding tests and lowering
passes to validate the final module.

Where These Utilities Fit
-------------------------
Use inline macros and meta classes for reusable kernel structure. Use hints for
non-semantic annotations and experiments. Use external kernels when reusing
existing CUDA or Triton code is more useful than re-expressing it as tile
primitives. Use analysis helpers in tests, debugging scripts, and custom
lowering flows.
