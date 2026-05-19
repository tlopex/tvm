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

.. _tirx-architecture:

Internal Architecture
=====================
This section describes how TIRX is built — the six architectural layers from
C++ IR nodes to the final lowering pipeline. Understanding these layers helps
when extending TIRX with new tile primitives, new backends, or new IR
constructs.

.. code:: text

    Layer 6: Lowering Pipeline     (C++)    TilePrimitiveDispatch → cleanup → strip scopes
    Layer 5: Dispatch Framework    (Python) TilePrimitiveCall → target-specific PrimFunc
    Layer 4: Printer / Debugger    (C++)    IR → readable TVMScript text
    Layer 3: TVMScript DSL         (Python) @Tx.prim_func → IR tree construction
    Layer 2: FFI Bridge            (C++)    Type-erased PackedFunc for C++↔Python
    Layer 1: IR Node Core          (C++)    Data definitions: nodes, refs, types

Each layer builds on the one below it. Users interact at Layer 3 (DSL); dispatch
authors work at Layer 5; compiler developers touch all layers.

Layer 1: IR Node Core
---------------------
TIRX's IR is a tree of C++ objects following TVM's **FooNode + Foo** pattern:

- **FooNode**: The actual data struct (inherits from ``Object``). Holds fields
  like ``op``, ``args``, ``body``. Allocated on the heap with reference counting.
- **Foo**: A lightweight handle (``ObjectRef``) that wraps a pointer to
  ``FooNode``. Cheap to copy, share, and pass through FFI.

.. code:: cpp

    // include/tvm/tirx/tirx_stmt.h
    class TilePrimitiveCallNode : public StmtNode {
     public:
      tvm::Op op;                      // which operator (e.g., "tirx.copy_async")
      ffi::Array<ffi::Any> args;       // operands (buffers, scalars)
      ffi::Map<ffi::String, Buffer> workspace;  // named temporary buffers
      ffi::Map<ffi::String, ffi::Any> config;   // keyword arguments
      ffi::Optional<ffi::String> dispatch;      // force a specific variant

      static void RegisterReflection() {
        namespace refl = tvm::ffi::reflection;
        refl::ObjectDef<TilePrimitiveCallNode>()
            .def_ro("op", &TilePrimitiveCallNode::op)
            .def_ro("args", &TilePrimitiveCallNode::args)
            .def_ro("workspace", &TilePrimitiveCallNode::workspace)
            .def_ro("config", &TilePrimitiveCallNode::config)
            .def_ro("dispatch", &TilePrimitiveCallNode::dispatch);
      }

      TVM_FFI_DECLARE_OBJECT_INFO_FINAL("tirx.TilePrimitiveCall", TilePrimitiveCallNode, StmtNode);
    };

    class TilePrimitiveCall : public Stmt {
     public:
      TVM_DLL TilePrimitiveCall(tvm::Op op, ffi::Array<ffi::Any> args,
                                ffi::Map<ffi::String, Buffer> workspace = {},
                                ffi::Map<ffi::String, ffi::Any> config = {},
                                ffi::Optional<ffi::String> dispatch = std::nullopt);
    };

Key design properties:

- **Immutable by default**: Once constructed, nodes are not mutated. To "modify"
  a node, you create a new one (``CopyOnWrite`` for in-place mutation when the
  reference count is 1).
- **Reflection**: ``RegisterReflection`` uses ``ObjectDef`` to expose fields as
  read-only properties, enabling Python access, serialization, and the printer.
- **Type hierarchy**: ``TilePrimitiveCallNode`` inherits from ``StmtNode``, which
  inherits from ``Object``. The visitor pattern (``StmtFunctor``,
  ``StmtExprMutator``) dispatches on these types.

Other key IR nodes:

.. list-table::
   :header-rows: 1
   :widths: 25 40

   * - Node
     - Purpose
   * - ``ExecScopeStmtNode``
     - Wraps a body in an execution scope (``cta``, ``warpgroup``, etc.)
   * - ``TilePrimitiveCallNode``
     - An operator invocation: op + args + config
   * - ``ExecScopeNode``
     - Stores the current ``ScopeKind`` and any ``ScopeIdDef`` declarations
   * - ``ScopeIdDefNode``
     - Declares scope-id variables, optional extents, scope binding, and
       preferred extents. Dispatch later resolves them into concrete bindings.

Layer 2: FFI Bridge
-------------------
The FFI (Foreign Function Interface) connects C++ and Python via **PackedFunc**
— a type-erased callable that can accept and return any TVM type.

**How it works:**

1. C++ registers a function by string name:

   .. code:: cpp

       // src/tirx/transform/lower_tirx.cc
       TVM_FFI_STATIC_INIT_BLOCK() {
         namespace refl = tvm::ffi::reflection;
         refl::GlobalDef()
             .def("tirx.transform.LowerTIRx", LowerTIRx);
       }

2. Python declares a matching FFI module with the same prefix:

   .. code:: python

       # python/tvm/tirx/transform/_ffi_api.py
       import tvm_ffi
       tvm_ffi.init_ffi_api("tirx.transform", __name__)

       # Now tirx.transform._ffi_api.LowerTIRx() calls the C++ function

   The prefix in ``init_ffi_api`` must match the C++ registration prefix.
   For example, ``"tirx.transform"`` maps to functions registered as
   ``"tirx.transform.*"``.

**What crosses the FFI boundary:**

- Primitive types (int, float, string) → passed by value.
- Object references (``Stmt``, ``PrimFunc``, ``Buffer``) → passed as
  ``ObjectRef`` with reference counting. Python holds a ref; C++ holds a ref.
  The object lives until all refs are dropped.
- Functions → wrapped as ``PackedFunc``. This is how C++ calls Python dispatch
  implementations and how Python calls C++ lowering passes.

**The dispatch callback path** (Layer 5 → Layer 2 → Layer 5):

When the C++ lowering pass encounters a ``TilePrimitiveCall``, it calls a Python
function through FFI:

.. code:: text

    C++ TilePrimitiveDispatch
      → PackedFunc("tirx.f_op_dispatcher")  [FFI call to Python]
        → Python run_dispatch()
          → predicate checks
          → call schedule impl (e.g., copy_tma_impl)
          → return PrimFunc  [FFI return to C++]
    C++ replaces TilePrimitiveCall with the returned body

This round-trip is the core of TIRX's extensibility: new dispatch implementations
are written in Python without touching C++.

Layer 3: TVMScript DSL
----------------------
The DSL layer lets users write TIRX kernels as Python functions. The
``@Tx.prim_func`` decorator triggers IR construction via the **IR Builder**
and **Frame Stack** pattern.

**Frame Stack pattern:**

Each Python ``with`` block pushes a **frame** onto a thread-local stack. When the
``with`` block exits, the frame pops and attaches its collected IR to the parent
frame.

.. code:: python

    @Tx.prim_func
    def kernel(
        A: Tx.Buffer((M, N), "float16"),
        B: Tx.Buffer((M, N), "float16"),
    ):
        # PrimFuncFrame pushed
        with Tx.kernel():
            # ExecScopeFrame("kernel") pushed
            cta_m, cta_n = Tx.cta_id([128, 128])
            # ScopeIdDef nodes created for cta_m and cta_n
            with Tx.cta():
                # ExecScopeFrame("cta") pushed
                Tx.copy(B, A)
                # TilePrimitiveCall node created and attached to current frame
            # ExecScopeFrame("cta") popped → wraps body in ExecScopeStmt
        # ExecScopeFrame("kernel") popped
    # PrimFuncFrame popped → produces PrimFunc

**Frame types:**

.. list-table::
   :header-rows: 1
   :widths: 25 50

   * - Frame
     - Created By
   * - ``PrimFuncFrame``
     - ``@Tx.prim_func`` decorator
   * - ``ExecScopeFrame``
     - ``Tx.kernel()``, ``Tx.cluster()``, ``Tx.cta()``, ``Tx.warpgroup()``, etc.
   * - ``ForFrame``
     - ``Tx.serial()``, ``Tx.grid()``
   * - ``AllocBuffer`` statement
     - ``Tx.alloc_buffer()`` returns a ``Buffer`` and inserts allocation IR
   * - ``Allocate`` statement
     - ``Tx.allocate()`` emits low-level allocation IR when used directly

**Operator calls:**

When you write ``Tx.copy_async(dst, src)``, the IR Builder:

1. Looks up the ``tvm.Op`` named ``"tirx.copy_async"``.
2. Packs the arguments into an ``Array<Any>``.
3. Creates a ``TilePrimitiveCall`` node.
4. Attaches it to the current frame's body.

The operator has **no implementation at this point** — it is purely declarative.
The implementation is selected later by the dispatch framework (Layer 5).

Layer 4: Printer
----------------
The printer converts IR trees back into readable TVMScript syntax. It is the
inverse of the DSL parser: given a ``PrimFunc``, it produces a string that looks
like valid ``@Tx.prim_func`` Python code.

**Primary use case: debugging.**

During development, you frequently need to inspect IR at intermediate stages:

.. code:: python

    # Print the IR before lowering
    print(ir_module.script())

    # Print after TIRX lowering
    mod = tirx.transform.LowerTIRx()(mod)
    print(mod.script())

    # Environment variable to print after dispatch
    # (set TVM_PRINT_AFTER_TIRX_DISPATCH_OPS=1)

The printer handles TIRX-specific nodes:

- ``ScopeIdDef`` → ``cta_m, cta_n = Tx.cta_id([...])``
- ``ExecScopeStmt`` → ``with Tx.cta():``
- ``TilePrimitiveCall`` → ``Tx.copy_async(dst, src)``
- Custom buffer scopes → ``scope="shared"``, ``scope="tmem"``

Without the printer, the IR would only be representable as a C++ object tree —
impractical for debugging kernels that can be hundreds of lines.

Layer 5: Dispatch Framework
---------------------------
The dispatch framework maps each ``TilePrimitiveCall`` to a target-specific
implementation. This is the **strategy pattern**: multiple implementations
are registered, and predicates select the best one at compile time.

**Data structures:**

.. code:: python

    # Global dispatch table
    _DISPATCH_TABLE: dict[tuple[Op, str], list[DispatchCase]]

    # Each entry
    @dataclass
    class DispatchCase:
        variant: str           # human-readable name ("tma", "vec_load")
        priority: int          # higher = tried first
        preds: list[Predicate] # conditions that must all pass
        impl: Callable         # function(TilePrimitiveCall, DispatchContext) → PrimFunc

**Registration example:**

This excerpt omits the helper predicates and implementation imports; see the
CUDA ``copy_async`` schedules for complete dispatch modules.

.. code:: python

    @register_dispatch(
        "copy_async", "cuda",
        variant="tma",
        priority=10,
        when=[
            predicate("validate_copy_op",
                      lambda op, sctx: (validate_copy_op(op, sctx), "not a valid copy op")),
            predicate("single_thread",
                      lambda op, sctx: (single_thread(op, sctx),
                                        f"unsupported exec_scope {sctx.scope_kind}")),
        ],
    )
    def copy_async_dispatch_tma(op: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
        return copy_tma_impl(op, sctx)

**What a dispatch implementation produces:**

The returned ``PrimFunc`` contains **device-side** IR only — no more
``TilePrimitiveCall`` nodes. For example, ``copy_tma_impl`` returns a function body
with ``ptx.cp_async.bulk.tensor`` instructions and mbarrier signaling.

**Host-side** setup (e.g., ``call_packed("runtime.cuTensorMapEncodeTiled", ...)``)
is injected separately via ``sctx.add_init_stmt(stmt, host=True)``. The C++
lowering pass collects these callbacks and inserts the host statements before
the kernel launch.

The C++ lowering pass splices the returned function body into the IR tree,
replacing the original ``TilePrimitiveCall``.

**Failure diagnostics:**

When no variant matches, the dispatcher produces a formatted table showing each
variant and which predicate failed:

.. code:: text

    RuntimeError: TIRX schedule dispatch failed: op=tirx.copy_async target=cuda
    +----------------------------+----------------------------------------------+
    | Variant                    | Error                                        |
    +----------------------------+----------------------------------------------+
    | non-bulk-copy (prio=20)    | rejected: validate_copy_op — not a valid     |
    |                            |   copy op                                    |
    |                            | opcall:                                      |
    |                            |   Tx.copy_async(dst_smem, src_local[0:64])   |
    | tma (prio=10)              | rejected: single_thread — unsupported        |
    |                            |   exec_scope ...                             |
    |                            | opcall:                                      |
    |                            |   Tx.copy_async(dst_smem, src_local[0:64])   |
    +----------------------------+----------------------------------------------+

Layer 6: Lowering Pipeline
--------------------------
``LowerTIRx`` removes tile-primitive calls and execution-scope wrappers through
three top-level stages. Its output no longer contains ``TilePrimitiveCall`` or
``ExecScopeStmt`` nodes, but it still flows through the TIRX backend pipeline
before final codegen.

.. code:: text

    Input: TIR + ExecScopeStmt + TilePrimitiveCall nodes
      │
      ├─ Stage 1: TilePrimitiveDispatch
      │    Resolves ScopeIdDef declarations, launch parameters, and
      │    execution-scope slices; for each TilePrimitiveCall, builds a
      │    DispatchContext(target, exec_scope, launch_params, var_range_map,
      │    alloc_only=False, callbacks={}, shared_state, inter, intra,
      │    scope_kind), calls Python "tirx.f_op_dispatcher" via FFI, and
      │    replaces the TilePrimitiveCall with the returned PrimFunc body.
      │    Collects alloc_buffers, device_init_stmts, host_init_stmts,
      │    and post_buffer_def_stmts from dispatch callbacks.
      │
      ├─ Stage 2: LowerTIRxCleanup
      │    Flattens layout-aware buffers and removes buffer offsets introduced
      │    while expanding tile primitives. It also retains legacy cleanup for
      │    old dispatch-only annotations such as scope_id_extent_map,
      │    thread_var_map, and tirx.warp_id_in_cta.
      │
      └─ Stage 3: LowerTIRxStripExecScope
           Strips ExecScopeStmt wrappers. These were needed during dispatch
           for scope tracking but are not valid final TIR nodes.

    Output: TIRX backend IR without TilePrimitiveCall / ExecScopeStmt

The pipeline is defined in ``src/tirx/transform/lower_tirx.cc``:

.. code:: cpp

    Pass LowerTIRx() {
      std::vector<tvm::transform::Pass> passes = {TilePrimitiveDispatch()};
      if (std::getenv("TVM_PRINT_AFTER_TIRX_DISPATCH_OPS")) {
        passes.push_back(tvm::transform::PrintIR());  // debug hook
      }
      passes.push_back(LowerTIRxCleanup());
      passes.push_back(LowerTIRxStripExecScope());
      return tvm::transform::Sequential(passes, "tirx.LowerTIRx");
    }

**Important property**: The input to Stage 1 contains TIR + TIRX nodes. The
output of Stage 3 no longer contains tile primitives or exec-scope wrappers.
Subsequent TIRX and TVM passes handle buffer flattening, loop unrolling,
vectorization, finalization, and codegen. This keeps TIRX localized to its own
pipeline entry points rather than changing the base TVM compilation pipeline.

Relationship to S-TIR and Relax
-------------------------------
TIRX exists alongside (not replacing) the existing TVM IRs:

.. list-table::
   :header-rows: 1
   :widths: 22 28 32 28

   * - Source level
     - Authoring form
     - Lowering result
     - Codegen path
   * - Relax regular tensor ops
     - S-TIR blocks plus schedule primitives
     - Scheduled and lowered TIR
     - Standard TVM target codegen
   * - Explicit accelerator kernels
     - TIRX scopes, layouts, and tile primitives
     - TIR with target intrinsics, host/device metadata, and generated launch code
     - TIRX finalization, then target-specific codegen

- **S-TIR** uses ``SBlock`` + schedule primitives (tile, vectorize, bind).
  Good for simple, regular computations.
- **TIRX** uses ``ExecScopeStmt`` + ``TilePrimitiveCall`` + dispatch. Designed for
  operations that need hardware-specific features (TMA, tensor cores, async
  pipelines, TMEM) that are awkward to express with schedule primitives alone.
- **Relax** is the graph-level IR. It handles operator fusion, memory planning,
  and scheduling across operators. Integration that chooses TIRX for suitable
  high-performance kernels is a direction of the stack rather than a completed
  contract for every operator.
- **Both paths eventually become codegen-ready TIR modules**, because TVM's
  target code generators consume TIR. This does not mean the paths are
  equivalent: TIRX lowers tile primitives through ``LowerTIRx`` before standard
  target codegen sees the module.

In the current codebase, kernels are authored directly in the TIRX DSL
(``@Tx.prim_func``). The dispatch framework is the integration point for higher
layers: a frontend can generate ``TilePrimitiveCall`` nodes and let the dispatch
system choose the implementation for the target GPU.
