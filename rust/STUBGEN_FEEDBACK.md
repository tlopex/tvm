<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Stubgen feedback from handwritten Rust TVM IR

## Purpose and stopping rule

This document records requirements found by constructing real TVM IR and
running Rust structural walk/map with handwritten opaque bindings.  The
handwritten crate is a reference output for stubgen, not a proposed manually
maintained TVM Rust frontend.  It lives in a TVM checkout only because the
experiment needs a real compiler library and real registered IR types.

The first stubgen milestone is deliberately small.  Generate `Expr`, `Var`,
`IntImm`, `Add`, `Stmt`, `Evaluate`, `BaseFunc`, and `PrimFunc`, including the
base types required by their inheritance chains.  Then delete the corresponding
handwritten definitions and run `tests/stubgen_acceptance.rs` unchanged.  The
milestone is complete when both its walk and map tests pass.

Do not expand the handwritten surface merely to approach complete TVM
coverage.  Add a type only when a focused experiment is needed to answer a
specific generator, runtime, ownership, or schema question.

## Actionable requirements

The important distinction is between an ABI-safe object wrapper, which
stubgen can generate mechanically, and a semantically safe TVM constructor or
transformation, which may require runtime support or a curated registered
function.

| Handwritten mechanism | Expected generated result | Current status or blocker | Acceptance signal |
| --- | --- | --- | --- |
| Opaque `Obj` plus owning `ObjectArc` reference | Generate the C-compatible base prefix, reference type, `Deref`, checked casts, and consuming upcasts | Mechanically available from the type hierarchy | Typed walk/map callbacks match `Var`, `IntImm`, and `Add` |
| `#[type_key]` and `#[type_final]` | Emit both properties on generated object types | The schema exposes the key but not C++ `_type_final` | Exact-type callback dispatch uses the final-type fast path |
| Reflected field getter | Generate a safe typed getter backed by one cached offset/getter pair | Old output calls a removed unchecked helper | `IntImm::value`, `Var::name`, `Add::lhs/rhs`, and function/body getters work |
| Registered constructor/global | Generate typed wrappers and cache successful lookup per call site | Rust generation currently disables globals | The acceptance test constructs all nodes without handwritten packed calls |
| Constructor nullability and defaults | Use non-optional arguments where TVM guarantees non-null and preserve defaults | Current schema is more conservative than constructor invariants | Generated `PrimFunc` construction is ergonomic and valid |
| Heterogeneous `Map<String, Any>` | Preserve the surrounding typed container instead of falling back to untyped `Any` | Rust `Any` is not currently `AnyCompatible` | Non-empty `DictAttrs` can be represented safely |
| C++ `RValueRef<T>` parameter | Generate a safe consuming argument holder | Packed Rust arguments currently encode only ordinary `AnyView` | Consuming pass calls preserve intended COW behavior |
| Semantic methods such as persistent module update | Select a curated registered function or override rather than raw field initialization | Cannot be inferred mechanically from field reflection | Updating one module handle does not mutate aliases or desynchronize indexes |

## Experiment loop

For each newly discovered requirement:

1. Handwrite the smallest expected Rust binding and exercise it in a focused
   walk/map test.
2. Classify the missing piece as generator logic, `tvm-ffi` runtime support,
   TVM reflection metadata, or a curated override.
3. Change stubgen/runtime/schema at the owning layer.
4. Generate the binding into a temporary output directory.
5. Remove the matching handwritten definition and rerun the unchanged
   acceptance test.

This makes the handwritten code an executable specification: success is not
"more handwritten IR", but less handwritten IR with the same walk/map result.

## What worked

The minimal generated object shape is sufficient:

- a `#[repr(C)]` opaque `Obj` containing only its offset-zero base prefix;
- an owning reference containing `ObjectArc<Obj>`;
- `Deref` along the C++ inheritance chain and consuming upcasts;
- owning field getters backed by registered reflection getters;
- `#[type_final]` on final C++ node types;
- constructors and pass factories backed by registered TVM functions.

With that surface, all four structural APIs handled real TIR and Relax graphs.
The prototype covers definition regions, FreeVar identity remapping, Relax DAG
memoization, map-value traversal, copy-on-write container reuse, explicit loop
body recursion with `structural_visit`, and scope-sensitive rebuilding with
`structural_mutate`.

## Blocking generator/runtime coupling

The previous generated TVM bindings were produced against the `tvm-ffi` fork
commit `9a3c83d2595147dafbde2139a0222a7e6b3706cf`.  They do not compile against
the current TVM submodule revision.  A direct compatibility compile reported
558 errors.  The main causes are:

- 308 calls to the removed `object::get_reflected_field_unchecked` helper;
- 119 calls to the unavailable `cached_type_attr!` macro;
- 112 calls to the unavailable `Function::call_packed_with_kwargs` method;
- generated modules do not import `ObjectArc`, while the current `ObjectRef`
  derive expansion refers to it without a fully qualified path.

The Rust generator and its required Rust runtime support therefore need to
land and be tested together.  A generated-code compile test against the exact
workspace `tvm-ffi` revision should be mandatory.

The preferred runtime field support is a checked owning getter.  A generated
method knows the object type that declares the field, so it can resolve that
type and its ancestors once, cache a copy of the field offset and getter
function, invoke the getter, and return an owning `Any`.  Caching the two values
instead of a pointer into registry-owned field storage avoids a lifetime
assumption.  The low-level operation may remain unsafe because `ObjectCore`
alone only promises a valid header; generated safe methods can contain that
unsafe operation for their opaque, runtime-owned handles and then perform
`TryFrom<Any>` for the declared result type.  The local private reflection
prototype implements that contract for the handwritten opaque types.

## Type metadata needed by codegen

The old generator does not emit `#[type_final]`.  For final types such as
`ir.IntImm`, `tirx.Add`, and `relax.expr.If`, this loses the exact-type fast path
used by typed walk/map dispatch.  `TVMFFITypeInfo` currently does not expose the
C++ `_type_final` property.  A type metadata bit or an immutable type attribute
should publish it so every language generator can make the same decision.

The reflection schema correctly describes nullable C++ object handles, so it
conservatively produces types such as `Option<Expr>` and
`Array<Option<Var>>`.  TVM constructors often enforce stronger non-null
invariants than the carrier type.  Stubgen cannot infer those invariants from
the current schema.  If ergonomic non-optional Rust APIs are desired, TVM must
publish field/parameter non-null metadata or provide curated wrappers.

## Constructors and global functions

The old Rust generator sets `supports_global_funcs = False` and generates an
automatic `ffi_new` from reflected `__ffi_init__` fields.  This is not enough
for pass authoring:

- useful constructors such as `ir.IntImm`, `tirx.Add`, `tirx.PrimFunc`, and
  `ir.Module_FromExpr` are registered global functions;
- pass factories and runners such as `transform.MakeModulePass`,
  `tirx.transform.CreatePrimFuncPass`, and `transform.RunPass` are globals;
- automatic reflected initialization exposes inherited storage fields and
  internal fields instead of the normal TVM API;
- defaults and keyword-only metadata are collected but the Rust constructor
  currently requires every reflected init field.

`IRModule` demonstrates why this matters.  Its generated `ffi_new` asks the
caller to provide both `functions` and the internal derived
`global_var_map_`.  Those values can disagree.  A normal constructor should
accept the public inputs and let TVM establish its invariants.

Recommended split:

1. Generate opaque types, inheritance, casts, and field getters mechanically.
2. Generate registered global functions with typed positional signatures.
3. Treat auto-reflection initialization as a low-level API, not automatically
   as the canonical safe constructor for every external C++ class.
4. Allow a small override table to attach idiomatic constructors and pass
   helpers to generated reference types.

Registered functions used by generated constructors should be cached per call
site.  Looking up `ir.IntImm` by string for every integer replaced by a map is
avoidable hot-path work.  The prototype uses a fallible `OnceLock`: successful
lookups become lock-free, while a failed lookup is not cached so loading the
TVM library later can still succeed.  The existing `cached_global_func!` has
the right fast path but panics on a missing registration, which is not suitable
inside APIs that otherwise return `Result`.

A local release probe quantified both caches.  Across 200,000 iterations,
looking up `ir.IntImm` cost 75--78 ns; constructing it cost 128--131 ns with a
fresh lookup and 50--51 ns with a cached function.  Across 500,000 reads, the
integer `value` getter cost about 17 ns when it resolved and scanned reflection
metadata each time and 1.6--1.7 ns with a cached offset/getter pair.  These are
microbenchmarks rather than end-to-end pass timings, but they show that codegen
should not repeat string-based lookup in per-node callbacks.

The generated representation also needs an answer for heterogeneous
containers.  `DictAttrs` stores `Map<String, Any>`, but Rust's `Any` currently
does not implement `AnyCompatible`, so that map type cannot be expressed and
the old generator falls back to an untyped `Any`.  The prototype can create an
empty `DictAttrs` only through a low-level reflection constructor with an empty
map whose element type is irrelevant.  Non-empty attributes need either
heterogeneous container support or a curated registered constructor.

## Structural-map semantics exposed by real passes

`structural_map` rebuilds according to the language-agnostic structural
protocol; it is not automatically equivalent to every existing C++ semantic
mutator.  Replacing `AssertStmt` with `Evaluate(0)` leaves that no-op inside a
`SeqStmt` unless the Rust pass also performs the normalization implemented by
C++ `StmtMutator`.  The Rust `SkipAssert` prototype now flattens nested
sequences, removes no-op evaluates, and collapses one-element sequences before
it is compared with the C++ pass.

Map and Dict keys are structural anchors: walk/map recursively processes only
their values.  This preserves lookup identity and was verified with a
`Map<GlobalVar, Expr>`.

That rule also exposes a TVM-side `IRModule` issue.  Directly mapping a module
with a callback that replaces `GlobalVar` leaves the keys in `functions`
unchanged but mutates the `GlobalVar` values in reflected
`global_var_map_`.  An experiment produced a functions key named `main` and a
name-map value named `main_mapped`.  Possible TVM fixes are to mark the derived
name map as structurally ignored or register an `IRModule` mutation hook that
rebuilds internal indexes.  Until then, the module-pass prototype maps each
function value and updates it through `ir.Module_Add`, allowing TVM to maintain
the derived table.

`ir.Module_Add` mutates the module node's container fields and returns the same
module handle.  Consuming one Rust handle is not sufficient: another cloned
handle may still share the same C++ node.  The public prototype method therefore
implements persistent `with_updated_function(&self)` as `ir.Module_Clone`
followed by `ir.Module_Add`.  The module pass clones the module node once and
then uses an internal owned-update path for every function.  Keeping a snapshot
of the original function map causes one expected container COW on the first
update, not one full-module copy per function.  In generated docs, `Clone` on
an object reference must be described as cloning a handle, not cloning the C++
object.

There is a separate ownership gap at packed-function boundaries.  C++ pass
entry points accept `RValueRef<IRModule>` and pass callbacks deliver
`RValueRef<PrimFunc>`/`RValueRef<IRModule>`.  Rust correctly normalizes an
incoming rvalue view to an owning `Any`, but `TupleAsPackedArgs` can only encode
ordinary `AnyView` arguments.  Therefore `Pass::run(module)` consumes the Rust
handle yet sends an lvalue, and C++ must copy the handle and may trigger COW.
The Rust runtime should expose a safe consuming `RValueRef<T>` argument holder;
stubgen should use it for parameters whose schema declares an rvalue reference.

## Recommended implementation order

1. **Make the minimal slice compile.** Keep generator output and its exact
   `tvm-ffi` runtime helpers in one compile contract, and fully qualify paths
   emitted by derive macros.
2. **Pass the focused walk test.** Generate opaque wrappers, inheritance,
   final-type metadata, and typed reflected getters for the acceptance types.
3. **Pass the focused map test.** Generate cached registered constructors so
   TVM can rebuild mapped nodes without handwritten packed calls.
4. **Add one normal pass boundary.** Generate the relevant pass factory and
   runner only after direct walk/map works.
5. **Address ownership and containers separately.** Add rvalue argument holders
   and heterogeneous `Any` containers with their own focused tests.
6. **Add semantic overrides last.** Curate APIs such as persistent
   `IRModule` updates only where reflection cannot express the invariant.

## Supporting evidence suite

The larger integration suite is supporting evidence, not the initial stubgen
acceptance surface.  It covers distinct protocol behavior rather than
repeating node construction:

- exact pre-order and post-order event sequences;
- subtree pruning with `WalkResult::Skip`;
- early termination with `WalkResult::Interrupt` and an owning interrupt payload;
- typed tuple callbacks and generated dispatch objects;
- reflected inherited and optional fields;
- recursive definition regions for PrimFunc parameters;
- FreeVar replacement memoization across definition and use;
- DAG memoization for a shared `relax.If`;
- pre-order map replacement pruning versus post-order mapping;
- shared-input preservation and unique-Array in-place reuse;
- Map key preservation with value replacement;
- nested `IfThenElse`/`SeqStmt` normalization;
- PrimFunc and module pass integration;
- Relax FunctionPass integration with definition/use identity checks;
- explicit `For.body` recursion for loop-depth analysis;
- a scope-sensitive mutator that simplifies nested-loop bounds and bodies but
  leaves top-level loop bounds unchanged;
- typed BufferType, BufferLoad/Store, BufferRegion, IterVar, SBlock, and
  SBlockRealize construction and reflected access;
- unit-loop variable substitution through real buffer indices, checked against
  C++ `LowerTIRxOpaque`;
- structural-equality differential checks against C++ `SkipAssert` and
  `StmtSimplify`.
