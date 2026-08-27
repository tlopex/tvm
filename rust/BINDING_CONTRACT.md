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

# Rust TVM IR binding contract

This document defines when a handwritten or generated Rust IR binding is
correct. The prototype is executable evidence for stubgen; the C++ declaration
and the shared TVM FFI ABI remain the sources of truth.

## Core rule

TVM FFI objects are not inherently C++-owned. A language may allocate an
object when it can reproduce the complete object layout and constructor
semantics. For an ordinary data node, generated Rust should therefore:

1. emit the complete inheritance prefix and every physical field in C++ order;
2. use ABI-equivalent Rust field types under `#[repr(C)]`;
3. allocate through the generated crate's `allocate_object` guard, which
   requires both `ObjectLayout` evidence and an explicit `RustAllocatable`
   marker and, once per type, compares the generated type key, inheritance
   depth, total size, total alignment, finality, reflected-field layouts, and
   native fingerprint with the loaded
   runtime before using `ObjectArc::new` to install the runtime type index,
   reference counts, and Rust deleter;
4. initialize the same defaults and validate the same invariants as C++; and
5. expose physical fields directly through the reference wrapper's `Deref`, so
   reading borrows and callers explicitly `clone()` only when they need an
   owning handle.

This lets Rust construct `AddObj { a, b, ... }` without a packed global call
while C++ reflection, structural traversal, reference counting, and destruction
continue to work.

A packed C++ constructor is not an acceptable final fallback for generated IR
bindings. Stubgen must instead distinguish two generated APIs:

- a uniform lossless complete-field path that allocates every ABI-complete
  concrete object from all of its physical state, named `new(...)` for an
  ordinary class and `from_complete_fields(...)` when a reviewed semantic
  constructor already owns `new(...)`; this path stays internal when its fields
  contain registry identity or another invariant callers cannot validate
  locally; and
- a semantic convenience constructor whose defaults, validation,
  normalization, and derived fields come from a machine-readable recipe.

If either the physical layout or semantic recipe is unavailable, stubgen must
omit that API and report a named blocker. It must not silently route `new()`
through a registered global. Interning tables and other shared runtime
services may still cross the ABI, but they are runtime capabilities rather than
per-node C++ allocation.

## Sources of truth

Use these sources together:

1. C++ declarations establish base layout, field order, and exact scalar widths.
2. C++ constructor bodies establish defaults, validation, normalization,
   interning, and derived state.
3. FFI type/reflection metadata establishes runtime type identity, inheritance,
   field schemas, structural flags, and language-independent traversal.
4. `tvm-ffi` establishes object allocation, ownership, casting, container, and
   packed-call rules.
5. Differential tests establish parity for Rust passes that claim to match a
   C++ pass.

Stubgen must never infer a physical layout from reflected fields alone. A C++
class may contain an unreflected field, an STL member, or a vptr.

## Minimal acceptance slice

The first generated slice remains intentionally small:

| Rust object | Type key | Direct fields | Construction |
| --- | --- | --- | --- |
| `ExprObj` | `ir.Expr` | `span`, `ty` | base prefix |
| `VarObj` | `ir.Var` | `name` | direct Rust allocation |
| `IntImmObj` | `ir.IntImm` | `value` | direct after integer validation |
| `AddObj` | `tirx.Add` | `a`, `b` | direct after dtype validation |
| `StmtObj` | `tirx.Stmt` | `span` | base prefix |
| `EvaluateObj` | `tirx.Evaluate` | `value` | direct after value validation |
| `BaseFuncObj` | `ir.BaseFunc` | `attrs` | base prefix |
| `PrimFuncObj` | `tirx.PrimFunc` | `params`, `ret_type`, `body` | reflected preparation plus direct Rust allocation |

`PrimType` and `TupleType` are constructed directly in Rust. `Type::Missing`
continues to use TVM's existing native singleton because its semantics include
allocation identity, not merely the bytes of `TypeNode`.

## Required checks

A binding is accepted only when every applicable check passes:

| Check | Required evidence |
| --- | --- |
| Runtime identity | exact type key, parent, depth, and finality |
| Physical layout | complete base prefix, field order, size, alignment, and exact scalar widths |
| Loaded-library compatibility | an explicit native completeness certificate exists and its size, alignment, finality, field count, and fingerprint all match before direct allocation |
| Reflected surface | exact reflected names, schemas, defaults, and structural flags |
| Complete allocator API | exact stored types by value, direct `Self` return, no hidden clone/conversion work, and visibility that preserves external invariants |
| Constructor parity | matching defaults, rejection cases, normalization, and derived state |
| Cross-language ABI | a C++ registered field getter can read a Rust allocation |
| Native behavior boundary | reuse an existing registered FFI operation when one exists; add a reflected type method only for genuinely type-owned behavior such as constructor preparation |
| Cross-language semantics | C++ structural equality accepts Rust- and C++-created equivalents |
| Ownership | Rust and C++ may clone/drop the handle without leaks, double drops, or dangling fields |
| Walk/map behavior | exact callback selection, order, definition regions, identity remapping, and COW behavior |
| Pass parity | structural equality with the named C++ pass on representative IR |

The focused checks live in `tests/stubgen_acceptance.rs`. It explicitly invokes
a C++ field getter on a Rust-created `Add` and compares that node with a
C++-created `Add` using C++ structural equality. For every ABI-complete object,
`tests/binding_contract.rs` compares Rust's total size and every reflected
direct field's offset, size, and alignment with C++ runtime metadata. It checks
Rust's object alignment against the alignment implied by the native parent and
registered fields, the exact reflected schema, flags, and registered default
values, and compile-checks every public owned `from_complete_fields` signature.
`ObjectDef::def_complete_layout()` is the
explicit native proof that no physical field is unreflected and publishes the
authoritative alignment, finality, and fingerprint. The production allocator recursively
validates every generated parent prefix and repeats the reflection-exposed
subset once per type before its first allocation, so a different loaded library
fails before C++ can read a mismatched Rust object.
Broader pass behavior is in
`tests/structural_passes.rs`.

## Pattern classification

| Pattern | Representative types | Owner/status |
| --- | --- | --- |
| Complete ordinary data layout | `Expr`, `Var`, `IntImm`, `Add`, `Stmt`, `Evaluate`, `Span`, `Range` | **GENERATE / verified** |
| Owning object reference and checked casts | all reference wrappers | **GENERATE / verified** |
| Direct scalar/object/optional/array/map fields | `IntImm`, `Call`, `For`, `SBlock`, Relax bindings | **GENERATE / verified** |
| Heterogeneous `Array<Any>` / `Map<K, Any>` | schedule values, `DictAttrs`, annotations | **RUNTIME / verified via shared container-element support** |
| Direct construction with validation | `IntImm`, binary arithmetic, `SeqStmt`, `SBlockRealize` | **GENERATE or reviewed template** |
| Complete layout, build-dependent defaults | `BufferType` | **reflected static prepare method + Rust allocation / verified** |
| Native registry identity | `Axis` | **opaque wrapper + existing `tirx.AxisGet` singleton lookup / verified** |
| Native interned identity | `SourceName` | **opaque wrapper + existing `ir.SourceName` lookup / verified** |
| Native polymorphic behavior | `Layout`, `PrimExprConvertible`, `DataProducer`, `IterVar`, `BufferRegion` | **opaque wrapper + native allocation + reflected Rust access / verified** |
| Native STL storage | `Source` | **opaque wrapper + existing `SourceMapAdd` construction / verified** |
| Complex semantic constructor | `BufferType`, `PrimFunc`, Relax `Function`, match buffer | **reflected static preparation + complete-field Rust allocation / verified** |
| Derived mutable indexes | `IRModule` construction/update | **GENERATE rebuild logic / verified** |
| Consuming `RValueRef<T>` packed argument | pass boundaries | **RUNTIME / verified without an extra reference-count increment** |
| Pass examples and analyses | `analysis`, `transform/*` | **PROTOTYPE ONLY** |

An incomplete type is safe only as a runtime-owned handle. Stubgen must not
expose `ObjectArc::new` for it or pretend that its reflected fields are the
complete physical object. A type blocked by a native vptr remains opaque unless
a separately reviewed C++ ABI migration removes that blocker.

## Important ABI details

- Rust `Option<ObjectRef>` represents a nullable C++ object handle; a required
  C++ object reference uses the non-optional wrapper.
- Do not infer a field's optionality from the referenced C++ `ObjectRef`
  class's `_type_is_nullable` flag.  Most C++ handles support an undefined
  value even when a particular node field is required.  Stubgen maps an
  explicit `ffi::Optional<T>` field schema to `Option<T>` and uses the
  constructor recipe to reject a missing required handle.
- C++ `int` and `enum class ... : int` use an `i32` representation, not `i64`.
- Native enum fields use a `#[repr(transparent)]` integer newtype with named
  constants, not a closed Rust `enum`; this keeps unknown values from a newer
  C++ library representable without undefined behavior. A conversion from
  `i64` checks only native-width narrowing; it does not reject an unknown value
  that fits the underlying integer.
- Field order includes inherited physical fields before derived fields.
- Representing inheritance as a nested Rust base is valid only when every
  derived C++ field offset agrees with that `#[repr(C)]` composition. C++ is
  allowed to reuse base tail padding, so matching field types and total size is
  not enough; the native-layout manifest must make any offset mismatch a hard
  blocker.
- Direct field access borrows through `Deref`; it does not change reference
  counts. `node.field.clone()` explicitly clones an owning handle, not the
  object node.
- A lossless complete-field allocator accepts each exact stored field type by
  value, ordered from the rootmost represented base to the concrete node and
  in native declaration order within each class. Its parameter names match the
  generated Rust fields after deterministic identifier sanitization (for
  example, native `global_var_map_` becomes Rust `global_var_map`). It moves
  those values into the guarded allocator without cloning handles, converting
  strings, or rebuilding containers. Convenience constructors may expose a
  semantic argument order, accept borrowed inputs, validate or derive state,
  and explicitly clone while delegating to this owned path.
- An ordinary generated class exposes that allocator as one direct
  `Type::new(a, b, c)` call. It must not require a builder chain. When a
  reviewed semantic constructor already owns `new(...)`, the mechanical path
  is named `from_complete_fields(...)` because Rust has no function
  overloading; both forms still perform the same direct Rust allocation.
- A preparation-backed constructor has a generated `ConstructorRecipe`
  contract containing its exact ordered input names and complete derived-field
  key set. Runtime calls compare that contract with the native
  `__ffi_constructor_recipe__` attribute and reject a stale binding or a
  returned map whose shape differs.
- A native layout certificate proves physical compatibility; it does not by
  itself authorize construction. Registry identity, interning, sentinels, and
  native resource ownership are constructor semantics. `Axis`, `SourceName`,
  `Source`, and `Type::Missing` therefore use their existing native operations
  and expose no direct Rust allocator.
- Generated object references must not implement unconditional `DerefMut`.
  Handles can share one allocation, so mutation requires structural mutation
  or an explicit uniqueness/COW mechanism.
- A polymorphic behavior base remains opaque. Additional reflected methods let
  Rust call behavior without changing the native virtual interface, but do not
  authorize Rust to manufacture a vptr or allocate a concrete subclass.
- `ObjectLayout` and `RustAllocatable` answer different questions: the former
  says Rust knows the bytes of a prefix, while the latter says the complete
  native type may be allocated in Rust. A layout-only base such as `TypeObj`
  implements only the former. Polymorphic, registry-owned, interned, and
  STL-backed objects expose opaque wrappers and implement neither.
- A Rust-created node carries a Rust deleter. C++ must release it through the
  header rather than assuming a C++ `delete` expression.
- A generated crate must not assume that a dynamically loaded TVM library has
  the layout used during generation. Before the first direct allocation of a
  type, the current runtime guard recursively validates every generated parent
  prefix, then checks the concrete type's identity, inheritance depth,
  registered total size, every reflected field's offset, size, and alignment,
  and the `__ffi_native_object_layout__` certificate. The certificate supplies
  native alignment, finality, completeness, and a fingerprint over the parent,
  size, alignment, finality, and ordered direct fields. A missing or mismatched
  value stops allocation because continuing would be memory-unsafe.
- A C++-created node keeps its C++ deleter; the same Rust wrapper can reference
  either origin.
- Reflection describes structural fields, not necessarily all physical fields.
- A field stored in an ABI-complete Rust struct is public and directly
  borrowable; stubgen must not emit one cloning getter per field.
- Convenience constructors should delegate to a complete constructor that
  exposes optional metadata such as `Span`; they must not silently make that
  metadata impossible to preserve.
- A complete-field allocator and any convenience constructor that only fills
  Rust fields return the object directly. `Result<Self>` is reserved for a
  real failure source: parsing, checked narrowing, semantic validation, a
  fallible cast, or a reflected preparation method. Generated callers must not need
  `?` or `unwrap()` around plain `ObjectArc::new` allocation.
- A `.cc` file may implement a reflected preparation method because the
  underlying compiler algorithm lives in C++, but it must return only the
  validation or derived fields needed before Rust performs the final allocation.
- Existing registered functions take precedence over adding duplicate methods.
  Existing C++ virtual dispatch remains internal to C++ and Rust crosses the
  standard packed-function ABI.

## Stubgen output ownership

- **GENERATE:** ABI-complete ordinary object structs with public physical
  fields, reference wrappers, read-only `Deref`, inheritance, casts, and
  constructor bodies whose semantics are fully available.
- **RUNTIME:** `ObjectArc`, heterogeneous `Array<Any>`/`Map<K, Any>` support,
  `RValueRef<T>` packed argument holders, safe object identity,
  `Function::from_type_method`, and reflected constructor-preparation calls.
- **RECIPE:** reviewed constructor semantics that generate Rust validation,
  defaults, normalization, and derived fields.
- **ABI BLOCKER:** native vptrs, unreflected members, or non-ABI-shareable
  storage. They expose no Rust allocator until the shared ABI is made
  constructible.
- **PROTOTYPE ONLY:** analyses and example transformations used to evaluate the
  generated surface.

## Golden-reference freeze gate

The handwritten output is ready to freeze as stubgen's replacement target only
when all of the following are true:

1. every handwritten `Object` definition passes the runtime identity,
   reflection, size, alignment, and field-offset checks;
2. every ABI-complete allocatable node has a direct complete-field allocator
   whose signature follows flattened native field order and whose body performs
   only Rust allocation, takes exact field types by value without hidden
   conversions or clones, and returns `Self`; identity-bound allocators remain
   private behind their validated or registry-backed constructor;
3. semantic constructors either contain reviewed Rust validation or call a
   registered reflected preparation method before delegating to that allocator;
4. no ABI-complete ordinary IR constructor calls a packed global; an explicitly
   opaque blocker may call its existing native constructor, and no generated
   wrapper exposes unconditional `DerefMut`;
5. C++ can inspect, traverse, compare, retain, and release Rust allocations,
   while Rust can consume C++ allocations through the same wrappers;
6. both languages exercise every reflected type method used by the handwritten
   slice;
7. behavior-only bases expose no reflection creator and cannot produce a
   method-less standalone object; and
8. the generated/layout-input fingerprint is checked against the loaded
   library before direct allocation; and
9. the complete C++, Rust, and Python suites pass against one build.

This freezes the expected generated Rust surface, not the handwritten files.
The actual stubgen is complete only when one invocation emits that surface from
layout and constructor-recipe inputs and the handwritten definitions can be
deleted without changing the acceptance tests.

The current prototype is **not yet frozen**: the runtime now publishes explicit
layout certificates and constructor input/output recipes, but stubgen does not
yet consume them, and enum declarations plus full semantic validation/default
recipes still need generator input. Passing the handwritten conformance tests
proves the current selected build and surface; the next gate is reproducing it
from generated code.

The acceptance slice is ready to replace only when stubgen can generate the
same complete layouts, complete-field allocators, and semantic preparation
recipes;
the handwritten definitions can then be deleted and the acceptance tests must
pass unchanged against the exact workspace `tvm-ffi` revision. A generated
`new()` that invokes `__ffi_init__` or another packed global does not pass this
gate.
