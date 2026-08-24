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
3. allocate with `ObjectArc::new`, which installs the runtime type index,
   reference counts, and a Rust deleter in `TVMFFIObject`;
4. initialize the same defaults and validate the same invariants as C++; and
5. expose direct typed field access.

This lets Rust construct `AddObj { a, b, ... }` without a packed global call
while C++ reflection, structural traversal, reference counting, and destruction
continue to work.

A registered C++ function is an explicit override, not the default constructor
strategy. Keep an override when direct Rust allocation cannot preserve the C++
contract, for example:

- interned or singleton objects (`SourceName`, `Axis`, `Type::Missing`);
- cached canonical values (`PrimType`);
- C++ polymorphic nodes with a vptr or hidden non-ABI state (`Layout`,
  `PrimExprConvertible`, `Source`);
- constructors with substantial derived state or semantic checks (`PrimFunc`,
  Relax `Function`, `For`, buffer load/store, match-buffer declarations);
- mutable objects with derived indexes (`IRModule`).

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
| `PrimFuncObj` | `tirx.PrimFunc` | `params`, `ret_type`, `body` | semantic C++ override |

`PrimType` and `Type::Missing` are constructor dependencies and intentionally
remain canonical C++ overrides.

## Required checks

A binding is accepted only when every applicable check passes:

| Check | Required evidence |
| --- | --- |
| Runtime identity | exact type key, parent, depth, and finality |
| Physical layout | complete base prefix, field order, size, alignment, and exact scalar widths |
| Reflected surface | exact reflected names, schemas, defaults, and structural flags |
| Constructor parity | matching defaults, rejection cases, normalization, and derived state |
| Cross-language ABI | a C++ registered field getter can read a Rust allocation |
| Cross-language semantics | C++ structural equality accepts Rust- and C++-created equivalents |
| Ownership | Rust and C++ may clone/drop the handle without leaks, double drops, or dangling fields |
| Walk/map behavior | exact callback selection, order, definition regions, identity remapping, and COW behavior |
| Pass parity | structural equality with the named C++ pass on representative IR |

The focused checks live in `tests/stubgen_acceptance.rs`. It explicitly invokes
a C++ field getter on a Rust-created `Add` and compares that node with a
C++-created `Add` using C++ structural equality. The complete runtime metadata
surface is checked in `tests/binding_contract.rs`; broader pass behavior is in
`tests/structural_passes.rs`.

## Pattern classification

| Pattern | Representative types | Owner/status |
| --- | --- | --- |
| Complete ordinary data layout | `Expr`, `Var`, `IntImm`, `Add`, `Stmt`, `Evaluate`, `Span`, `Range` | **GENERATE / verified** |
| Owning object reference and checked casts | all reference wrappers | **GENERATE / verified** |
| Direct scalar/object/optional/array/map fields | `IntImm`, `Call`, `For`, `SBlock`, Relax bindings | **GENERATE / verified** |
| Heterogeneous `Map<String, Any>` | `DictAttrs`, annotations | **RUNTIME / verified via `AnyMap`** |
| Direct construction with validation | `IntImm`, binary arithmetic, `SeqStmt`, `SBlockRealize` | **GENERATE or reviewed template** |
| Interned/canonical object | `SourceName`, `Axis`, `PrimType`, missing type | **OVERRIDE** |
| C++ polymorphic or hidden layout | `Layout`, `PrimExprConvertible`, `Source` | **OPAQUE + OVERRIDE** |
| Complex semantic constructor | `PrimFunc`, Relax `Function`, `For`, buffer access nodes | **OVERRIDE** |
| Derived mutable indexes | `IRModule` construction/update | **OVERRIDE** |
| Consuming `RValueRef<T>` packed argument | pass boundaries | **RUNTIME gap** |
| Pass examples and analyses | `analysis`, `transform/*` | **PROTOTYPE ONLY** |

An opaque type is safe only as a runtime-owned handle. Stubgen must not expose
`ObjectArc::new` for it or pretend that its reflected fields are the complete
physical object.

## Important ABI details

- Rust `Option<ObjectRef>` represents a nullable C++ object handle; a required
  C++ object reference uses the non-optional wrapper.
- C++ `int` and `enum class ... : int` map to `i32`/`#[repr(i32)]`, not `i64`.
- Field order includes inherited physical fields before derived fields.
- `Clone` clones an owning handle, not the object node.
- A Rust-created node carries a Rust deleter. C++ must release it through the
  header rather than assuming a C++ `delete` expression.
- A C++-created node keeps its C++ deleter; the same Rust wrapper can reference
  either origin.
- Reflection describes structural fields, not necessarily all physical fields.

## Stubgen output ownership

- **GENERATE:** ABI-complete ordinary object structs, reference wrappers,
  inheritance, casts, direct getters, and constructor bodies whose semantics are
  fully available.
- **RUNTIME:** `ObjectArc`, `AnyMap`, reusable packed argument holders, checked
  reflection fallback for opaque types, and fallible cached global lookup.
- **OVERRIDE:** a small reviewed table for interning, vtables/hidden layout,
  complex semantic construction, and persistent module updates.
- **PROTOTYPE ONLY:** analyses and example transformations used to evaluate the
  generated surface.

The acceptance slice is ready to replace when stubgen can generate the same
complete layouts and direct constructors, the handwritten definitions can be
deleted, and all acceptance tests pass unchanged against the exact workspace
`tvm-ffi` revision.
