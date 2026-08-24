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

This document defines how to decide whether a handwritten or generated Rust
IR binding is correct.  The handwritten prototype is an executable reference
for stubgen, not the source of truth by itself.

## Sources of truth

Use these sources in order:

1. The C++ node declaration and registration define the type key, inheritance,
   finality, reflected fields, structural flags, and constructor semantics.
2. TVM FFI reflection defines the language-independent type and field metadata
   that Rust can inspect at runtime.
3. Existing registered functions define the public construction and pass
   boundaries.  Rust should not reproduce C++ object initialization rules.
4. Differential tests against C++ define behavior when a Rust pass claims to
   implement the same transformation.
5. Rust adds ownership and lifetime rules around that ABI; it must not expose a
   safe API when the available metadata is insufficient to prove safety.

Python APIs are useful ergonomic references, but they do not override the C++
or FFI contract.

Every generated item must therefore be traceable to one of three things:

- published reflection or registered-function metadata;
- a stable `tvm-ffi` ABI rule;
- a small, reviewed override for semantics that metadata cannot express.

## Minimal acceptance slice

The first stubgen milestone is deliberately limited to these eight object
types.  “Fields” lists fields declared directly by that type, not inherited
fields.

| Rust object | C++ declaration | Type key | Parent | Final | Direct reflected fields | Canonical constructor |
| --- | --- | --- | --- | --- | --- | --- |
| `ExprObj` | `include/tvm/ir/base_expr.h` | `ir.Expr` | `ffi.Object` | no | `span`, `ty` | none |
| `VarObj` | `include/tvm/ir/expr.h` | `ir.Var` | `ir.Expr` | no | `name` | `ir.Var` |
| `IntImmObj` | `include/tvm/ir/expr.h` | `ir.IntImm` | `ir.Expr` | yes | `value` | `ir.IntImm` |
| `AddObj` | `include/tvm/tirx/expr.h` | `tirx.Add` | `ir.Expr` | yes | `a`, `b` | `tirx.Add` |
| `StmtObj` | `include/tvm/tirx/stmt.h` | `tirx.Stmt` | `ffi.Object` | no | `span` | none |
| `EvaluateObj` | `include/tvm/tirx/stmt.h` | `tirx.Evaluate` | `tirx.Stmt` | yes | `value` | `tirx.Evaluate` |
| `BaseFuncObj` | `include/tvm/ir/function.h` | `ir.BaseFunc` | `ir.Expr` | no | `attrs` | none |
| `PrimFuncObj` | `include/tvm/tirx/function.h` | `tirx.PrimFunc` | `ir.BaseFunc` | yes | `params`, `ret_type`, `body` | `tirx.PrimFunc` |

The minimal constructors also depend on `ir.PrimType` and `ir.TypeMissing`.
The structural flags are part of the contract: `Expr.span`, `Var.name`, and
`Stmt.span` use `SEqHashIgnore`; `PrimFunc.params` uses
`SEqHashDefRecursive`.

## Required checks

A binding is accepted only when all applicable checks pass:

| Check | What it prevents | Current signal |
| --- | --- | --- |
| Type identity and inheritance | A wrong type key, depth, or parent cast | Runtime `TVMFFITypeInfo` assertions |
| Complete field surface | Silently omitting or inventing a reflected field | Exact direct-field-name and getter assertions |
| Structural field flags | Incorrect definition/use or structural-equality behavior | Runtime field-flag assertions |
| Canonical construction | Reimplementing C++ initialization or calling the wrong global | Global-existence and constructor round-trip tests |
| Typed access | A getter returning the wrong Rust type or ownership form | Construct-and-read round-trip tests |
| Walk behavior | Incorrect node selection, order, or definition region | Exact event and `DefRegionKind` assertions |
| Map behavior | Incorrect rewriting, identity remapping, or input mutation | Result-value and object-identity assertions |
| C++ parity | A Rust pass with different semantics from its named C++ equivalent | Structural-equality differential tests |
| Rust safety | Dangling borrowed values, invalid ownership, or unsafe public contracts | Compile-time types plus ownership tests |

The focused minimal checks live in `tests/stubgen_acceptance.rs`.  The exact
type hierarchy, direct fields, flags, and structural kind of every handwritten
wrapper are checked in `tests/binding_contract.rs`.  The larger behavioral and
C++ differential suite lives in `tests/structural_passes.rs`.

## Current prototype rating

- **Verified:** the minimal eight-type stubgen slice passes metadata,
  construction, typed-access, walk, and map checks.
- **Verified metadata:** every current handwritten wrapper matches the runtime
  type key, parent, depth, reflected fields, field flags, and structural kind.
- **Partial API:** fields whose dependency type is not yet handwritten, such as
  `SourceMap`, `GlobalInfo`, `Layout`, and heterogeneous annotation maps, are
  returned as owning `Any` values rather than pretending to have a more precise
  Rust type.
- **Prototype only:** conveniences such as `Expr::int` and `Expr::add`, and the
  example passes, demonstrate usage but are not a complete generated API.
- **Blocked:** the schema/runtime gaps below still prevent mechanically
  generating the entire surface without overrides.

## What cannot yet be generated mechanically

Passing the current tests does not mean that arbitrary TVM bindings can be
generated safely.  These gaps require additional schema/runtime work or an
explicit override:

- `TVMFFITypeInfo` does not publish C++ `_type_final`.  The acceptance test
  records finality from the C++ declarations, but cannot independently recover
  it from runtime metadata.
- Reflection carrier types do not express every constructor's non-null
  invariant or default argument.
- `Map<String, Any>` cannot currently be represented by the typed Rust
  container API because `Any` is not `AnyCompatible`.
- Packed Rust calls do not yet have a safe consuming argument for C++
  `RValueRef<T>` parameters.
- Types such as `IRModule` have semantic update rules and derived indexes that
  cannot be inferred from field reflection alone.

These are blockers or override points, not permission for stubgen to guess.

## Ownership of generated code

Classify each handwritten mechanism before moving it into production:

- **GENERATE:** opaque object/reference wrappers, inheritance, checked casts,
  reflected getters, and typed registered-function wrappers.
- **RUNTIME:** reusable ABI operations such as cached field lookup and safe
  consuming packed arguments.
- **OVERRIDE:** a small curated API where TVM invariants are not represented by
  metadata, for example persistent `IRModule` updates.
- **PROTOTYPE ONLY:** pass examples and experiments that demonstrate a need but
  should not be copied into generated bindings.

The milestone is complete when stubgen generates the minimal slice, the
handwritten definitions are deleted, and the same acceptance tests pass
without modification.  Generated output should also be compiled against the
exact `tvm-ffi` revision used by TVM so generator/runtime drift fails in CI.
