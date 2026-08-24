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

# Feedback for Rust TVM stubgen

This crate handwrites a representative Rust IR surface, then runs real
`structural_walk`, `structural_map`, pass integration, and C++ differential
tests. Its purpose is to establish what stubgen must generate, what belongs in
`tvm-ffi`, and which operations need reviewed overrides.

## Main correction from the experiment

The first prototype treated every IR node as an opaque header and routed every
constructor through a registered packed function. That is unnecessarily slow
and is not the intended object-ABI model.

For an ordinary C++ data node, Rust can use the same memory layout and allocate
the object directly:

```rust
#[repr(C)]
#[derive(Object)]
#[type_key = "tirx.Add"]
#[type_final]
pub struct AddObj {
    base: ExprObj,
    a: Expr,
    b: Expr,
}

pub fn new(a: &Expr, b: &Expr) -> Result<Add> {
    // Validate the same dtype rule as C++, then:
    Ok(Add {
        data: ObjectArc::new(AddObj {
            base: ExprObj::new(a.ty()?, None),
            a: a.clone(),
            b: b.clone(),
        }),
    })
}
```

`ObjectArc::new` installs a common `TVMFFIObject` header and a Rust deleter.
C++ sees the normal runtime type index and field offsets, while destruction
returns through the deleter stored in the header. This was verified by invoking
a C++ reflection getter on a Rust-created `Add`, running C++ structural
traversal on it, and comparing it with a C++-created `Add` through C++
structural equality.

## What stubgen should generate

For a directly constructible node, generate:

- the complete `#[repr(C)]` inheritance prefix;
- all physical fields in exact C++ order, including unreflected fields when
  their layout is part of the supported ABI;
- exact-width scalar types (`int` is `i32`, not `i64`);
- the reference wrapper with `ObjectArc`, `Deref`, casts, and upcasts;
- direct typed getters;
- a direct Rust constructor only when its full validation/default logic is
  known and reproduced;
- finality and structural metadata used by typed walk/map dispatch.

The generator cannot recover complete physical layout from reflection alone.
It needs a layout source derived from C++ declarations or an explicitly curated
schema. Reflection is still the authority for field schemas, flags, and the
language-independent structural protocol.

## Constructor classification

Do not make either “direct allocation” or “registered global” unconditional.
Classify each node:

| Class | Examples | Generated behavior |
| --- | --- | --- |
| Plain data node | `Span`, `Range`, `Var`, `IntImm`, `Add`, `Evaluate`, `VarBinding`, `SBlock` | complete layout and direct Rust allocation |
| Plain node with local validation | integer literals, binary ops, `SeqStmt`, `SBlockRealize` | direct allocation plus equivalent Rust validation |
| Interned/singleton/cached | `SourceName`, `Axis`, `Type::Missing`, `PrimType` | registered C++ override |
| Polymorphic/hidden layout | `Layout`, `PrimExprConvertible`, `Source` | opaque runtime-owned handle plus registered C++ override |
| Complex semantic constructor | `For`, `PrimFunc`, Relax `Function`, buffer load/store, match buffer | registered C++ override unless its semantics are deliberately ported |
| Derived mutable state | `IRModule` | semantic C++ construction/update override |

For override calls, use fallible cached lookup: cache successful registrations,
but do not permanently cache failure because a TVM library may be loaded later.

## Runtime support learned from the experiment

### Heterogeneous maps

TVM fields such as `DictAttrs::__dict__`, loop annotations, and block
annotations are `Map<String, Any>`. A blanket `AnyCompatible for Any` conflicts
with Rust's identity conversion rules, so the prototype adds `AnyMap<K>`: an
ABI-compatible map wrapper whose public values are owning `Any` objects. This
belongs in `tvm-ffi`, not in generated IR bindings.

### Object origins and deletion

The same Rust reference wrapper must accept both origins:

- a Rust allocation has a Rust object deleter in its header;
- a C++ allocation has a C++ object deleter;
- refcount operations always dispatch through the common header.

Generated safe direct construction must be disabled for incomplete or
polymorphic layouts. Creating a header-only Rust allocation and labeling it as
a larger C++ type is unsound even if field access is done through reflection.

### Consuming packed arguments

C++ pass APIs use `RValueRef<T>` at some boundaries. Rust still needs a reusable
safe owning packed-argument holder so a consumed Rust handle is encoded as an
rvalue rather than copied as an ordinary `AnyView`. This remains a runtime gap.

## Metadata gaps

The current experiment still needs explicit decisions for:

- publishing C++ `_type_final` in language-independent metadata;
- obtaining complete physical layout, including unreflected fields and vptr
  classification;
- constructor parameter nullability/defaults and validation semantics;
- enum names and values, including their underlying C++ width;
- `RValueRef<T>` code generation.

These gaps require metadata or reviewed overrides. They are not permission for
stubgen to guess.

## Structural-pass lessons

The IR binding and the pass algorithm are separate concerns:

- `structural_walk`/`structural_map` control recursion through pre/post order;
- `structural_visit`/`structural_mutate` let callbacks explicitly recurse;
- maps preserve keys and recursively transform values;
- FreeVar and DAG identity semantics come from registered structural metadata;
- `IRModule` cannot be rebuilt by blindly mapping reflected storage because
  `global_var_map_` is a derived index;
- a Rust pass claiming C++ parity must be checked with C++ structural equality,
  not only a few field assertions.

The example Rust passes remain prototype evidence. Stubgen should generate the
IR surface they consume, not generate those transformations.

## Recommended implementation order

1. Add a reliable C++-layout input and generate the minimal complete structs.
2. Generate direct constructors for `Var`, `IntImm`, `Add`, and `Evaluate`.
3. Keep `PrimType`, missing type, and `PrimFunc` as reviewed overrides.
4. Run `tests/stubgen_acceptance.rs` unchanged, including the C++-getter-on-Rust
   allocation test.
5. Generate container fields using `AnyMap` and the normal typed `Array`/`Map`
   wrappers.
6. Add the packed rvalue holder and pass boundaries separately.
7. Expand the type slice only after each new layout/constructor pattern has a
   focused test and a named owner.

The goal is not to preserve handwritten bindings. The goal is for generated
code to replace them while retaining the same cross-language ABI and pass
behavior.
