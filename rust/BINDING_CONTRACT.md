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

# Handwritten Rust binding contract

The C++ declaration and registered TVM FFI metadata are authoritative. A
handwritten Rust binding is accepted only when all applicable rules below hold.

## Object representation

- Every object wrapper has the exact native type key and parent chain.
- A Rust node struct exposes only the shared object header or a previously
  declared opaque parent prefix.
- Rust never allocates a header-only wrapper as though it were the complete
  C++ object.
- Object upcasts use tvm-ffi's standard zero-cost handle cast support.

## Construction

- Prefer an existing registered native constructor.
- For non-constructor behavior owned by a type, reuse its existing registered
  type method or type attribute.
- Reuse a registered global function for an existing native constructor or a
  compiler-wide service; do not introduce a global function for type-local
  behavior.
- Rust does not reproduce C++ validation, normalization, defaulting, or
  derived-field logic when the native constructor already owns it.
- The prototype must not add a private C ABI or a parallel constructor table.

## Field access

- A public field accessor resolves native reflection metadata by runtime type
  and field name, including inherited fields.
- It invokes the registered C ABI getter with the registry-provided offset;
  handwritten Rust code never defines or infers the C++ layout.
- The result is returned as an owning Rust FFI value, so callers cannot retain
  a dangling borrowed field pointer.
- Missing fields, incompatible dynamic types, getter errors, and conversion
  errors remain ordinary `Result` failures.

## Passes

- Use `structural_walk`/`structural_map` when the framework should recurse in a
  selected pre- or post-order.
- Use `structural_visit`/`structural_mutate` only when a callback needs to
  choose children or update state around a recursive call.
- Typed handlers use the standard `#[tvm_ffi::dispatch(...)]` mechanism.
- Definition/use identity and mutation remapping stay inside the structural
  protocol; passes do not rebuild that protocol themselves.

## Verification

Tests must cover behavior that crosses an important boundary:

1. native construction followed by typed reflected field access;
2. representative common IR, TIRx, buffer/block, and Relax objects;
3. structural walk/map order and replacement behavior;
4. callback-controlled visit/mutate behavior where used by a pass;
5. object identity remapping for definitions and uses;
6. Rust-backed TVM Pass execution through the existing function ABI;
7. invalid native constructor input and reflection conversion failures.

Tests must not assert private C++ sizes, offsets, alignment, or a Rust-owned
copy of native object layout. Those are implementation details of the model we
explicitly do not use.
