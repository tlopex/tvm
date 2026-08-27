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

# Handwritten TVM Rust IR and pass prototype

This crate tests whether Rust analyses and passes can use TVM IR through the
existing language-independent TVM FFI protocols. Stub generation is not part
of this prototype.

The binding model is deliberately small:

- Rust object structs contain only the shared TVM FFI object prefix. They do
  not copy C++ object layouts or allocate native IR nodes themselves.
- Public constructors call TVM's existing registered native constructors.
  TVM therefore remains responsible for validation, normalization, derived
  fields, and allocation.
- Typed Rust accessors use registered reflection field getters. The metadata
  lookup is cached, while each returned object/value has normal FFI ownership.
- Concrete type behavior uses existing registered type methods or type
  attributes. Compiler-wide services use existing registered global
  functions.
- Framework-controlled passes use `structural_walk` and `structural_map`.
  Passes that must control recursion use `structural_visit` and
  `structural_mutate`.

The handwritten slice covers common IR, representative TIRx and Relax nodes,
buffers and blocks, analyses, expression/statement transformations, PrimFunc
passes, Relax function passes, and module passes. Its tests focus on native
construction, reflected access, structural recursion, definition/use identity,
and pass integration rather than private object layout.

Build TVM first, then run:

```bash
cargo test --manifest-path rust/Cargo.toml
```

By default tests load `build/lib/libtvm_compiler.so`. Set
`TVM_COMPILER_LIBRARY` to test another build.
