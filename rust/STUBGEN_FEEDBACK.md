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
tests. Its purpose is to establish what a one-command, Rust-native stubgen must
generate, what belongs in `tvm-ffi`, and which C++ declarations must be changed
before they can be safely constructed by Rust.

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
    pub a: Expr,
    pub b: Expr,
}

pub fn from_complete_fields(
    span: Option<Span>,
    ty: Type,
    a: Expr,
    b: Expr,
) -> Add {
    Add {
        data: crate::abi::allocate_object(AddObj {
            base: ExprObj::new(span, ty),
            a,
            b,
        }),
    }
}

pub fn new(a: &Expr, b: &Expr) -> Result<Add> {
    // Validate the same dtype rule as C++, then delegate to the owned path.
    let result_type = matching_binary_type(a, b)?;
    Ok(from_complete_fields(
        None,
        result_type,
        a.clone(),
        b.clone(),
    ))
}
```

The guarded allocator requires generated `ObjectLayout` evidence plus an
explicit `RustAllocatable` marker and then uses `ObjectArc::new` to install a
common `TVMFFIObject` header and a Rust deleter.
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
- open integer newtypes for native enum fields, with named associated constants
  and raw conversion; a Rust `enum` is unsound when a newer C++ library can
  produce an enumerator unknown to the generated crate; conversions from a
  wider integer check only that the value fits the native width and must not
  reject an otherwise representable unknown enumerator;
- the reference wrapper with `ObjectArc`, read-only `Deref`, casts, and
  upcasts;
- public physical fields, so `node.a` borrows through `Deref` and
  `node.a.clone()` is an explicit ownership decision by the caller;
- no cloning getter boilerplate and no second reflection-backed field path;
- no unconditional `DerefMut`, because two reference wrappers may share one
  allocation; mutation belongs in structural mutation or a checked COW API;
- no public standalone allocator for a behavior-only base: after replacing a
  C++ abstract vtable with a type-registered C ABI table, keep the base native
  constructor protected and publish no reflection creator; only concrete types
  that register the table may be allocated;
- a direct Rust constructor only when its full validation/default logic is
  known and reproduced;
- a lossless `from_complete_fields` path for every ABI-complete concrete node;
  names such as `with_span` and `with_metadata` are semantic convenience APIs,
  never substitutes for this uniform generated entry;
- internal visibility for that lossless allocator when raw fields encode an
  external identity invariant; for example, `Axis` exposes only `get(name)`
  publicly because a caller-chosen name/index pair could address the wrong C++
  registry entry;
- complete-construction parameters use the exact stored Rust field types and
  take them by value (`String`, `Array<T>`, `Option<T>`, and object handles),
  so moving a field performs no hidden clone, conversion, or allocation;
- a complete constructor variant that accepts optional metadata such as
  `Span`, with convenience constructors delegating to it;
- direct `Self` returns for complete-field allocation and other infallible
  Rust-only constructors; use `Result<Self>` only when generated code performs
  parsing, checked conversion, validation, casting, or a fallible ABI call;
- finality and structural metadata used by typed walk/map dispatch.
- allocation through a helper bounded by `ObjectLayout` and
  `RustAllocatable`, so neither an opaque wrapper nor a layout-only base prefix
  can accidentally be constructed merely because it has an object header;
- a generated `ConstructorRecipe` contract for each preparation-backed
  constructor. It records input arity and the complete returned field-key set,
  and the runtime helper rejects a native table that differs instead of
  silently ignoring new derived state.

The generator cannot recover complete physical layout from reflection alone.
It needs a layout source derived from C++ declarations or an explicitly curated
schema. Reflection is still the authority for field schemas, flags, and the
language-independent structural protocol.

## One-invocation input and output contract

One-command generation must consume a merged manifest with four independently
auditable sources.  No source is allowed to fill in another source's missing
facts by guessing:

| Manifest section | Required facts | Authority |
| --- | --- | --- |
| Runtime type | type key, parent chain, reflected field names/schemas/flags, structural attributes | TVM FFI registry |
| Native layout | finality, total size/alignment, ordered physical fields, exact C++ widths/offsets, explicit vptr/STL/unreflected blockers, and a stable per-type fingerprint | C++ build-generated layout data plus a matching runtime-published fingerprint |
| Rust mapping | object/reference names, module path, exact Rust type for every physical field, nullable/container mapping, and upcasts | reviewed mapping rules |
| Constructor recipe | public name, ordered inputs, defaults, failure conditions, normalized/derived fields, and either a Rust template or a named C ABI preparation table | reviewed constructor semantics |

For each type the generator first joins these sections by type key, verifies
that the parent layout and every reflected field agree, and chooses exactly one
outcome:

1. **complete:** emit the `#[repr(C)]` object, reference wrapper, read-only
   `Deref`, layout evidence, casts, and owned `from_complete_fields`;
2. **complete plus semantic:** additionally emit reviewed convenience
   constructors that validate/prepare and delegate to the owned allocator; or
3. **blocked:** emit only a safe opaque reference wrapper and a diagnostic that
   names the missing layout fact, recipe fact, or ABI blocker.

The generated `from_complete_fields` signature is mechanical: flatten the
stored fields from the rootmost supported base to the concrete node and keep
each class's native declaration order. Parameter names and Rust types match the
generated Rust fields after deterministic identifier sanitization (`span, ty,
a, b` for `Add`; native `global_var_map_` becomes Rust `global_var_map`), and
each parameter moves exactly once into the object. It contains no `clone`,
`String::from`, `Array::new`, packed call, validation, or `Result`. Conversions
and semantic argument order belong in a separately named convenience
constructor.

Field nullability belongs to the field/constructor contract, not to the C++
reference wrapper in isolation.  C++ commonly gives an `ObjectRef` type a
default undefined state, while nodes still require a defined value in a field
of that type.  An explicit `ffi::Optional<T>` schema maps to Rust `Option<T>`;
a plain object field stays non-optional and its semantic constructor recipe
must enforce any required-value invariant.

The invocation is successful only if every requested type reaches one of those
three explicit outcomes, generated files are deterministic, and the emitted
coverage manifest records the source and status of every field, constructor,
enum, behavior table, and blocker.  An omitted type or silently generated
packed-constructor fallback is a generation error.

The generated crate is valid only for a library that publishes the same
per-type layout fingerprints.  Runtime support must check each fingerprint
before the first Rust allocation of that type and return an error on a missing
or mismatched value.  Matching only type keys is insufficient: the same key may
survive a field, alignment, base-layout, or finality change that would make
Rust allocation unsafe.

## Integration with the existing stubgen

This should extend `tvm-ffi-stubgen`, not create a second TVM-only generator.
The existing pipeline already:

- loads the requested shared libraries from `--dlls`;
- collects registered type keys and global functions;
- constructs language-neutral `ObjectInfo`/`FuncInfo` values;
- topologically sorts objects by runtime inheritance; and
- delegates rendering to a pluggable language `Generator`.

Today only `PythonGenerator` is registered and the CLI restricts `--target` to
`python`. The Rust implementation should register `RustGenerator`, add `.rs`
marker/scaffolding support, and remove that hard-coded target restriction. The
language-neutral object model must be extended rather than bypassed: current
`ObjectInfo` keeps field schemas but drops offsets, sizes, alignments, flags,
defaults, and total size that are already available on runtime `TypeInfo`.
Rust generation needs those facts plus the independently generated native
layout/finality/blocker section and constructor recipes described above.

The intended command shape is therefore one existing stubgen invocation, for
example:

```text
tvm-ffi-stubgen rust/src --dlls build/libtvm_compiler.so --target rust \
  --native-layout build/rust_native_layout.json \
  --constructor-recipes rust/constructor_recipes.json
```

The exact option names can follow the upstream CLI review, but the ownership is
fixed: runtime registry collection remains shared, native layout and semantic
recipes are merged before rendering, and `RustGenerator` only emits code. It
must not rediscover C++ layout or constructor semantics while formatting Rust.

## Constructor classification

Direct Rust allocation is the target. Classification determines what input the
generator needs; it does not authorize a packed C++ constructor fallback:

| Class | Examples | Generated behavior |
| --- | --- | --- |
| Plain data node | `Span`, `Range`, `Var`, `IntImm`, `Add`, `Evaluate`, `VarBinding`, `SBlock` | complete layout and direct Rust allocation |
| Plain node with local validation | integer literals, binary ops, `SeqStmt`, `SBlockRealize` | direct allocation plus equivalent Rust validation |
| Shared registry metadata | `Axis` | call the registered C ABI constructor-preparation table, then allocate the final handle in Rust |
| Cross-origin value key | `SourceName` | allocate in Rust and define equality/hash by immutable value rather than allocation identity or a language-local cache |
| Former C++ polymorphic base | `Layout`, `PrimExprConvertible`, `DataProducer` | remove the C++ vptr and dispatch behavior through an opaque C ABI vtable type attribute |
| Former hidden STL storage | `Source` | replace it with an ABI-shareable field representation and allocate in Rust |
| Complex semantic constructor | `PrimFunc`, Relax `Function`, match buffer | call the direct C ABI preparation table for analysis/validation, then allocate complete fields in Rust |
| Build-dependent defaults | `BufferType` | read normalized physical fields through its C ABI preparation table, then allocate in Rust |
| Derived mutable state | `IRModule` | rebuild and validate derived indexes in generated Rust code |

`Type::Missing`, `PrimType`, `TupleType`, `For`, `BufferType`, buffer load/store,
Relax `Tuple`, and `IRModule` now demonstrate that these formerly classified
exceptions can be Rust-native once their semantics are made explicit.

## Language-neutral behavior and constructor preparation

The handwritten IR constructors no longer call packed global constructors or
packed migration helpers. Four direct C ABI tables cover the cases that cannot
be expressed as plain field initialization:

| Type attribute | Purpose | Final allocation |
| --- | --- | --- |
| `__constructor_vtable__` | validate inputs and return a uniform `Map<String, Any>` of derived physical fields for `Axis`, `BufferType`, `PrimFunc`, Relax `Function`, and `MatchBufferRegion` | Rust |
| `__to_prim_expr__` | convert `IterVar`, `BufferRegion`, or `Tensor` through one direct function pointer | conversion result follows the shared owning-result ABI |
| `__layout_vtable__` | dispatch all `Layout` operations without a C++ virtual table or `ffi.Function` | Rust-created layouts remain ordinary FFI objects |
| `__data_producer_vtable__` | dispatch shape, element type, and name queries for `DataProducer` subclasses without a C++ virtual table | result follows the shared owning-result ABI |

Each type attribute stores an opaque pointer to a C-compatible struct whose
entries are ABI function pointers. Rust or a nonvirtual native base shim finds
the column and calls the pointer directly; Rust uses
`TVMFFIGetTypeAttrColumn`. C++ may implement a registered compiler algorithm,
but neither the object layout nor the call boundary depends on the C++ object
ABI. A constructor `prepare` hook is not an allocator: it returns
validation/derived data and Rust performs the only final
node allocation. Its vtable also publishes the exact argument count, while its
result always uses physical field names such as `ret_type`, `ret_ty`, or the
inherited `ty`. This lets generated constructors share one decoding template
instead of embedding a different return convention for every node type.

The distinction is about the boundary, not the implementation file extension:
a hook implemented in `.cc` is acceptable when its public table is declared in
a header that also compiles as C and its signature contains only C ABI values.
Crossing the boundary through a C++ virtual method, C++ reference, STL object,
exception, or packed `ffi.Function` is not acceptable. The tests exercise both
directions: Rust calls registered tables directly, and C++ entry points receive
Rust-created objects and dispatch through the same tables.

Calls that execute passes, structural protocols, and container runtime
operations remain cross-language services. They are not hidden implementations
of generated IR constructors.

Moving a native virtual interface to one of these tables is a coordinated C++
ABI migration, not work that Rust stubgen can perform by itself.  Every
concrete native subclass must register all required table entries, and an
out-of-tree subclass that previously overrode the virtual methods must migrate
at the same time.  The table makes future objects language-neutral; it does
not preserve binary compatibility with the former C++ vtable layout.

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

Generated safe direct construction must be disabled for incomplete layouts.
Creating a header-only Rust allocation and labeling it as a larger native type
is unsound even if field access is done through reflection. A native virtual
base must first be converted to an explicit C ABI behavior table, as done for
`Layout`, `PrimExprConvertible`, and `DataProducer`. Removing the virtual
functions is not enough by itself: the former abstract base must remain
non-instantiable, otherwise reflection can create an empty object that has no
behavior table.

### Consuming packed arguments

C++ pass APIs use `RValueRef<T>` at some boundaries. Rust still needs a reusable
safe owning packed-argument holder so a consumed Rust handle is encoded as an
rvalue rather than copied as an ordinary `AnyView`. This remains a runtime gap.

## Metadata gaps

The current experiment still needs explicit decisions for:

- publishing C++ `_type_final` in language-independent metadata;
- obtaining complete physical layout, including unreflected fields and
  identifying native vptr/STL blockers and C++ base-tail-padding reuse that
  require an ABI refactor or an opaque binding;
- publishing the corresponding per-type layout fingerprint at runtime so a
  generated crate cannot allocate against a different native layout;
- constructor parameter type schemas, names, nullability/defaults, and
  validation semantics; preparation output shape is now fixed, but stubgen
  still needs machine-readable input recipes and the set of derived field keys;
- build-configuration values used by constructor defaults;
- enum names and values, including their underlying C++ width; generated
  bindings represent these as open integer newtypes rather than closed Rust
  enums;
- `RValueRef<T>` code generation.

These gaps require metadata, reviewed recipes, or an ABI refactor. They are not
permission for stubgen to guess or to hide allocation behind a packed global.

## Layout evidence

Each ABI-complete Rust object implements the hidden `ObjectLayout` contract.
The current conformance test compares total size and every reflected direct
field's offset, size, and alignment with native runtime metadata, and checks
Rust alignment against the value implied by the parent and those fields. That
is enough to catch errors in the selected handwritten slice, whose physical
members are all registered, but it is not authoritative generation input:
runtime metadata does not publish type-level C++ alignment or prove that no
member is unreflected. The build-generated native-layout section must supply
those facts and explicitly mark vptr/STL/unreflected blockers.

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
2. Always generate a lossless complete-field Rust allocator for ABI-complete
   nodes.
3. Generate semantic recipes first for `Type::Missing`, `PrimType`, `Var`,
   `IntImm`, `Add`, `Evaluate`, `For`, buffers, tuples, and modules.
4. Run `tests/stubgen_acceptance.rs` unchanged, including the C++-getter-on-Rust
   allocation test.
5. Generate container fields using `AnyMap` and the normal typed `Array`/`Map`
   wrappers.
6. Generate direct C ABI behavior/preparation-table calls for reviewed recipes,
   and reject unrefactored vptr/STL-backed nodes.
7. Add the packed rvalue holder and pass boundaries separately.
8. Expand the type slice only after each new layout/constructor pattern has a
   focused test and a named owner.

The goal is not to preserve handwritten bindings. The goal is for one stubgen
invocation to replace them with Rust structs and Rust allocation code while
retaining the same cross-language ABI and pass behavior. Packed calls may still
run compiler services and passes; they must not be the hidden implementation of
ordinary generated IR constructors. The handwritten slice demonstrates the
intended Rust surface and cross-language behavior. It does not yet supply every
one-command generator input listed in the metadata gaps; the next stubgen step
is to emit the same code from an authoritative native-layout and
constructor-recipe manifest rather than preserve the handwritten files.
