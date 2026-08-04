#!/usr/bin/env bash
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Generate into a staging directory, validate it, then either compare with the
# checked-in tree (default) or replace that tree (`--write`).  A broken backend
# can no longer leave src/generated half-overwritten.
set -euo pipefail

cd "$(dirname "$0")"

mode="${1:---check}"
if [[ "$mode" != "--check" && "$mode" != "--write" ]]; then
    echo "usage: $0 [--check|--write]" >&2
    exit 2
fi

stubgen="${TVM_FFI_STUBGEN:-tvm-ffi-stubgen}"
compiler_lib="${TVM_COMPILER_LIB:-$PWD/../build/lib/libtvm_compiler.so}"
generator_source="${TVM_FFI_SOURCE_DIR:-}"

command -v "$stubgen" >/dev/null || {
    echo "Rust stub generator not found: $stubgen (set TVM_FFI_STUBGEN)" >&2
    exit 1
}
if [[ ! -f "$compiler_lib" ]]; then
    echo "TVM compiler library not found: $compiler_lib (set TVM_COMPILER_LIB)" >&2
    exit 1
fi
if ! "$stubgen" --help 2>&1 | rg -q -- '--target.*rust|rust.*--target'; then
    echo "$stubgen does not advertise the fork-only Rust backend" >&2
    exit 1
fi

work_dir="$(mktemp -d "$PWD/src/.rust-stubgen.XXXXXX")"
candidate="$work_dir/generated"
mkdir -p "$candidate"
cleanup() {
    rm -rf -- "$work_dir"
}
trap cleanup EXIT

compiler_dir="$(dirname "$compiler_lib")"
loader_paths="$compiler_dir"
if [[ -n "${TVM_TOOLCHAIN_LIB_DIR:-}" ]]; then
    loader_paths="$loader_paths:$TVM_TOOLCHAIN_LIB_DIR"
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    loader_paths="$loader_paths:$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH="$loader_paths"

for prefix in ir tirx target transform instrument arith; do
    echo "generating prefix: $prefix"
    "$stubgen" "$candidate" --target rust \
        --dlls "$compiler_lib" \
        --init-lib tvm_compiler --init-pypkg tvm --init-prefix "$prefix."
done

# This intentionally rejects the current fork's safe native builders.  Fixing
# the backend is required before `--write` can update checked-in bindings.
./check_generated_safety.sh "$candidate"

tvm_commit="$(git -C .. rev-parse HEAD 2>/dev/null || echo unknown)"
generator_commit="unknown"
if [[ -n "$generator_source" && -d "$generator_source/.git" ]]; then
    generator_commit="$(git -C "$generator_source" rev-parse HEAD 2>/dev/null || echo unknown)"
fi
compiler_sha256="$(sha256sum "$compiler_lib" | awk '{print $1}')"
{
    echo "tvm=$tvm_commit"
    echo "tvm_ffi_generator=$generator_commit"
    echo "compiler_sha256=$compiler_sha256"
    echo "prefixes=ir,tirx,target,transform,instrument,arith"
} > "$candidate/STAMP"

if [[ "$mode" == "--check" ]]; then
    if diff -qr src/generated "$candidate"; then
        echo "generated bindings are up to date"
    else
        echo "generated bindings differ; inspect with TVM_FFI_SOURCE_DIR set, then use --write" >&2
        exit 1
    fi
else
    previous="$work_dir/previous"
    mv src/generated "$previous"
    if ! mv "$candidate" src/generated; then
        mv "$previous" src/generated
        echo "failed to install staged bindings; restored previous tree" >&2
        exit 1
    fi
    echo "installed validated bindings in src/generated"
fi
