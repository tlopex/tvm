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

# One-click regeneration of src/generated/ (rust_cpusim_plan.md §四 / BLOCKERS C2).
#
# Usage: from an env with editable tvm-ffi (rust_stubgen branch) on PATH:
#     cd ~/tvm/rust && ./regen.sh
#
# Stamps src/generated/STAMP with the tvm / tvm-ffi commits the mirrors were
# generated against — the #[repr(C)] layouts are only valid against that exact
# libtvm_compiler.so build.
set -euo pipefail
cd "$(dirname "$0")"

command -v tvm-ffi-stubgen >/dev/null || {
    echo "tvm-ffi-stubgen not on PATH (activate the env with editable tvm-ffi)" >&2
    exit 1
}

export LD_LIBRARY_PATH="${CONDA_PREFIX:-/nonexistent}/lib:$PWD/../build/lib:${LD_LIBRARY_PATH:-}"

for pfx in ir tirx target transform instrument arith; do
    echo "== stubgen prefix: $pfx"
    tvm-ffi-stubgen src/generated --target rust \
        --dlls ../build/lib/libtvm_compiler.so \
        --init-lib tvm_compiler --init-pypkg tvm --init-prefix "$pfx."
done

TVM_FFI_DIR="$(dirname "$(dirname "$(command -v tvm-ffi-stubgen)")")"
{
    echo "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "tvm:       $(git -C .. rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "tvm-ffi:   $(git -C "$HOME/tvm-ffi" rev-parse --short HEAD 2>/dev/null || echo unknown)"
} > src/generated/STAMP
echo "== stamped:"
cat src/generated/STAMP
