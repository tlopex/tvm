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

set -euo pipefail

generated_dir="${1:-src/generated}"

if [[ ! -d "$generated_dir" ]]; then
    echo "generated Rust directory does not exist: $generated_dir" >&2
    exit 2
fi

if rg -n 'impl DerefMut for' "$generated_dir"; then
    echo "unsafe aliasing surface: generated ObjectRefs must not implement DerefMut" >&2
    exit 1
fi

safe_native_ctors="$(rg -n -U 'pub fn new\([^;{]*\)\s*->\s*Self\s*\{' "$generated_dir" -g '*.rs' || true)"
if [[ -n "$safe_native_ctors" ]]; then
    echo "$safe_native_ctors" >&2
    echo "native allocation returning Self must be new_unchecked or disabled; safe new() is prohibited" >&2
    exit 1
fi

if rg -n 'type_index == <N as tvm_ffi::ObjectCore>::type_index' "$generated_dir"; then
    echo "downcast must use subtype-aware runtime ancestry" >&2
    exit 1
fi

if rg -n -U 'let raw = ObjectArc::as_raw\([^;]+;\n\s*let header' "$generated_dir" -g '*.rs'; then
    echo "downcast must return None before dereferencing a null ObjectRef" >&2
    exit 1
fi

echo "generated safety checks passed: no DerefMut, no safe native allocation, null-safe subtype-aware downcast"
