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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

generated_dir="${1:-src/generated}"
if [[ ! -d "$generated_dir" ]]; then
    echo "generated Rust directory does not exist: $generated_dir" >&2
    exit 2
fi
if ! command -v rg >/dev/null 2>&1; then
    echo "required command not found: rg" >&2
    exit 2
fi

fail_on_match() {
    local message="$1"
    local pattern="$2"
    if rg -n -U -P "$pattern" "$generated_dir" -g '*.rs'; then
        echo "$message" >&2
        exit 1
    fi
}

fail_on_match \
    "unsafe aliasing surface: generated ObjectRefs must not implement DerefMut" \
    '\bDerefMut\b'
fail_on_match \
    "native Rust allocation is prohibited for reflected C++ objects" \
    'ObjectArc\s*(?:::\s*<[^;\n]+>)?\s*::\s*new(?:_with_extra_items)?\s*\('
fail_on_match \
    "safe self-consuming builders are prohibited; use unsafe build_unchecked" \
    'pub\s+fn\s+build(?:_unchecked)?\s*\(\s*(?:mut\s+)?self\b'
fail_on_match \
    "safe native new() returning an object is prohibited" \
    'pub\s+fn\s+new\s*\([^;{]*\)\s*->\s*(?:Self|Result\s*<\s*Self\s*>)\s*\{'
fail_on_match \
    "generated Object node structs must not expose mirrored data fields" \
    '(?s)pub struct [A-Za-z_][A-Za-z0-9_]*Obj\s*\{(?:(?!\n\}).)*?\n\s*pub(?:\([^\n)]*\))?\s+(?:r#)?[A-Za-z_][A-Za-z0-9_]*\s*:'

object_count="$(rg -o 'pub struct [A-Za-z_][A-Za-z0-9_]*Obj\s*\{' "$generated_dir" -g '*.rs' | wc -l || true)"
marker_count="$(rg -o '_not_send_sync\s*:\s*PhantomData\s*<\s*Rc\s*<\s*\(\s*\)\s*>\s*>' "$generated_dir" -g '*.rs' | wc -l || true)"
if [[ "$object_count" == 0 ]]; then
    echo "generated tree contains no Object node declarations" >&2
    exit 1
fi
if [[ "$marker_count" != "$object_count" ]]; then
    echo "every generated Object node must default to !Send + !Sync" >&2
    echo "object nodes: $object_count; thread-safety markers: $marker_count" >&2
    exit 1
fi

getter_count="$(rg -o 'tvm_ffi::object::get_object_field\s*::' "$generated_dir" -g '*.rs' | wc -l || true)"
if [[ "$getter_count" == 0 ]]; then
    echo "generated fields must use owning reflection getters" >&2
    exit 1
fi

unchecked_count="$(rg -o 'pub\s+unsafe\s+fn\s+build_unchecked\s*\(' "$generated_dir" -g '*.rs' | wc -l || true)"
all_build_unchecked_count="$(rg -o 'pub\s+(?:unsafe\s+)?fn\s+build_unchecked\s*\(' "$generated_dir" -g '*.rs' | wc -l || true)"
if [[ "$unchecked_count" == 0 || "$unchecked_count" != "$all_build_unchecked_count" ]]; then
    echo "generic reflected builders must expose only pub unsafe fn build_unchecked" >&2
    echo "unsafe builders: $unchecked_count; all build_unchecked methods: $all_build_unchecked_count" >&2
    exit 1
fi

echo "generated safety checks passed: $object_count opaque !Send/!Sync objects, $getter_count reflection getters, $unchecked_count unsafe builders"
