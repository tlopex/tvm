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
    if rg --no-ignore -n -U -P "$pattern" "$generated_dir" -g '*.rs'; then
        echo "$message" >&2
        exit 1
    fi
}

fail_on_match \
    "unsafe aliasing surface: generated ObjectRefs must not implement DerefMut" \
    '\bDerefMut\b'
fail_on_match \
    "generated objects cannot opt into Send or Sync without a reflected contract" \
    'unsafe\s+impl(?:\s*<[^>{}]*>)?\s+(?:Send|Sync)\s+for'
fail_on_match \
    "native Rust allocation is prohibited for reflected C++ objects" \
    'ObjectArc\s*(?:::\s*<[^;\n]+>)?\s*::\s*new(?:_with_extra_items)?\s*\('
fail_on_match \
    "generic reflected builders are prohibited; constructor invariants are unknown" \
    '\b(?:ffi_new_unchecked|build_unchecked|get_kwargs_object)\b'
fail_on_match \
    "pointer identity belongs on ObjectRefCore, not every generated type" \
    'pub\s+fn\s+same_as\s*<'
fail_on_match \
    "checked casts belong on ObjectRefCast, not every generated type" \
    'pub\s+fn\s+downcast\s*<'
fail_on_match \
    "generated Object node structs must not expose mirrored data fields" \
    '(?s)pub struct [A-Za-z_][A-Za-z0-9_]*Obj\s*\{(?:(?!\n\}).)*?\n\s*pub(?:\([^\n)]*\))?\s+(?:r#)?[A-Za-z_][A-Za-z0-9_]*\s*:'
fail_on_match \
    "generated Object node structs may contain only the base prefix and thread marker" \
    '(?s)pub struct [A-Za-z_][A-Za-z0-9_]*Obj\s*\{(?:(?!\n\}).)*?\n\s*(?!(?:base|_not_send_sync)\s*:)(?:r#)?[A-Za-z_][A-Za-z0-9_]*\s*:'

object_count="$(rg --no-ignore -o 'pub struct [A-Za-z_][A-Za-z0-9_]*Obj\s*\{' "$generated_dir" -g '*.rs' | wc -l || true)"
marker_count="$(rg --no-ignore -o '_not_send_sync\s*:\s*PhantomData\s*<\s*Rc\s*<\s*\(\s*\)\s*>\s*>' "$generated_dir" -g '*.rs' | wc -l || true)"
if [[ "$object_count" == 0 ]]; then
    echo "generated tree contains no Object node declarations" >&2
    exit 1
fi
if [[ "$marker_count" != "$object_count" ]]; then
    echo "every generated Object node must default to !Send + !Sync" >&2
    echo "object nodes: $object_count; thread-safety markers: $marker_count" >&2
    exit 1
fi

getter_count="$(rg --no-ignore -o 'tvm_ffi::object::get_object_field\s*::' "$generated_dir" -g '*.rs' | wc -l || true)"
if [[ "$getter_count" == 0 ]]; then
    echo "generated fields must use owning reflection getters" >&2
    exit 1
fi

echo "generated safety checks passed: $object_count opaque !Send/!Sync objects, $getter_count reflection getters, no generic builders"
