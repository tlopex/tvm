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
#
# macOS wheel repair: run delocate, then ad-hoc re-sign every Mach-O and repack.
#
# delocate / install_name_tool edits invalidate a Mach-O's code signature. On
# Apple Silicon (arm64) dyld refuses to load an invalidly-signed library and
# SIGKILLs the process, so `import tvm` dies with no Python traceback. delocate
# does not re-sign libraries whose dependencies it skips (here libtvm_runtime
# depends on the excluded libtvm_ffi), so they ship with a stale signature.
# Re-sign every .dylib/.so in the repaired wheel ad-hoc, then repack so the
# RECORD hashes match the re-signed files.
#
# cibuildwheel invokes this as the macOS repair-wheel-command, run from the
# project root, with:  $1 = {dest_dir}   $2 = {wheel}   $3 = {delocate_archs}
set -euxo pipefail

dest_dir="$1"
wheel="$2"
archs="$3"

python -m pip install -q wheel

repaired_dir="$(mktemp -d)"
delocate-wheel --ignore-missing-dependencies --exclude libtvm_ffi.dylib \
  --require-archs "${archs}" -w "${repaired_dir}" -v "${wheel}"
repaired_wheel="$(ls "${repaired_dir}"/*.whl)"

unpack_dir="$(mktemp -d)"
python -m wheel unpack "${repaired_wheel}" -d "${unpack_dir}"

# Ad-hoc re-sign every Mach-O so dyld accepts it on arm64. Re-signing an
# already-valid library is idempotent, so signing all of them is safe.
find "${unpack_dir}" -type f \( -name '*.dylib' -o -name '*.so' \) -print0 \
  | xargs -0 -n1 codesign --force --sign -

# Repack: regenerates RECORD with the post-signing file hashes.
python -m wheel pack "${unpack_dir}"/*/ -d "${dest_dir}"
