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

# Regenerate into a same-filesystem staging tree, validate that tree as an
# independent crate, and only then compare or install it.  The checked-in tree
# is never used as generator input.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_dir="$(cd "$script_dir/.." && pwd -P)"
cd "$script_dir"
generated_dir="$script_dir/src/generated"
runtime_source="$repo_dir/3rdparty/tvm-ffi"
runtime_crate="$runtime_source/rust/tvm-ffi"
compiler_lib="${TVM_COMPILER_LIB:-$repo_dir/build/lib/libtvm_compiler.so}"
prefixes=(ir tirx target transform instrument arith)
prefix_csv="$(IFS=,; printf '%s' "${prefixes[*]}")"
schema_version="tvm-ffi-reflection-rust-v6"

mode="${1:---check}"
if [[ "$mode" != "--check" && "$mode" != "--write" ]]; then
    echo "usage: $0 [--check|--write]" >&2
    exit 2
fi

for command_name in git rg diff cargo rustfmt; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "required command not found: $command_name" >&2
        exit 1
    fi
done
if [[ ! -f "$compiler_lib" ]]; then
    echo "TVM compiler library not found: $compiler_lib" >&2
    echo "set TVM_COMPILER_LIB to an absolute libtvm_compiler path" >&2
    exit 1
fi
if [[ ! -f "$runtime_crate/Cargo.toml" ]]; then
    echo "tvm-ffi Rust runtime submodule is missing: $runtime_crate" >&2
    exit 1
fi

# The compile gate consumes the runtime worktree directly.  Development
# regeneration is allowed from a dirty checkout, but the STAMP must say so
# explicitly instead of presenting the HEAD commit as exact provenance.
runtime_commit="$(git -C "$runtime_source" rev-parse 'HEAD^{commit}')"
runtime_dirty=false
if [[ -n "$(git -C "$runtime_source" status --porcelain=v1 --untracked-files=normal)" ]]; then
    runtime_dirty=true
    echo "warning: tvm-ffi runtime worktree is dirty; STAMP will record runtime_dirty=true" >&2
fi
cargo_runtime_commit="$(sed -nE '/^tvm-ffi = /s/.*rev = "([0-9a-f]{40})".*/\1/p' "$script_dir/Cargo.toml")"
if [[ "$cargo_runtime_commit" != "$runtime_commit" ]]; then
    echo "Cargo.toml tvm-ffi rev must match the runtime submodule HEAD" >&2
    echo "Cargo.toml: ${cargo_runtime_commit:-<missing>}; submodule: $runtime_commit" >&2
    exit 1
fi

work_dir="$(mktemp -d "$script_dir/src/.rust-stubgen.XXXXXX")"
candidate="$work_dir/generated"
backup="$work_dir/previous"
restore_needed=0
keep_work_dir=0
mkdir -p "$candidate"

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ "$restore_needed" == 1 && -d "$backup" ]]; then
        # A signal or failed rename during --write restores the exact previous
        # tree.  Preserve the work directory if restoration itself fails.
        if [[ -e "$generated_dir" ]]; then
            mv -- "$generated_dir" "$work_dir/failed-install" || keep_work_dir=1
        fi
        if ! mv -- "$backup" "$generated_dir"; then
            echo "failed to restore generated tree; recovery copy: $backup" >&2
            keep_work_dir=1
        fi
    fi
    if [[ "$keep_work_dir" == 0 ]]; then
        rm -rf -- "$work_dir"
    else
        echo "preserved regeneration workspace: $work_dir" >&2
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

generator_source="${TVM_FFI_SOURCE_DIR:-$runtime_source}"
generator_commit=""
python_bin="${TVM_FFI_PYTHON:-python3}"

generator_source="$(cd "$generator_source" && pwd -P)"
generator_build="${TVM_FFI_BUILD_DIR:-$generator_source/build}"
if [[ ! -d "$generator_build" ]]; then
    echo "local tvm-ffi build directory does not exist: $generator_build" >&2
    exit 1
fi
generator_build="$(cd "$generator_build" && pwd -P)"
if [[ ! -f "$generator_source/python/tvm_ffi/stub/cli.py" ]]; then
    echo "TVM_FFI_SOURCE_DIR is not a tvm-ffi source tree: $generator_source" >&2
    exit 1
fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python interpreter not found: $python_bin" >&2
    exit 1
fi
generator_commit="$(git -C "$generator_source" rev-parse 'HEAD^{commit}')"
generator_dirty=false
if [[ -n "$(git -C "$generator_source" status --porcelain=v1 --untracked-files=normal)" ]]; then
    generator_dirty=true
    echo "warning: tvm-ffi generator worktree is dirty; STAMP will record generator_dirty=true" >&2
fi
if [[ ! -f "$generator_build/lib/libtvm_ffi.so" ]]; then
    echo "local tvm-ffi runtime is not built: $generator_build/lib/libtvm_ffi.so" >&2
    echo "set TVM_FFI_BUILD_DIR to its configured build directory" >&2
    exit 1
fi
if ! compgen -G "$generator_build/core.*.so" >/dev/null; then
    echo "local tvm-ffi Python extension is not built under: $generator_build" >&2
    exit 1
fi

run_stubgen() {
    TVM_FFI_SOURCE="$generator_source" TVM_FFI_BUILD="$generator_build" \
        "$python_bin" - "$@" <<'PY'
import ctypes
import importlib.util
import os
import sys
from pathlib import Path

source = Path(os.environ["TVM_FFI_SOURCE"]).resolve()
build = Path(os.environ["TVM_FFI_BUILD"]).resolve()
package_dir = source / "python" / "tvm_ffi"

# An editable install can otherwise redirect this import to a different
# checkout.  Load the requested source and its already-built extension by
# absolute path instead.
sys.meta_path[:] = [
    finder
    for finder in sys.meta_path
    if type(finder).__name__ != "ScikitBuildRedirectingFinder"
]
package_spec = importlib.util.spec_from_file_location(
    "tvm_ffi",
    package_dir / "__init__.py",
    submodule_search_locations=[str(package_dir)],
)
if package_spec is None or package_spec.loader is None:
    raise RuntimeError(f"cannot load tvm_ffi package from {package_dir}")
package = importlib.util.module_from_spec(package_spec)
sys.modules["tvm_ffi"] = package

libinfo_spec = importlib.util.spec_from_file_location(
    "tvm_ffi.libinfo", package_dir / "libinfo.py"
)
if libinfo_spec is None or libinfo_spec.loader is None:
    raise RuntimeError("cannot load local tvm_ffi.libinfo")
libinfo = importlib.util.module_from_spec(libinfo_spec)
sys.modules["tvm_ffi.libinfo"] = libinfo
libinfo_spec.loader.exec_module(libinfo)
package.libinfo = libinfo

library = ctypes.CDLL(str(build / "lib" / "libtvm_ffi.so"), ctypes.RTLD_GLOBAL)
original_loader = libinfo.load_lib_ctypes
libinfo.load_lib_ctypes = lambda *args, **kwargs: library
core_paths = sorted(build.glob("core.*.so"))
if len(core_paths) != 1:
    raise RuntimeError(f"expected one local tvm_ffi core extension, got {core_paths}")
core_spec = importlib.util.spec_from_file_location("tvm_ffi.core", core_paths[0])
if core_spec is None or core_spec.loader is None:
    raise RuntimeError(f"cannot load {core_paths[0]}")
core = importlib.util.module_from_spec(core_spec)
sys.modules["tvm_ffi.core"] = core
core_spec.loader.exec_module(core)
package.core = core
package_spec.loader.exec_module(package)
libinfo.load_lib_ctypes = original_loader
package.core = core

from tvm_ffi.stub.cli import __main__

raise SystemExit(__main__())
PY
}

loader_paths="$(dirname "$compiler_lib")"
loader_paths="$generator_build/lib:$loader_paths"
if [[ -n "${TVM_TOOLCHAIN_LIB_DIR:-}" ]]; then
    loader_paths="$loader_paths:$TVM_TOOLCHAIN_LIB_DIR"
fi
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    loader_paths="$loader_paths:$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH="$loader_paths"

if ! stubgen_help="$(run_stubgen --help 2>&1)"; then
    echo "selected stubgen failed its --help preflight:" >&2
    printf '%s\n' "$stubgen_help" >&2
    exit 1
fi
if ! rg -q -- '--target' <<<"$stubgen_help" || \
    ! rg -q -- 'rust' <<<"$stubgen_help"; then
    echo "selected stubgen does not advertise the required Rust interface" >&2
    printf '%s\n' "$stubgen_help" >&2
    exit 1
fi

prefix_args=()
for prefix in "${prefixes[@]}"; do
    prefix_args+=(--init-prefix "$prefix.")
done
echo "generating dependency-closed prefixes: $prefix_csv"
run_stubgen "$candidate" --target rust \
    --dlls "$compiler_lib" \
    "${prefix_args[@]}"

mapfile -d '' rust_files < <(find "$candidate" -type f -name '*.rs' -print0 | sort -z)
if [[ "${#rust_files[@]}" == 0 ]]; then
    echo "stubgen produced no Rust files" >&2
    exit 1
fi
rustfmt --edition 2021 "${rust_files[@]}"
"$script_dir/check_generated_safety.sh" "$candidate"

rustfmt_version="$(rustfmt --version)"
generated_source_sha256="$(
    CANDIDATE_DIR="$candidate" "$python_bin" - <<'PY'
import hashlib
import os
from pathlib import Path

root = Path(os.environ["CANDIDATE_DIR"])
digest = hashlib.sha256()
for path in sorted(root.rglob("*.rs")):
    relative = path.relative_to(root).as_posix().encode()
    source = path.read_bytes()
    digest.update(len(relative).to_bytes(8, "big"))
    digest.update(relative)
    digest.update(len(source).to_bytes(8, "big"))
    digest.update(source)
print(digest.hexdigest())
PY
)"
{
    printf 'format_version=1\n'
    printf 'schema_version=%s\n' "$schema_version"
    printf 'generator_commit=%s\n' "$generator_commit"
    printf 'generator_dirty=%s\n' "$generator_dirty"
    printf 'runtime_commit=%s\n' "$runtime_commit"
    printf 'runtime_dirty=%s\n' "$runtime_dirty"
    printf 'generated_source_sha256=%s\n' "$generated_source_sha256"
    printf 'prefixes=%s\n' "$prefix_csv"
    printf 'rustfmt=%s\n' "$rustfmt_version"
} > "$candidate/STAMP"

# Compile only the staged modules in a fresh crate.  This prevents handwritten
# compatibility code in the main crate from hiding a broken generator output.
compile_dir="$work_dir/compile-check"
mkdir -p "$compile_dir/src"
ln -s "$candidate" "$compile_dir/src/generated"
ln -s "$script_dir/src/prim_expr.rs" "$compile_dir/src/prim_expr.rs"
{
    printf '[package]\nname = "tvm-generated-candidate-check"\n'
    printf 'version = "0.0.0"\nedition = "2021"\npublish = false\n\n'
    printf '[dependencies]\n'
    printf 'tvm-ffi = { path = "%s" }\n' "$runtime_crate"
    printf '\n[lib]\npath = "src/lib.rs"\n'
} > "$compile_dir/Cargo.toml"
{
    printf 'pub mod generated;\nmod prim_expr;\npub use prim_expr::PrimExpr;\n\n'
    printf '#[cfg(test)]\nmod runtime_smoke {\n'
    printf '    use super::generated::{ir, tirx};\n'
    printf '    use tvm_ffi::{DLDataType, DLDataTypeExt, Module, ObjectRefCast, Result};\n\n'
    printf '    #[test]\n'
    printf '    fn generated_int_imm_getters_match_native_ir() -> Result<()> {\n'
    printf '        let compiler = std::env::var("TVM_COMPILER_LIB")\n'
    printf '            .expect("TVM_COMPILER_LIB must be set by rust/regen.sh");\n'
    printf '        let _compiler = Module::load_from_file(&compiler)?;\n'
    printf '        let expected_dtype = DLDataType::try_from_str("int32")?;\n'
    printf '        let value = ir::int_imm(expected_dtype, 37, None)?\n'
    printf '            .expect("ir.IntImm returned an undefined object");\n'
    printf '        assert_eq!(value.value()?, 37);\n'
    printf '        let primitive_type: ir::PrimType = value.ty()?.downcast()?;\n'
    printf '        assert_eq!(primitive_type.dtype()?, expected_dtype);\n'
    printf '        let expr: ir::Expr = value.clone().into();\n'
    printf '        let evaluate = tirx::evaluate(Some(expr), None)?\n'
    printf '            .expect("tirx.Evaluate returned an undefined object");\n'
    printf '        let seq = tirx::seq_stmt(\n'
    printf '            tvm_ffi::Array::new(vec![Some(evaluate.clone().into()), Some(evaluate.into())]),\n'
    printf '            None,\n'
    printf '        )?\n'
    printf '            .expect("tirx.SeqStmt returned an undefined object");\n'
    printf '        assert_eq!(seq.seq()?.len(), 2);\n'
    printf '        Ok(())\n'
    printf '    }\n'
    printf '}\n'
} > "$compile_dir/src/lib.rs"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$script_dir/target}/stubgen-candidate" \
    TVM_FFI_LIB_DIR="$generator_build/lib" \
    RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-D warnings" \
    cargo check --quiet --manifest-path "$compile_dir/Cargo.toml" --all-targets
echo "independent candidate Cargo check passed"

CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$script_dir/target}/stubgen-candidate" \
    TVM_FFI_LIB_DIR="$generator_build/lib" \
    TVM_COMPILER_LIB="$compiler_lib" \
    RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-D warnings" \
    cargo test --quiet --manifest-path "$compile_dir/Cargo.toml" \
        runtime_smoke::generated_int_imm_getters_match_native_ir -- --exact
echo "generated IntImm runtime getter smoke passed"

if [[ "$mode" == "--check" ]]; then
    if diff -qr --no-dereference "$generated_dir" "$candidate"; then
        echo "generated bindings are byte-for-byte up to date"
    else
        echo "generated bindings differ; run $0 --write after reviewing the generator/schema" >&2
        exit 1
    fi
else
    restore_needed=1
    if [[ -e "$generated_dir" ]]; then
        mv -- "$generated_dir" "$backup"
    else
        restore_needed=0
    fi
    if ! mv -- "$candidate" "$generated_dir"; then
        echo "failed to install staged bindings; restoring previous tree" >&2
        exit 1
    fi
    restore_needed=0
    rm -rf -- "$backup"
    echo "installed validated bindings in $generated_dir"
fi
