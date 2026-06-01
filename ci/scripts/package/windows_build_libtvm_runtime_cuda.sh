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
# Build tvm_runtime_cuda.dll on a Windows runner, run by the build_cuda_runtime CI
# job (on the host; unlike Linux there is no build container on Windows). Installs
# the pinned CUDA toolkit via conda and builds the sidecar into build-wheel-cuda/lib/
# for the wheel build to bundle. Windows mirror of manylinux_build_libtvm_runtime_cuda.sh.
#
# Usage: windows_build_libtvm_runtime_cuda.sh
set -euxo pipefail

# Keep a unix-style path for bash file operations and a mixed (forward-slash)
# path for CMake/cmd, which dislike the /c/... msys form.
repo_root_unix="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
repo_root="$(cygpath -m "${repo_root_unix}")"
build_dir_unix="${repo_root_unix}/build-wheel-cuda"
build_dir="$(cygpath -m "${build_dir_unix}")"
cuda_prefix_unix="/c/opt/cuda"
cuda_prefix="C:/opt/cuda"

# Locate conda: this script runs under a non-login bash, so conda may not be on
# PATH even though the runner ships Miniconda (exposed via $CONDA).
conda_exe="$(command -v conda || true)"
if [[ -z "${conda_exe}" ]]; then
  conda_exe="${CONDA:-/c/Miniconda}/Scripts/conda.exe"
fi

# Install the pinned CUDA toolkit via conda from the nvidia channel, mirroring the
# LLVM-via-conda install used elsewhere in the publish action. The win-64 channel
# caps at 13.0.x, matching the Linux hook's CUDA 13.0.2.
if [[ ! -e "${cuda_prefix_unix}/Library/bin/nvcc.exe" ]]; then
  "${conda_exe}" create -q -p "${cuda_prefix}" -c nvidia/label/cuda-13.0.2 cuda-toolkit -y \
    || "${conda_exe}" create -q -p "${cuda_prefix}" -c nvidia/label/cuda-13.0.2 cuda-toolkit --use-only-tar-bz2 -y
fi

# conda lays the Windows toolkit out under <prefix>/Library (bin/nvcc.exe,
# lib/x64/cudart.lib, include/...). Discover the root from nvcc.exe so TVM's
# FindCUDA MSVC branch resolves against the real layout instead of a hardcode.
nvcc_unix="$(find "${cuda_prefix_unix}" -iname nvcc.exe | head -n1)"
test -n "${nvcc_unix}"
nvcc_exe="$(cygpath -m "${nvcc_unix}")"
cuda_root="$(cygpath -m "$(dirname "$(dirname "${nvcc_unix}")")")"   # <prefix>/Library
export CUDA_PATH="${cuda_root}"

python -m pip install -U pip cmake ninja
"${nvcc_exe}" --version

# nvcc needs the MSVC host compiler (cl.exe) on PATH, but this bash is not a VS
# Developer shell. Locate VS via vswhere and run the cmake configure+build inside
# vcvars64.
vswhere="C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
vs_path="$("${vswhere}" -latest -products '*' \
  -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
  -property installationPath | tr -d '\r')"
test -n "${vs_path}"
vcvars="${vs_path}\\VC\\Auxiliary\\Build\\vcvars64.bat"

rm -rf "${build_dir_unix}"

# CMAKE_CUDA_COMPILER only tells CMake which nvcc to use (load-bearing here: the
# conda nvcc is not on PATH); it does not affect the resulting tvm_runtime_cuda.dll,
# which is built only from .cc host sources (no .cu device code). CMAKE_CUDA_ARCHITECTURES
# is intentionally not set -- a no-op for the same reason, and modern CMake fills a
# default. -allow-unsupported-compiler guards against the runner's MSVC being newer
# than the CUDA toolkit officially supports.
cmd_script="$(mktemp --suffix=.bat)"
cat > "${cmd_script}" <<EOF
call "${vcvars}" || exit /b 1
cmake -S "${repo_root}" -B "${build_dir}" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_TESTING=OFF ^
  -DTVM_BUILD_PYTHON_MODULE=ON ^
  -DUSE_CUDA="${cuda_root}" ^
  -DUSE_LLVM=OFF ^
  -DUSE_CUBLAS=OFF -DUSE_CUDNN=OFF -DUSE_CUTLASS=OFF -DUSE_NCCL=OFF -DUSE_NVTX=OFF ^
  -DCMAKE_CUDA_COMPILER="${nvcc_exe}" ^
  -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler" || exit /b 1
cmake --build "${build_dir}" --target tvm_runtime tvm_runtime_cuda --config Release || exit /b 1
EOF
cmd //C "$(cygpath -w "${cmd_script}")"
rm -f "${cmd_script}"

cuda_lib_unix="${build_dir_unix}/lib/tvm_runtime_cuda.dll"
test -f "${cuda_lib_unix}"
# No patchelf/rpath step on Windows; delvewheel vendors dependencies at repair time.
echo "CUDA runtime: ${build_dir}/lib/tvm_runtime_cuda.dll"
