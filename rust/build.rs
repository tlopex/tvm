/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

//! Re-emit loader paths so plain `cargo run` finds libtvm_ffi (from the
//! tvm-ffi python install) and the conda libstdc++ that ~/tvm's libraries
//! were built against (CXXABI_1.3.15).
use std::env;
use std::process::Command;

fn main() {
    let mut dirs: Vec<String> = Vec::new();

    // libtvm_ffi from the active tvm-ffi install (needs the venv activated).
    if let Ok(out) = Command::new("tvm-ffi-config").arg("--libdir").output() {
        if out.status.success() {
            dirs.push(String::from_utf8(out.stdout).unwrap().trim().to_string());
        }
    }
    // conda toolchain's libstdc++ (gcc-15) used to build ~/tvm.
    let home = env::var("HOME").unwrap_or_default();
    let conda_lib = format!("{home}/miniforge3/envs/tvm-build-venv/lib");
    if std::path::Path::new(&conda_lib).exists() {
        dirs.push(conda_lib);
    }
    // ~/tvm/build/lib itself (libtvm_compiler.so and friends).
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
    dirs.push(format!("{manifest}/../build/lib"));

    let loader_var = match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => "PATH",
        Ok("macos") => "DYLD_LIBRARY_PATH",
        _ => "LD_LIBRARY_PATH",
    };
    let sep = if loader_var == "PATH" { ";" } else { ":" };
    let prev = env::var(loader_var).unwrap_or_default();
    if !prev.is_empty() {
        dirs.push(prev);
    }
    println!("cargo:rustc-env={loader_var}={}", dirs.join(sep));
}
