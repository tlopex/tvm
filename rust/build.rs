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

//! Configure link and loader paths from the TVM build itself, so the Rust
//! bindings and `libtvm_compiler` use the same `libtvm_ffi`.
use std::env;
use std::path::{Path, PathBuf};

fn resolve_from_manifest(manifest_dir: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        manifest_dir.join(path)
    }
}

fn push_unique(dirs: &mut Vec<PathBuf>, dir: PathBuf) {
    if !dirs.contains(&dir) {
        dirs.push(dir);
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=TVM_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=TVM_COMPILER_LIB");
    println!("cargo:rerun-if-env-changed=TVM_TOOLCHAIN_LIB_DIR");

    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by Cargo"),
    );
    let compiler_lib = env::var_os("TVM_COMPILER_LIB")
        .map(PathBuf::from)
        .map(|path| resolve_from_manifest(&manifest_dir, path));
    let build_lib_dir = if let Some(compiler_lib) = compiler_lib.as_ref() {
        compiler_lib
            .parent()
            .expect("TVM_COMPILER_LIB must include a parent directory")
            .to_path_buf()
    } else {
        let build_dir = env::var_os("TVM_BUILD_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("../build"));
        resolve_from_manifest(&manifest_dir, build_dir).join("lib")
    };

    let mut dirs = Vec::new();
    if build_lib_dir.is_dir() {
        println!("cargo:rustc-link-search=native={}", build_lib_dir.display());
        push_unique(&mut dirs, build_lib_dir.clone());
    } else {
        println!(
            "cargo:warning=TVM library directory `{}` does not exist; cargo check can proceed, but linking or running requires a configured TVM build (set TVM_BUILD_DIR)",
            build_lib_dir.display()
        );
    }
    if let Some(compiler_lib) = compiler_lib {
        if !compiler_lib.is_file() {
            println!(
                "cargo:warning=TVM_COMPILER_LIB `{}` does not exist",
                compiler_lib.display()
            );
        }
    }

    if let Some(toolchain_lib_dir) = env::var_os("TVM_TOOLCHAIN_LIB_DIR") {
        let toolchain_lib_dir =
            resolve_from_manifest(&manifest_dir, PathBuf::from(toolchain_lib_dir));
        if toolchain_lib_dir.is_dir() {
            println!(
                "cargo:rustc-link-search=native={}",
                toolchain_lib_dir.display()
            );
            push_unique(&mut dirs, toolchain_lib_dir);
        } else {
            println!(
                "cargo:warning=TVM_TOOLCHAIN_LIB_DIR `{}` does not exist and will be ignored",
                toolchain_lib_dir.display()
            );
        }
    }

    let loader_var = match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("windows") => "PATH",
        Ok("macos") => "DYLD_LIBRARY_PATH",
        _ => "LD_LIBRARY_PATH",
    };
    println!("cargo:rerun-if-env-changed={loader_var}");

    if let Some(previous) = env::var_os(loader_var) {
        for dir in env::split_paths(&previous) {
            push_unique(&mut dirs, dir);
        }
    }

    if !dirs.is_empty() {
        match env::join_paths(&dirs) {
            Ok(paths) => println!(
                "cargo:rustc-env={loader_var}={}",
                paths.to_string_lossy()
            ),
            Err(err) => println!(
                "cargo:warning=failed to construct {loader_var} from configured library paths: {err}"
            ),
        }
    }
}
