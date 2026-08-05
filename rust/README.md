<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Rust bindings and pass SDK

这个 crate 让 Rust 代码通过 `tvm-ffi` 使用 TVM IR。它包含两个边界不同的部分：

- `src/generated/` 是 stubgen 根据 TVM reflection schema 生成的绑定。
- `visitor.rs`、`mutator.rs`、`analyzer.rs`、`passes/` 和相关辅助代码是手写 pass SDK。

stubgen 只负责把已有 ABI 暴露成 Rust API。visitor 的 child role、definition scope、重写规则、
Analyzer 的状态约束和具体 pass 算法都不是 reflection 可以推导的内容，也不应算作 stubgen 的能力。

Rust 使用者不需要编写 C++，但这不是一套脱离 native TVM 的 IR 实现。TVM 的 reflection schema、
对象生命周期、constructor invariant、global function 和 pass runtime 仍由 C++/tvm-ffi 提供。

## Stubgen 生成什么

- 每个 IR 类型生成 opaque typed handle，而不是可由 Rust 分配的 C++ struct 镜像。
- reflected field 生成 `value.field() -> tvm_ffi::Result<T>` getter。getter 通过 runtime reflection
  返回 owning Rust 值，不公开可变字段。
- nullable ObjectRef 映射成 `Option<T>`；异构容器保留 owning `AnyValue`，不会把 scalar 强行
  转成 ObjectRef。
- 有可表达 schema 的 global 生成 typed wrapper；无法静态表达的 callable 使用明确的 packed
  fallback，而不是伪造函数签名。
- generated handle 默认不声称 `Send` 或 `Sync`。reflection 没有提供足够的线程安全证明。

stubgen 不生成 Rust-side object allocation、逐字段 builder、visitor、mutator、Analyzer 语义或
pass 算法。创建和重建 IR 必须调用 TVM 已注册的 native global，不能写 reflected field 来冒充
native constructor。

`src/prim_expr.rs` 也是手写 bridge：`PrimExpr` 是带 `PrimType` 约束的 `Expr` view，不是一个独立
runtime node。它被生成 API 引用，但它本身不属于 stubgen 输出。

详细的边界和验收说明见
[STUBGEN_PASS_EVALUATION.md](STUBGEN_PASS_EVALUATION.md)。

## 手写 pass SDK

pass SDK 在 generated bindings 之上提供可审查的编译器语义：

- visitor 明确列出要遍历的 Expr/Stmt child，而不是递归所有 ObjectRef 字段；
- mutator 把 Rust rewrite policy 接到 TVM 的 canonical transform/global；
- Analyzer wrapper 约束共享状态、constraint scope 和错误恢复；
- pass wrapper 负责 callback ownership、panic 转换和 native pass 调用。

这些能力要靠各自的行为测试证明。generated bindings 可以编译或 getter 可以读取，不等于某个
Rust pass 已与 C++ pass 语义等价。

## 构建和测试

开发时让 TVM、Python extension 和 Rust 链接同一套 `libtvm_ffi`：

```bash
cd rust
export TVM_BUILD_DIR=/path/to/tvm/build
# 仅在 TVM 使用非系统 libstdc++ 时设置：
export TVM_TOOLCHAIN_LIB_DIR=/path/to/toolchain/lib

cargo check --all-targets
cargo test --all-targets
cargo run --bin tirx_demo
```

也可以用 `TVM_COMPILER_LIB=/absolute/path/libtvm_compiler.so` 代替 `TVM_BUILD_DIR`。不要把另一
Python 环境的 `libtvm_ffi.so` 放到 TVM build 前面；相同 SONAME 不代表相同 ABI。

## 重新生成

`regen.sh` 从空的、同文件系统 staging 目录开始，在一次 Rust stubgen 调用中选择六个 schema
root：`ir`、`tirx`、`target`、`transform`、`instrument` 和 `arith`。Rust generator 的 schema
选择只需要这些 `--init-prefix`，不需要额外的初始化配置。

脚本默认使用仓库内的 tvm-ffi source/build，并通过绝对路径加载 Python package、
`libtvm_ffi.so` 和已构建的 extension：

```bash
cd rust
export TVM_FFI_SOURCE_DIR="$PWD/../3rdparty/tvm-ffi"
export TVM_FFI_BUILD_DIR="$TVM_FFI_SOURCE_DIR/build"
export TVM_FFI_PYTHON=/path/to/python
export TVM_COMPILER_LIB="$PWD/../build/lib/libtvm_compiler.so"
./regen.sh --check
```

生成后的唯一完整 E2E gate 是：

1. 对 fresh candidate 执行 `rustfmt` 和 generated-source safety scan。
2. 创建 fresh Cargo crate，放入 candidate generated tree 和实际必需的手写 `prim_expr.rs` bridge。
3. 以 `RUSTFLAGS=-D warnings` 对该 crate 执行 `cargo check --all-targets`。
4. 加载真实 `libtvm_compiler`，创建 `IntImm(int32, 37)`，并读取三类 reflected field：scalar
   `value()`、object `ty()`（再读取 `PrimType::dtype()`）和 `Array` `SeqStmt::seq()`。

这个 smoke 的目的很窄：证明当前生成的 scalar、object 和 `Array` getter 能通过真实 runtime
工作。它不证明所有字段、所有类型或任何 pass 算法正确。

`--check` 比较包含 `STAMP` 的完整目录，不写 checked-in tree；`--write` 只安装通过上述 gate 的
candidate，并对脚本可处理的错误和信号保留恢复路径。`STAMP` 记录 generator/runtime commit、
dirty 状态、generated Rust source hash、prefix 和 rustfmt 版本。dirty worktree 会被如实记录，
但不会被脚本强制拒绝。

不要对 TVM 或 tvm-ffi 执行 `pip install -e`。多 worktree 开发时，editable redirector 可能让
命令从另一 checkout 导入同名 package；`regen.sh` 会绕过该 redirector，但普通开发环境仍应避免
这种混用。
