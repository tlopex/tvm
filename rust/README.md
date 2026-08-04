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

这个 crate 是 TVM IR 的纯 Rust 使用层。Rust 代码通过 `tvm-ffi` ABI 持有对象、读取字段、
调用 global function 和实现 pass callback；使用者不需要写 C++。C++ 在这里仍然重要，但职责
只有两项：它是反射 schema 的权威来源，也是 TVM 对象和 pass 的 ABI/runtime 实现。

stubgen v2 **不再逐字段复制 C++ struct layout**。每个 generated `*Obj` 只保存对象头前缀和
`!Send + !Sync` marker；字段方法通过运行时 reflection getter 返回拥有所有权的 Rust 值。
因此 C++ 的隐藏字段、padding、非平凡 constructor/destructor 不会被 Rust 假装成可本地分配的
结构体。

详细设计、原始问题和验收依据见
[STUBGEN_PASS_EVALUATION.md](STUBGEN_PASS_EVALUATION.md)。

## Generated API 的约定

- 131 个反射对象生成 typed ObjectRef，目标 prefix 的 395 个 globals 全部生成；对象继承通过
  transparent handle upcast/downcast 表达。
- 反射字段生成 `value.field() -> tvm_ffi::Result<T>`，不会公开可变镜像字段。
- C++ nullable ObjectRef 生成 `Option<T>`。FFI `None` 的入站、出站和容器 round-trip 使用同一
  nullable 语义。
- `Array<Any>`/`Map<K, Any>` 使用 owning `AnyValue`，不会把标量错误收窄成 ObjectRef。
- 可精确表达的 global 生成 cached typed wrapper。没有函数 schema 的 callable 生成
  `*_packed(&[AnyView<'_>]) -> Result<Any>`；复杂 union/tuple/list/dict 边界也安全地类型擦除，
  不伪造错误的 Rust 签名。
- generic reflection builder 入口叫 `ffi_new_unchecked`，最终分配只能通过
  `pub unsafe fn build_unchecked`。它可能绕过类型专用 constructor 的语义检查；pass 应优先调用
  generated canonical global constructor。
- reflection 没有证明 C++ 类型线程安全，所以 generated handle 默认 `!Send + !Sync`。以后只有
  schema 明确给出线程安全契约，才能逐类型放宽。

## Pass 层

generated globals 已覆盖 Analyzer、结构化辅助函数和 pass factory 的底层调用。`visitor.rs`、
`mutator.rs` 和 `passes/` 在其上提供 compiler-pass 语义：child role、definition scope、重建和
panic containment 都属于 pass SDK，而不是 object binding generator 猜出来的规则。

高层 `analyzer::Analyzer` 不实现 `Clone`：`as_raw().clone()` 会共享可变 C++ 状态，真正的独立副本
必须显式调用 `fork()`。约束只通过 `with_constraint` closure 暴露，保证 native recovery callback
严格 LIFO；active scope 中禁止 fork，exit 失败后 wrapper 会 poison 并拒绝继续使用。C++ `Clone`
同时复制传统分析器和 Z3 prover 状态。

这一区分很关键：generic reflection walk 适合调试、统计和序列化，但 compiler visitor 需要知道
哪些字段是 child、definition、annotation 或 source span，不能把“所有 ObjectRef 字段”都当作
语义子节点。

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

也可以用 `TVM_COMPILER_LIB=/absolute/path/libtvm_compiler.so` 代替 `TVM_BUILD_DIR`。
不要把另一 Python 环境的 `libtvm_ffi.so` 放到 TVM build 前面，否则相同 SONAME 可能对应不同
ABI。

## 确定性重新生成

`regen.sh` 始终从空的、同文件系统 staging tree 生成六组 prefix：
`ir,tirx,target,transform,instrument,arith`。之后依次执行 `rustfmt`、安全 gate 和一个只包含
candidate generated modules 的独立 Cargo check。任何一步失败都不会改动 checked-in tree。

推荐直接使用本仓库的 tvm-ffi source/build。脚本通过绝对路径加载指定 Python source 和已构建
Cython extension，避免环境里同名包劫持 import：

```bash
export TVM_FFI_SOURCE_DIR="$PWD/../3rdparty/tvm-ffi"
export TVM_FFI_BUILD_DIR="$TVM_FFI_SOURCE_DIR/build"
export TVM_FFI_PYTHON=/path/to/python
export TVM_COMPILER_LIB="$PWD/../build/lib/libtvm_compiler.so"
./regen.sh --check
```

也支持已安装的 CLI。wheel/console script 通常不携带可验证的 git provenance，所以必须显式给出
精确的 40 位 commit：

```bash
export TVM_FFI_STUBGEN=/path/to/tvm-ffi-stubgen
export TVM_FFI_GENERATOR_COMMIT=<exact-40-hex-commit>
export TVM_COMPILER_LIB=/path/to/libtvm_compiler.so
./regen.sh --check
```

不要对 TVM 或 tvm-ffi 执行 `pip install -e`。editable install 会让一个 worktree 静默导入另一个
worktree；TVM 开发使用 `PYTHONPATH`，本脚本的 local-source 模式则完全绕过 editable redirector。

`--check` 对包含 `STAMP` 的完整目录做逐字节比较。`--write` 只安装已验证的 candidate，并在
rename 或 signal 失败时恢复原树。`STAMP` 只包含确定性输入：format/schema version、generator
exact commit、runtime exact commit、prefix 集合和 rustfmt version。它特意不记录当前 TVM HEAD
或 compiler ELF hash；这两者会让相同 schema 的输出随本地 build 改变，真实 schema 变化已经由
generated 内容的字节差异捕获。

## “10/10”的含义

这里的 10/10 是一组可以在 clean checkout 重放的验收门槛，不是“所有未来 TVM pass 已自动
实现”的宣传语。只有同时满足以下条件，生成层才记为 10/10：

1. 全量对象和目标 globals 严格生成；不支持的 schema 必须 fail closed 或明确 packed fallback。
2. 无 `DerefMut`、`ObjectArc::new`、公开 mirrored data field 或 safe native builder。
3. 所有对象默认 `!Send + !Sync`，字段读取走 owning reflection getter，generic builder 只能
   `unsafe build_unchecked`。
4. nullable、heterogeneous Any container、subtype cast 和无签名 callable 有明确且可测试的表示。
5. fresh generation 确定、格式化、安全检查通过，并可作为独立 Cargo crate 零 warning 编译。
6. `--check` 不写文件；`--write` staged、可恢复，且 generator/runtime provenance 精确。

`check_generated_safety.sh` 和 `regen.sh --check` 把这些生成层约束固化为 CI 可执行检查。某个具体
pass 是否与 C++ 实现语义等价，仍需该 pass 自己的端到端测试证明。
