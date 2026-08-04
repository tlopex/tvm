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

# Rust bindings and TIRx pass prototype

这个 crate 用 stubgen 生成的 131 个 `#[repr(C)]` IR 镜像读取 TIRx，并在其上增加一层
手写、可测试的 pass API。当前定位是“验证并推动 stubgen 设计”的原型，不是已经稳定的
Rust SDK；完整审计见 [STUBGEN_PASS_EVALUATION.md](STUBGEN_PASS_EVALUATION.md)。

## 已实现

- `src/ffi_api.rs`：C++ constructor/global 的 typed wrapper，包含 IR、`PassInfo`、
  `PrimFuncPass`、`ModulePass`、`Sequential` 和 `RunPass`；callback panic 会转成 TVM Error。
- `src/visitor.rs`：按 TIRx 语义遍历的 fail-closed `StmtExprVisitor`，不是把反射到的所有字段
  都当 child；支持 subtype，并能恢复地处理 bindings/runtime schema 不一致。
- `src/mutator.rs`：statement-structural mutator。发生变化时走 C++ constructor，保留校验和
  `SeqStmt::Flatten` 规范化；不调用 generated native allocation。
- `src/passes/verify_ssa.rs`：Rust `VerifySSA`，包括 Let 的弱 SSA 规则、buffer definition 和
  match-buffer scope。
- `src/passes/skip_assert.rs`：Rust `SkipAssert`，可包装成普通 `PrimFuncPass`。
- `src/passes/remove_no_op.rs`：保守的第一阶段 RemoveNoOp；未知 Call 一律保留，名称明确为
  `RustRemoveNoOpConservative`，不冒充依赖 Analyzer 的完整 C++ 实现。
- `tests/passes.rs`：真实加载 `libtvm_compiler.so` 的端到端测试。

端到端测试不是只构造 Pass 对象：它把三个 pass 都交给 C++ `RunPass`，覆盖 C++ 传入
`RValueRef`、Rust callback 解码、返回值所有权以及 panic 转 TVM Error 的完整边界。

## 构建和运行

纯静态检查不要求 TVM build：

```bash
cd rust
cargo check --all-targets
```

运行 demo/test 时，让 C++、Python runtime 和 Rust 使用同一份 `libtvm_ffi`：

```bash
export TVM_BUILD_DIR=/path/to/tvm/build
# 仅当 TVM 使用了非系统 libstdc++ 时设置：
export TVM_TOOLCHAIN_LIB_DIR=/path/to/toolchain/lib

cargo test --all-targets
cargo run --bin tirx_demo
```

也可以用 `TVM_COMPILER_LIB=/absolute/path/libtvm_compiler.so` 代替
`TVM_BUILD_DIR`。不要把另一套 Python 环境中的 `libtvm_ffi.so` 放到 TVM build 前面，
否则同一 SONAME 会产生 ABI/undefined-symbol 错误。

## 安全策略

审计确认原 generator 不能从“已反射字段之间没有 padding”推断对象可由 Rust 原生分配。
`Analyzer`、`Target`、`UniqueNameSupply` 和 Python visitor/mutator 都有未反射的尾部状态或
非平凡构造逻辑。

本分支因此做了以下阻断：

- 删除五个已确定会少分配/漏构造的 API。
- 其余实验性本地构造从安全 `new` 改为 `unsafe new_unchecked`；pass 代码禁止调用它们。
- 删除所有 generated `DerefMut`，避免 cloned `ObjectRef` 在 safe Rust 中制造别名 `&mut`。
- generated `downcast` 改为 subtype-aware；visitor/mutator 另有不会因缺 type key 而 panic 的
  dispatch。
- C++ nullable `ObjectRef` 由 tvm-ffi fork `4bee9f37` 原位表示；提供 `is_defined/is_null`，null
  clone/drop/Any 输出安全，null `Array/Map` 访问会先诊断，不能形成 Rust null reference。
- `Function::register_global` 拒绝 undefined Function，避免把 null handle 存入 C++ 全局表。
- 删除会泄漏 `AnyView<'static>` 的 `Array` 索引 API，改成 `get_any(&self) -> AnyView<'_>`；
  undefined Array/Map 的 `len/is_empty` 与 C++ 一样按空容器处理。
- `check_generated_safety.sh` 固化上述最低要求。

长期方案仍应在 stubgen 中默认不生成 native allocation：命名 builder 可以保留，但
`build()` 必须调用 C++ `__ffi_init__` 或公开 constructor global。

nullable 支持目前是验证性 core 改动，不是最终 API：`ObjectArc<T>` 从 `NonNull<T>` 改成 raw
pointer 后，`Option<ObjectArc<T>>` 不再有单指针 niche 布局保证。更适合 upstream 的方案是区分
non-null 与 nullable handle，并让 generator 从 C++ `_type_is_nullable` 生成正确类型。当前 derive
也还不能让 `kTVMFFINone` 按“nullable/non-nullable”分别做入站 Any 转换；因此 null 字段读取与
出站传参已覆盖，但 `Any(None) -> nullable ObjectRef` round-trip 仍是明确的 P1 缺口。

## 重新生成

当前 Rust backend 只存在于历史 tvm-ffi fork，官方 `tvm-ffi-stubgen` 尚不支持
`--target rust`。原脚本还会原地分六次覆盖，失败后留下半生成树；现在 `regen.sh` 改为：

1. 在同一文件系统的临时目录生成全部六个 prefix。
2. 先执行安全检查。
3. `--check` 比较；只有 `--write` 才替换原树，安装失败会恢复旧树。
4. 写入 TVM/generator/compiler provenance。

```bash
export TVM_FFI_STUBGEN=/path/to/fork/bin/tvm-ffi-stubgen
export TVM_FFI_SOURCE_DIR=/path/to/fork
export TVM_COMPILER_LIB=/path/to/build/lib/libtvm_compiler.so
./regen.sh --check
```

当前 `src/generated` 是有 provenance 的历史 snapshot，并应用了批量安全 postprocess；旧
generator 仍会重新生成 safe native builder，且不会重放 null guard/subtype 等变换，因此
`regen.sh` 会主动拒绝它，`--write` 目前也不应作为可用工作流。应先把这些规则移入 generator
模板并 rebase 到 `4bee9f37`，再把 regeneration 变成 CI source of truth。历史生成版本、当前
runtime pin 和 postprocess 清单记录在 `src/generated/STAMP`。

## 仍然阻塞完整 pass SDK 的问题

- `Map<String, Any>`/`Optional<Any>` 被错误收窄成 `ObjectRef`，标量 metadata 无法可靠读取。
- generator 不知道 C++ ObjectRef wrapper 的 nullable 标记，`kTVMFFINone` 入站转换仍不对称。
- global functions、Analyzer、effect-kind、structural helpers 没有自动生成 typed wrapper。
- enum 仍是裸 `i32`，缺 Rust 文档、默认参数和 kw-only builder 语义。
- 没有全量 size/alignment/offset ABI manifest；当前测试只覆盖实际使用到的路径。
- 完整 `RemoveNoOp` 还需要 Analyzer、side-effect metadata、PassContext config 和 buffer
  等价证明。

这些问题的优先级和推荐实现顺序见详细评估文档。
