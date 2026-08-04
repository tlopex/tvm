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

# Rust stubgen pass 开发评估

日期：2026-08-03

TVM branch：`main-dev/2026-07-09/rust_stubgen` (`b831b1ae`)

Runtime/Cargo tvm-ffi fork：`4bee9f37`

历史 Rust generator fork：`317fd810`

## 结论

原始 bindings 适合“只读验证部分 IR layout”，还不适合作为稳定的 pass SDK。最危险的
问题不是 boilerplate，而是 safe API 能成功构造一个内存不足或语义非法的 C++ 对象。

原始体验评分：

| 场景 | 评分 | 主要原因 |
|---|---:|---|
| 读取已知 IR 字段 | 7/10 | 字段访问直接，131 个镜像覆盖面较广；但 ABI 没有完整 guard |
| 写真实 compiler pass | 2/10 | 无 typed globals、visitor/mutator、Analyzer、Pass factory |
| 安全性/错误可发现性 | 1/10 | native `new`、`DerefMut`、schema mismatch panic 都藏在安全 API 后 |
| regeneration/clean checkout | 1/10 | 绝对 symlink、版本分裂、手工 postprocess、非原子脚本 |

本分支增加 safe facade、visitor/mutator 和三项 pass 后，支持范围内的 statement pass 已经能
形成可运行闭环；但 Any container、Analyzer 和全量 ABI validation 仍是 generator 级 blocker。
按同一维度，当前“已支持的 statement pass 原型”约为 6/10：最危险的入口已被阻断，错误也
能携带上下文返回，但覆盖范围和 regeneration 仍不够支撑稳定 SDK。这个分数不能外推到 131
个类型的全部 API。

## 实际实现的 pass

### VerifySSA

这是 visitor 的语义验收测试，不只是计数 demo。Rust 实现逐项对齐
`src/tirx/analysis/verify_ssa.cc`：

- PrimFunc 参数是 definition。
- 重复 Let 绑定只在 value deep-equal 时允许。
- Bind、For loop var、AllocBuffer data 各有独立 definition 规则。
- match-buffer 描述里的变量在特殊 scope 中允许重复 definition。
- 遍历发生错误或遇到未知 node 时 fail closed。

它也证明 generic reflection walk 不能直接充当 compiler visitor：definition 字段、annotation
和 source span 都需要不同策略。

### RustSkipAssert

对齐 `src/tirx/transform/skip_assert.cc` 的核心规则：`AssertStmt -> Evaluate(0)`。它覆盖：

- If/For/While/Seq/AttrStmt 递归。
- SBlock init/body 和 SBlockRealize。
- singleton/empty/nested SeqStmt 规范化。
- PrimFunc body 重建。
- Rust closure -> C++ `PrimFuncPass` callback。

实际 `RunPass` 暴露了一个只在真实 callback 中出现的问题：C++ 对右值参数使用
`kTVMFFIObjectRValueRef`，直接套 typed Rust callback 会按普通 ObjectRef 解码失败。当前实现
用 packed callback 消费 `RValueRef` 后再做 typed conversion，并用 `catch_unwind` 阻止 panic
跨 C ABI。

### RustRemoveNoOpConservative

这是完整 `RemoveNoOp` 的可证明安全第一阶段：

- 删除不含 Call 的 Evaluate。
- 折叠直接的常量假 If/While。
- 删除 non-positive literal extent 的 For。
- 删除 `pragma_debug_skip_region`。
- 保留所有未知 Call，并在删除 control-flow wrapper 时保留可能有 Call 的表达式求值。
- 对 visitor 尚不认识的非 TIRx `Expr` 保留整个 Evaluate，而不是把“遍历不了”误当作 pure
  或让保守 pass 整体失败。

没有把它命名成完整 `RemoveNoOp`。C++ 实现还依赖：

- `arith::Analyzer` 的 contextual proof/simplify。
- Op effect kind。
- PassContext 的 `RemoveNoOpConfig`。
- BufferLoad/BufferStore 等价证明。
- profiler call 特例。

当前 stubgen 对这些能力没有 typed API，因此保守分阶段比伪装等价更可靠。

## P0：内存安全

### native allocation 的判断依据不成立

generator 的 native blocker 只检查 reflected field 之间的 padding。它没有使用：

- `TVMFFITypeInfo.total_size`。
- 第一个 field 前的隐藏状态。
- 最后一个 field 后的隐藏状态。
- field 的真实 alignment。
- C++ constructor/destructor 和节点语义校验。

确定错误的例子：

- generated `AnalyzerObj` 只有 object header；C++ Analyzer 包含多个子 analyzer 和非平凡状态。
- generated `TargetObj` 在 attrs 结束；C++ 尾部还有 `std::string str_repr_`。
- `UniqueNameSupply` 有隐藏 name set。
- `PyStmtExprVisitor/Mutator` 有多组 callback Function。
- `For::new` 绕过 dtype/defined/promotion 检查。
- 原 demo 构造 singleton `SeqStmt`，而 C++ 明确禁止长度 0/1。

本分支的临时阻断：

- 已知少分配的五个 constructor 直接删除。
- 其余 native allocation 改为 `unsafe new_unchecked`。
- 所有 pass 只调用 `ffi_api.rs` 中的 C++ constructor wrapper。

generator 的长期规则应是：默认不 native allocate。只有显式证明完整 layout、平凡构造析构、
没有 hidden state 和没有 constructor invariant 的类型才能 opt in。

### DerefMut + Clone 是 safe Rust aliasing UB

原生成物对四个 mutable type 同时实现 `Clone` 和 `DerefMut`。两个 clone 指向同一 Object，
却可以在 safe Rust 中同时产生共享引用和独占引用。本分支删除所有 generated `DerefMut`。
mutable C++ object 应通过受控 FFI 方法或 interior-mutability API 表达。

### nullable ObjectRef 暴露了 core 表示问题

C++ 常用 null `Span` 表示无 source location，`Call.attrs`、`IterVar.dom`、`Buffer.strides` 也有
合法 undefined 状态。原 `ObjectArc<SpanObj>` 内部使用 `NonNull`；仅仅从 C++ 读入对象就已经
让 Rust 字段持有非法 bit pattern，之后 `AnyView::from` 还会读 null object header。

配套 tvm-ffi 分支做了可执行的 core 方案：`ObjectArc` 以单个 raw pointer 原位表示 null，null
clone/drop 不触碰 header，`Deref` 先诊断；derived ObjectRef、Array、Map 的出站 Any 把 null
编码为 FFI None，并提供 `ObjectRefCore::is_defined/is_null`。同时修了 null Array 的 safe
`as_container` 解引用、禁止 undefined Function 进入 C++ global registry。`ObjectArc` 自身原有
的 `Clone + DerefMut` 也会制造 aliased `&mut`，现已删除；初始化时的可变 raw access只留在
显式 unsafe 区域。原 `Array::index` 返回可复制的 `AnyView<'static>`，array drop 后会形成 UAF；
该入口已删除，替换为生命周期绑定的 `get_any(&self) -> AnyView<'_>`，null Array/Map 的
`len/is_empty` 则与 C++ 空 handle 语义对齐。

这是原型验证，不应原样当成最终 upstream ABI：raw pointer 使 `Option<ObjectArc<T>>` 失去
`NonNull` niche 的单指针布局保证。长期更合适的是区分 non-null/nullable handle，并让 stubgen
从 C++ `_type_is_nullable` 生成对应类型。

还有一个已确认但未在本原型解决的 P1：derive 不知道 `_type_is_nullable`，所以虽能把 null ref
输出成 `kTVMFFINone`，却仍拒绝把入站 FFI None 转回 nullable ObjectRef/Array/Map；如果对全部
类型一律接受 None，又会错误放宽 `Op`、`Type`、`SourceMap` 等 non-nullable wrapper。generator
和 derive 必须一起携带该标记，不能靠全局默认值猜测。

## P1：类型和 API 正确性

### Any container 被错误收窄

14 个字段把 `Any` 映射成 `ObjectRef`。例如 C++ `Target.attrs` 和 `For.annotations` 是
`Map<String, Any>`，其中合法值包括 bool/int/None；Rust 的 `Map<String, ObjectRef>`：

- 读取 scalar 会报类型错误。
- `Optional<Any>` 中的 scalar 可能被静默看成 None。

本分支重建 For/SBlock 时只透传原 map handle，绝不以错误 marker 读取 value。长期需要
`Array<Any>`/`Map<K, Any>` 的真正 type-erased container 支持；在此之前应 skip/fail closed，
不能假装成 ObjectRef。

### downcast 原来不是 C++ `as<N>()`

原生成代码只比较 type index 相等，派生对象 cast 到中间基类会失败。本分支把 131 个
generated downcast 改成 ancestry check；visitor/mutator 另用可恢复 lookup，避免某个
generated type key 不存在于运行库时 `LazyLock` 永久 panic。

### `__ffi_init__` lookup 缺 TypeAttr fallback

generator 检测 constructor 时接受 TypeMethod 或 TypeAttrColumn，但生成的调用只找 methods。
auto dataclass init 实际常只存在于 `__ffi_init__` TypeAttrColumn。本分支的 `ffi_compat.rs`
实现 method-first、column-fallback，并通过 `DictAttrs` 的真实构造路径验证。

### builder/default/kw-only 丢失

generator 收集 `kw_only/has_default`，却把全部字段作为位置参数调用 init；C++ auto-init 会拒绝
kw-only positional input。后来再手工把 61 个 builder 改成长 flat `new(...)`，还会丢默认值
并让同型参数容易交换。

推荐 API：保留命名 builder 和默认值，`build()` 永远调用 C++ init/global；不要本地分配。

## Pass SDK 缺失面

原生成树的 324 个 public function 恰好是：

- 131 个 `same_as`。
- 131 个 `downcast`。
- 62 个 constructor。

`RustGenerator.generate_global_funcs_block` 是 no-op。因此原作者仍需手写 global 名字、
`AnyView` 参数和返回转换。MVP 应自动生成：

1. cached typed global wrappers（constructor、Analyzer、structural helpers、Pass factory）。
2. 从 inheritance graph 生成的 fail-closed Expr/Stmt kind dispatch。
3. visitor trait；少量 child-role 语义表由 TIRx 维护。
4. C++ constructor-backed mutator/rebuilder。
5. PrimFuncPass/ModulePass/Sequential/RunPass facade。
6. `Bindings::load(path)` 和 ABI manifest validation。
7. ForKind 等 enum、C++ doc、默认值和 kw-only builder。

generic reflection walker 仍有价值，但应限定为 debug、统计、序列化类任务，不能默认替代
compiler semantic walk。

## 构建和 regeneration 审计

原始问题：

- `3rdparty/tvm-ffi` 是 `/home/linzhanl/tvm-ffi` 绝对 symlink。
- Cargo、Python generator、C++ runtime 使用不同 provenance。
- build.rs 把 Python 环境的 `libtvm_ffi` 放到 TVM build 前，实测可产生 CXXABI 或
  undefined-symbol 错误。
- 未配置 default rustup toolchain，plain `cargo` 立即失败。
- Cargo.lock 被忽略。
- regen 在 `src/generated` 原地运行六次；任一步失败都会留下半生成树。
- PATH 上的官方 stubgen 存在，但没有 Rust backend；原脚本只做 command existence check。
- 生成后还有不可重放的 constructor/compat 手改。

本分支已经：

- 把绝对 symlink 恢复为真实 git submodule，指向公开 tvm-ffi fork，并让 gitlink/Cargo 都 pin
  `4bee9f37`。
- build.rs 使用 `TVM_BUILD_DIR`/`TVM_COMPILER_LIB`，TVM build library 排第一；toolchain
  library 显式可选。
- 增加 `rust-toolchain.toml` 和 tracked Cargo.lock。
- regen 先 staging、再 safety validation，支持 check/write 和失败恢复。
- 恢复 provenance STAMP，并增加 generated safety gate。

`src/generated` 仍是来自 `317fd810` 的历史 snapshot。当前安全 gate 能验证该 snapshot，但旧
generator 会重新产生 safe native allocation，且不会重放 DerefMut 删除、subtype/null downcast
guard 等变换，所以会被 gate 拒绝；这意味着 `regen.sh --write` 目前仍不是可用的日常工作流。
generator 必须 rebase 到 runtime commit，并把 postprocess 变成模板规则，之后 regeneration
才能真正作为 CI source of truth。

## 验证结果

静态验证：

```text
cargo check --all-targets
Finished successfully
check_generated_safety.sh
passed: no DerefMut, no safe native allocation, null-safe subtype-aware downcast

tvm-ffi: cargo test -p tvm-ffi
all unit/integration/doc tests passed
```

使用 TVM `b831b1ae` build、tvm-ffi `4bee9f37` 的运行时验证：

```text
cargo test --all-targets
1 integration test passed

cargo run --bin tirx_demo
IntImm/IfThenElse/Op read passed
SkipAssert result = Evaluate(7)
RemoveNoOp result = Evaluate(0)
VerifySSA = true
all three C++ RunPass -> Rust callbacks passed
intentional Rust callback panic returned as TVM Error
```

测试还覆盖了 null Span 的 FFI None 编码/null-safe downcast、undefined Range 的 contextual Error、
undefined SourceMap 的反例检查、未知 Expr 的保守保留，以及 C++ `RValueRef` callback ownership。

反过来，用另一份 schema 不同的 `libtvm_compiler.so` 时在 `ir.TupleType` 参数数量上失败。这不是
当前 pass wrapper 的逻辑错误，却直接证明 `STAMP` 只是 provenance 文本，并不会在形成 Rust
reference 前拒绝错误 DSO；完整 ABI manifest/load guard 仍是 production blocker。

测试覆盖的是当前 pass 所需的真实路径，不等于 131 个类型的完整 ABI 证明。下一步应对所有
类型在加载时验证 type key、total size、alignment、field offset/size/alignment，并增加 hidden
leading/interior/trailing state、custom ctor/dtor、mixed Any container、kw-only/default 和 stale
cleanup 测试。

## 推荐后续顺序

1. generator 与 runtime/Cargo rebase 到同一 tvm-ffi commit，把 postprocess 移入模板。
2. upstream 禁用 native allocation、DerefMut；加入全量 ABI manifest 和加载时强制校验。
3. 把 C++ `_type_is_nullable` 带入 schema/derive，补齐 FFI None 的双向转换。
4. 修 Any container、subtype cast、TypeAttr init lookup。
5. 增加 strict/no-skip/check/dry-run/atomic regeneration 和 CI。
6. 自动生成 typed globals、visitor/mutator/pass facade。
7. 给 Analyzer/effect metadata 加 wrapper，再把 conservative RemoveNoOp 扩成完整等价实现。
8. 第二阶段实现 ConvertSSA、NarrowDataType/ForceNarrowIndex 等需要完整 expression mutator 的 pass。
