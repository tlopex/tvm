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

# Rust stubgen 和 pass 开发评估

日期：2026-08-04

v3 generator/runtime：`tvm-ffi e7f9fc593c8aa417924eb40ec2778e72b1e148a9`

## 结论

原始 stubgen 的长处是“覆盖面”：131 个类型大体能读到字段。它的问题是把 C++ 对象的反射字段
误当成完整 Rust layout，并把未证明安全的分配、可变引用和 null handle 暴露在 safe API 后面。
这就是原始评分“字段 7/10、安全 1/10、写 pass 2/10、可维护性差”的根因。

v3 改变了生成模型，而不只是给旧结果打补丁：

```text
C++ reflection schema ──stubgen──> opaque Rust handles + typed API
         │                                  │
         └── TVM/tvm-ffi ABI runtime <──────┘
```

Rust consumer 是纯 Rust；C++ 不需要被开发者“再抄一遍”。C++ 只提供权威 schema 和已有对象/Pass
ABI。generated node 不声称拥有完整 C++ layout，字段通过 reflection getter 读取，构造只通过
typed C++ global 或明确反射出的 canonical static factory 完成。

按本文定义的六组可执行 gate，v3 生成层达到 10/10。这个分数只表示 generator 的安全、类型、
可重放和可诊断契约全部达标；它不等于 131 个 C++ 类型的每项语义、或所有 TVM pass，都已经在
Rust 中重新实现。

## 原始问题与 v3 处理

| 原始问题 | 风险 | v3 规则 |
|---|---|---|
| 按反射字段生成完整 `#[repr(C)]` data struct | hidden tail/padding/析构状态导致少分配或错误 drop | `*Obj` 只含继承前缀和 thread marker；字段走 reflection getter |
| `ObjectArc::new` 原生分配 C++ 对象 | 绕过 constructor invariant，甚至分配尺寸不足 | generated tree 完全禁止；优先调用 canonical global |
| `Clone + DerefMut` | safe Rust 可制造同一对象的 aliased `&mut` | generated/runtime ObjectRef 不实现 `DerefMut` |
| nullable ObjectRef 当作必非空指针 | null Span/attrs/container 触发非法表示或解引用 | schema 携带 nullability，Rust 生成 `Option<T>`，FFI None 双向 round-trip |
| `PrimExpr` schema 退化成普通 `Expr` | pass 无法区分 primitive result，甚至访问错误 child role | schema 保留 `TypedExpr<Expr, PrimType>`；Rust wrapper 在所有入站路径检查 reflected `ty` |
| `Array<Any>`/`Map<String, Any>` 收窄为 ObjectRef | bool/int/string 等合法值读取失败或静默丢失 | 容器内使用 owning `AnyValue` |
| exact type-index downcast | 派生类不能 cast 到中间基类 | runtime ancestry-aware cast |
| 没有 global 生成 | pass 作者手写字符串、packed 参数和返回转换 | 生成 cached typed globals；无 schema callable 明确 packed fallback |
| union/tuple 强行套一个 Rust 类型 | 签名看似 typed、实际 ABI 错误 | `Any`/`AnyView` type erasure，值不丢失 |
| generic builder 调反射 auto-init | 绕过类型专用语义检查 | 不生成 generic builder；只接受 typed global/static canonical factory |
| 默认允许 Send/Sync | reflection 并未证明 C++ 线程安全，擦除到通用 ObjectRef 还可能绕过 node marker | 每个 node 和 runtime Object 擦除根都保持 `!Send + !Sync`；callback closure 要求 `Send + Sync` |
| 六次原地生成 | 中途失败留下半棵 module tree | fresh same-filesystem staging、preflight、staging tree 写入、handled error/signal 下可恢复替换 |
| 手工后处理 | generator 无法复现 checked-in 内容 | 规则进入 generator/runtime，fresh generation 两次字节一致 |

## 为什么不能“照 C++ struct 抄成 Rust”

reflection metadata 描述的是可见字段，不是 C++ object-layout manifest。它没有证明：

- 第一个字段前、最后字段后没有隐藏状态；
- 所有 padding、size 和 alignment 都完整；
- constructor/destructor 是 trivial；
- 字段组合满足类型语义约束；
- 对象可由 Rust allocator 创建并由 Rust drop glue 销毁。

`Analyzer` 包含分析器状态，`Target` 有非反射字符串表示，`UniqueNameSupply` 有名称集合，visitor/
mutator 节点有 callback，`For`、`SeqStmt` 等 constructor 还检查语义 invariant。即使某个版本的
field offset 恰好对齐，也不能推出 native allocation 安全。

因此 v3 的生成对象是 ABI handle，不是 C++ value struct mirror。`#[repr(C)]` 只保证已声明的
base 前缀可用于 ObjectCore 关系；读取 reflected field 时，tvm-ffi 查找
`TVMFFIFieldInfo.getter`，检查 object type/layout，再把返回的 owning Any 转成目标 Rust 类型。

## 类型系统边界

### Nullable

C++ 的 `_type_is_nullable` 被写入 Optional schema。Rust 顶层和容器转换统一使用 `Option<T>`：
`None` 对应 `kTVMFFINone`，defined object 对应 ObjectRef。连续 Optional 在 Rust 侧归一成一层
`Option`，因为 packed ABI 只有一个 None 状态。

这解决的不只是 `Span`。`Call.attrs`、`IterVar.dom`、optional Array/Map 和返回 null 的 global
都需要同一规则。non-nullable wrapper 仍拒绝 None，不能为了方便全局放宽。

### TypedExpr refinement

C++ `PrimExpr` 不是独立 runtime node，而是 `ExprNode::ty` 为 `PrimType` 的受限 view。v3 schema
把它编码为 `TypedExpr<ir.Expr, ir.PrimType>`；Rust `TypedExpr<Base, Expected>` 保持同一对象指针，
并在 `try_from_base`、`Any`、`AnyView` 和 object-ref cast 的每个入站路径通过 reflection 检查
`ty`。它不能只凭 Expr runtime type index 匹配，也没有公开 unchecked constructor。Python 类型
系统没有对应 refinement，因此 annotation 仍显示 base `Expr`，最终由 C++ TypeTraits 验证语义。

### Any 和复杂 schema

`AnyValue` 是 owning type-erased value，适合 `Array<AnyValue>` 和 `Map<K, AnyValue>`；它可以安全
保存 scalar、string、None 和 ObjectRef。借用的 packed 参数使用 `AnyView<'_>`，返回未知值使用
owning `Any`。

Rust 没有与 Python `Union`、动态 tuple/list/dict 完全相同的单一原生 ABI 表示。v3 对这些边界
显式类型擦除。类型擦除降低静态便利性，但它保留全部值并让调用正确；生成一个看似漂亮但错误的
typed signature 才是不可接受的行为。

### Callable schema fallback

有完整 callable schema 的 global 生成 typed wrapper，并用 thread-local `OnceCell<Function>` 缓存
registry lookup。只有 `ffi.Function` 而没有参数/返回 schema 的 18 项 API 生成
`*_packed(&[AnyView<'_>]) -> Result<Any>`。它们仍然可调用，但不会被伪装成零参数函数。

## 写 pass 的 API 分层

pass SDK 不应是“把 C++ pass 逐行翻译成 Rust”，而应分四层：

1. **ABI/runtime 层**：ObjectRef、Any/AnyValue、Function、Array/Map、错误和 callback ownership。
2. **generated schema 层**：对象继承、字段 getter、typed/packed globals、canonical constructor。
3. **IR traversal 层**：Expr/Stmt visitor、statement-structural mutator、definition scope、重建和
   Seq normalization；当前不包含通用 expression mutator。
4. **pass 层**：Analyzer/effect metadata、PassInfo、PrimFuncPass/ModulePass、Sequential 和具体算法。

前两层可以由 schema 系统生成。第三层只能部分生成：继承 dispatch 可以自动化，但 child role、
definition、annotation 和 source span 的语义需要小型、可审查的 IR 规则表。第四层是 pass 本身，
必须用测试证明与目标语义一致。

有状态 API 还需要比 generated handle 更强的语义封装。例如 generated `Analyzer::clone()` 只是共享
同一可变 C++ 状态；高层 SDK 用显式 `fork()` 表示深拷贝，并让 C++ Clone 同时复制 Z3 prover。
constraint recovery 必须严格 LIFO，因此只暴露 `with_constraint` closure，不公开可被乱序 drop 的
共享借用 guard；scope 内禁止 fork，exit 失败会 poison wrapper。

generic reflection walker 因而不是 compiler visitor 的替代品。它适合 debug/statistics；若把所有
ObjectRef field 都递归访问，可能把 definition 当 use、把 metadata 当 executable child，甚至
在 schema 扩展时悄悄改变 pass 行为。

## Pass callback 的 FFI 注意事项

真实 `RunPass` 还涉及三个只靠静态 bindings 看不出的边界：

- C++ 可能用 `kTVMFFIObjectRValueRef` 传递右值；callback 必须消费正确的 owning 表示。
- Rust panic 不得跨 C ABI；callback wrapper 必须 `catch_unwind` 并转为 TVM Error。
- 返回 ObjectRef 的引用计数和临时 Any 生命周期必须由 runtime 转换统一管理。

因此验收 pass 不能只测试“factory 返回了一个 Pass 对象”。端到端测试应把 pass 交给 C++
`RunPass`，验证 C++ -> Rust callback -> C++ 返回值、错误和 panic 的完整闭环。

## Regeneration 和可维护性

`regen.sh` 的 staged flow 是：

1. 从空目录依次生成 `ir,tirx,target,transform,instrument,arith`。
2. `rustfmt` 全部 Rust 文件。
3. 执行 safety gate。
4. 在临时独立 crate 中只编译 candidate generated tree。
5. 写入确定性 `STAMP`。
6. `--check` 做完整目录字节比较，或 `--write` 以备份/恢复协议替换。

本地 source 模式按绝对路径加载指定 tvm-ffi Python package、`libtvm_ffi.so` 和 Cython core，主动
移除 scikit-build editable redirector。生成器/runtime worktree 必须 clean，这样 `STAMP` 中的
source commit 对应明确的 clean checkout；脚本没有独立证明已有 build artifact 正好由该 commit
编译。installed CLI 模式要求调用者声明 `TVM_FFI_GENERATOR_COMMIT`，该值会被记录但无法由 CLI
binary 反向验证。

`STAMP` 不包含当前 TVM HEAD 或 compiler ELF hash。它们会受重编译、linker 和本地路径影响，
不是生成格式的稳定输入。schema 真的变化时，generated Rust 自身就会发生字节变化；STAMP 记录
的是 format/schema version、generator/runtime source commit 字段、prefix 和 rustfmt version；
其可验证性边界如上所述。

禁止 `pip install -e` 同样是 correctness 规则，而非偏好。TVM/tvm-ffi 多 worktree 开发时，
editable redirector 会让命令行路径看似正确却导入另一 checkout，最终得到不可复现的 schema、
runtime 和 Cargo 组合。

去掉 119 份 generic builder 以及每类型重复的 `same_as/downcast` 后，checked-in generated `.rs`
从 20,644 行降为 10,583 行，减少 10,061 行；安全规则更强，但审查面缩小约 49%。

## 10/10 可执行验收表

| Gate | 通过条件 | 失败方式 |
|---|---|---|
| 覆盖 | 131/131 objects 和 395/395 目标 globals 全量生成 | unsupported schema strict failure；无签名 callable 明确 packed fallback；忽略规则不得跳过 candidate 子目录 |
| 内存/别名安全 | 无 `ObjectArc::new`、`DerefMut`、public mirrored field | `check_generated_safety.sh` 立即拒绝 |
| 构造安全 | generated tree 不含 generic builder | `ffi_new_unchecked/build_unchecked` 立即拒绝；pass 用 canonical global |
| 类型正确 | nullable Option、AnyValue container、subtype cast、type-erased complex schema | runtime/generator 单元测试和 full-tree compile |
| 线程契约 | generated node 与通用 ObjectRef 擦除入口默认 `!Send + !Sync`；callback 要求 `Send + Sync` | marker 数检查、compile-fail contract 与 callback bound 测试 |
| 可重放 | fresh tree 两次一致、rustfmt、policy safety gate、独立 Cargo check、byte check | 验证失败不改 checked-in tree；write 对 handled error/signal 恢复 |

这里的 10/10 是封闭、可审计的 acceptance profile。它不声称：

- reflection 能证明 arbitrary C++ constructor invariant；
- generic visitor 能自动获得 compiler semantics；
- Rust 已经复刻全部 C++ pass；
- 未经端到端测试的具体 pass 与 C++ 版本等价；
- 今后新增 schema 不需要新增测试。

这种定义比主观“代码看起来完整”更有用：每项都能在 CI 中失败，也能明确指出应该修 generator、
runtime、pass SDK 还是具体 pass。源码 pattern gate 本身不是形式化安全证明，因此还必须和
compile-fail、runtime、C++ schema 及端到端 pass 测试一起使用。

## 后续演进原则

1. schema 若要放宽 Send/Sync，必须逐类型携带并验证线程安全属性，不能删掉全局 marker。
2. 类型专用 constructor 必须来自 typed global 或 canonical static factory；不恢复 generic auto-init。
3. 新 schema form 先选择正确的 type-erased fallback，再增加 ergonomic typed representation。
4. 自动生成 inheritance dispatch，但把 child-role/definition 语义保留为显式规则表。
5. Analyzer、effect kind 和 PassContext config wrapper 应直接来自 typed globals，不再维护字符串
   compatibility shim。
6. 每次 generator/runtime 变更都从空目录生成、独立编译、运行 safety gate，并验证两次输出一致。
