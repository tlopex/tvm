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

## 结论

stubgen 应被看作 reflection-to-Rust binding generator，而不是 pass generator。它的职责是稳定地
暴露 TVM 已注册的 IR 类型、字段和 global function；它不应该复制 C++ object layout，也不应该
根据字段列表猜 constructor、遍历或 pass 语义。

```text
TVM reflection schema ──stubgen──> generated Rust bindings
          │                            │
          └──── native TVM ABI <───────┘
                                       │
                               handwritten pass SDK
```

因此要分别评估两件事：

- generated 层是否忠实、安全、可重放地暴露 ABI；
- 手写 SDK 和具体 pass 是否有正确的 compiler semantics。

后一项做得再多，也不能提高 stubgen 本身的覆盖率；前一项通过编译，也不能证明 pass 正确。

## 生成模型

reflection metadata 是可见 API schema，不是完整的 C++ memory-layout 描述。它没有证明对象的
完整 size/alignment、隐藏状态、constructor/destructor、字段组合 invariant 或线程安全性。因此
generated object 必须是 opaque ABI handle：

- inheritance 由 typed handle 的 upcast/downcast 表达；
- field access 走 runtime reflection getter；
- nullable ObjectRef 使用 `Option<T>`；
- 异构值使用 owning `AnyValue`，复杂 callable 边界使用明确的 type erasure；
- object construction 只调用已有 native global；
- generated handle 不默认获得 `Send`/`Sync`。

不应重新引入 Rust-side allocation、可变 mirrored field、generic field-wise builder 或根据字段
metadata 猜 constructor。这些做法会绕过 native allocation 和类型专用 invariant。

## Stubgen 与手写 SDK 的边界

| 层 | 负责内容 | 不负责内容 |
|---|---|---|
| tvm-ffi/runtime | ObjectRef、Any、Array/Map、registry call、引用计数和 callback ABI | TVM IR child role 和 pass 算法 |
| stubgen generated | typed object handle、reflected getter、typed/packed global wrapper、module tree | visitor、mutator、Analyzer 状态规则、pass 语义 |
| handwritten bridge/SDK | `PrimExpr` refinement、child role、definition scope、rewrite policy、Analyzer 和 pass wrapper | 自动发现未来 schema 的 compiler semantics |
| concrete pass | 具体转换、分析和不变量 | 由“bindings 能编译”替代行为测试 |

`PrimExpr` 是这个边界的一个实际例子。它是 `Expr` 加 `PrimType` 约束的 view，不是独立 runtime
node，所以 `src/prim_expr.rs` 是手写 bridge。fresh generated crate 要显式包含这个依赖，不能在
验收输入中省略它。

generic reflection walk 也不能代替 compiler visitor。所有 ObjectRef field 并不都是 executable
child；其中可能有 definition、annotation、source span 或 metadata。把它们一律递归会让 pass 在
schema 新增字段时静默改变行为。

同理，Analyzer 的 fork/constraint scope、mutator 的重建策略、callback panic containment 和具体
pass 算法都要由手写 SDK 定义，并由各自测试验证。stubgen 最多提供它们调用的对象和 global。

## 当前生成流程

`rust/regen.sh` 当前执行一条清楚的路径：

1. 创建 fresh、同文件系统 staging tree。
2. 一次调用 Rust stubgen，传入 `ir`、`tirx`、`target`、`transform`、`instrument`、`arith` 六个
   `--init-prefix`。Rust schema 选择不需要额外的初始化配置。
3. 对 candidate 执行 `rustfmt` 和 generated-source safety scan。
4. 创建 fresh Cargo crate，加入 candidate generated tree 和所需的手写 `prim_expr.rs` bridge。
5. 在 `RUSTFLAGS=-D warnings` 下执行 `cargo check --all-targets`。
6. 加载真实 `libtvm_compiler`，运行一个 getter runtime smoke。
7. `--check` 比较完整 candidate；`--write` 只替换已经通过验证的 tree。

脚本从指定的本地 tvm-ffi source/build 按绝对路径加载 generator、runtime library 和 Python
extension，避免 editable install 把 import 重定向到另一 worktree。它不要求 generator/runtime
worktree clean；dirty 状态会写入 `STAMP`。当前 `STAMP` 记录 commit、dirty 状态、generated Rust
source hash、prefix 和 rustfmt 版本。

固定 object/global 数量不是可靠的正确性条件。schema 合法新增或删减时，这些数字必然变化；真正
需要检查的是生成是否失败、输出是否可编译、ABI 类型是否正确，以及有代表性的 runtime 行为是否
成立。

## 当前唯一完整 E2E 在验证什么

runtime smoke 创建真实的 `IntImm(int32, 37)`，随后验证三类 getter：

- scalar：`IntImm::value()` 返回 `37`；
- object：`IntImm::ty()` 能 downcast 为 `PrimType`，其 `dtype()` 为 `int32`；
- `Array`：把该表达式放入两个 `Evaluate`，构造 `SeqStmt`，`SeqStmt::seq()` 返回长度为 `2` 的数组。

这三个断言不是随意的示例。它们分别覆盖 getter lowering 中最容易走不同转换路径的 scalar、typed
ObjectRef 和 owning `Array`。fresh-crate `cargo check -D warnings` 则负责发现生成文件的语法、类型、
module wiring、手写 bridge 接口和 warning 回归。

它们的证明范围也必须写清楚：

- 不证明每个 generated type 或 field 都在 runtime 被调用过；
- 不证明 nullable、Map、packed fallback 和 callback 的全部组合；
- 不证明 visitor/mutator/Analyzer 或任何具体 pass 的语义；
- 不证明某个 Rust pass 与 C++ pass 等价。

若要扩大范围，应按一个可观察的失败模式增加小测试，例如 nullable round-trip、异构容器或真实
pass callback 闭环；不应为了“测试数量”复制只检查文件存在、硬编码计数或重复编译同一棵 tree
的测试。

## 维护原则

1. generator 只编码可从 schema 证明的事实；compiler semantics 留在小而显式的手写规则中。
2. 不用固定生成数量充当完成度，也不使用生成后补丁来制造 checked-in 结果。
3. 每个测试写明它覆盖的转换路径和不能证明的范围。
4. 新 schema form 先选择正确的 packed/type-erased fallback，再考虑 ergonomic typed API。
5. pass 能力单独用真实 native callback 和 IR 行为测试验收，不与 stubgen 绑定验收混在一起。
