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

#include <gtest/gtest.h>

#include <unordered_map>
#include <vector>

#include "../../src/arith/constr_visitor.h"

namespace tvm {
namespace arith {
namespace {
using namespace tirx;

class SnapshotCollector : public ConstrVisitor {
 public:
  using ConstrVisitor::VisitExpr_;
  using ConstrVisitor::VisitStmt_;

  void VisitStmt_(const EvaluateNode* op) override {
    snapshots.push_back(GetConstrSet(op->value.as_or_throw<PrimExpr>()));
    ConstrVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) override {
    loads.push_back(GetConstrSet(op->indices[0]));
    ConstrVisitor::VisitExpr_(op);
  }

  std::vector<ConstrSet> snapshots;
  std::vector<ConstrSet> loads;
};

TEST(ConstrVisitor, SnapshotsOutliveBranchAndVisitor) {
  PrimVar x("x", PrimType::Int(32));
  std::vector<ConstrSet> snapshots;
  {
    SnapshotCollector visitor;
    visitor(SeqStmt({IfThenElse(x < 8, Evaluate(x), Evaluate(x)), Evaluate(x)}));
    snapshots = visitor.snapshots;
  }
  ASSERT_EQ(snapshots.size(), 3);
  EXPECT_TRUE(snapshots[0].CanProve(x < 8));
  EXPECT_TRUE(snapshots[1].CanProve(x >= 8));
  EXPECT_FALSE(snapshots[2].CanProve(x < 8));
  EXPECT_FALSE(snapshots[2].CanProve(x >= 8));
}

TEST(ConstrVisitor, LaterBindingsDoNotChangeEarlierSnapshots) {
  PrimVar x("x", PrimType::Int(32));
  PrimVar y("y", PrimType::Int(32));
  SnapshotCollector visitor;
  visitor(SeqStmt({Bind(x, PrimExpr(3)), Evaluate(x), Bind(y, x + 1), Evaluate(y)}));
  ASSERT_EQ(visitor.snapshots.size(), 2);
  EXPECT_TRUE(visitor.snapshots[0].CanProve(x == 3));
  EXPECT_FALSE(visitor.snapshots[0].CanProve(y == 4));
  EXPECT_TRUE(visitor.snapshots[1].CanProve(y == 4));
}

TEST(ConstrVisitor, AddressSnapshotKeepsTransitiveBindings) {
  PrimVar x("x", PrimType::Int(32));
  PrimVar y("y", PrimType::Int(32));
  PrimVar z("z", PrimType::Int(32));
  ffi::Array<Stmt> statements{Bind(x, PrimExpr(3)), Bind(y, x + 1), Bind(z, y * 2)};
  for (int i = 0; i < 128; ++i) {
    statements.push_back(Bind(PrimVar("unused", PrimType::Int(32)), x + i));
  }
  statements.push_back(Evaluate(z));
  SnapshotCollector visitor;
  visitor(SeqStmt(statements));
  ASSERT_EQ(visitor.snapshots.size(), 1);
  EXPECT_EQ(visitor.snapshots[0].constraints.size(), 3);
  EXPECT_TRUE(visitor.snapshots[0].CanProve(z == 8));
  EXPECT_EQ(visitor.GetConstrSet().constraints.size(), 131);
}

TEST(ConstrVisitor, AddressSnapshotKeepsIndirectPredicateDependencies) {
  PrimVar x("x", PrimType::Int(32));
  PrimVar query("query", PrimType::Int(32));
  PrimVar y("y", PrimType::Int(32));
  SnapshotCollector visitor;
  visitor(SeqStmt({Bind(query, x + 1), Bind(y, x * 2), IfThenElse(y < 16, Evaluate(query))}));
  ASSERT_EQ(visitor.snapshots.size(), 1);
  EXPECT_TRUE(visitor.snapshots[0].CanProve(query < 9));
}

TEST(ConstrVisitor, RangeDependenciesSurviveLoopExit) {
  PrimVar n("n", PrimType::Int(32));
  PrimVar i("i", PrimType::Int(32));
  SnapshotCollector visitor;
  visitor(SeqStmt({Bind(n, PrimExpr(4)), For(i, 0, n, ForKind::kSerial, Evaluate(i))}));
  ASSERT_EQ(visitor.snapshots.size(), 1);
  EXPECT_TRUE(visitor.snapshots[0].CanProve(i >= 0 && i < 4));
  EXPECT_FALSE(visitor.GetConstrSet().CanProve(i < 4));
}

TEST(ConstrVisitor, LetBindingsAreScopedToTheBody) {
  PrimVar x("x", PrimType::Int(32));
  auto buffer = decl_buffer({16}, PrimType::Int(32));
  SnapshotCollector visitor;
  visitor(Evaluate(Let(x, 3, BufferLoad(buffer, {x}))));
  visitor(Evaluate(BufferLoad(buffer, {0})));
  ASSERT_EQ(visitor.loads.size(), 2);
  EXPECT_TRUE(visitor.loads[0].CanProve(x == 3));
  EXPECT_FALSE(visitor.loads[1].CanProve(x == 3));
  EXPECT_FALSE(visitor.GetConstrSet().CanProve(x == 3));
}

TEST(ConstrVisitor, MutableReadDoesNotBecomeAPersistentBinding) {
  PrimVar x("x", PrimType::Int(32));
  auto buffer = decl_buffer({16}, PrimType::Int(32));
  SnapshotCollector visitor;
  visitor(SeqStmt({Bind(x, BufferLoad(buffer, {0})), Evaluate(x)}));
  EXPECT_TRUE(visitor.GetConstrSet().constraints.empty());
  EXPECT_FALSE(visitor.snapshots[0].CanProve(x == 0));
}

TEST(ConstrSet, RenameDistinguishesLocalValuesAcrossExecutions) {
  PrimVar tx("tx", PrimType::Int(32));
  PrimVar j("j", PrimType::Int(32));
  ConstrSet first{{Constr(tx, Range::FromMinExtent(0, 32)), Constr(j, 2 * tx)}};
  std::unordered_map<const VarNode*, Var> vars;
  auto rename = [&](const Var& var) {
    auto [it, inserted] = vars.emplace(var.get(), var);
    if (inserted) it->second = var.CopyWithSuffix("_other");
    return it->second;
  };
  auto second = first.RenameVars(rename);
  auto other_j = rename(j).as_or_throw<PrimExpr>();
  auto merged = first.Merge(second);
  EXPECT_TRUE(merged.CanProve(j != other_j + 1));
  EXPECT_FALSE(merged.CanProve(j != other_j));
  EXPECT_FALSE(merged.CanProve(j == other_j));
  EXPECT_TRUE(first.CanProve(j < 64));
}

TEST(ConstrSet, MergeDoesNotSilentlyDiscardDuplicateBindings) {
  PrimVar x("x", PrimType::Int(32));
  ConstrSet first{{Constr(x, PrimExpr(0))}};
  ConstrSet second{{Constr(x, PrimExpr(1))}};
  EXPECT_FALSE(first.Merge(second).CanProve(x == 0));
  EXPECT_FALSE(first.Merge(first).CanProve(x == 0));
}

TEST(ConstrSet, ManuallyConstructedUnsupportedFactsAreRejected) {
  PrimVar x("x", PrimType::Int(32));
  PrimVar y("y", PrimType::Int(32));
  PrimExpr wrapped = Cast(PrimType::Int(32), Cast(PrimType::UInt(8), x + 128));
  ConstrSet bindings{{Constr(y, wrapped)}};
  ConstrSet predicates{{Constr(wrapped < 128)}};
  ConstrSet ranges{{Constr(y, Range::FromMinExtent(0, wrapped))}};
  EXPECT_FALSE(bindings.CanProve(y == y));
  EXPECT_FALSE(predicates.CanProve(x == x));
  EXPECT_FALSE(ranges.CanProve(y == y));
  EXPECT_FALSE(ConstrSet{}.CanProve(wrapped == y));
}

TEST(ConstrSet, SymbolicGrowthAcrossBindingsIsRejected) {
  PrimVar x("x", PrimType::Int(64));
  PrimVar j("j", PrimType::Int(64));
  PrimVar k("k", PrimType::Int(64));
  PrimExpr scale = IntImm::Int64(5000000000);
  PrimExpr inner = scale * x - IntImm::Int64(9999999999);
  // x == 2 gives inner == 1, and scale * inner == scale. All runtime
  // intermediates fit int64, but distributing the products needs 25 * 10^18.
  EXPECT_FALSE(ConstrSet{}.CanProve(scale * inner != scale));
  ConstrSet bindings{{Constr(j, inner), Constr(k, scale * j)}};
  EXPECT_FALSE(bindings.CanProve(k != scale));
  EXPECT_FALSE(bindings.CanProve(k - scale == 0));
}

TEST(ConstrVisitor, ReductionInitFactsDoNotReachTheUpdate) {
  PrimVar i("i", PrimType::Int(32));
  SBlock block({IterVar(Range::FromMinExtent(0, 4), i, IterVarType::kCommReduce)}, {}, {}, "reduce",
               Evaluate(i),
               SeqStmt({AssertStmt(i == 0, StringImm("AssertionError"), {}), Evaluate(i)}));
  SnapshotCollector visitor;
  visitor(block);
  ASSERT_EQ(visitor.snapshots.size(), 2);
  EXPECT_TRUE(visitor.snapshots[0].CanProve(i == 0));
  EXPECT_FALSE(visitor.snapshots[1].CanProve(i == 0));
  EXPECT_FALSE(visitor.GetConstrSet().CanProve(i == 0));
}

TEST(ConstrSet, ExpansionMustFitEachIntegerNode) {
  PrimVar x("x", PrimType::Int(32));
  PrimVar j("j", PrimType::Int(32));
  PrimVar k("k", PrimType::Int(32));
  PrimExpr inner = 50000 * x - 99999;
  PrimExpr expr = 50000 * inner - 50000;
  EXPECT_FALSE(IsSupportedConstraintExpr(expr));
  EXPECT_FALSE(IsSupportedConstraintExpr(Cast(PrimType::Int(64), expr)));
  // Boolean comparisons can introduce a difference in their operands' dtype.
  EXPECT_TRUE(IsSupportedConstraintExpr(1500000000 * x));
  EXPECT_FALSE(IsSupportedConstraintExpr(1500000000 * x != -1500000000 * x));
  ConstrSet bindings{{Constr(j, inner), Constr(k, 50000 * j)}};
  EXPECT_FALSE(bindings.CanProve(k != 50000));
  ConstrSet ranges{{Constr(j, inner), Constr(k, Range::FromMinExtent(50000 * j, 1))}};
  EXPECT_FALSE(ranges.CanProve(k != 50000));
  PrimExpr wide = IntImm::Int64(50000) *
                  (IntImm::Int64(50000) * Cast(PrimType::Int(64), x) - IntImm::Int64(99999));
  EXPECT_TRUE(IsSupportedConstraintExpr(wide));
}

TEST(ConstrVisitor, SharedExpressionGrowthIsBounded) {
  PrimVar x("x", PrimType::Int(64));
  PrimExpr expr = x;
  for (int i = 0; i < 80; ++i) {
    expr = Add(expr, expr);
  }
  // The IR is small, but expanding it as a tree has 2^80 leaves.
  EXPECT_FALSE(IsSupportedConstraintExpr(expr));
}

}  // namespace
}  // namespace arith
}  // namespace tvm
