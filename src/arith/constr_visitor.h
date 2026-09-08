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

/*!
 * \file constr_visitor.h
 * \brief Save arithmetic premises at different points of an IR traversal.
 */
#ifndef TVM_ARITH_CONSTR_VISITOR_H_
#define TVM_ARITH_CONSTR_VISITOR_H_

#include <tvm/arith/analyzer.h>
#include <tvm/ir/with_context.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace arith {

/*! \brief One premise, retaining bindings for the analyzer's rewrite and bound tables. */
struct Constr {
  enum Kind { kPredicate, kBindValue, kBindRange };

  explicit Constr(PrimExpr predicate) : kind(kPredicate), value(std::move(predicate)) {}
  Constr(tirx::Var var, PrimExpr value)
      : kind(kBindValue), var(std::move(var)), value(std::move(value)) {}
  Constr(tirx::Var var, Range range)
      : kind(kBindRange), var(std::move(var)), range(std::move(range)) {}

  Kind kind;
  tirx::Var var;
  PrimExpr value{ffi::UnsafeInit{}};
  Range range;
};

/*!
 * \brief An ordered snapshot of pure scalar premises.
 *
 * Unlike IRVisitorWithAnalyzer's current analyzer context, a snapshot can outlive
 * the scope in which it was collected.  Consumers must distinguish the variables
 * of different executions before combining snapshots.  Merge means conjunction
 * of premises, not a control-flow join.
 */
struct ConstrSet {
  std::vector<Constr> constraints;

  ConstrSet RenameVars(const std::function<tirx::Var(const tirx::Var&)>& rename) const {
    auto substitute = [&](const tirx::Var& var) -> ffi::Optional<Expr> { return rename(var); };
    ConstrSet result;
    for (const auto& c : constraints) {
      switch (c.kind) {
        case Constr::kPredicate:
          result.constraints.emplace_back(tirx::Substitute(c.value, substitute));
          break;
        case Constr::kBindValue:
          result.constraints.emplace_back(rename(c.var), tirx::Substitute(c.value, substitute));
          break;
        case Constr::kBindRange:
          result.constraints.emplace_back(rename(c.var), tirx::Substitute(c.range, substitute));
          break;
      }
    }
    return result;
  }

  ConstrSet Merge(const ConstrSet& other) const {
    ConstrSet result = *this;
    result.constraints.insert(result.constraints.end(), other.constraints.begin(),
                              other.constraints.end());
    return result;
  }

  bool CanProve(const PrimExpr& predicate) const {
    // Rebinding a variable can discard information or introduce inconsistent
    // premises.  Require the consumer to rename independent executions first.
    std::unordered_set<const tirx::VarNode*> bound;
    for (const auto& c : constraints) {
      if (c.kind != Constr::kPredicate && !bound.insert(c.var.get()).second) {
        return false;
      }
    }
    Analyzer analyzer;
    WithGroup<ConstraintContext> contexts;
    for (const auto& c : constraints) {
      switch (c.kind) {
        case Constr::kPredicate:
          // Bound analysis needs normalized comparisons, e.g. !(x < n) -> x >= n.
          contexts.Emplace(analyzer, analyzer->rewrite_simplify(c.value));
          break;
        case Constr::kBindValue:
          analyzer->Bind(c.var, c.value);
          break;
        case Constr::kBindRange:
          analyzer->Bind(c.var, c.range);
          break;
      }
    }
    return analyzer->CanProve(predicate);
  }
};

/*!
 * \brief Collect constraints that can safely be replayed at another access point.
 *
 * Only pure scalar integer expressions are retained.  In particular, a Bind of
 * a mutable read does not create a persistent rewrite to that read.  Its variable
 * remains an unconstrained symbol, which a consumer must rename per execution.
 * This deliberately loses information instead of modeling memory snapshots.
 *
 * Derived visitors that override control flow must use WithConstrScope and add
 * the appropriate premises after visiting conditions and bounds.
 */
class ConstrVisitor : public tirx::StmtExprVisitor {
 public:
  using StmtExprVisitor::VisitExpr_;
  using StmtExprVisitor::VisitStmt_;

  ConstrSet GetConstrSet() const { return {constraints_}; }

  static bool IsPureScalar(const PrimExpr& value) {
    auto ty = value.ty();
    return ty.IsScalar() && ty.MatchesCode(kDLInt, kDLUInt, kDLBool) &&
           tirx::SideEffect(value) <= tirx::CallEffectKind::kPure;
  }

  void VisitStmt_(const tirx::BindNode* op) override {
    this->VisitExpr(op->value);
    AddBinding(op->var, op->value);
  }

  void VisitStmt_(const tirx::AssertStmtNode* op) override {
    StmtExprVisitor::VisitStmt_(op);
    AddConstraint(op->condition);
  }

  // SeqStmt does not introduce a scope.  Bind/Assert facts also remain visible
  // to following siblings when the sequence contains another SeqStmt.

  void VisitStmt_(const tirx::IfThenElseNode* op) override {
    this->VisitExpr(op->condition);
    WithConstrScope([&]() {
      AddConstraint(op->condition);
      this->VisitStmt(op->then_case);
    });
    if (op->else_case) {
      WithConstrScope([&]() {
        AddConstraint(tirx::Not(op->condition));
        this->VisitStmt(op->else_case.value());
      });
    }
  }

  void VisitStmt_(const tirx::AttrStmtNode* op) override {
    this->VisitExpr(op->value);
    WithConstrScope([&]() {
      if (op->attr_key == tirx::attr::thread_extent ||
          op->attr_key == s_tir::attr::virtual_thread) {
        auto iv = op->node.as_or_throw<tirx::IterVar>();
        AddRange(iv->var, Range::FromMinExtent(IntImm(op->value.ty(), 0), op->value));
      }
      this->VisitStmt(op->body);
    });
  }

  void VisitStmt_(const tirx::ForNode* op) override {
    this->VisitExpr(op->min);
    this->VisitExpr(op->extent);
    if (op->step) this->VisitExpr(op->step.value());
    WithConstrScope([&]() {
      if (!op->step || tirx::is_one(op->step.value())) {
        AddRange(op->loop_var, Range::FromMinExtent(op->min, op->extent));
      }
      AddConstraint(op->extent > IntImm(op->extent.ty(), 0));
      this->VisitStmt(op->body);
    });
  }

  void VisitStmt_(const tirx::WhileNode* op) override {
    this->VisitExpr(op->condition);
    WithConstrScope([&]() {
      AddConstraint(op->condition);
      this->VisitStmt(op->body);
    });
  }

  void VisitStmt_(const tirx::SBlockNode* op) override {
    WithConstrScope([&]() { StmtExprVisitor::VisitStmt_(op); });
  }

  void VisitExpr_(const tirx::LetNode* op) override {
    this->VisitExpr(op->value);
    WithConstrScope([&]() {
      AddBinding(op->var, op->value);
      this->VisitExpr(op->body);
    });
  }

  // Select may evaluate both operands, so neither gets a branch constraint.
  void VisitExpr_(const CallNode* op) override {
    if (op->op.same_as(tirx::builtin::if_then_else())) {
      auto condition = op->args[0].as_or_throw<PrimExpr>();
      this->VisitExpr(condition);
      WithConstrScope([&]() {
        AddConstraint(condition);
        this->VisitExpr(op->args[1]);
      });
      WithConstrScope([&]() {
        AddConstraint(tirx::Not(condition));
        this->VisitExpr(op->args[2]);
      });
    } else {
      StmtExprVisitor::VisitExpr_(op);
    }
  }

 protected:
  template <typename F>
  void WithConstrScope(F&& body) {
    struct Guard {
      std::vector<Constr>& constraints;
      size_t size;
      ~Guard() { constraints.erase(constraints.begin() + size, constraints.end()); }
    } guard{constraints_, constraints_.size()};
    body();
  }

  void AddConstraint(const PrimExpr& predicate) {
    if (IsPureScalar(predicate)) constraints_.emplace_back(predicate);
  }

  void AddBinding(const tirx::Var& var, const Expr& value) {
    if (auto scalar = value.as<PrimExpr>(); scalar && IsPureScalar(scalar.value())) {
      constraints_.emplace_back(var, scalar.value());
    }
  }

  void AddRange(const tirx::Var& var, const Range& range) {
    if (IsPureScalar(range->min) && IsPureScalar(range->extent)) {
      constraints_.emplace_back(var, range);
    }
  }

 private:
  std::vector<Constr> constraints_;
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_CONSTR_VISITOR_H_
