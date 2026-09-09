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
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr_functor.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace arith {

/*!
 * \brief Validate the restricted arithmetic domain used by snapshot proofs.
 *
 * Use a closed set of scalar signed int32/int64 arithmetic and boolean operators.
 * Casts must preserve the signed integer value; multiplication requires a constant
 * factor, and division a positive constant divisor. Calls and loads are excluded.
 * Signed overflow follows the existing TIR undefined semantics.
 *
 * Also limit symbolic expansion: even a defined execution can have coefficients
 * that overflow the analyzer's int64 arithmetic when products are distributed.
 * Track a conservative magnitude through expressions and Bind definitions. Add
 * magnitudes for sums/comparisons and multiply them for products/divisions (which
 * can combine divisors). Reject growth beyond int64 rather than asking the
 * analyzer to simplify it. This deliberately excludes some valid large indices.
 */
class ConstraintExprValidator : public tirx::ExprFunctor<uint64_t(const Expr&)> {
 public:
  bool IsSupported(const PrimExpr& expr) {
    expression_bounds_.clear();
    return VisitExpr(expr) < kUnsupported;
  }

  bool Bind(const tirx::Var& var, const PrimExpr& value) {
    expression_bounds_.clear();
    uint64_t bound = VisitExpr(value);
    if (bound == kUnsupported) return false;
    bindings_[var.get()] = bound;
    return true;
  }

  bool Bind(const tirx::Var& var, const Range& range) {
    expression_bounds_.clear();
    uint64_t bound = Add(VisitExpr(range->min), VisitExpr(range->extent));
    if (bound == kUnsupported) return false;
    // A singleton range can become a value binding. Retain its dependencies
    // even when the extent is only known to be one after simplification.
    bindings_[var.get()] = bound;
    return true;
  }

 private:
  static constexpr uint64_t kUnsupported = uint64_t{1} << 63;

  static uint64_t Add(uint64_t a, uint64_t b) {
    return a >= kUnsupported - b ? kUnsupported : a + b;
  }

  static uint64_t Mul(uint64_t a, uint64_t b) {
    return a > (kUnsupported - 1) / b ? kUnsupported : a * b;
  }

  uint64_t VisitExpr(const Expr& expr) final {
    auto it = expression_bounds_.find(expr.get());
    if (it != expression_bounds_.end()) return it->second;
    auto value = expr.as<PrimExpr>();
    if (!value) return kUnsupported;
    PrimType ty = value.value().ty();
    if (!ty.IsScalar() || !(ty.MatchesCode(kDLBool) ||
                            (ty.MatchesCode(kDLInt) && (ty.bits() == 32 || ty.bits() == 64)))) {
      return kUnsupported;
    }
    if (ty.MatchesCode(kDLBool) &&
        !(expr.as<tirx::VarNode>() || expr.as<IntImmNode>() || expr.as<tirx::EQNode>() ||
          expr.as<tirx::NENode>() || expr.as<tirx::LTNode>() || expr.as<tirx::LENode>() ||
          expr.as<tirx::GTNode>() || expr.as<tirx::GENode>() || expr.as<tirx::AndNode>() ||
          expr.as<tirx::OrNode>() || expr.as<tirx::NotNode>())) {
      return kUnsupported;
    }
    uint64_t bound = ExprFunctor::VisitExpr(expr);
    expression_bounds_.emplace(expr.get(), bound);
    return bound;
  }

  uint64_t VisitExprDefault_(const ffi::Object*) final { return kUnsupported; }

  uint64_t VisitExpr_(const tirx::VarNode* op) final {
    auto it = bindings_.find(op);
    return it == bindings_.end() ? 1 : it->second;
  }

  uint64_t VisitExpr_(const IntImmNode* op) final {
    if (op->value == std::numeric_limits<int64_t>::min()) return kUnsupported;
    return std::max<int64_t>(1, op->value < 0 ? -op->value : op->value);
  }

  uint64_t VisitExpr_(const tirx::CastNode* op) final {
    PrimType from = op->value.ty();
    PrimType to = op->ty.as_or_throw<PrimType>();
    if (!(from.MatchesCode(kDLInt) && to.MatchesCode(kDLInt) && from.bits() <= to.bits())) {
      return kUnsupported;
    }
    return VisitExpr(op->value);
  }

#define TVM_CONSTR_ADDITIVE_BOUND(Node)             \
  uint64_t VisitExpr_(const tirx::Node* op) final { \
    return Add(VisitExpr(op->a), VisitExpr(op->b)); \
  }
  TVM_CONSTR_ADDITIVE_BOUND(AddNode)
  TVM_CONSTR_ADDITIVE_BOUND(SubNode)
  TVM_CONSTR_ADDITIVE_BOUND(MinNode)
  TVM_CONSTR_ADDITIVE_BOUND(MaxNode)
  TVM_CONSTR_ADDITIVE_BOUND(EQNode)
  TVM_CONSTR_ADDITIVE_BOUND(NENode)
  TVM_CONSTR_ADDITIVE_BOUND(LTNode)
  TVM_CONSTR_ADDITIVE_BOUND(LENode)
  TVM_CONSTR_ADDITIVE_BOUND(GTNode)
  TVM_CONSTR_ADDITIVE_BOUND(GENode)
  TVM_CONSTR_ADDITIVE_BOUND(AndNode)
  TVM_CONSTR_ADDITIVE_BOUND(OrNode)
#undef TVM_CONSTR_ADDITIVE_BOUND

  uint64_t VisitExpr_(const tirx::NotNode* op) final { return VisitExpr(op->a); }

  uint64_t VisitExpr_(const tirx::MulNode* op) final {
    if (!(op->a.as<IntImmNode>() || op->b.as<IntImmNode>())) return kUnsupported;
    return Mul(VisitExpr(op->a), VisitExpr(op->b));
  }

#define TVM_CONSTR_DIVISION_BOUND(Node)                       \
  uint64_t VisitExpr_(const tirx::Node* op) final {           \
    const auto* divisor = op->b.as<IntImmNode>();             \
    if (!divisor || divisor->value <= 0) return kUnsupported; \
    return Mul(VisitExpr(op->a), VisitExpr(op->b));           \
  }
  TVM_CONSTR_DIVISION_BOUND(DivNode)
  TVM_CONSTR_DIVISION_BOUND(ModNode)
  TVM_CONSTR_DIVISION_BOUND(FloorDivNode)
  TVM_CONSTR_DIVISION_BOUND(FloorModNode)
#undef TVM_CONSTR_DIVISION_BOUND

  std::unordered_map<const tirx::VarNode*, uint64_t> bindings_;
  // Memoize within one check to avoid expanding shared expression DAGs.
  // Reset between checks, since bindings and expression lifetimes can change.
  std::unordered_map<const ffi::Object*, uint64_t> expression_bounds_;
};

inline bool IsSupportedConstraintExpr(const PrimExpr& expr) {
  return ConstraintExprValidator().IsSupported(expr);
}

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
    ConstraintExprValidator validator;
    // Rebinding a variable can discard information or introduce inconsistent
    // premises.  Require the consumer to rename independent executions first.
    std::unordered_set<const tirx::VarNode*> bound;
    for (const auto& c : constraints) {
      if (c.kind != Constr::kPredicate && !bound.insert(c.var.get()).second) {
        return false;
      }
      // Validate replay as well as collection: a consumer can construct or
      // rename snapshots without going through ConstrVisitor.
      if (c.kind != Constr::kPredicate) {
        auto var = c.var.as<PrimExpr>();
        if (!var || !IsSupportedConstraintExpr(var.value())) return false;
      }
      if (c.kind == Constr::kBindRange) {
        if (!validator.Bind(c.var, c.range)) return false;
      } else if (c.kind == Constr::kBindValue) {
        if (!validator.Bind(c.var, c.value)) return false;
      } else if (!validator.IsSupported(c.value)) {
        return false;
      }
    }
    if (!validator.IsSupported(predicate)) return false;
    Analyzer analyzer;
    // Congruence alone often separates flat addresses, e.g. even and odd
    // indices.  Try this inexpensive sufficient condition before replaying
    // bindings and entering predicate scopes.  No snapshot facts are assumed.
    if (const auto* ne = predicate.as<tirx::NENode>(); ne && ne->a.ty().MatchesCode(kDLInt)) {
      if (analyzer->modular_set(ne->a - ne->b)->base != 0) return true;
    }
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
 * Only expressions accepted by IsSupportedConstraintExpr are retained.  A Bind
 * of a mutable read or an unsupported conversion does not create a persistent
 * rewrite.  Its variable remains an unconstrained symbol, which a consumer must
 * rename per execution.  Dropping such facts weakens the premises without
 * introducing assumptions about memory snapshots or wrapping arithmetic.
 *
 * Derived visitors that override control flow must use WithConstrScope and add
 * the appropriate premises after visiting conditions and bounds.
 */
class ConstrVisitor : public tirx::StmtExprVisitor {
 public:
  using StmtExprVisitor::VisitExpr_;
  using StmtExprVisitor::VisitStmt_;

  ConstrSet GetConstrSet() const { return {constraints_}; }

  /*!
   * \brief Save predicates and the bindings needed by an expression or predicate.
   *
   * Bindings are in definition order, so a backward pass also finds transitive
   * dependencies.  Keep every predicate, including ones after the definition of
   * a query variable: they may constrain it indirectly through another variable.
   * Unused bindings (often unrelated loop/thread axes or scalar temporaries) do
   * not need to be copied, renamed, validated, and replayed for an address proof.
   * Omitting a premise only weakens the snapshot.
   */
  ConstrSet GetConstrSet(const PrimExpr& expr) const {
    std::unordered_set<const tirx::VarNode*> needed;
    auto add_vars = [&](const PrimExpr& value) {
      tirx::PostOrderVisit(value, [&](const ffi::ObjectRef& node) {
        if (const auto* var = node.as<tirx::VarNode>()) needed.insert(var);
      });
    };
    add_vars(expr);
    for (const auto& c : constraints_) {
      if (c.kind == Constr::kPredicate) add_vars(c.value);
    }
    ConstrSet result;
    for (auto it = constraints_.rbegin(); it != constraints_.rend(); ++it) {
      const auto& c = *it;
      if (c.kind != Constr::kPredicate && !needed.count(c.var.get())) continue;
      result.constraints.push_back(c);
      if (c.kind == Constr::kBindRange) {
        add_vars(c.range->min);
        add_vars(c.range->extent);
      } else if (c.kind == Constr::kBindValue) {
        add_vars(c.value);
      }
    }
    std::reverse(result.constraints.begin(), result.constraints.end());
    return result;
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
    if (IsSupportedConstraintExpr(predicate)) constraints_.emplace_back(predicate);
  }

  void AddBinding(const tirx::Var& var, const Expr& value) {
    if (auto scalar = value.as<PrimExpr>(); scalar && IsSupportedConstraintExpr(scalar.value())) {
      constraints_.emplace_back(var, scalar.value());
    }
  }

  void AddRange(const tirx::Var& var, const Range& range) {
    if (IsSupportedConstraintExpr(range->min) && IsSupportedConstraintExpr(range->extent)) {
      constraints_.emplace_back(var, range);
    }
  }

 private:
  std::vector<Constr> constraints_;
};

}  // namespace arith
}  // namespace tvm
#endif  // TVM_ARITH_CONSTR_VISITOR_H_
