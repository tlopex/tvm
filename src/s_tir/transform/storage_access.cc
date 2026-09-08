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
 * \file storage_access.cc
 */
#include "storage_access.h"

#include <tvm/ffi/cast.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/op.h>

#include <string>
#include <utility>

#include "../../tirx/transform/ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

void StorageAccessVisitor::VisitExpr_(const BufferLoadNode* op) {
  Var buf = op->buffer->data;
  StorageScope scope = GetScope(buf);
  if (Enabled(buf.get(), scope)) {
    TVM_FFI_ICHECK(allow_append_) << op << " " << scope.to_string();
    AccessEntry e;
    e.threads = env_threads();
    e.buffer = buf;
    e.dtype = op->ty.as_or_throw<PrimType>().WithLanes(1);
    e.can_prove_disjoint =
        CanProveDisjoint(op->buffer, op->indices, op->ty.as_or_throw<PrimType>());
    if (e.can_prove_disjoint) e.constraints = GetConstrSet();
    for (const auto& index : op->indices) {
      e.touched.push_back(arith::IntSet::Vector(index));
    }
    e.type = kRead;
    e.scope = scope;
    curr_stmt_.access.emplace_back(std::move(e));
  }
  // traverse child
  StmtExprVisitor::VisitExpr_(op);
}

void StorageAccessVisitor::VisitStmt_(const BufferStoreNode* op) {
  allow_append_ = true;
  TVM_FFI_ICHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;

  Var buf = op->buffer->data;
  StorageScope scope = GetScope(buf);
  if (Enabled(buf.get(), scope)) {
    AccessEntry e;
    e.threads = env_threads();
    e.buffer = buf;
    e.dtype = op->value.ty().WithLanes(1);
    e.can_prove_disjoint = CanProveDisjoint(op->buffer, op->indices, op->value.ty());
    if (e.can_prove_disjoint) e.constraints = GetConstrSet();
    for (const auto& index : op->indices) {
      e.touched.push_back(arith::IntSet::Vector(index));
    }
    e.type = kWrite;
    e.scope = scope;
    curr_stmt_.access.emplace_back(std::move(e));
  }
  // traverse child
  StmtExprVisitor::VisitStmt_(op);
  // push to the scope
  scope_.back().push_back(curr_stmt_);
  // clear access entry.
  curr_stmt_.access.clear();
  allow_append_ = false;
}

void StorageAccessVisitor::VisitStmt_(const EvaluateNode* op) {
  allow_append_ = true;
  TVM_FFI_ICHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;
  StmtExprVisitor::VisitStmt_(op);
  // push to the scope
  if (curr_stmt_.access.size() != 0) {
    scope_.back().push_back(curr_stmt_);
    curr_stmt_.access.clear();
  }
  allow_append_ = false;
}

void StorageAccessVisitor::VisitStmt_(const BindNode* op) {
  allow_append_ = true;
  TVM_FFI_ICHECK_EQ(curr_stmt_.access.size(), 0U);
  curr_stmt_.stmt = op;
  arith::ConstrVisitor::VisitStmt_(op);
  // push to the scope
  scope_.back().push_back(curr_stmt_);
  // clear access entry.
  curr_stmt_.access.clear();
  allow_append_ = false;
}

void StorageAccessVisitor::VisitStmt_(const AssertStmtNode* op) {
  allow_append_ = true;
  TVM_FFI_ICHECK(curr_stmt_.access.empty());
  curr_stmt_.stmt = op;
  arith::ConstrVisitor::VisitStmt_(op);
  if (!curr_stmt_.access.empty()) {
    scope_.back().push_back(curr_stmt_);
    curr_stmt_.access.clear();
  }
  allow_append_ = false;
}

void StorageAccessVisitor::VisitStmt_(const AttrStmtNode* op) {
  if (op->attr_key == s_tir::attr::double_buffer_write) {
    TVM_FFI_ICHECK(double_buffer_write_ == nullptr);
    double_buffer_write_ = op->node.as<VarNode>();
    scope_.push_back(std::vector<StmtEntry>());
    arith::ConstrVisitor::VisitStmt_(op);
    StmtEntry s;
    s.stmt = op;
    s.access = Summarize(std::move(scope_.back()), nullptr);
    scope_.pop_back();
    if (!s.access.empty()) {
      for (AccessEntry& e : s.access) {
        if (e.type == kWrite && e.buffer.get() == double_buffer_write_) {
          e.double_buffer_write = true;
        }
      }
      scope_.back().emplace_back(std::move(s));
    }
    double_buffer_write_ = nullptr;
  } else if (op->attr_key == tirx::attr::thread_extent) {
    IterVar iv = op->node.as_or_throw<IterVar>();
    env_threads_.push_back(iv);
    if (!in_device_env_) {
      in_device_env_ = true;
      scope_.push_back(std::vector<StmtEntry>());
      arith::ConstrVisitor::VisitStmt_(op);
      // no need to take the result as the thread barrier automatically syncs.
      Summarize(std::move(scope_.back()), nullptr);
      in_device_env_ = false;
      scope_.pop_back();
    } else {
      arith::ConstrVisitor::VisitStmt_(op);
    }
    env_threads_.pop_back();
  } else if (op->attr_key == s_tir::attr::hand_threaded) {
    // skip this pass on blocks that were hand_threaded
    // this avoids control flow and read/write conflicts
    // between hand-threaded kernels and automatic threading
  } else {
    arith::ConstrVisitor::VisitStmt_(op);
  }
}

void StorageAccessVisitor::VisitStmt_(const ForNode* op) {
  bool was_inside_loop = inside_loop_;
  inside_loop_ = true;
  scope_.push_back(std::vector<StmtEntry>());
  arith::ConstrVisitor::VisitStmt_(op);
  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), op);
  scope_.pop_back();
  if (s.access.size() != 0) {
    // relax the touched set to contain all ranges in the loop.
    std::unordered_map<const VarNode*, arith::IntSet> relax_map;
    relax_map[op->loop_var.get()] =
        arith::IntSet::FromRange(Range::FromMinExtent(op->min, op->extent));
    for (AccessEntry& e : s.access) {
      if (e.buffer.defined()) {
        TVM_FFI_ICHECK(e.touched.size());
        ffi::Array<arith::IntSet> new_touched;
        for (const auto& touched : e.touched) {
          new_touched.push_back(arith::EvalSet(touched, relax_map));
        }
        e.touched = std::move(new_touched);
      }
    }
  }
  if (!s.access.empty()) {
    scope_.back().emplace_back(std::move(s));
  }
  inside_loop_ = was_inside_loop;
}

bool IsThreadInvariant(const PrimExpr& cond) {
  if (auto call = cond.as<CallNode>()) {
    if (auto opt_call_op = call->op.as<Op>()) {
      auto call_op = opt_call_op.value();
      if (call_op.same_as(builtin::tvm_thread_invariant())) {
        return true;
      }
    }
  }
  return false;
}

void StorageAccessVisitor::VisitStmt_(const IfThenElseNode* op) {
  auto condition_reads = VisitCondition(op->condition);
  bool is_thread_invariant = IsThreadInvariant(op->condition);
  if (!is_thread_invariant) {
    ++condition_counter_;
  }
  scope_.push_back(std::vector<StmtEntry>());
  WithConstrScope([&]() {
    AddConstraint(op->condition);
    this->VisitStmt(op->then_case);
  });
  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), nullptr);
  s.access.insert(s.access.begin(), condition_reads.begin(), condition_reads.end());
  scope_.pop_back();
  if (op->else_case) {
    scope_.push_back(std::vector<StmtEntry>());
    WithConstrScope([&]() {
      AddConstraint(Not(op->condition));
      this->VisitStmt(op->else_case.value());
    });
    auto v = Summarize(std::move(scope_.back()), nullptr);
    scope_.pop_back();
    s.access.insert(s.access.end(), v.begin(), v.end());
  }
  scope_.back().emplace_back(std::move(s));
  if (!is_thread_invariant) {
    --condition_counter_;
  }
}

void StorageAccessVisitor::VisitStmt_(const WhileNode* op) {
  bool was_inside_loop = inside_loop_;
  inside_loop_ = true;
  auto condition_reads = VisitCondition(op->condition);
  bool is_thread_invariant = IsThreadInvariant(op->condition);
  if (!is_thread_invariant) {
    ++condition_counter_;
  }
  scope_.push_back(std::vector<StmtEntry>());
  WithConstrScope([&]() {
    AddConstraint(op->condition);
    this->VisitStmt(op->body);
  });
  StmtEntry s;
  s.stmt = op;
  s.access = Summarize(std::move(scope_.back()), nullptr);
  s.access.insert(s.access.begin(), condition_reads.begin(), condition_reads.end());
  scope_.pop_back();
  scope_.back().emplace_back(std::move(s));
  if (!is_thread_invariant) {
    --condition_counter_;
  }
  inside_loop_ = was_inside_loop;
}

void StorageAccessVisitor::VisitExpr_(const CallNode* op) {
  if (op->op.same_as(builtin::address_of())) {
    if (const auto* load = op->args[0].as<BufferLoadNode>()) {
      // Taking an address does not read the buffer value.  Visit only the
      // load's children so index expressions still contribute accesses.
      StmtExprVisitor::VisitExpr_(load);
    } else {
      // address_of also accepts scalar variables (e.g. tcgen registers).
      // Recurse without assuming the argument is a BufferLoad.
      StmtExprVisitor::VisitExpr_(op);
    }
  } else if (op->op.same_as(builtin::tvm_access_ptr())) {
    TVM_FFI_ICHECK_EQ(op->args.size(), 5U);
    PrimType dtype = op->args[0].as_or_throw<PrimExpr>().ty();
    const VarNode* buffer = op->args[1].as<VarNode>();
    if (buffer == nullptr) {
      // args[1] is not a raw Var — e.g. a nested tvm_access_ptr or some
      // other PrimExpr. Recurse into sub-exprs so any inner buffer var
      // refs still get visited, but don't try to record an access entry
      // here (GetScope on a null Var would dereference a null pointer).
      StmtExprVisitor::VisitExpr_(op);
      return;
    }
    PrimExpr offset = op->args[2].as_or_throw<PrimExpr>();
    PrimExpr extent = op->args[3].as_or_throw<PrimExpr>();
    const IntImmNode* flag = op->args[4].as<IntImmNode>();
    StorageScope scope = GetScope(ffi::GetRef<Var>(buffer));
    // The buffer scope.
    if (Enabled(buffer, scope)) {
      TVM_FFI_ICHECK(allow_append_);
      AccessEntry e;
      e.threads = env_threads();
      e.dtype = dtype;
      e.buffer = ffi::GetRef<Var>(buffer);
      e.touched = {arith::IntSet::FromRange(Range::FromMinExtent(offset, extent))};
      e.scope = scope;
      if (flag->value & 1) {
        e.type = kRead;
        curr_stmt_.access.emplace_back(e);
      }
      if (flag->value & 2) {
        e.type = kWrite;
        curr_stmt_.access.emplace_back(e);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  } else if (op->op.same_as(builtin::tvm_storage_sync())) {
    TVM_FFI_ICHECK(allow_append_);
    const std::string& s = op->args[0].as<StringImmNode>()->value;
    if (s != "warp") {
      StorageScope scope = StorageScope::Create(s);
      AccessEntry e;
      e.threads = env_threads();
      e.type = kSync;
      e.scope = StorageScope::Create(s);
      curr_stmt_.access.emplace_back(std::move(e));
    }
  } else {
    arith::ConstrVisitor::VisitExpr_(op);
  }
}

std::vector<StorageAccessVisitor::AccessEntry> StorageAccessVisitor::VisitCondition(
    const PrimExpr& condition) {
  TVM_FFI_ICHECK(curr_stmt_.access.empty());
  allow_append_ = true;
  this->VisitExpr(condition);
  std::vector<AccessEntry> result;
  result.swap(curr_stmt_.access);
  allow_append_ = false;
  return result;
}

bool StorageAccessVisitor::CanProveDisjoint(const Buffer& buffer,
                                            const ffi::Array<PrimExpr>& indices,
                                            PrimType access_type) const {
  // Strided/offset aliases, vectors, access_ptr and dynamic shared allocations
  // need a byte-address model.  Loop accesses also need an iteration model.
  return !inside_loop_ && access_type.IsScalar() && indices.size() == 1 &&
         buffer->strides.empty() && is_zero(buffer->elem_offset) &&
         GetScope(buffer->data).tag.empty() && IsPureScalar(indices[0]);
}

StorageScope StorageAccessVisitor::GetScope(Var buffer_var) const {
  if (buffer_var->ty.as<PointerTypeNode>()) {
    return StorageScope::Create(GetPtrStorageScope(buffer_var));
  }
  return StorageScope();  // global by default
}

}  // namespace s_tir
}  // namespace tvm
