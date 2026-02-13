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
#include <tvm/script/ir_builder/ir/ir.h>
#include <tvm/script/ir_builder/tir/frame.h>
#include <tvm/tir/function.h>

#include "../../../tir/ir/script/script_complete.h"
#include "./utils.h"

namespace tvm {
namespace script {
namespace ir_builder {
namespace tir {

TVM_FFI_STATIC_INIT_BLOCK() {
  TIRFrameNode::RegisterReflection();
  PrimFuncFrameNode::RegisterReflection();
  SBlockFrameNode::RegisterReflection();
  ExecScopeFrameNode::RegisterReflection();
  BlockInitFrameNode::RegisterReflection();
  ForFrameNode::RegisterReflection();
  AssertFrameNode::RegisterReflection();
  LaunchThreadFrameNode::RegisterReflection();
  AttrFrameNode::RegisterReflection();
  WhileFrameNode::RegisterReflection();
  IfFrameNode::RegisterReflection();
  ThenFrameNode::RegisterReflection();
  ElseFrameNode::RegisterReflection();
  ComposeOpFrameNode::RegisterReflection();
  DeclBufferFrameNode::RegisterReflection();
  ComposeOpFrameNode::RegisterReflection();
  AllocBufferFrameNode::RegisterReflection();
}

void PrimFuncFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  // if the prim func is not private and there isn't already a global symbol,
  // add a global symbol
  auto insert_attr = [&](ffi::String key, ffi::Any value) {
    if (!attrs.defined()) {
      attrs = {{key, value}};
    } else if (!attrs.count(key)) {
      // copy over attributes (can't mutate the dict inside the optional in-place)
      ffi::Map<ffi::String, ffi::Any> new_attrs;
      for (auto kv : attrs) {
        new_attrs.Set(kv.first, kv.second);
      }
      new_attrs.Set(key, value);
      attrs = std::move(new_attrs);
    }
  };
  if (!is_private && name.has_value() && !attrs.count(tvm::attr::kGlobalSymbol)) {
    insert_attr(tvm::attr::kGlobalSymbol, name.value());
  }
  if (is_tirx) {
    insert_attr(tvm::attr::kIsTIRx, tvm::Bool(true));
  }
  tvm::tir::PrimFunc func(
      /*params=*/args,
      /*body=*/AsStmt(stmts),
      /*ret_type=*/ret_type.value_or(TupleType::Empty()),
      /*buffer_map=*/buffer_map,
      /*attrs=*/attrs.defined() ? DictAttrs(attrs) : NullValue<DictAttrs>(),
      /*span=*/tvm::Span());
  func = tvm::tir::ScriptComplete(func, root_alloc_buffers, is_tirx);
  IRBuilder builder = IRBuilder::Current();
  if (builder->frames.empty()) {
    TVM_FFI_CHECK(!builder->result.defined(), ValueError) << "Builder.result has already been set";
    builder->result = func;
  } else if (ffi::Optional<ir::IRModuleFrame> opt_frame = builder->FindFrame<ir::IRModuleFrame>()) {
    TVM_FFI_CHECK(name.has_value(), ValueError)
        << "The function name must be defined before exiting the "
           "function scope, if it's defined in a Module";
    const ir::IRModuleFrame& frame = opt_frame.value();
    const ffi::String& func_name = name.value_or("");
    if (!frame->global_var_map.count(func_name)) {
      // Case. First time visiting the function.
      ir::DeclFunction(func_name, func);
    }
    // Define the function.
    // Note we do checks to disallow redefinition of functions inside the `DefFunction`.
    ir::DefFunction(func_name, func);
  } else {
    TVM_FFI_THROW(ValueError) << "Cannot find where to insert PrimFunc";
  }
}

void SBlockFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  ffi::Array<tvm::tir::Buffer> tir_alloc_buffers;
  for (const tvm::tir::Buffer& buffer : alloc_buffers) {
    tir_alloc_buffers.push_back(buffer);
  }
  ffi::Map<ffi::String, Any> attrs = annotations.value_or({});

  ffi::Optional<PrimFuncFrame> frame = IRBuilder::Current()->FindFrame<PrimFuncFrame>();
  ICHECK(frame.defined()) << "ValueError: Block must be defined within a PrimFunc";
  if (!frame.value()->is_tirx) {
    if (int detect_access = (!reads.defined()) | (!writes.defined() << 1)) {
      attrs.Set("tir.script_parsing_detect_access", tvm::IntImm(DataType::Int(64), detect_access));
    }
  }
  tvm::tir::SBlock block(iter_vars, reads.value_or(ffi::Array<tvm::tir::BufferRegion>()),
                         writes.value_or(ffi::Array<tvm::tir::BufferRegion>()), name, AsStmt(stmts),
                         init, tir_alloc_buffers, match_buffers, attrs, tvm::Span());
  if (no_realize) {
    TVM_FFI_CHECK(iter_values.empty(), ValueError)
        << "Block bindings are not allowed when `no_realize=True`";
    TVM_FFI_CHECK(!predicate.defined(), ValueError)
        << "`T.where` is not allowed when `no_realize=True`";
    AddToParent(block);
  } else {
    AddToParent(tvm::tir::SBlockRealize(iter_values, predicate.value_or(Bool(true)), block));
  }
}

void ExecScopeFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  ICHECK(exec_scope.defined()) << "InternalError: ExecScopeFrame must have an execution scope";
  tvm::tir::Stmt body = AsStmt(stmts);
  if (annotations.defined() && !annotations.value().empty()) {
    // If annotations are present (e.g., tirx.scope_partition), wrap the body in an SBlock
    ffi::Map<ffi::String, ffi::Any> attrs = annotations.value();
    tvm::tir::SBlock block({}, {}, {}, "", body, std::nullopt, {}, {}, attrs, tvm::Span());
    body = tvm::tir::SBlockRealize({}, Bool(true), block);
  }
  AddToParent(tvm::tir::ExecScopeStmt(exec_scope.value(), body));
}

void BlockInitFrameNode::EnterWithScope() {
  SBlockFrame frame = FindSBlockFrame("T.init");
  if (frame->init.defined()) {
    TVM_FFI_THROW(ValueError) << "Duplicate block init declaration";
  }
  TIRFrameNode::EnterWithScope();
}

void BlockInitFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  SBlockFrame frame = FindSBlockFrame("T.init");
  frame->init = AsStmt(stmts);
}

void ForFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(this->f_make_for_loop(vars, doms, steps, AsStmt(stmts)));
}

void AssertFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  if (stmts.empty()) {
    AddToParent(tvm::tir::AssertStmt(condition, error_kind, message_parts));
  } else {
    ffi::Array<tvm::tir::Stmt> seq;
    seq.push_back(tvm::tir::AssertStmt(condition, error_kind, message_parts));
    for (const auto& stmt : stmts) {
      seq.push_back(stmt);
    }
    AddToParent(tvm::tir::SeqStmt(seq));
  }
}

void LaunchThreadFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::AttrStmt(iter_var, attr_key, extent, AsStmt(stmts)));
}

void AttrFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::AttrStmt(node, attr_key, value, AsStmt(stmts)));
}

void WhileFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::While(condition, AsStmt(stmts)));
}

void IfFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  if (!stmts.empty()) {
    TVM_FFI_THROW(InternalError)
        << "stmt within IfThenElse frame should be either in ThenFrame or ElseFrame";
  }
  if (!then_stmts.defined()) {
    TVM_FFI_THROW(InternalError) << "IfThenElse frame should have at least one then branch";
  }
  AddToParent(tvm::tir::IfThenElse(
      condition, AsStmt(then_stmts.value()),
      else_stmts.defined() ? AsStmt(else_stmts.value()) : tvm::tir::Stmt(nullptr)));
}

void ThenFrameNode::EnterWithScope() {
  IfFrame frame = FindIfFrame("T.then_");
  if (frame->then_stmts.defined()) {
    TVM_FFI_THROW(ValueError) << "Duplicate then branch declaration, previous one is "
                              << frame->then_stmts.value();
  }
  TIRFrameNode::EnterWithScope();
}

void ThenFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  FindIfFrame("T.then_")->then_stmts = stmts;
}

void ElseFrameNode::EnterWithScope() {
  IfFrame frame = FindIfFrame("T.else_");
  if (!frame->then_stmts.defined()) {
    TVM_FFI_THROW(InternalError) << "The else branch should follow then branch";
  }
  if (frame->else_stmts.defined()) {
    TVM_FFI_THROW(ValueError) << "Duplicate else branch declaration, previous one is "
                              << frame->else_stmts.value();
  }
  TIRFrameNode::EnterWithScope();
}

void ElseFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  FindIfFrame("T.else_")->else_stmts = stmts;
}

void DeclBufferFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  if (allocated) {
    AddToParent(tvm::tir::DeclBuffer(buffer, AsStmt(stmts)));
  } else {
    AddToParent(tvm::tir::Allocate(buffer->data, buffer->dtype, buffer->shape,
                                   tvm::IntImm(DataType::Bool(), 1),
                                   tvm::tir::DeclBuffer(buffer, AsStmt(stmts))));
  }
}

void ComposeOpFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  ffi::Array<ObjectRef> ops;
  for (const auto& stmt : stmts) {
    auto op_call = stmt.as<tvm::tir::tirx::OpCallNode>();
    ICHECK(op_call) << "ValueError: Only TIRx op calls allowed in ComposeOp. Violated by " << stmt;
    ops.push_back(ffi::GetRef<tvm::tir::tirx::OpCall>(op_call));
  }
  auto compose_op_op = tvm::Op::Get("tirx.compose_op");
  AddToParent(tvm::tir::tirx::OpCall(compose_op_op, ops, workspace, config, dispatch));
}

void AllocBufferFrameNode::ExitWithScope() {
  TIRFrameNode::ExitWithScope();
  AddToParent(tvm::tir::AllocBuffer(buffer, AsStmt(stmts)));
}

}  // namespace tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm
