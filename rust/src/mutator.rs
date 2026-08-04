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

//! Safe, statement-structural mutation over generated TIRx bindings.
//!
//! Rebuilt nodes are created through [`crate::ffi_api`] so C++ validation and
//! normalization still run.  The generated native `new` methods are never
//! used.  This first layer intentionally leaves expressions and buffer
//! descriptors unchanged; it is enough for statement-only passes such as
//! SkipAssert and establishes the API shape for a future full StmtExprMutator.

use crate::ffi_api;
use crate::generated::tirx::{
    AllocBufferObj, AssertStmtObj, AttrStmtObj, BindObj, BreakObj, BufferStoreObj, ContinueObj,
    DeclBufferObj, EvaluateObj, ForObj, IfThenElseObj, SBlockObj, SBlockRealizeObj,
    ScopeIdDefStmtObj, SeqStmtObj, Stmt, TilePrimitiveCallObj, WhileObj,
};
use tvm_ffi::{Error, ObjectRefCast, Result};

fn unsupported_stmt(stmt: &Stmt) -> Error {
    let type_index = unsafe {
        let ptr = tvm_ffi::ObjectArc::as_raw(<Stmt as tvm_ffi::ObjectRefCore>::data(stmt))
            as *const tvm_ffi::tvm_ffi_sys::TVMFFIObject;
        (!ptr.is_null()).then(|| (*ptr).type_index)
    };
    let Some(type_index) = type_index else {
        return Error::new(
            tvm_ffi::error::TYPE_ERROR,
            "cannot mutate a null statement ObjectRef",
            "",
        );
    };
    Error::new(
        tvm_ffi::error::TYPE_ERROR,
        &format!(
            "generated Rust statement mutator has no dispatch entry for runtime type index {type_index}"
        ),
        "",
    )
}

macro_rules! dispatch {
    ($mutator:expr, $stmt:expr, $( $node:ty => $method:ident ),+ $(,)?) => {
        $(
            if let Some(node) = $crate::visitor::try_downcast::<_, $node>($stmt) {
                return $mutator.$method($stmt, node);
            }
        )+
    };
}

/// Mutate the statement-bearing part of a TIRx tree.
///
/// Typed overrides own recursion for their node, as with C++ `StmtMutator`.
/// Default implementations preserve pointer identity when no child changed.
pub trait StatementMutator {
    fn mutate_stmt(&mut self, stmt: &Stmt) -> Result<Stmt> {
        dispatch!(
            self,
            stmt,
            BindObj => mutate_bind,
            AttrStmtObj => mutate_attr_stmt,
            IfThenElseObj => mutate_if_then_else,
            ForObj => mutate_for,
            WhileObj => mutate_while,
            BreakObj => mutate_break,
            ContinueObj => mutate_continue,
            AllocBufferObj => mutate_alloc_buffer,
            DeclBufferObj => mutate_decl_buffer,
            BufferStoreObj => mutate_buffer_store,
            AssertStmtObj => mutate_assert_stmt,
            SeqStmtObj => mutate_seq_stmt,
            EvaluateObj => mutate_evaluate,
            SBlockObj => mutate_sblock,
            SBlockRealizeObj => mutate_sblock_realize,
            ScopeIdDefStmtObj => mutate_scope_id_def_stmt,
            TilePrimitiveCallObj => mutate_tile_primitive_call,
        );
        Err(unsupported_stmt(stmt))
    }

    fn mutate_bind(&mut self, original: &Stmt, _node: &BindObj) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_attr_stmt(&mut self, original: &Stmt, node: &AttrStmtObj) -> Result<Stmt> {
        let body = self.mutate_stmt(&node.body)?;
        if body.same_as(&node.body) {
            Ok(original.clone())
        } else {
            Ok(ffi_api::attr_stmt(
                &node.node,
                &node.attr_key,
                &node.value,
                &body,
                Some(&node.span),
            )?
            .into())
        }
    }

    fn mutate_if_then_else(&mut self, original: &Stmt, node: &IfThenElseObj) -> Result<Stmt> {
        let then_case = self.mutate_stmt(&node.then_case)?;
        let old_else = node.else_case.get();
        let else_case = match old_else.as_ref() {
            Some(stmt) => Some(self.mutate_stmt(stmt)?),
            None => None,
        };
        let else_unchanged = match (old_else.as_ref(), else_case.as_ref()) {
            (None, None) => true,
            (Some(before), Some(after)) => before.same_as(after),
            _ => false,
        };
        if then_case.same_as(&node.then_case) && else_unchanged {
            Ok(original.clone())
        } else {
            Ok(ffi_api::if_then_else(
                &node.condition,
                &then_case,
                else_case.as_ref(),
                Some(&node.span),
            )?
            .into())
        }
    }

    fn mutate_for(&mut self, original: &Stmt, node: &ForObj) -> Result<Stmt> {
        let body = self.mutate_stmt(&node.body)?;
        if body.same_as(&node.body) {
            Ok(original.clone())
        } else {
            Ok(ffi_api::for_loop(
                &node.loop_var,
                &node.min,
                &node.extent,
                node.kind,
                &body,
                &node.thread_binding,
                &node.annotations,
                &node.step,
                Some(&node.span),
            )?
            .into())
        }
    }

    fn mutate_while(&mut self, original: &Stmt, node: &WhileObj) -> Result<Stmt> {
        let body = self.mutate_stmt(&node.body)?;
        if body.same_as(&node.body) {
            Ok(original.clone())
        } else {
            Ok(ffi_api::while_loop(&node.condition, &body, Some(&node.span))?.into())
        }
    }

    fn mutate_break(&mut self, original: &Stmt, _node: &BreakObj) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_continue(&mut self, original: &Stmt, _node: &ContinueObj) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_alloc_buffer(&mut self, original: &Stmt, _node: &AllocBufferObj) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_decl_buffer(&mut self, original: &Stmt, _node: &DeclBufferObj) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_buffer_store(&mut self, original: &Stmt, _node: &BufferStoreObj) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_assert_stmt(&mut self, original: &Stmt, _node: &AssertStmtObj) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_seq_stmt(&mut self, original: &Stmt, node: &SeqStmtObj) -> Result<Stmt> {
        let mut changed = false;
        let mut sequence = Vec::with_capacity(node.seq.len());
        for before in &node.seq {
            let after = self.mutate_stmt(&before)?;
            changed |= !before.same_as(&after);
            sequence.push(after);
        }
        if changed {
            ffi_api::normalize_seq(sequence, Some(&node.span))
        } else {
            // C++ StmtMutator always runs SeqStmt::Flatten, even when every
            // child is pointer-identical.  Passing the original as the single
            // root lets normalize_seq preserve identity when it is already
            // canonical while still removing latent nested/no-op entries.
            ffi_api::normalize_seq([original.clone()], Some(&node.span))
        }
    }

    fn mutate_evaluate(&mut self, original: &Stmt, _node: &EvaluateObj) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_sblock(&mut self, original: &Stmt, node: &SBlockObj) -> Result<Stmt> {
        let old_init = node.init.get();
        let init = match old_init.as_ref() {
            Some(stmt) => Some(self.mutate_stmt(stmt)?),
            None => None,
        };
        let body = self.mutate_stmt(&node.body)?;
        let init_unchanged = match (old_init.as_ref(), init.as_ref()) {
            (None, None) => true,
            (Some(before), Some(after)) => before.same_as(after),
            _ => false,
        };
        if init_unchanged && body.same_as(&node.body) {
            Ok(original.clone())
        } else {
            Ok(ffi_api::sblock(
                &node.iter_vars,
                &node.reads,
                &node.writes,
                &node.name_hint,
                &body,
                init.as_ref(),
                &node.alloc_buffers,
                &node.match_buffers,
                &node.annotations,
                Some(&node.span),
            )?
            .into())
        }
    }

    fn mutate_sblock_realize(&mut self, original: &Stmt, node: &SBlockRealizeObj) -> Result<Stmt> {
        let old_block: Stmt = node.block.clone().into();
        let block = self.mutate_stmt(&old_block)?;
        if block.same_as(&old_block) {
            Ok(original.clone())
        } else {
            let block = block.try_cast()?;
            Ok(ffi_api::sblock_realize(
                &node.iter_values,
                &node.predicate,
                &block,
                Some(&node.span),
            )?
            .into())
        }
    }

    fn mutate_scope_id_def_stmt(
        &mut self,
        original: &Stmt,
        _node: &ScopeIdDefStmtObj,
    ) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_tile_primitive_call(
        &mut self,
        _original: &Stmt,
        _node: &TilePrimitiveCallObj,
    ) -> Result<Stmt> {
        Err(Error::new(
            tvm_ffi::error::TYPE_ERROR,
            "TilePrimitiveCall mutation requires Array<Any>/Map<String, Any>; generated bindings currently narrow these fields to ObjectRef",
            "",
        ))
    }
}
