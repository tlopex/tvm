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
    AllocBuffer, AssertStmt, AttrStmt, Bind, Break, BufferStore, Continue, DeclBuffer, Evaluate,
    For, IfThenElse, SBlock, SBlockRealize, ScopeIdDefStmt, SeqStmt, Stmt, TilePrimitiveCall,
    While,
};
use tvm_ffi::{AnyValue, Array, Error, ObjectRefCast, ObjectRefCore, Result};

fn type_error(message: impl AsRef<str>) -> Error {
    Error::new(tvm_ffi::error::TYPE_ERROR, message.as_ref(), "")
}

fn required<T>(value: Option<T>, path: &str) -> Result<T> {
    value.ok_or_else(|| type_error(format!("required IR field `{path}` is null")))
}

fn required_stmt_item(values: &Array<Option<Stmt>>, index: usize, path: &str) -> Result<Stmt> {
    let value = values.get(index).map_err(|error| {
        type_error(format!(
            "cannot decode required IR array element `{path}[{index}]`: {error}"
        ))
    })?;
    required(value, &format!("{path}[{index}]"))
}

fn any_array_item(values: &Array<AnyValue>, index: usize, path: &str) -> Result<AnyValue> {
    values.get(index).map_err(|error| {
        type_error(format!(
            "cannot decode heterogeneous IR array element `{path}[{index}]`: {error}"
        ))
    })
}

fn mutate_tile_primitive_value<M: StatementMutator + ?Sized>(
    mutator: &mut M,
    value: AnyValue,
) -> Result<(AnyValue, bool)> {
    let Some(object) = value.try_as::<tvm_ffi::object::ObjectRef>() else {
        return Ok((value, false));
    };
    let Some(before) = crate::visitor::try_downcast::<_, Stmt>(&object) else {
        // This is a statement-only mutator. Expressions, buffers, scalars,
        // configuration objects, and unknown objects remain opaque payloads.
        return Ok((value, false));
    };

    let after = mutator.mutate_stmt(&before)?;
    if before.same_as(&after) {
        Ok((value, false))
    } else {
        Ok((AnyValue::from_value(after), true))
    }
}

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
                return $mutator.$method($stmt, &node);
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
            Bind => mutate_bind,
            AttrStmt => mutate_attr_stmt,
            IfThenElse => mutate_if_then_else,
            For => mutate_for,
            While => mutate_while,
            Break => mutate_break,
            Continue => mutate_continue,
            AllocBuffer => mutate_alloc_buffer,
            DeclBuffer => mutate_decl_buffer,
            BufferStore => mutate_buffer_store,
            AssertStmt => mutate_assert_stmt,
            SeqStmt => mutate_seq_stmt,
            Evaluate => mutate_evaluate,
            SBlock => mutate_sblock,
            SBlockRealize => mutate_sblock_realize,
            ScopeIdDefStmt => mutate_scope_id_def_stmt,
            TilePrimitiveCall => mutate_tile_primitive_call,
        );
        Err(unsupported_stmt(stmt))
    }

    fn mutate_bind(&mut self, original: &Stmt, _node: &Bind) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_attr_stmt(&mut self, original: &Stmt, node: &AttrStmt) -> Result<Stmt> {
        let attr_node = node.node()?;
        let attr_key = node.attr_key()?;
        let value = required(node.value()?, "AttrStmt::value")?;
        let old_body = required(node.body()?, "AttrStmt::body")?;
        let span = node.span()?;

        let body = self.mutate_stmt(&old_body)?;
        if body.same_as(&old_body) {
            Ok(original.clone())
        } else {
            Ok(ffi_api::attr_stmt(&attr_node, &attr_key, &value, &body, span.as_ref())?.into())
        }
    }

    fn mutate_if_then_else(&mut self, original: &Stmt, node: &IfThenElse) -> Result<Stmt> {
        let condition = required(node.condition()?, "IfThenElse::condition")?;
        let old_then = required(node.then_case()?, "IfThenElse::then_case")?;
        let old_else = node.else_case()?;
        let span = node.span()?;

        let then_case = self.mutate_stmt(&old_then)?;
        let else_case = match old_else.as_ref() {
            Some(stmt) => Some(self.mutate_stmt(stmt)?),
            None => None,
        };
        let else_unchanged = match (old_else.as_ref(), else_case.as_ref()) {
            (None, None) => true,
            (Some(before), Some(after)) => before.same_as(after),
            _ => false,
        };
        if then_case.same_as(&old_then) && else_unchanged {
            Ok(original.clone())
        } else {
            Ok(
                ffi_api::if_then_else(&condition, &then_case, else_case.as_ref(), span.as_ref())?
                    .into(),
            )
        }
    }

    fn mutate_for(&mut self, original: &Stmt, node: &For) -> Result<Stmt> {
        let loop_var = required(node.loop_var()?, "For::loop_var")?;
        let min = required(node.min()?, "For::min")?;
        let extent = required(node.extent()?, "For::extent")?;
        let kind = node.kind()?;
        let old_body = required(node.body()?, "For::body")?;
        let thread_binding = node.thread_binding()?;
        let annotations = node.annotations()?;
        let step = node.step()?;
        let span = node.span()?;

        let body = self.mutate_stmt(&old_body)?;
        if body.same_as(&old_body) {
            Ok(original.clone())
        } else {
            Ok(ffi_api::for_loop(
                &loop_var,
                &min,
                &extent,
                kind,
                &body,
                thread_binding.as_ref(),
                &annotations,
                step.as_ref(),
                span.as_ref(),
            )?
            .into())
        }
    }

    fn mutate_while(&mut self, original: &Stmt, node: &While) -> Result<Stmt> {
        let condition = required(node.condition()?, "While::condition")?;
        let old_body = required(node.body()?, "While::body")?;
        let span = node.span()?;

        let body = self.mutate_stmt(&old_body)?;
        if body.same_as(&old_body) {
            Ok(original.clone())
        } else {
            Ok(ffi_api::while_loop(&condition, &body, span.as_ref())?.into())
        }
    }

    fn mutate_break(&mut self, original: &Stmt, _node: &Break) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_continue(&mut self, original: &Stmt, _node: &Continue) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_alloc_buffer(&mut self, original: &Stmt, _node: &AllocBuffer) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_decl_buffer(&mut self, original: &Stmt, _node: &DeclBuffer) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_buffer_store(&mut self, original: &Stmt, _node: &BufferStore) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_assert_stmt(&mut self, original: &Stmt, _node: &AssertStmt) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_seq_stmt(&mut self, original: &Stmt, node: &SeqStmt) -> Result<Stmt> {
        let old_sequence = node.seq()?;
        let span = node.span()?;
        let mut changed = false;
        let mut sequence = Vec::with_capacity(old_sequence.len());
        for index in 0..old_sequence.len() {
            let before = required_stmt_item(&old_sequence, index, "SeqStmt::seq")?;
            let after = self.mutate_stmt(&before)?;
            changed |= !before.same_as(&after);
            sequence.push(after);
        }
        if changed {
            ffi_api::normalize_seq(sequence, span.as_ref())
        } else {
            Ok(original.clone())
        }
    }

    fn mutate_evaluate(&mut self, original: &Stmt, _node: &Evaluate) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_sblock(&mut self, original: &Stmt, node: &SBlock) -> Result<Stmt> {
        let iter_vars = node.iter_vars()?;
        let reads = node.reads()?;
        let writes = node.writes()?;
        let name_hint = node.name_hint()?;
        let old_body = required(node.body()?, "SBlock::body")?;
        let old_init = node.init()?;
        let alloc_buffers = node.alloc_buffers()?;
        let match_buffers = node.match_buffers()?;
        let annotations = node.annotations()?;
        let span = node.span()?;

        let init = match old_init.as_ref() {
            Some(stmt) => Some(self.mutate_stmt(stmt)?),
            None => None,
        };
        let body = self.mutate_stmt(&old_body)?;
        let init_unchanged = match (old_init.as_ref(), init.as_ref()) {
            (None, None) => true,
            (Some(before), Some(after)) => before.same_as(after),
            _ => false,
        };
        if init_unchanged && body.same_as(&old_body) {
            Ok(original.clone())
        } else {
            Ok(ffi_api::sblock(
                &iter_vars,
                &reads,
                &writes,
                &name_hint,
                &body,
                init.as_ref(),
                &alloc_buffers,
                &match_buffers,
                &annotations,
                span.as_ref(),
            )?
            .into())
        }
    }

    fn mutate_sblock_realize(&mut self, original: &Stmt, node: &SBlockRealize) -> Result<Stmt> {
        let iter_values = node.iter_values()?;
        let predicate = required(node.predicate()?, "SBlockRealize::predicate")?;
        let old_block = required(node.block()?, "SBlockRealize::block")?;
        let span = node.span()?;

        let old_block: Stmt = old_block.into();
        let block = self.mutate_stmt(&old_block)?;
        if block.same_as(&old_block) {
            Ok(original.clone())
        } else {
            let block = block.try_cast()?;
            Ok(ffi_api::sblock_realize(&iter_values, &predicate, &block, span.as_ref())?.into())
        }
    }

    fn mutate_scope_id_def_stmt(
        &mut self,
        original: &Stmt,
        _node: &ScopeIdDefStmt,
    ) -> Result<Stmt> {
        Ok(original.clone())
    }

    fn mutate_tile_primitive_call(
        &mut self,
        original: &Stmt,
        node: &TilePrimitiveCall,
    ) -> Result<Stmt> {
        let op = node.op()?;
        let old_args = node.args()?;
        let workspace = node.workspace()?;
        let config = node.config()?;
        let dispatch_token = node.dispatch()?;
        let scope = node.scope()?;

        let mut changed = false;
        let mut args = Vec::with_capacity(old_args.len());
        for index in 0..old_args.len() {
            let value = any_array_item(&old_args, index, "TilePrimitiveCall::args")?;
            let (value, value_changed) = mutate_tile_primitive_value(self, value)?;
            changed |= value_changed;
            args.push(value);
        }

        let mut config_changed = false;
        let mut config_entries = Vec::with_capacity(config.len());
        for (key, value) in &config {
            let (value, value_changed) = mutate_tile_primitive_value(self, value)?;
            config_changed |= value_changed;
            config_entries.push((key, value));
        }

        if !changed && !config_changed {
            return Ok(original.clone());
        }

        let args = if changed { Array::new(args) } else { old_args };
        let config = if config_changed {
            config_entries.into_iter().collect()
        } else {
            config
        };
        Ok(ffi_api::tile_primitive_call(
            &op,
            &args,
            &workspace,
            &config,
            dispatch_token.as_ref(),
            scope.as_ref(),
        )?
        .into())
    }
}
