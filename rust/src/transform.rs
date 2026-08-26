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

use tvm_ffi::derive::{Object, ObjectRef};
use tvm_ffi::{Any, Array, Error, Function, ObjectArc, Result, String, RUNTIME_ERROR};

use crate::ir::IRModule;
use crate::relax::RelaxFunction;
use crate::tirx::PrimFunc;

mod eliminate_unit_loops;
mod fold_integer_constants;
mod increment_int_immediates;
mod prune_unreachable_functions;
mod rename_bound_variables;
mod simplify_add_zero;
mod simplify_known_control_flow;
mod simplify_neutral_elements;
mod skip_assert;
mod utils;

pub use eliminate_unit_loops::{eliminate_unit_loops, eliminate_unit_loops_prim_func};
pub use fold_integer_constants::{
    fold_integer_constants, fold_integer_constants_expr, fold_integer_constants_prim_func,
};
pub use increment_int_immediates::increment_int_immediates;
pub use prune_unreachable_functions::{
    prune_unreachable_functions, prune_unreachable_functions_from_main,
    prune_unreachable_functions_pass,
};
pub use rename_bound_variables::{
    rename_bound_variables, rename_bound_variables_function, rename_bound_variables_pass,
};
pub use simplify_add_zero::{
    simplify_add_zero, simplify_add_zero_expr, simplify_add_zero_module,
    simplify_add_zero_module_pass, simplify_add_zero_prim_func,
};
pub use simplify_known_control_flow::{
    simplify_known_control_flow, simplify_known_control_flow_prim_func,
};
pub use simplify_neutral_elements::{
    simplify_neutral_elements_expr, simplify_neutral_elements_in_loop_bodies,
    simplify_neutral_elements_prim_func,
};
pub use skip_assert::{skip_assert, skip_assert_prim_func};

/// Opaque Rust view of TVM's `PassNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "transform.Pass"]
pub struct PassObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to a TVM pass.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Pass {
    data: ObjectArc<PassObj>,
}

impl std::ops::Deref for Pass {
    type Target = PassObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Opaque Rust view of TVM's `PassContextNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "transform.PassContext"]
#[type_final]
pub struct PassContextObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to the active TVM pass context.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct PassContext {
    data: ObjectArc<PassContextObj>,
}

impl std::ops::Deref for PassContext {
    type Target = PassContextObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl Pass {
    /// Run this pass on an IRModule using TVM's current PassContext.
    ///
    /// This consumes the Rust module handle. The current Rust FFI still
    /// transports it as an lvalue, so C++ may perform one copy-on-write step at
    /// the boundary.
    pub fn run(&self, module: IRModule) -> Result<IRModule> {
        tvm_ffi::cached_global_func!("transform.RunPass")
            .call_tuple_with_len::<2, _>((self, module))?
            .try_into()
    }
}

/// Construct a TVM PrimFunc pass backed by a Rust callback.
///
/// The callback signature mirrors C++ `CreatePrimFuncPass`: the function is
/// passed as an ABI rvalue reference, followed by the module and pass context.
pub fn create_prim_func_pass<F>(
    name: &str,
    opt_level: i64,
    required: Vec<&str>,
    traceable: bool,
    pass_func: F,
) -> Result<Pass>
where
    F: Fn(PrimFunc, IRModule, PassContext) -> Result<PrimFunc> + 'static,
{
    let pass_func = Function::from_packed(move |args| {
        if args.len() != 3 {
            return Err(Error::new(
                RUNTIME_ERROR,
                "Rust PrimFunc pass expected (PrimFunc, IRModule, PassContext)",
                "",
            ));
        }
        // PrimFunc passes receive their first argument as an ABI RValueRef.
        // Owning the view normalizes that transport wrapper to the object.
        let func = PrimFunc::try_from(Any::from(args[0]))?;
        let module = IRModule::try_from(args[1])?;
        let context = PassContext::try_from(args[2])?;
        Ok(Any::from(pass_func(func, module, context)?))
    });

    let pass_info = create_pass_info(name, opt_level, required, traceable)?;

    tvm_ffi::cached_global_func!("tirx.transform.CreatePrimFuncPass")
        .call_packed(&[
            tvm_ffi::AnyView::from(&pass_func),
            tvm_ffi::AnyView::from(&pass_info),
        ])?
        .try_into()
}

/// Construct a TVM Relax FunctionPass backed by a Rust callback.
pub fn create_relax_function_pass<F>(
    name: &str,
    opt_level: i64,
    required: Vec<&str>,
    traceable: bool,
    pass_func: F,
) -> Result<Pass>
where
    F: Fn(RelaxFunction, IRModule, PassContext) -> Result<RelaxFunction> + 'static,
{
    let pass_func = Function::from_packed(move |args| {
        if args.len() != 3 {
            return Err(Error::new(
                RUNTIME_ERROR,
                "Rust Relax function pass expected (Function, IRModule, PassContext)",
                "",
            ));
        }
        let function = RelaxFunction::try_from(Any::from(args[0]))?;
        let module = IRModule::try_from(args[1])?;
        let context = PassContext::try_from(args[2])?;
        Ok(Any::from(pass_func(function, module, context)?))
    });
    let pass_info = create_pass_info(name, opt_level, required, traceable)?;

    tvm_ffi::cached_global_func!("relax.transform.MakeFunctionPass")
        .call_packed(&[
            tvm_ffi::AnyView::from(&pass_func),
            tvm_ffi::AnyView::from(&pass_info),
        ])?
        .try_into()
}

/// Construct a TVM module pass backed by a Rust callback.
pub fn create_module_pass<F>(
    name: &str,
    opt_level: i64,
    required: Vec<&str>,
    traceable: bool,
    pass_func: F,
) -> Result<Pass>
where
    F: Fn(IRModule, PassContext) -> Result<IRModule> + 'static,
{
    let pass_func = Function::from_packed(move |args| {
        if args.len() != 2 {
            return Err(Error::new(
                RUNTIME_ERROR,
                "Rust module pass expected (IRModule, PassContext)",
                "",
            ));
        }
        // Module passes receive their first argument through the RValueRef ABI.
        let module = IRModule::try_from(Any::from(args[0]))?;
        let context = PassContext::try_from(args[1])?;
        Ok(Any::from(pass_func(module, context)?))
    });
    let pass_info = create_pass_info(name, opt_level, required, traceable)?;

    tvm_ffi::cached_global_func!("transform.MakeModulePass")
        .call_packed(&[
            tvm_ffi::AnyView::from(&pass_func),
            tvm_ffi::AnyView::from(&pass_info),
        ])?
        .try_into()
}

fn create_pass_info(
    name: &str,
    opt_level: i64,
    required: Vec<&str>,
    traceable: bool,
) -> Result<Any> {
    let required = Array::<String>::new(required.into_iter().map(String::from).collect());
    let name = String::from(name);
    tvm_ffi::cached_global_func!("transform.PassInfo").call_packed(&[
        tvm_ffi::AnyView::from(&opt_level),
        tvm_ffi::AnyView::from(&name),
        tvm_ffi::AnyView::from(&required),
        tvm_ffi::AnyView::from(&traceable),
    ])
}
