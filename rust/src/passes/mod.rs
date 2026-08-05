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

//! Passes implemented in Rust on top of the generated TIRx bindings.

mod simplify;
mod skip_assert;
mod verify_ssa;

pub use simplify::{simplify_pass, simplify_prim_func, simplify_stmt_expressions};
pub use skip_assert::{skip_assert, skip_assert_pass, skip_assert_prim_func};
pub use verify_ssa::{verify_ssa, verify_ssa_module, verify_ssa_or_error, verify_ssa_pass};
