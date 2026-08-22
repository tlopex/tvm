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
use tvm_ffi::{
    AnyCompatible, AnyView, Array, DLDataType, DLDataTypeExt, Map, ObjectArc, ObjectRefCast,
    Result, String,
};

/// Opaque Rust view of TVM's `ExprNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Expr"]
pub struct ExprObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to any TVM expression.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Expr {
    data: ObjectArc<ExprObj>,
}

impl std::ops::Deref for Expr {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl ExprObj {
    /// Return the static type annotation carried by this expression.
    pub fn ty(&self) -> Result<Type> {
        crate::reflected_field!(self, "ty")?.try_into()
    }
}

/// Opaque Rust view of TVM's `BaseFuncNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.BaseFunc"]
pub struct BaseFuncObj {
    base: ExprObj,
}

/// Reference-counted handle to any TVM base function.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct BaseFunc {
    data: ObjectArc<BaseFuncObj>,
}

impl std::ops::Deref for BaseFunc {
    type Target = BaseFuncObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for BaseFuncObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl BaseFuncObj {
    /// Return the function attributes.
    pub fn attrs(&self) -> Result<DictAttrs> {
        crate::reflected_field!(self, "attrs")?.try_into()
    }
}

/// Opaque Rust view of TVM's `GlobalVarNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.GlobalVar"]
#[type_final]
pub struct GlobalVarObj {
    base: ExprObj,
}

/// Reference-counted handle to a global function name.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct GlobalVar {
    data: ObjectArc<GlobalVarObj>,
}

impl std::ops::Deref for GlobalVar {
    type Target = GlobalVarObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for GlobalVarObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl GlobalVarObj {
    /// Return the module-level symbol name hint.
    pub fn name_hint(&self) -> Result<String> {
        crate::reflected_field!(self, "name_hint")?.try_into()
    }
}

/// Opaque Rust view of TVM's `VarNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Var"]
pub struct VarObj {
    base: ExprObj,
}

/// Reference-counted handle to a TVM variable.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Var {
    data: ObjectArc<VarObj>,
}

impl std::ops::Deref for Var {
    type Target = VarObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for VarObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl VarObj {
    /// Return the display name hint.  Variable identity is still pointer-based.
    pub fn name(&self) -> Result<String> {
        crate::reflected_field!(self, "name")?.try_into()
    }
}

/// Opaque Rust view of TVM's source-span metadata.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Span"]
pub struct SpanObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to source-span metadata.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Span {
    data: ObjectArc<SpanObj>,
}

/// Opaque prefix for TVM objects that can convert to a primitive expression.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.PrimExprConvertible"]
pub struct PrimExprConvertibleObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to a primitive-expression-convertible object.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct PrimExprConvertible {
    data: ObjectArc<PrimExprConvertibleObj>,
}

impl std::ops::Deref for PrimExprConvertible {
    type Target = PrimExprConvertibleObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Opaque Rust view of an integer range.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Range"]
#[type_final]
pub struct RangeObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to `min .. min + extent`.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Range {
    data: ObjectArc<RangeObj>,
}

impl std::ops::Deref for Range {
    type Target = RangeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl RangeObj {
    /// Return the first value in the range.
    pub fn minimum(&self) -> Result<Expr> {
        crate::reflected_field!(self, "min")?.try_into()
    }

    /// Return the number of values in the range.
    pub fn extent(&self) -> Result<Expr> {
        crate::reflected_field!(self, "extent")?.try_into()
    }

    /// Return optional source metadata.
    pub fn span(&self) -> Result<Option<Span>> {
        crate::reflected_field!(self, "span")?.try_into()
    }
}

impl Range {
    /// Construct a range from its minimum and extent.
    pub fn from_min_extent(minimum: &Expr, extent: &Expr) -> Result<Self> {
        let none = ();
        crate::global_function!("ir.Range_from_min_extent")?
            .call_packed(&[
                AnyView::from(minimum),
                AnyView::from(extent),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

impl std::ops::Deref for Span {
    type Target = SpanObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Opaque Rust view of TVM's `CallNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Call"]
#[type_final]
pub struct CallObj {
    base: ExprObj,
}

/// Reference-counted handle to a call expression shared by TIR and Relax.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Call {
    data: ObjectArc<CallObj>,
}

impl std::ops::Deref for Call {
    type Target = CallObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for CallObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl CallObj {
    /// Return the callable expression.
    pub fn operator(&self) -> Result<Expr> {
        crate::reflected_field!(self, "op")?.try_into()
    }

    /// Return the call arguments.
    pub fn arguments(&self) -> Result<Array<Expr>> {
        crate::reflected_field!(self, "args")?.try_into()
    }

    /// Return optional operator-specific attributes.
    pub fn attrs(&self) -> Result<Option<Attrs>> {
        crate::reflected_field!(self, "attrs")?.try_into()
    }

    /// Return explicit type arguments.
    pub fn type_arguments(&self) -> Result<Array<Type>> {
        crate::reflected_field!(self, "ty_args")?.try_into()
    }
}

/// Opaque Rust view of TVM's `TypeNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Type"]
pub struct TypeObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to any TVM type.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Type {
    data: ObjectArc<TypeObj>,
}

impl std::ops::Deref for Type {
    type Target = TypeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Opaque Rust view of TVM's `PrimTypeNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.PrimType"]
#[type_final]
pub struct PrimTypeObj {
    base: TypeObj,
}

/// Reference-counted handle to a primitive TVM type.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct PrimType {
    data: ObjectArc<PrimTypeObj>,
}

impl std::ops::Deref for PrimType {
    type Target = PrimTypeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for PrimTypeObj {
    type Target = TypeObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl PrimTypeObj {
    /// Return the DLPack dtype stored by this primitive type.
    pub fn dtype(&self) -> Result<DLDataType> {
        crate::reflected_field!(self, "dtype")?.try_into()
    }
}

/// Opaque Rust view of TVM's `IntImmNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.IntImm"]
#[type_final]
pub struct IntImmObj {
    base: ExprObj,
}

/// Reference-counted handle to an integer literal.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct IntImm {
    data: ObjectArc<IntImmObj>,
}

impl std::ops::Deref for IntImm {
    type Target = IntImmObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for IntImmObj {
    type Target = ExprObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl IntImmObj {
    /// Return the literal value.
    pub fn value(&self) -> Result<i64> {
        crate::reflected_field!(self, "value")?.try_into()
    }
}

/// Opaque Rust view of TVM's `AttrsNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Attrs"]
pub struct AttrsObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to any TVM attributes object.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Attrs {
    data: ObjectArc<AttrsObj>,
}

impl std::ops::Deref for Attrs {
    type Target = AttrsObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Opaque Rust view of TVM's `DictAttrsNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.DictAttrs"]
#[type_final]
pub struct DictAttrsObj {
    base: AttrsObj,
}

/// Reference-counted handle to dictionary-backed TVM attributes.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct DictAttrs {
    data: ObjectArc<DictAttrsObj>,
}

impl std::ops::Deref for DictAttrs {
    type Target = DictAttrsObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for DictAttrsObj {
    type Target = AttrsObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// Opaque Rust view of TVM's `IRModuleNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.IRModule"]
#[type_final]
pub struct IRModuleObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to a TVM IRModule.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct IRModule {
    data: ObjectArc<IRModuleObj>,
}

impl std::ops::Deref for IRModule {
    type Target = IRModuleObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl IRModuleObj {
    /// Return the module's global-function table.
    pub fn functions(&self) -> Result<Map<GlobalVar, BaseFunc>> {
        crate::reflected_field!(self, "functions")?.try_into()
    }
}

impl Expr {
    /// Construct an integer literal through TVM's registered IR constructor.
    pub fn int(dtype: &str, value: i64) -> Result<Self> {
        Ok(IntImm::new(dtype, value)?.into())
    }

    /// Construct an addition expression through TVM's registered IR constructor.
    pub fn add(lhs: &Expr, rhs: &Expr) -> Result<Self> {
        Ok(crate::tirx::Add::new(lhs, rhs)?.into())
    }
}

impl IntImm {
    /// Construct an integer literal through TVM's registered IR constructor.
    pub fn new(dtype: &str, value: i64) -> Result<Self> {
        Self::from_dtype(DLDataType::try_from_str(dtype)?, value)
    }

    /// Construct an integer literal from a parsed DLPack dtype.
    pub fn from_dtype(dtype: DLDataType, value: i64) -> Result<Self> {
        let none = ();
        crate::global_function!("ir.IntImm")?
            .call_packed(&[
                AnyView::from(&dtype),
                AnyView::from(&value),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

impl PrimType {
    /// Construct a primitive type from a DLPack dtype string.
    pub fn new(dtype: &str) -> Result<Self> {
        let dtype = DLDataType::try_from_str(dtype)?;
        crate::global_function!("ir.PrimType")?
            .call_packed(&[AnyView::from(&dtype)])?
            .try_into()
    }
}

impl Type {
    /// Construct TVM's sentinel for an unavailable static type.
    pub fn missing() -> Result<Self> {
        crate::global_function!("ir.TypeMissing")?
            .call_packed(&[])?
            .try_into()
    }
}

impl Var {
    /// Construct a variable with an explicit primitive type.
    pub fn new(name: &str, dtype: &str) -> Result<Self> {
        Self::with_type(name, &PrimType::new(dtype)?.into())
    }

    /// Construct a variable with an arbitrary TVM type annotation.
    pub fn with_type(name: &str, ty: &Type) -> Result<Self> {
        let name = String::from(name);
        let ty = Some(ty.clone());
        let none = ();
        crate::global_function!("ir.Var")?
            .call_packed(&[
                AnyView::from(&name),
                AnyView::from(&ty),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

impl GlobalVar {
    /// Construct a module-level symbol.
    pub fn new(name_hint: &str) -> Result<Self> {
        let name_hint = String::from(name_hint);
        crate::global_function!("ir.GlobalVar")?
            .call_packed(&[AnyView::from(&name_hint)])?
            .try_into()
    }
}

impl Call {
    /// Construct a call with no attributes or explicit type arguments.
    pub fn new(ret_type: &Type, operator: &Expr, arguments: Vec<Expr>) -> Result<Self> {
        Self::with_metadata(ret_type, operator, arguments, None, Vec::new())
    }

    /// Construct a call with all reflected metadata supplied explicitly.
    pub fn with_metadata(
        ret_type: &Type,
        operator: &Expr,
        arguments: Vec<Expr>,
        attrs: Option<&Attrs>,
        type_arguments: Vec<Type>,
    ) -> Result<Self> {
        let arguments = Array::new(arguments);
        let attrs = attrs.cloned();
        let type_arguments = Array::new(type_arguments);
        let none = ();
        crate::global_function!("ir.Call")?
            .call_packed(&[
                AnyView::from(ret_type),
                AnyView::from(operator),
                AnyView::from(&arguments),
                AnyView::from(&attrs),
                AnyView::from(&type_arguments),
                AnyView::from(&none),
            ])?
            .try_into()
    }
}

impl From<IntImm> for Expr {
    fn from(value: IntImm) -> Self {
        value
            .try_cast()
            .expect("ir.IntImm must be a subtype of ir.Expr")
    }
}

impl From<PrimType> for Type {
    fn from(value: PrimType) -> Self {
        value
            .try_cast()
            .expect("ir.PrimType must be a subtype of ir.Type")
    }
}

impl From<Var> for Expr {
    fn from(value: Var) -> Self {
        value
            .try_cast()
            .expect("ir.Var must be a subtype of ir.Expr")
    }
}

impl From<GlobalVar> for Expr {
    fn from(value: GlobalVar) -> Self {
        value
            .try_cast()
            .expect("ir.GlobalVar must be a subtype of ir.Expr")
    }
}

impl From<Call> for Expr {
    fn from(value: Call) -> Self {
        value
            .try_cast()
            .expect("ir.Call must be a subtype of ir.Expr")
    }
}

impl From<DictAttrs> for Attrs {
    fn from(value: DictAttrs) -> Self {
        value
            .try_cast()
            .expect("ir.DictAttrs must be a subtype of ir.Attrs")
    }
}

impl IRModule {
    /// Wrap a function expression in an IRModule whose entry is `main`.
    pub fn from_expr<E: AnyCompatible>(expr: &E) -> Result<Self> {
        let empty_functions = Map::<GlobalVar, BaseFunc>::new();
        crate::global_function!("ir.Module_FromExpr")?
            .call_packed(&[AnyView::from(expr), AnyView::from(&empty_functions)])?
            .try_into()
    }

    /// Return an independently updatable module with one function replaced.
    ///
    /// The module node is copied before TVM updates its function table, so
    /// other Rust handles that share `self` remain unchanged.  Structural
    /// mapping the reflected maps directly would also fail to maintain TVM's
    /// derived global-name index.
    pub fn with_updated_function(
        &self,
        global_var: &GlobalVar,
        function: &BaseFunc,
    ) -> Result<Self> {
        self.copy_for_update()?
            .update_function_owned(global_var, function)
    }

    pub(crate) fn copy_for_update(&self) -> Result<Self> {
        crate::global_function!("ir.Module_Clone")?
            .call_packed(&[AnyView::from(self)])?
            .try_into()
    }

    pub(crate) fn update_function_owned(
        self,
        global_var: &GlobalVar,
        function: &BaseFunc,
    ) -> Result<Self> {
        let update = true;
        crate::global_function!("ir.Module_Add")?
            .call_packed(&[
                AnyView::from(&self),
                AnyView::from(global_var),
                AnyView::from(function),
                AnyView::from(&update),
            ])?
            .try_into()
    }
}

impl DictAttrs {
    /// Construct a defined, empty DictAttrs object.
    pub fn empty() -> Result<Self> {
        let empty = Map::<String, String>::new();
        let type_key = String::from("ir.DictAttrs");
        let field_name = String::from("__dict__");
        crate::global_function!("ffi.MakeObjectFromPackedArgs")?
            .call_packed(&[
                AnyView::from(&type_key),
                AnyView::from(&field_name),
                AnyView::from(&empty),
            ])?
            .try_into()
    }
}
