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
    Any, AnyView, Array, DLDataType, DLDataTypeCode, DLDataTypeExt, Error, Function, Map,
    ObjectArc, ObjectCore, ObjectRefCast, ObjectRefCore, Result, String, VALUE_ERROR,
};

/// Opaque Rust view of TVM's `ExprNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Expr"]
pub struct ExprObj {
    base: tvm_ffi::Object,
}
crate::abi::reflected_fields!(ExprObj {
    span => "span": Option<Span>,
    ty => "ty": Type,
});

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

/// Opaque Rust view of TVM's `BaseFuncNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.BaseFunc"]
pub struct BaseFuncObj {
    base: ExprObj,
}
crate::abi::reflected_fields!(BaseFuncObj {
    attrs => "attrs": DictAttrs,
});

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

/// Opaque Rust view of TVM's `GlobalVarNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.GlobalVar"]
#[type_final]
pub struct GlobalVarObj {
    base: ExprObj,
}
crate::abi::reflected_fields!(GlobalVarObj {
    name_hint => "name_hint": String,
});

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

/// Opaque Rust view of TVM's `VarNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Var"]
pub struct VarObj {
    base: ExprObj,
}
crate::abi::reflected_fields!(VarObj {
    name => "name": String,
});

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

/// Opaque Rust view of a TVM source name.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.SourceName"]
#[type_final]
pub struct SourceNameObj {
    base: tvm_ffi::Object,
}
crate::abi::reflected_fields!(SourceNameObj {
    name => "name": String,
});

/// Reference-counted handle to a source name.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct SourceName {
    data: ObjectArc<SourceNameObj>,
}

impl std::ops::Deref for SourceName {
    type Target = SourceNameObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl SourceName {
    /// Return TVM's interned source name.
    pub fn get(name: &str) -> Result<Self> {
        let name = String::from(name);
        tvm_ffi::cached_global_func!("ir.SourceName")
            .call_tuple((&name,))?
            .try_into()
    }
}

/// Opaque Rust view of TVM's `SourceNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Source"]
#[type_final]
pub struct SourceObj {
    base: tvm_ffi::Object,
}
crate::abi::reflected_fields!(SourceObj {
    source_name => "source_name": SourceName,
    source => "source": String,
});

/// Reference-counted handle to one program source fragment.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Source {
    data: ObjectArc<SourceObj>,
}

impl std::ops::Deref for Source {
    type Target = SourceObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Opaque Rust view of TVM's `SourceMapObj`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.SourceMap"]
#[type_final]
pub struct SourceMapObj {
    base: tvm_ffi::Object,
}
crate::abi::reflected_fields!(SourceMapObj {
    source_map => "source_map": Map<SourceName, Source>,
});

/// Reference-counted handle to TVM's source-name-to-source mapping.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct SourceMap {
    data: ObjectArc<SourceMapObj>,
}

impl std::ops::Deref for SourceMap {
    type Target = SourceMapObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl SourceMap {
    /// Construct an empty native source map.
    pub fn new() -> Result<Self> {
        Function::from_type_attr(SourceMapObj::type_index(), "__ffi_init__")?
            .call_tuple((Map::<SourceName, Source>::new(),))?
            .try_into()
    }

    /// Add source text using TVM's native `Source` constructor and line map logic.
    pub fn add(&self, name: &str, content: &str) -> Result<SourceName> {
        let name = String::from(name);
        let content = String::from(content);
        tvm_ffi::cached_global_func!("SourceMapAdd")
            .call_tuple((self, &name, &content))?
            .try_into()
    }
}

/// Opaque Rust view of TVM's source-span metadata.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Span"]
pub struct SpanObj {
    base: tvm_ffi::Object,
}
crate::abi::reflected_fields!(SpanObj {
    source_name => "source_name": SourceName,
    line => "line": i32,
    column => "column": i32,
    end_line => "end_line": i32,
    end_column => "end_column": i32,
});

/// Reference-counted handle to source-span metadata.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Span {
    data: ObjectArc<SpanObj>,
}

impl Span {
    /// Construct source-location metadata in TVM's `(line, end_line, column, end_column)` order.
    pub fn new<S>(
        source_name: S,
        line: i64,
        end_line: i64,
        column: i64,
        end_column: i64,
    ) -> Result<Self>
    where
        S: Into<SourceName>,
    {
        let line = i32::try_from(line).map_err(|_| integer_field_overflow("line", line))?;
        let end_line =
            i32::try_from(end_line).map_err(|_| integer_field_overflow("end_line", end_line))?;
        let column = i32::try_from(column).map_err(|_| integer_field_overflow("column", column))?;
        let end_column = i32::try_from(end_column)
            .map_err(|_| integer_field_overflow("end_column", end_column))?;
        tvm_ffi::cached_global_func!("ir.Span")
            .call_tuple((source_name.into(), line, end_line, column, end_column))?
            .try_into()
    }
}

fn integer_field_overflow(field: &str, value: i64) -> Error {
    Error::new(
        VALUE_ERROR,
        &format!("{field} value {value} does not fit TVM's 32-bit integer field"),
        "",
    )
}

/// Opaque view of TVM objects that can convert to a primitive expression.
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

impl PrimExprConvertible {
    /// Convert through the concrete type's reflected method.
    pub fn to_prim_expr(&self) -> Result<Expr> {
        Function::from_type_method(PrimExprConvertibleObj::type_index(), "to_prim_expr")?
            .call_tuple((self,))?
            .try_into()
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
crate::abi::reflected_fields!(RangeObj {
    min => "min": Expr,
    extent => "extent": Expr,
    span => "span": Option<Span>,
});

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

impl Range {
    /// Construct a range from its minimum and extent.
    pub fn from_min_extent<M, E>(minimum: M, extent: E) -> Result<Self>
    where
        M: Into<Expr>,
        E: Into<Expr>,
    {
        Self::from_min_extent_with_span(minimum, extent, None)
    }

    /// Construct a range with source metadata through TVM's native constructor.
    pub fn from_min_extent_with_span<M, E>(
        minimum: M,
        extent: E,
        span: Option<&Span>,
    ) -> Result<Self>
    where
        M: Into<Expr>,
        E: Into<Expr>,
    {
        tvm_ffi::cached_global_func!("ir.Range_from_min_extent")
            .call_tuple((minimum.into(), extent.into(), span.cloned()))?
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
crate::abi::reflected_fields!(CallObj {
    op => "op": Expr,
    args => "args": Array<Expr>,
    attrs => "attrs": Option<Attrs>,
    ty_args => "ty_args": Array<Type>,
});

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

/// Opaque Rust view of TVM's `TypeNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Type"]
pub struct TypeObj {
    base: tvm_ffi::Object,
}
crate::abi::reflected_fields!(TypeObj {
    span => "span": Option<Span>,
});

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
crate::abi::reflected_fields!(PrimTypeObj {
    dtype => "dtype": DLDataType,
});

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

/// Opaque Rust view of TVM's `TupleTypeNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.TupleType"]
#[type_final]
pub struct TupleTypeObj {
    base: TypeObj,
}
crate::abi::reflected_fields!(TupleTypeObj {
    fields => "fields": Array<Type>,
});

/// Reference-counted handle to a tuple type.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct TupleType {
    data: ObjectArc<TupleTypeObj>,
}

impl std::ops::Deref for TupleType {
    type Target = TupleTypeObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for TupleTypeObj {
    type Target = TypeObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl TupleType {
    /// Construct a tuple type through TVM's native constructor.
    pub fn new(fields: Vec<Type>) -> Result<Self> {
        Self::with_span(fields, None)
    }

    /// Construct a tuple type with optional source metadata.
    pub fn with_span(fields: Vec<Type>, span: Option<&Span>) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.TupleType")
            .call_tuple((Array::new(fields), span.cloned()))?
            .try_into()
    }

    /// Construct the empty tuple type used as TVM's void type.
    pub fn empty() -> Result<Self> {
        Self::new(Vec::new())
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
crate::abi::reflected_fields!(IntImmObj {
    value => "value": i64,
});

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

/// Opaque base for TVM attribute objects.
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

/// Opaque Rust view of TVM's dictionary-backed attributes.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.DictAttrs"]
#[type_final]
pub struct DictAttrsObj {
    base: AttrsObj,
}
crate::abi::reflected_fields!(DictAttrsObj {
    dict => "__dict__": Map<String, Any>,
});

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

/// Opaque base for module-level global metadata.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.GlobalInfo"]
pub struct GlobalInfoObj {
    base: tvm_ffi::Object,
}

/// Reference-counted handle to module-level global metadata.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct GlobalInfo {
    data: ObjectArc<GlobalInfoObj>,
}

impl std::ops::Deref for GlobalInfo {
    type Target = GlobalInfoObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// Opaque Rust view of TVM's fieldless test global-info object.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.DummyGlobalInfo"]
#[type_final]
pub struct DummyGlobalInfoObj {
    base: GlobalInfoObj,
}

/// Reference-counted handle to a dummy global-info value.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct DummyGlobalInfo {
    data: ObjectArc<DummyGlobalInfoObj>,
}

impl std::ops::Deref for DummyGlobalInfo {
    type Target = DummyGlobalInfoObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::Deref for DummyGlobalInfoObj {
    type Target = GlobalInfoObj;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl DummyGlobalInfo {
    /// Construct TVM's fieldless global-info test value natively.
    pub fn new() -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.DummyGlobalInfo")
            .call_tuple(())?
            .try_into()
    }
}

tvm_ffi::impl_object_upcast!(DummyGlobalInfo => GlobalInfo);

/// Opaque Rust view of TVM's `IRModuleNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.IRModule"]
#[type_final]
pub struct IRModuleObj {
    base: tvm_ffi::Object,
}
crate::abi::reflected_fields!(IRModuleObj {
    functions => "functions": Map<GlobalVar, BaseFunc>,
    source_map => "source_map": SourceMap,
    attrs => "attrs": DictAttrs,
    global_infos => "global_infos": Map<String, Array<GlobalInfo>>,
    global_var_map => "global_var_map_": Map<String, GlobalVar>,
});

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

impl IntImm {
    /// Construct an integer literal through TVM's native constructor.
    pub fn new(dtype: &str, value: i64) -> Result<Self> {
        Self::from_dtype(DLDataType::try_from_str(dtype)?, value)
    }

    /// Construct an integer literal from a parsed DLPack dtype.
    pub fn from_dtype(dtype: DLDataType, value: i64) -> Result<Self> {
        Self::from_dtype_with_span(dtype, value, None)
    }

    /// Construct an integer literal through TVM's native constructor with source metadata.
    pub fn from_dtype_with_span(
        dtype: DLDataType,
        value: i64,
        span: Option<&Span>,
    ) -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.IntImm")
            .call_tuple((dtype, value, span.cloned()))?
            .try_into()
    }
}

impl PrimType {
    /// Construct a primitive type from a DLPack dtype string.
    pub fn new(dtype: &str) -> Result<Self> {
        Self::from_dtype(DLDataType::try_from_str(dtype)?)
    }

    /// Construct a primitive type from a parsed DLPack dtype.
    pub fn from_dtype(dtype: DLDataType) -> Result<Self> {
        Self::from_dtype_with_span(dtype, None)
    }

    /// Construct a primitive type with optional source metadata.
    pub fn from_dtype_with_span(dtype: DLDataType, span: Option<&Span>) -> Result<Self> {
        if span.is_some() {
            return Function::from_type_attr(PrimTypeObj::type_index(), "__ffi_init__")?
                .call_tuple((span.cloned(), dtype))?
                .try_into();
        }
        tvm_ffi::cached_global_func!("ir.PrimType")
            .call_tuple((dtype,))?
            .try_into()
    }

    /// Construct TVM's void primitive type without parsing a user-supplied dtype string.
    pub fn void() -> Result<Self> {
        Self::from_dtype(DLDataType {
            code: DLDataTypeCode::kDLOpaqueHandle as u8,
            bits: 0,
            lanes: 0,
        })
    }
}

impl Type {
    /// Construct TVM's language-independent sentinel for an unavailable static type.
    pub fn missing() -> Result<Self> {
        tvm_ffi::cached_global_func!("ir.TypeMissing")
            .call_tuple(())?
            .try_into()
    }

    /// Return whether this value is the exact `ir.Type` missing-type sentinel.
    pub fn is_missing(&self) -> bool {
        AnyView::from(self).type_index() == <TypeObj as tvm_ffi::ObjectCore>::type_index()
    }
}

impl Var {
    /// Construct a variable through TVM's native constructor with an explicit primitive type.
    pub fn new(name: &str, dtype: &str) -> Result<Self> {
        Self::with_type(name, PrimType::new(dtype)?)
    }

    /// Construct a variable with an arbitrary TVM type annotation.
    pub fn with_type<T>(name: &str, ty: T) -> Result<Self>
    where
        T: Into<Type>,
    {
        Self::with_type_and_span(name, ty, None)
    }

    /// Construct a variable through TVM's native constructor with source metadata.
    pub fn with_type_and_span<T>(name: &str, ty: T, span: Option<&Span>) -> Result<Self>
    where
        T: Into<Type>,
    {
        Self::with_optional_type_and_span(name, Some(ty.into()), span)
    }

    /// Construct a variable with the native constructor's optional type annotation.
    pub fn with_optional_type_and_span(
        name: &str,
        ty: Option<Type>,
        span: Option<&Span>,
    ) -> Result<Self> {
        let name = String::from(name);
        tvm_ffi::cached_global_func!("ir.Var")
            .call_tuple((&name, ty, span.cloned()))?
            .try_into()
    }
}

impl GlobalVar {
    /// Construct a module-level symbol through TVM's native constructor.
    pub fn new(name_hint: &str) -> Result<Self> {
        Self::with_span(name_hint, None)
    }

    /// Construct a module-level symbol with optional source metadata.
    pub fn with_span(name_hint: &str, span: Option<&Span>) -> Result<Self> {
        let name_hint = String::from(name_hint);
        if span.is_none() {
            return tvm_ffi::cached_global_func!("ir.GlobalVar")
                .call_tuple((&name_hint,))?
                .try_into();
        }
        Function::from_type_attr(GlobalVarObj::type_index(), "__ffi_init__")?
            .call_tuple((span.cloned(), Type::missing()?, &name_hint))?
            .try_into()
    }
}

impl Call {
    /// Construct a call through TVM's native constructor with no extra metadata.
    pub fn new<T, O>(ret_type: T, operator: O, arguments: Vec<Expr>) -> Result<Self>
    where
        T: Into<Type>,
        O: Into<Expr>,
    {
        Self::with_metadata(ret_type, operator, arguments, None, Vec::new(), None)
    }

    /// Construct a call with all reflected metadata supplied explicitly.
    pub fn with_metadata<T, O>(
        ret_type: T,
        operator: O,
        arguments: Vec<Expr>,
        attrs: Option<Attrs>,
        type_arguments: Vec<Type>,
        span: Option<&Span>,
    ) -> Result<Self>
    where
        T: Into<Type>,
        O: Into<Expr>,
    {
        tvm_ffi::cached_global_func!("ir.Call")
            .call_tuple((
                ret_type.into(),
                operator.into(),
                Array::new(arguments),
                attrs,
                Array::new(type_arguments),
                span.cloned(),
            ))?
            .try_into()
    }
}

tvm_ffi::impl_object_upcast!(
    BaseFunc => Expr,
    IntImm => Expr,
    PrimType => Type,
    TupleType => Type,
    Var => Expr,
    GlobalVar => Expr,
    Call => Expr,
    DictAttrs => Attrs,
);

impl IRModule {
    /// Wrap a function expression in an IRModule whose entry is `main`.
    pub fn from_expr<E>(expr: E) -> Result<Self>
    where
        E: Into<Expr>,
    {
        let function = expr.into().try_cast::<BaseFunc>()?;
        let global_symbol = function
            .attrs()?
            .dict()?
            .get(&String::from("global_symbol"))?
            .map(String::try_from)
            .transpose()?;
        let global_name = global_symbol
            .as_deref()
            .filter(|name| !name.is_empty())
            .unwrap_or("main");
        let global_var = GlobalVar::new(global_name)?;
        Self::with_metadata(
            [(global_var, function)].into_iter().collect(),
            SourceMap::new()?,
            DictAttrs::empty()?,
            Map::new(),
        )
    }

    /// Construct a module through its standard reflected initializer.
    ///
    /// The derived name-to-global-variable index is rebuilt and checked here;
    /// callers never supply it independently.
    pub fn with_metadata(
        functions: Map<GlobalVar, BaseFunc>,
        source_map: SourceMap,
        attrs: DictAttrs,
        global_infos: Map<String, Array<GlobalInfo>>,
    ) -> Result<Self> {
        let mut indexed_globals = Vec::with_capacity(functions.len());
        let mut names = std::collections::HashSet::with_capacity(functions.len());
        for (global_var, _) in functions.iter() {
            let name = global_var.name_hint()?;
            if !names.insert(name.as_str().to_owned()) {
                return Err(Error::new(
                    VALUE_ERROR,
                    &format!("duplicate global function name {}", name.as_str()),
                    "",
                ));
            }
            indexed_globals.push((name, global_var));
        }
        Function::from_type_attr(IRModuleObj::type_index(), "__ffi_init__")?
            .call_tuple((
                functions,
                indexed_globals
                    .into_iter()
                    .collect::<Map<String, GlobalVar>>(),
                source_map,
                attrs,
                global_infos,
            ))?
            .try_into()
    }

    /// Return an independently updatable module with one function replaced.
    ///
    /// Rust rebuilds both immutable maps and the module node, so other handles
    /// that share `self` remain unchanged and the derived global-name index
    /// cannot become stale.
    pub fn with_updated_function(
        &self,
        global_var: &GlobalVar,
        function: &BaseFunc,
    ) -> Result<Self> {
        self.copy_for_update()?
            .update_function_owned(global_var, function)
    }

    /// Return an independently updatable module with one global-info group set.
    pub fn with_updated_global_info(
        &self,
        name: &str,
        global_info: Vec<GlobalInfo>,
    ) -> Result<Self> {
        let name = String::from(name);
        let mut global_infos = self
            .global_infos()?
            .iter()
            .filter(|(existing, _)| existing.as_str() != name.as_str())
            .collect::<Vec<_>>();
        global_infos.push((name, Array::new(global_info)));
        Self::with_metadata(
            self.functions()?,
            self.source_map()?,
            self.attrs()?,
            global_infos.into_iter().collect(),
        )
    }

    pub(crate) fn copy_for_update(&self) -> Result<Self> {
        Self::with_metadata(
            self.functions()?,
            self.source_map()?,
            self.attrs()?,
            self.global_infos()?,
        )
    }

    pub(crate) fn update_function_owned(
        self,
        global_var: &GlobalVar,
        function: &BaseFunc,
    ) -> Result<Self> {
        let name = global_var.name_hint()?;
        let current_functions = self.functions()?;
        let mut functions = Vec::with_capacity(current_functions.len() + 1);
        for (existing_var, existing_function) in current_functions.iter() {
            if existing_var.name_hint()?.as_str() == name.as_str() {
                if !existing_var.same_as(global_var) {
                    return Err(Error::new(
                        VALUE_ERROR,
                        &format!("duplicate global function name {}", name.as_str()),
                        "",
                    ));
                }
            } else {
                functions.push((existing_var, existing_function));
            }
        }
        functions.push((global_var.clone(), function.clone()));
        Self::with_metadata(
            functions.into_iter().collect(),
            self.source_map()?,
            self.attrs()?,
            self.global_infos()?,
        )
    }
}

impl DictAttrs {
    /// Construct a defined, empty DictAttrs object.
    pub fn empty() -> Result<Self> {
        Self::from_dictionary(Map::new())
    }

    /// Construct DictAttrs from a heterogeneous string-to-value map.
    pub fn from_dictionary(dictionary: Map<String, Any>) -> Result<Self> {
        Function::from_type_attr(DictAttrsObj::type_index(), "__ffi_init__")?
            .call_tuple((dictionary,))?
            .try_into()
    }
}
