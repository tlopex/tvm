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
    AnyCompatible, AnyMap, AnyView, Array, DLDataType, DLDataTypeCode, DLDataTypeExt, Error, Map,
    ObjectArc, ObjectRefCast, Result, String, TYPE_ERROR, VALUE_ERROR,
};

/// ABI-complete Rust representation of TVM's `ExprNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Expr"]
pub struct ExprObj {
    base: tvm_ffi::Object,
    span: Option<Span>,
    ty: Type,
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
    pub(crate) fn new(ty: Type, span: Option<Span>) -> Self {
        Self {
            base: tvm_ffi::Object::new(),
            span,
            ty,
        }
    }

    /// Return optional source metadata carried by this expression.
    pub fn span(&self) -> Result<Option<Span>> {
        Ok(self.span.clone())
    }

    /// Return the static type annotation carried by this expression.
    pub fn ty(&self) -> Result<Type> {
        Ok(self.ty.clone())
    }
}

/// ABI-complete Rust representation of TVM's `BaseFuncNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.BaseFunc"]
pub struct BaseFuncObj {
    base: ExprObj,
    attrs: DictAttrs,
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
        Ok(self.attrs.clone())
    }
}

/// ABI-complete Rust representation of TVM's `GlobalVarNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.GlobalVar"]
#[type_final]
pub struct GlobalVarObj {
    base: ExprObj,
    name_hint: String,
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
        Ok(self.name_hint.clone())
    }
}

/// ABI-complete Rust representation of TVM's `VarNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Var"]
pub struct VarObj {
    base: ExprObj,
    name: String,
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
        Ok(self.name.clone())
    }
}

/// ABI-complete Rust view of TVM's interned source name.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.SourceName"]
#[type_final]
pub struct SourceNameObj {
    base: tvm_ffi::Object,
    name: String,
}

/// Reference-counted handle to an interned source name.
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

impl SourceNameObj {
    /// Return the interned source name text.
    pub fn name(&self) -> Result<String> {
        Ok(self.name.clone())
    }
}

impl SourceName {
    /// Return TVM's process-wide interned source name for `name`.
    pub fn get(name: &str) -> Result<Self> {
        let name = String::from(name);
        crate::global_function!("ir.SourceName")?
            .call_packed(&[AnyView::from(&name)])?
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

impl SourceObj {
    /// Return the interned name of this source fragment.
    pub fn source_name(&self) -> Result<SourceName> {
        crate::reflected_field!(self, "source_name")?.try_into()
    }

    /// Return the complete source text.
    pub fn source(&self) -> Result<String> {
        crate::reflected_field!(self, "source")?.try_into()
    }
}

/// ABI-complete Rust representation of TVM's `SourceMapObj`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.SourceMap"]
#[type_final]
pub struct SourceMapObj {
    base: tvm_ffi::Object,
    source_map: Map<SourceName, Source>,
}

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

impl SourceMapObj {
    /// Return the typed source-name-to-source mapping.
    pub fn source_map(&self) -> Result<Map<SourceName, Source>> {
        Ok(self.source_map.clone())
    }
}

impl SourceMap {
    /// Construct an empty source map directly in Rust.
    pub fn new() -> Self {
        Self {
            data: ObjectArc::new(SourceMapObj {
                base: tvm_ffi::Object::new(),
                source_map: Map::new(),
            }),
        }
    }

    /// Add a named source fragment through TVM's registered source-map API.
    pub fn add(&self, name: &str, content: &str) -> Result<SourceName> {
        let name = String::from(name);
        let content = String::from(content);
        crate::global_function!("SourceMapAdd")?
            .call_packed(&[
                AnyView::from(self),
                AnyView::from(&name),
                AnyView::from(&content),
            ])?
            .try_into()
    }
}

/// ABI-complete Rust representation of TVM's source-span metadata.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Span"]
pub struct SpanObj {
    base: tvm_ffi::Object,
    source_name: SourceName,
    line: i32,
    column: i32,
    end_line: i32,
    end_column: i32,
}

/// Reference-counted handle to source-span metadata.
#[repr(C)]
#[derive(ObjectRef, Clone)]
pub struct Span {
    data: ObjectArc<SpanObj>,
}

impl SpanObj {
    /// Return the source fragment containing this span.
    pub fn source_name(&self) -> Result<SourceName> {
        Ok(self.source_name.clone())
    }

    /// Return the first source line.
    pub fn line(&self) -> Result<i64> {
        Ok(i64::from(self.line))
    }

    /// Return the first source column.
    pub fn column(&self) -> Result<i64> {
        Ok(i64::from(self.column))
    }

    /// Return the last source line.
    pub fn end_line(&self) -> Result<i64> {
        Ok(i64::from(self.end_line))
    }

    /// Return the last source column.
    pub fn end_column(&self) -> Result<i64> {
        Ok(i64::from(self.end_column))
    }
}

impl Span {
    /// Construct source-location metadata.
    pub fn new(
        source_name: &SourceName,
        line: i64,
        column: i64,
        end_line: i64,
        end_column: i64,
    ) -> Result<Self> {
        let line = i32::try_from(line).map_err(|_| integer_field_overflow("line", line))?;
        let column = i32::try_from(column).map_err(|_| integer_field_overflow("column", column))?;
        let end_line =
            i32::try_from(end_line).map_err(|_| integer_field_overflow("end_line", end_line))?;
        let end_column = i32::try_from(end_column)
            .map_err(|_| integer_field_overflow("end_column", end_column))?;
        Ok(Self {
            data: ObjectArc::new(SpanObj {
                base: tvm_ffi::Object::new(),
                source_name: source_name.clone(),
                line,
                column,
                end_line,
                end_column,
            }),
        })
    }
}

fn integer_field_overflow(field: &str, value: i64) -> Error {
    Error::new(
        VALUE_ERROR,
        &format!("{field} value {value} does not fit TVM's 32-bit integer field"),
        "",
    )
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

/// ABI-complete Rust representation of an integer range.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Range"]
#[type_final]
pub struct RangeObj {
    base: tvm_ffi::Object,
    min: Expr,
    extent: Expr,
    span: Option<Span>,
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
        Ok(self.min.clone())
    }

    /// Return the number of values in the range.
    pub fn extent(&self) -> Result<Expr> {
        Ok(self.extent.clone())
    }

    /// Return optional source metadata.
    pub fn span(&self) -> Result<Option<Span>> {
        Ok(self.span.clone())
    }
}

impl Range {
    /// Construct a range from its minimum and extent.
    pub fn from_min_extent(minimum: &Expr, extent: &Expr) -> Result<Self> {
        require_primitive_expr(minimum, "Range minimum")?;
        require_primitive_expr(extent, "Range extent")?;
        Ok(Self {
            data: ObjectArc::new(RangeObj {
                base: tvm_ffi::Object::new(),
                min: minimum.clone(),
                extent: extent.clone(),
                span: None,
            }),
        })
    }
}

fn require_primitive_expr(value: &Expr, context: &str) -> Result<()> {
    value.ty()?.try_cast::<PrimType>().map(|_| ()).map_err(|_| {
        Error::new(
            TYPE_ERROR,
            &format!("{context} must have a primitive type"),
            "",
        )
    })
}

impl std::ops::Deref for Span {
    type Target = SpanObj;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

/// ABI-complete Rust representation of TVM's `CallNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Call"]
#[type_final]
pub struct CallObj {
    base: ExprObj,
    op: Expr,
    args: Array<Expr>,
    attrs: Option<Attrs>,
    ty_args: Array<Type>,
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
        Ok(self.op.clone())
    }

    /// Return the call arguments.
    pub fn arguments(&self) -> Result<Array<Expr>> {
        Ok(self.args.clone())
    }

    /// Return optional operator-specific attributes.
    pub fn attrs(&self) -> Result<Option<Attrs>> {
        Ok(self.attrs.clone())
    }

    /// Return explicit type arguments.
    pub fn type_arguments(&self) -> Result<Array<Type>> {
        Ok(self.ty_args.clone())
    }
}

/// ABI-complete Rust representation of TVM's `TypeNode` prefix.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.Type"]
pub struct TypeObj {
    base: tvm_ffi::Object,
    span: Option<Span>,
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

impl TypeObj {
    pub(crate) fn new(span: Option<Span>) -> Self {
        Self {
            base: tvm_ffi::Object::new(),
            span,
        }
    }

    /// Return optional source metadata carried by this type.
    pub fn span(&self) -> Result<Option<Span>> {
        Ok(self.span.clone())
    }
}

/// ABI-complete Rust representation of TVM's `PrimTypeNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.PrimType"]
#[type_final]
pub struct PrimTypeObj {
    base: TypeObj,
    dtype: DLDataType,
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
        Ok(self.dtype)
    }
}

/// ABI-complete Rust representation of TVM's `IntImmNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.IntImm"]
#[type_final]
pub struct IntImmObj {
    base: ExprObj,
    value: i64,
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
        Ok(self.value)
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
    dict: AnyMap<String>,
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

impl DictAttrsObj {
    /// Return the heterogeneous string-to-value attribute dictionary.
    pub fn dictionary(&self) -> Result<AnyMap<String>> {
        Ok(self.dict.clone())
    }
}

/// Opaque Rust view of TVM's `GlobalInfoNode` base class.
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
    /// Construct TVM's fieldless global-info test value directly in Rust.
    pub fn new() -> Result<Self> {
        Ok(Self {
            data: ObjectArc::new(DummyGlobalInfoObj {
                base: GlobalInfoObj {
                    base: tvm_ffi::Object::new(),
                },
            }),
        })
    }
}

impl From<DummyGlobalInfo> for GlobalInfo {
    fn from(value: DummyGlobalInfo) -> Self {
        value
            .try_cast()
            .expect("ir.DummyGlobalInfo must be a subtype of ir.GlobalInfo")
    }
}

/// ABI-complete Rust representation of TVM's `IRModuleNode`.
#[repr(C)]
#[derive(Object)]
#[type_key = "ir.IRModule"]
#[type_final]
pub struct IRModuleObj {
    base: tvm_ffi::Object,
    functions: Map<GlobalVar, BaseFunc>,
    source_map: SourceMap,
    attrs: DictAttrs,
    global_infos: Map<String, Array<GlobalInfo>>,
    global_var_map: Map<String, GlobalVar>,
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
        Ok(self.functions.clone())
    }

    /// Return TVM's derived name-to-global-variable index.
    pub fn global_var_map(&self) -> Result<Map<String, GlobalVar>> {
        Ok(self.global_var_map.clone())
    }

    /// Return the module's typed source map.
    pub fn source_map(&self) -> Result<SourceMap> {
        Ok(self.source_map.clone())
    }

    /// Return the module attributes.
    pub fn attrs(&self) -> Result<DictAttrs> {
        Ok(self.attrs.clone())
    }

    /// Return typed module-level global metadata groups.
    pub fn global_infos(&self) -> Result<Map<String, Array<GlobalInfo>>> {
        Ok(self.global_infos.clone())
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
    /// Construct an integer literal directly in Rust.
    pub fn new(dtype: &str, value: i64) -> Result<Self> {
        Self::from_dtype(DLDataType::try_from_str(dtype)?, value)
    }

    /// Construct an integer literal from a parsed DLPack dtype.
    pub fn from_dtype(dtype: DLDataType, value: i64) -> Result<Self> {
        validate_integer_literal(dtype, value)?;
        let value_type = PrimType::from_dtype(dtype)?;
        Ok(Self {
            data: ObjectArc::new(IntImmObj {
                base: ExprObj::new(value_type.into(), None),
                value,
            }),
        })
    }
}

fn validate_integer_literal(dtype: DLDataType, value: i64) -> Result<()> {
    let dtype_text = dtype.to_string();
    if dtype.lanes != 1 {
        return Err(Error::new(
            VALUE_ERROR,
            &format!("IntImm can only take a scalar, but {dtype_text} was supplied"),
            "",
        ));
    }
    let is_int = dtype.code == DLDataTypeCode::kDLInt as u8;
    let is_uint = dtype.code == DLDataTypeCode::kDLUInt as u8;
    let is_bool = dtype.code == DLDataTypeCode::kDLBool as u8;
    if !is_int && !is_uint && !is_bool {
        return Err(Error::new(
            VALUE_ERROR,
            &format!("IntImm supports only int, uint, or bool, but {dtype_text} was supplied"),
            "",
        ));
    }
    let bits = u32::from(dtype.bits);
    if bits == 0 || bits > 64 {
        return Err(Error::new(
            VALUE_ERROR,
            &format!("invalid integer bit width in {dtype_text}"),
            "",
        ));
    }
    let in_range = if is_uint {
        value >= 0 && (bits == 64 || (value as u64) < (1_u64 << bits))
    } else if is_bool || bits == 1 {
        value == 0 || value == 1
    } else if bits == 64 {
        true
    } else {
        let bound = 1_i64 << (bits - 1);
        value >= -bound && value < bound
    };
    if in_range {
        Ok(())
    } else {
        Err(Error::new(
            VALUE_ERROR,
            &format!("literal value {value} is outside the range of {dtype_text}"),
            "",
        ))
    }
}

impl PrimType {
    /// Construct a primitive type from a DLPack dtype string.
    pub fn new(dtype: &str) -> Result<Self> {
        Self::from_dtype(DLDataType::try_from_str(dtype)?)
    }

    /// Return TVM's canonical primitive type for a parsed DLPack dtype.
    ///
    /// Primitive types are intentionally obtained from TVM because C++ interns
    /// these nodes; ordinary IR data nodes are allocated directly in Rust.
    pub fn from_dtype(dtype: DLDataType) -> Result<Self> {
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
    /// Construct a variable directly in Rust with an explicit primitive type.
    pub fn new(name: &str, dtype: &str) -> Result<Self> {
        Self::with_type(name, &PrimType::new(dtype)?.into())
    }

    /// Construct a variable with an arbitrary TVM type annotation.
    pub fn with_type(name: &str, ty: &Type) -> Result<Self> {
        Ok(Self {
            data: ObjectArc::new(VarObj {
                base: ExprObj::new(ty.clone(), None),
                name: String::from(name),
            }),
        })
    }
}

impl GlobalVar {
    /// Construct a module-level symbol directly in Rust.
    pub fn new(name_hint: &str) -> Result<Self> {
        Ok(Self {
            data: ObjectArc::new(GlobalVarObj {
                base: ExprObj::new(Type::missing()?, None),
                name_hint: String::from(name_hint),
            }),
        })
    }
}

impl Call {
    /// Construct a call directly in Rust with no attributes or explicit type arguments.
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
        Ok(Self {
            data: ObjectArc::new(CallObj {
                base: ExprObj::new(ret_type.clone(), None),
                op: operator.clone(),
                args: arguments,
                attrs,
                ty_args: type_arguments,
            }),
        })
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

    /// Return an independently updatable module with one global-info group set.
    pub fn with_updated_global_info(
        &self,
        name: &str,
        global_info: Vec<GlobalInfo>,
    ) -> Result<Self> {
        let module = self.copy_for_update()?;
        let name = String::from(name);
        let global_info = Array::new(global_info);
        crate::global_function!("ir.Module_UpdateGlobalInfo")?.call_packed(&[
            AnyView::from(&module),
            AnyView::from(&name),
            AnyView::from(&global_info),
        ])?;
        Ok(module)
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
        Self::from_dictionary(&AnyMap::new())
    }

    /// Construct DictAttrs from a heterogeneous string-to-value map.
    pub fn from_dictionary(dictionary: &AnyMap<String>) -> Result<Self> {
        Ok(Self {
            data: ObjectArc::new(DictAttrsObj {
                base: AttrsObj {
                    base: tvm_ffi::Object::new(),
                },
                dict: dictionary.clone(),
            }),
        })
    }
}

impl Default for SourceMap {
    fn default() -> Self {
        Self::new()
    }
}
