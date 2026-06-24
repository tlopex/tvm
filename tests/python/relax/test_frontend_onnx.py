# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-argument
# ruff: noqa: E501, F841
"""
ONNX testcases
================
This file is a test script to test Relax ONNX frontend coverage.
"""

from typing import Literal

import numpy as np
import pytest

pytest.importorskip("onnx")

import onnx
import tvm_ffi
from onnx import ModelProto, TensorProto, helper, numpy_helper

import tvm
import tvm.testing
import tvm.tirx
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T

bg = np.random.MT19937(0)
rg = np.random.Generator(bg)


def check_import(
    model: ModelProto,
    ir_version: int = 8,
    opset: int = 14,
) -> tvm.IRModule:
    """Import an ONNX model and verify that the frontend can construct Relax IR.

    Parameters
    ----------
    model: ModelProto
        The input onnx model that should be tested.
    ir_version: int
        Which version of the onnx IR to use.
    opset: int
        The opset version to use for the onnx importer.
    """
    configure_model_format(model, ir_version=ir_version, opset=opset)

    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
    assert tvm_model["main"] is not None
    return tvm_model


def configure_model_format(
    model: ModelProto,
    ir_version: int | None = 8,
    opset: int | None = 14,
) -> None:
    if ir_version is not None:
        model.ir_version = ir_version
    if opset is not None:
        for opset_import in model.opset_import:
            if opset_import.domain in ["", "ai.onnx"]:
                opset_import.version = opset
                break


def generate_random_inputs(
    model: ModelProto, inputs: dict[str, np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    input_values = {}
    for input_info in model.graph.input:
        if inputs is not None and input_info.name in inputs and inputs[input_info.name] is not None:
            input_values[input_info.name] = inputs[input_info.name]
            continue

        shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
        input_values[input_info.name] = generate_random_value(
            shape, input_info.type.tensor_type.elem_type
        )

    return input_values


def generate_random_value(shape, elem_type) -> np.ndarray:
    np_dtype = helper.tensor_dtype_to_np_dtype(elem_type) if elem_type else np.dtype("float32")
    dtype = np.dtype(np_dtype)

    if np.issubdtype(dtype, np.bool_):
        random_value = rg.choice(a=[False, True], size=shape)
    elif np.issubdtype(dtype, np.signedinteger):
        random_value = rg.integers(low=-63, high=63, size=shape).astype(dtype)
        random_value[random_value == 0] = 1
    elif np.issubdtype(dtype, np.unsignedinteger):
        random_value = rg.integers(low=1, high=63, size=shape).astype(dtype)
    else:
        random_value = rg.standard_normal(size=shape).astype(dtype)

    return random_value


def check_correctness(
    model: ModelProto,
    inputs: dict[str, np.ndarray] | None = None,
    ir_version: int = 8,
    opset: int = 14,
    rtol: float = 1e-7,
    atol: float = 1e-5,
    check_dtypes: bool = False,
) -> tvm.IRModule:
    """Compare an imported ONNX model against ONNX Runtime for selected core-op paths."""
    onnxruntime = pytest.importorskip("onnxruntime")

    configure_model_format(model, ir_version=ir_version, opset=opset)
    inputs = generate_random_inputs(model, inputs)

    ort_session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ort_output = ort_session.run([], inputs)

    imported_model = from_onnx(model, opset=opset, keep_params_in_input=True)
    tvm_model = relax.transform.DecomposeOpsForInference()(imported_model)
    tvm_model = relax.transform.LegalizeOps()(tvm_model)
    tvm_model, params = relax.frontend.detach_params(tvm_model)

    with tvm.transform.PassContext(opt_level=3):
        ex = tvm.compile(tvm_model, target="llvm")
        vm = relax.VirtualMachine(ex, tvm.cpu())

    input_list = [
        inputs[param.name_hint] for param in tvm_model["main"].params if param.name_hint in inputs
    ]
    if params:
        input_list += params["main"]

    vm.set_input("main", *input_list)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    if len(ort_output) == 1:
        tvm_output = [tvm_output]

    def _get_numpy_subdtype(narray):
        if np.issubdtype(narray.dtype, np.integer):
            return "integer"
        if np.issubdtype(narray.dtype, np.floating):
            return "floating"
        if np.issubdtype(narray.dtype, np.bool_):
            return "bool"
        if np.issubdtype(narray.dtype, np.complexfloating):
            return "complexfloating"
        return "other"

    def _check_output(tvm_out, ort_out):
        if isinstance(tvm_out, tuple) and isinstance(ort_out, tvm_ffi.Shape | list):
            assert len(tvm_out) == len(ort_out), "Unequal number of outputs"
            for tvm_out_i, ort_out_i in zip(tvm_out, ort_out):
                _check_output(tvm_out_i, ort_out_i)
        elif isinstance(tvm_out, tvm.runtime.Tensor) and isinstance(ort_out, np.ndarray):
            if check_dtypes:
                assert tvm_out.numpy().dtype == ort_out.dtype
            tvm.testing.assert_allclose(tvm_out.numpy(), ort_out, rtol=rtol, atol=atol)
        elif isinstance(tvm_out, tvm_ffi.Shape) and isinstance(ort_out, np.ndarray):
            shape_out = tvm.runtime.tensor([int(i) for i in tvm_out])
            if check_dtypes:
                assert _get_numpy_subdtype(shape_out.numpy()) == _get_numpy_subdtype(ort_out)
            tvm.testing.assert_allclose(shape_out.numpy(), ort_out, rtol=rtol, atol=atol)
        elif isinstance(tvm_out, int | float | bool) and isinstance(ort_out, np.ndarray):
            if check_dtypes:
                assert _get_numpy_subdtype(np.array(tvm_out)) == _get_numpy_subdtype(ort_out)
            tvm.testing.assert_allclose(np.array(tvm_out), ort_out, rtol=rtol, atol=atol)
        else:
            raise ValueError(f"Unsupported types: {type(tvm_out)}, {type(ort_out)}")

    assert len(tvm_output) == len(ort_output), "Unequal number of outputs"
    for tvm_out, ort_out in zip(tvm_output, ort_output):
        if ort_out is not None:
            _check_output(tvm_out, ort_out)

    return imported_model


def run_in_tvm(
    model: ModelProto,
    inputs: dict[str, np.ndarray] | None = None,
    ir_version: int = 8,
    opset: int = 14,
):
    configure_model_format(model, ir_version=ir_version, opset=opset)

    inputs = generate_random_inputs(model, inputs)
    tvm_model = from_onnx(model, opset=opset, keep_params_in_input=True)
    tvm_model = relax.transform.DecomposeOpsForInference()(tvm_model)
    tvm_model = relax.transform.LegalizeOps()(tvm_model)
    tvm_model, params = relax.frontend.detach_params(tvm_model)

    with tvm.transform.PassContext(opt_level=3):
        ex = tvm.compile(tvm_model, target="llvm")
        vm = relax.VirtualMachine(ex, tvm.cpu())

    input_list = [
        inputs[param.name_hint] for param in tvm_model["main"].params if param.name_hint in inputs
    ]
    if params:
        input_list += params["main"]

    vm.set_input("main", *input_list)
    vm.invoke_stateful("main")
    return vm.get_outputs("main")


def get_static_tensor_shape(tensor_sinfo) -> list[int] | None:
    shape = getattr(tensor_sinfo, "shape", None)
    values = getattr(shape, "values", None)
    if values is None:
        return None

    static_shape = []
    for value in values:
        if not isinstance(value, tvm.tirx.IntImm):
            return None
        static_shape.append(int(value))
    return static_shape


def assert_tensor_sinfo(
    tensor_sinfo, expected_shape=None, expected_dtype: str | None = None
) -> None:
    if expected_dtype is not None:
        dtype = getattr(tensor_sinfo, "dtype", None)
        if dtype is None and isinstance(tensor_sinfo, tvm.ir.PrimType):
            dtype = tensor_sinfo.dtype
        assert dtype == expected_dtype

    if expected_shape is not None:
        assert get_static_tensor_shape(tensor_sinfo) == list(expected_shape)


def static_shape_or_none(shape) -> list[int] | None:
    if shape is None:
        return None
    if all(isinstance(dim, int) for dim in shape):
        return list(shape)
    return None


def collect_relax_call_ops(func: relax.Function) -> list[str]:
    op_names: list[str] = []

    def fvisit(expr: relax.Expr) -> None:
        if isinstance(expr, relax.Call) and isinstance(expr.op, tvm.ir.Op):
            op_names.append(expr.op.name)

    relax.analysis.post_order_visit(func.body, fvisit)
    return list(op_names)


def collect_relax_calls(func: relax.Function, op_name: str) -> list[relax.Call]:
    calls: list[relax.Call] = []

    def fvisit(expr: relax.Expr) -> None:
        if (
            isinstance(expr, relax.Call)
            and isinstance(expr.op, tvm.ir.Op)
            and expr.op.name == op_name
        ):
            calls.append(expr)

    relax.analysis.post_order_visit(func.body, fvisit)
    return calls


def assert_has_relax_if(func: relax.Function, min_count: int = 1) -> None:
    if_count = 0

    def fvisit(expr: relax.Expr) -> None:
        nonlocal if_count
        if isinstance(expr, relax.If):
            if_count += 1

    relax.analysis.post_order_visit(func.body, fvisit)
    assert if_count >= min_count


def assert_tuple_tensor_sinfo(tuple_sinfo, expected_fields: list[tuple[list[int], str]]) -> None:
    fields = getattr(tuple_sinfo, "fields", None)
    assert fields is not None
    assert len(fields) == len(expected_fields)
    for field, (expected_shape, expected_dtype) in zip(fields, expected_fields):
        assert_tensor_sinfo(field, expected_shape, expected_dtype)


def assert_has_relax_ops(func: relax.Function, expected_ops: str | list[str] | None) -> None:
    if expected_ops is None:
        return
    if isinstance(expected_ops, str):
        expected_ops = [expected_ops]
    call_ops = collect_relax_call_ops(func)
    for expected_op in expected_ops:
        assert expected_op in call_ops


def collect_scalar_constants(func: relax.Function) -> list[bool | int | float]:
    values = []

    def fvisit(expr):
        if isinstance(expr, relax.Constant):
            value = expr.data.numpy()
            if value.shape == ():
                values.append(value.item())

    relax.analysis.post_order_visit(func.body, fvisit)
    return values


ONNX_RELAX_OPS = {
    "Abs": "relax.abs",
    "Acos": "relax.acos",
    "Acosh": "relax.acosh",
    "Add": "relax.add",
    "And": "relax.logical_and",
    "ArgMax": "relax.argmax",
    "ArgMin": "relax.argmin",
    "Asin": "relax.asin",
    "Asinh": "relax.asinh",
    "Atan": "relax.atan",
    "Atanh": "relax.atanh",
    "BatchNormalization": "relax.nn.batch_norm",
    "BitwiseAnd": "relax.bitwise_and",
    "BitwiseNot": "relax.bitwise_not",
    "BitwiseOr": "relax.bitwise_or",
    "BitwiseXor": "relax.bitwise_xor",
    "BitShiftLeft": "relax.left_shift",
    "BitShiftRight": "relax.right_shift",
    "Ceil": "relax.ceil",
    "Compress": "relax.take",
    "Concat": "relax.concat",
    "Cos": "relax.cos",
    "Cosh": "relax.cosh",
    "DequantizeLinear": "relax.dequantize",
    "DepthToSpace": ["relax.reshape", "relax.permute_dims"],
    "Div": "relax.divide",
    "Dropout": "relax.nn.dropout",
    "DynamicQuantizeLinear": ["relax.quantize", "relax.round", "relax.clip"],
    "Elu": ["relax.exp", "relax.nn.relu"],
    "Equal": "relax.equal",
    "Erf": "relax.erf",
    "Exp": "relax.exp",
    "Expand": "relax.broadcast_to",
    "EyeLike": "relax.eye_like",
    "Flatten": "relax.reshape",
    "Floor": "relax.floor",
    "Gather": "relax.take",
    "GatherElements": "relax.gather_elements",
    "GatherND": "relax.gather_nd",
    "Gemm": "relax.matmul",
    "GlobalAveragePool": "relax.mean",
    "GlobalLpPool": ["relax.abs", "relax.power", "relax.sum"],
    "GlobalMaxPool": "relax.max",
    "Greater": "relax.greater",
    "GreaterOrEqual": "relax.greater_equal",
    "HardSigmoid": ["relax.multiply", "relax.add", "relax.clip"],
    "HardSwish": ["relax.multiply", "relax.divide", "relax.clip"],
    "Hardmax": "relax.one_hot",
    "InstanceNormalization": ["relax.mean", "relax.sqrt"],
    "IsInf": "relax.isinf",
    "IsNaN": "relax.isnan",
    "LeakyRelu": "relax.nn.leakyrelu",
    "Less": "relax.less",
    "LessOrEqual": "relax.less_equal",
    "Log": "relax.log",
    "LogSoftmax": "relax.nn.log_softmax",
    "MatMul": "relax.matmul",
    "Max": "relax.max",
    "Mean": "relax.mean",
    "MeanVarianceNormalization": ["relax.mean", "relax.power", "relax.sqrt"],
    "Min": "relax.min",
    "Mish": ["relax.exp", "relax.log", "relax.tanh"],
    "Mul": "relax.multiply",
    "Neg": "relax.negative",
    "NonZero": "relax.nonzero",
    "Not": "relax.logical_not",
    "OneHot": "relax.one_hot",
    "Or": "relax.logical_or",
    "Pad": "relax.call_tir",
    "Pow": "relax.power",
    "PRelu": "relax.nn.prelu",
    "QuantizeLinear": "relax.quantize",
    "Reciprocal": "relax.divide",
    "ReduceL1": ["relax.abs", "relax.sum"],
    "ReduceL2": ["relax.multiply", "relax.sum", "relax.sqrt"],
    "ReduceLogSum": ["relax.sum", "relax.log"],
    "ReduceLogSumExp": ["relax.max", "relax.exp", "relax.sum", "relax.log"],
    "ReduceMax": "relax.max",
    "ReduceMean": "relax.mean",
    "ReduceMin": "relax.min",
    "ReduceProd": "relax.prod",
    "ReduceSum": "relax.sum",
    "ReduceSumSquare": ["relax.multiply", "relax.sum"],
    "Relu": "relax.nn.relu",
    "Reshape": "relax.reshape",
    "Round": "relax.round",
    "Scatter": "relax.scatter_elements",
    "ScatterElements": "relax.scatter_elements",
    "ScatterND": "relax.scatter_nd",
    "Selu": ["relax.exp", "relax.nn.relu"],
    "Shape": "relax.shape_of",
    "Shrink": ["relax.where", "relax.greater", "relax.less"],
    "Sign": "relax.sign",
    "Sigmoid": "relax.sigmoid",
    "Sin": "relax.sin",
    "Sinh": "relax.sinh",
    "Sqrt": "relax.sqrt",
    "Softmax": "relax.nn.softmax",
    "Softplus": "relax.nn.softplus",
    "Softsign": ["relax.divide", "relax.abs", "relax.add"],
    "SpaceToDepth": ["relax.reshape", "relax.permute_dims"],
    "Slice": "relax.strided_slice",
    "Split": "relax.split",
    "Squeeze": "relax.reshape",
    "Sub": "relax.subtract",
    "Sum": "relax.sum",
    "Tan": "relax.tan",
    "Tanh": "relax.tanh",
    "ThresholdedRelu": ["relax.greater", "relax.astype", "relax.multiply"],
    "Tile": "relax.call_tir",
    "TopK": "relax.topk",
    "Transpose": "relax.permute_dims",
    "Unsqueeze": "relax.expand_dims",
    "Unique": "relax.unique",
    "Where": "relax.where",
    "Xor": "relax.logical_xor",
}

ONNX_NUMERIC_RELAX_OPS = {
    "Abs",
    "Acos",
    "Acosh",
    "Add",
    "And",
    "ArgMax",
    "ArgMin",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "AveragePool",
    "BatchNormalization",
    "BiasGelu",
    "BitwiseAnd",
    "BitwiseNot",
    "BitwiseOr",
    "BitwiseXor",
    "BitShift",
    "Cast",
    "Ceil",
    "Clip",
    "Compress",
    "Concat",
    "Conv",
    "ConvTranspose",
    "Cos",
    "Cosh",
    "CumSum",
    "DequantizeLinear",
    "DepthToSpace",
    "Div",
    "Dropout",
    "DynamicQuantizeLinear",
    "Elu",
    "Equal",
    "Erf",
    "Exp",
    "Expand",
    "Flatten",
    "Floor",
    "FastGelu",
    "Gather",
    "GatherElements",
    "GatherND",
    "Gemm",
    "Gelu",
    "GlobalAveragePool",
    "GlobalLpPool",
    "GlobalMaxPool",
    "Greater",
    "GreaterOrEqual",
    "HardSigmoid",
    "HardSwish",
    "Hardmax",
    "InstanceNormalization",
    "IsInf",
    "IsNaN",
    "LayerNormalization",
    "LeakyRelu",
    "Less",
    "LessOrEqual",
    "Log",
    "LogSoftmax",
    "MatMul",
    "MatMulInteger",
    "MatMulInteger16",
    "Max",
    "MaxPool",
    "Mean",
    "MeanVarianceNormalization",
    "Min",
    "Mish",
    "Mul",
    "Neg",
    "NonZero",
    "Not",
    "OneHot",
    "Or",
    "Pow",
    "PRelu",
    "QuantizeLinear",
    "Reciprocal",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSum",
    "ReduceSumSquare",
    "Relu",
    "Reshape",
    "Round",
    "Scatter",
    "ScatterElements",
    "ScatterND",
    "Selu",
    "Shrink",
    "Sign",
    "Sigmoid",
    "Sin",
    "Sinh",
    "Shape",
    "Size",
    "Softmax",
    "Softplus",
    "Softsign",
    "SpaceToDepth",
    "Sqrt",
    "Slice",
    "Split",
    "Squeeze",
    "Sub",
    "Sum",
    "Tan",
    "Tanh",
    "ThresholdedRelu",
    "TopK",
    "Transpose",
    "Unsqueeze",
    "Unique",
    "Where",
    "Xor",
}

ONNX_NUMERIC_RELAX_OP_DOMAINS = {
    ("BiasGelu", "com.microsoft"),
    ("FastGelu", "com.microsoft"),
    ("Gelu", "com.microsoft"),
}

ONNX_SHAPE_PRESERVING_UNARY_OPS = {
    "Abs",
    "Acos",
    "Acosh",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "BitwiseNot",
    "Ceil",
    "Cos",
    "Cosh",
    "Erf",
    "Exp",
    "Floor",
    "HardSigmoid",
    "LeakyRelu",
    "Log",
    "Neg",
    "Not",
    "Reciprocal",
    "Selu",
    "Shrink",
    "Sigmoid",
    "Sin",
    "Sinh",
    "Sqrt",
    "Tan",
    "Tanh",
    "ThresholdedRelu",
}


def should_check_numeric(op_name: str, domain: str | None = None) -> bool:
    return op_name in ONNX_NUMERIC_RELAX_OPS and (
        domain in (None, "") or (op_name, domain) in ONNX_NUMERIC_RELAX_OP_DOMAINS
    )


@pytest.mark.parametrize(
    "input_names, expected_names",
    [
        ([".", "123"], ["_", "input_123"]),
        ([".", "_"], ["_", "__1"]),
        (["123", "input_123"], ["input_123", "input_123_1"]),
    ],
)
def test_sanitize(input_names, expected_names):
    node = helper.make_node("Add", inputs=input_names, outputs=["output"])
    graph = helper.make_graph(
        [node],
        "test",
        inputs=[
            helper.make_tensor_value_info(str(var), TensorProto.FLOAT, [32, 32])
            for var in input_names
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 32]),
        ],
    )
    model = helper.make_model(graph, producer_name="test_sanitizer")

    tvm_model = from_onnx(model)

    for i, param in enumerate(tvm_model["main"].params):
        assert param.name_hint == expected_names[i]


def verify_unary(
    op_name,
    shape,
    attrs={},
    domain=None,
    input_dtype=TensorProto.FLOAT,
    output_dtype=TensorProto.FLOAT,
    opset=14,
):
    test_node = helper.make_node(op_name, ["x"], ["y"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "elemwise_test",
        inputs=[
            helper.make_tensor_value_info("x", input_dtype, shape),
        ],
        outputs=[helper.make_tensor_value_info("y", output_dtype, shape)],
    )

    model = helper.make_model(graph, producer_name="elemwise_test")
    if should_check_numeric(op_name, domain):
        check_correctness(model, opset=opset)
    else:
        tvm_model = check_import(model, opset=opset)
        assert_tensor_sinfo(
            tvm_model["main"].ret_ty,
            static_shape_or_none(shape) if op_name in ONNX_SHAPE_PRESERVING_UNARY_OPS else None,
            str(helper.tensor_dtype_to_np_dtype(output_dtype)),
        )
        assert_has_relax_ops(tvm_model["main"], ONNX_RELAX_OPS.get(op_name))


def verify_unary_dynamic_shape(
    op_name,
    shape,
    shape_instance,
    attrs={},
    domain=None,
    input_dtype=TensorProto.FLOAT,
    output_dtype=TensorProto.FLOAT,
    opset=14,
):
    test_node = helper.make_node(op_name, ["x"], ["y"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "elemwise_test",
        inputs=[
            helper.make_tensor_value_info("x", input_dtype, shape),
        ],
        outputs=[helper.make_tensor_value_info("y", output_dtype, shape)],
    )

    model = helper.make_model(graph, producer_name="elemwise_test")
    if should_check_numeric(op_name, domain):
        inputs = {"x": generate_random_value(shape_instance, input_dtype)}
        check_correctness(model, inputs, opset=opset)
    else:
        tvm_model = check_import(model, opset=opset)
        assert_tensor_sinfo(
            tvm_model["main"].ret_ty,
            static_shape_or_none(shape) if op_name in ONNX_SHAPE_PRESERVING_UNARY_OPS else None,
            str(helper.tensor_dtype_to_np_dtype(output_dtype)),
        )
        assert_has_relax_ops(tvm_model["main"], ONNX_RELAX_OPS.get(op_name))


def verify_binary(
    op_name, shape_a, shape_b, shape_c, attrs={}, domain=None, dtype=TensorProto.FLOAT, opset=14
):
    test_node = helper.make_node(op_name, ["a", "b"], ["c"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "binary_test",
        inputs=[
            helper.make_tensor_value_info("a", dtype, shape_a),
            helper.make_tensor_value_info("b", dtype, shape_b),
        ],
        outputs=[helper.make_tensor_value_info("c", dtype, shape_c)],
    )

    model = helper.make_model(graph, producer_name="binary_test")
    if should_check_numeric(op_name, domain):
        check_correctness(model, opset=opset, check_dtypes=True)
    else:
        tvm_model = check_import(model, opset=opset)
        assert_tensor_sinfo(
            tvm_model["main"].ret_ty,
            static_shape_or_none(shape_c),
            str(helper.tensor_dtype_to_np_dtype(dtype)),
        )
        assert_has_relax_ops(tvm_model["main"], ONNX_RELAX_OPS.get(op_name))


def verify_binary_scalar(op_name, attrs={}, domain=None, dtype=TensorProto.INT32, opset=14):
    a = make_constant_node("a", dtype, [], [4])
    b = make_constant_node("b", dtype, [], [8])
    test_node = helper.make_node(op_name, ["a", "b"], ["c"], **attrs, domain=domain)
    graph = helper.make_graph(
        [a, b, test_node],
        "binary_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("c", dtype, ())],
    )

    model = helper.make_model(graph, producer_name="binary_test")
    if should_check_numeric(op_name, domain):
        check_correctness(model, opset=opset, check_dtypes=True)
    else:
        tvm_model = check_import(model, opset=opset)
        assert_tensor_sinfo(
            tvm_model["main"].ret_ty, [], str(helper.tensor_dtype_to_np_dtype(dtype))
        )


def verify_compare(op_name, shape, attrs={}, domain=None):
    test_node = helper.make_node(op_name, ["a", "b"], ["c"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "compare_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, shape),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.BOOL, shape)],
    )

    model = helper.make_model(graph, producer_name="compare_test")
    if should_check_numeric(op_name, domain):
        check_correctness(model)
    else:
        tvm_model = check_import(model)
        assert_tensor_sinfo(tvm_model["main"].ret_ty, static_shape_or_none(shape), "bool")
        assert_has_relax_ops(tvm_model["main"], ONNX_RELAX_OPS.get(op_name))


def verify_ternary(op_name, shape_a, shape_b, shape_c, shape_d, attrs={}, domain=None):
    test_node = helper.make_node(op_name, ["a", "b", "c"], ["d"], **attrs, domain=domain)
    graph = helper.make_graph(
        [test_node],
        "ternary_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, shape_a),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, shape_b),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, shape_c),
        ],
        outputs=[helper.make_tensor_value_info("d", TensorProto.FLOAT, shape_d)],
    )

    model = helper.make_model(graph, producer_name="ternary_test")
    if should_check_numeric(op_name, domain):
        check_correctness(model)
    else:
        tvm_model = check_import(model)
        assert_tensor_sinfo(tvm_model["main"].ret_ty, static_shape_or_none(shape_d), "float32")
        assert_has_relax_ops(tvm_model["main"], ONNX_RELAX_OPS.get(op_name))


@pytest.mark.parametrize("dynamic", [True, False])
def test_matmul(dynamic):
    matmul_node = helper.make_node("MatMul", ["a", "b"], ["c"])

    a_shape = [32, 48]
    b_shape = [48, 64]
    output_shape = [32, 64]

    if dynamic:
        a_shape = ["?", "?"]

    graph = helper.make_graph(
        [matmul_node],
        "matmul_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape),
        ],
        initializer=[
            helper.make_tensor(
                "b", TensorProto.FLOAT, b_shape, np.random.normal(size=b_shape).astype("float32")
            )
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, output_shape)],
    )

    model = helper.make_model(graph, producer_name="matmul_test")
    inputs = None
    if dynamic:
        inputs = {
            "a": np.random.normal(size=[32, 48]).astype("float32"),
        }
    check_correctness(model, inputs)


@pytest.mark.parametrize(
    ("a_dtype", "b_dtype", "a_shape", "b_shape"),
    [
        (np.int16, np.int16, [2, 3], [3, 4]),
        (np.uint16, np.uint16, [2, 3], [3, 4]),
        (np.int16, np.uint16, [2, 1, 3, 5], [1, 2, 5, 4]),
    ],
)
def test_matmulinteger16(a_dtype, b_dtype, a_shape, b_shape):
    a = np.arange(np.prod(a_shape), dtype=np.int64).reshape(a_shape)
    b = np.arange(np.prod(b_shape), dtype=np.int64).reshape(b_shape)
    if np.issubdtype(a_dtype, np.signedinteger):
        a -= a.size // 2
    if np.issubdtype(b_dtype, np.signedinteger):
        b -= b.size // 2
    a = a.astype(a_dtype)
    b = b.astype(b_dtype)

    out_dtype = np.uint32 if a_dtype == np.uint16 and b_dtype == np.uint16 else np.int32
    expected = np.matmul(a.astype(out_dtype), b.astype(out_dtype))

    node = helper.make_node("MatMulInteger16", ["a", "b"], ["y"], domain="com.microsoft")
    graph = helper.make_graph(
        [node],
        "matmulinteger16_test",
        inputs=[
            helper.make_tensor_value_info("a", helper.np_dtype_to_tensor_dtype(a.dtype), a_shape),
            helper.make_tensor_value_info("b", helper.np_dtype_to_tensor_dtype(b.dtype), b_shape),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "y", helper.np_dtype_to_tensor_dtype(np.dtype(out_dtype)), expected.shape
            )
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="matmulinteger16_test",
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 11

    tvm_output = run_in_tvm(model, inputs={"a": a, "b": b}, ir_version=11, opset=18)
    assert isinstance(tvm_output, tvm.runtime.Tensor)
    assert tvm_output.numpy().dtype == out_dtype
    tvm.testing.assert_allclose(tvm_output.numpy(), expected)

    tvm_model = check_import(model, ir_version=11, opset=18)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, expected.shape, str(np.dtype(out_dtype)))
    assert "relax.matmul" in collect_relax_call_ops(tvm_model["main"])


def test_matmulinteger16_ir():
    node = helper.make_node("MatMulInteger16", ["a", "b"], ["y"], domain="com.microsoft")
    graph = helper.make_graph(
        [node],
        "matmulinteger16_ir_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.UINT16, [2, 3]),
            helper.make_tensor_value_info("b", TensorProto.UINT16, [3, 4]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.UINT32, [2, 4])],
    )
    model = helper.make_model(
        graph,
        producer_name="matmulinteger16_ir_test",
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 11

    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)
    call_ops = collect_relax_call_ops(tvm_model["main"])
    assert call_ops.count("relax.astype") == 2
    assert "relax.matmul" in call_ops
    assert tvm_model["main"].ret_ty.dtype == "uint32"


def test_matmulinteger16_invalid_dtype_raises():
    node = helper.make_node("MatMulInteger16", ["a", "b"], ["y"], domain="com.microsoft")
    graph = helper.make_graph(
        [node],
        "matmulinteger16_invalid_dtype_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.INT8, [2, 3]),
            helper.make_tensor_value_info("b", TensorProto.UINT16, [3, 4]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.INT32, [2, 4])],
    )
    model = helper.make_model(
        graph,
        producer_name="matmulinteger16_invalid_dtype_test",
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 11

    with pytest.raises(ValueError, match="input A"):
        from_onnx(model, opset=18, keep_params_in_input=True)


def test_concat():
    verify_binary("Concat", [1, 32], [1, 32], [2, 32], attrs={"axis": 0})


def test_concat_with_param_shape_value():
    """Concat must handle a 1D-int64 initializer mixed with a ShapeExpr when
    keep_params_in_input=True. Standard pattern in PyTorch-exported ONNX
    models for dynamic-batch Reshape: Reshape(x, Concat(Shape(x)[:1], [12]))."""
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, 4])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 12])
    twelve = numpy_helper.from_array(np.array([12], dtype=np.int64), "twelve")
    starts = numpy_helper.from_array(np.array([0], dtype=np.int64), "starts")
    ends = numpy_helper.from_array(np.array([1], dtype=np.int64), "ends")
    nodes = [
        helper.make_node("Shape", ["x"], ["x_shape"]),
        helper.make_node("Slice", ["x_shape", "starts", "ends"], ["dyn_n"]),
        helper.make_node("Concat", ["dyn_n", "twelve"], ["new_shape"], axis=0),
        helper.make_node("Reshape", ["x", "new_shape"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "concat_param_shape",
        [inp],
        [out],
        initializer=[twelve, starts, ends],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    # Both modes should succeed; previously True crashed with
    # "Op(relax.concat) expects the input to be a Tuple of Tensors".
    from_onnx(model, keep_params_in_input=False)
    from_onnx(model, keep_params_in_input=True)


def test_concat_with_param_tensor_keeps_runtime_param():
    """Concat(input, weight) under keep_params_in_input=True must keep `weight`
    as a runtime param, not fold it into a constant."""
    weight_np = np.arange(8, dtype=np.float32).reshape(2, 4)
    graph = helper.make_graph(
        [helper.make_node("Concat", ["x", "w"], ["y"], axis=0)],
        "concat_param_tensor",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 4])],
        initializer=[numpy_helper.from_array(weight_np, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)

    mod, params = relax.frontend.detach_params(from_onnx(model, keep_params_in_input=True))
    assert "w" in [p.name_hint for p in mod["main"].params]
    assert len(params["main"]) == 1
    np.testing.assert_array_equal(params["main"][0].numpy(), weight_np)


@pytest.mark.parametrize("op_name", ["Add", "Sub", "Mul", "Div", "Pow"])
def test_binary(op_name: str):
    verify_binary(op_name, [1, 32], [1, 32], [1, 32])
    verify_binary_scalar(op_name)


def test_div_integer_constant_zero_divisor_raises_valueerror():
    b_init = numpy_helper.from_array(np.array([3, 0, -2, 1], dtype=np.int32), name="b")
    node = helper.make_node("Div", ["a", "b"], ["y"])
    graph = helper.make_graph(
        [node],
        "div_const_zero",
        [helper.make_tensor_value_info("a", TensorProto.INT32, [4])],
        [helper.make_tensor_value_info("y", TensorProto.INT32, [4])],
        initializer=[b_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9

    with pytest.raises(
        ValueError, match="ONNX Div with integer inputs encountered divisor value 0"
    ):
        from_onnx(model, opset=18, keep_params_in_input=False)


@pytest.mark.parametrize("int_mode", [True, False])
def test_mod(int_mode: bool):
    if int_mode:
        dtype, fmod = TensorProto.INT32, 0
    else:
        dtype, fmod = TensorProto.FLOAT, 1
    verify_binary("Mod", [1, 32], [1, 32], [1, 32], attrs={"fmod": fmod}, dtype=dtype)
    verify_binary_scalar("Mod", attrs={"fmod": fmod}, dtype=dtype)


SHAPE_PARAMS = [
    ([[32, 32], [32, 32]], [32, 32]),
    ([[32, 1], [1, 2]], [32, 2]),
    (
        [
            [
                32,
            ],
            [
                1,
            ],
        ],
        [
            32,
        ],
    ),
    ([[32, 32, 1, 1], [1, 32, 32]], [32, 32, 32, 32]),
    (
        [
            [32, 32, 1, 1],
            [1, 32, 1],
            [
                32,
            ],
        ],
        [32, 32, 32, 32],
    ),
]


@pytest.mark.parametrize("input_shapes, expected_output_shape", SHAPE_PARAMS)
@pytest.mark.parametrize("op_name", ["Min", "Max", "Sum", "Mean"])
def test_multi_input_broadcasting(op_name, input_shapes, expected_output_shape):
    num_inputs = len(input_shapes)
    input_names = [f"i{i}" for i in range(num_inputs)]

    input_values_info = []
    for name, shape in zip(input_names, input_shapes):
        input_values_info.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shape))
    test_node = helper.make_node(op_name, input_names, ["output"])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, expected_output_shape)
    graph = helper.make_graph(
        [test_node],
        f"multi_input_{op_name}_test",
        inputs=input_values_info,
        outputs=[output_info],
    )
    model = helper.make_model(graph, producer_name="multi_input_test")
    check_correctness(model)


@pytest.mark.parametrize("op_name", ["Less", "LessOrEqual", "Greater", "GreaterOrEqual"])
def test_compare(op_name: str):
    verify_compare(op_name, [1, 32])


@pytest.mark.parametrize("op_name", ["And", "Or", "Xor"])
def test_binary_bool(op_name: str):
    verify_binary(op_name, [32, 32], [32, 32], [32, 32], dtype=TensorProto.BOOL)


@pytest.mark.parametrize("op_name", ["BitwiseAnd", "BitwiseOr", "BitwiseXor"])
def test_bitwise(op_name: str):
    verify_binary(op_name, [32, 32], [32, 32], [32, 32], dtype=TensorProto.UINT64, opset=18)


def test_bitwise_not():
    verify_unary(
        "BitwiseNot",
        [32, 32],
        input_dtype=TensorProto.UINT64,
        output_dtype=TensorProto.UINT64,
        opset=18,
    )


@pytest.mark.parametrize("direction", ["LEFT", "RIGHT"])
def test_bitwise_shift(direction: str):
    shape = [32, 32]
    dtype = TensorProto.UINT64
    test_node = helper.make_node("BitShift", ["a", "b"], ["c"], direction=direction)
    graph = helper.make_graph(
        [test_node],
        "binary_test",
        inputs=[
            helper.make_tensor_value_info("a", dtype, shape),
            helper.make_tensor_value_info("b", dtype, shape),
        ],
        outputs=[helper.make_tensor_value_info("c", dtype, shape)],
    )

    model = helper.make_model(graph, producer_name="binary_test")
    check_correctness(model, inputs={"b": np.random.randint(0, 8, shape).astype("uint64")})


@pytest.mark.parametrize(
    "op_name",
    [
        "Sin",
        "Cos",
        "Tan",
        "Sinh",
        "Cosh",
        "Tanh",
        "Asin",
        "Acos",
        "Atan",
        "Asinh",
        "Acosh",
        "Atanh",
        "Neg",
        "Abs",
        "Log",
        "Exp",
        "Not",
        "Reciprocal",
        "Floor",
        "Ceil",
        "Round",
        "IsInf",
        "IsNaN",
        "Sqrt",
        "Relu",
        "Elu",
        "HardSwish",
        "Sign",
        "Softplus",
        "Softsign",
        "Erf",
        "Sigmoid",
        "Softmax",
        "LogSoftmax",
        "Hardmax",
        "Identity",
    ],
)
def test_unary(op_name: str):
    input_dtype = TensorProto.FLOAT
    if op_name in [
        "IsNaN",
        "IsInf",
    ]:
        pytest.skip(f"Skipping test {op_name} because current LegalizeOps does not support it.")
    elif op_name == "Not":
        input_dtype = TensorProto.BOOL
        output_dtype = TensorProto.BOOL
    else:
        output_dtype = TensorProto.FLOAT
    verify_unary(op_name, [8, 8, 8], input_dtype=input_dtype, output_dtype=output_dtype)


@pytest.mark.parametrize("op_name", ["Softmax", "LogSoftmax", "Hardmax"])
def test_softmax_family_opset11_default_axis_semantics(op_name: str):
    verify_unary(op_name, [2, 3, 4], opset=11)


@pytest.mark.parametrize("op_name", ["Softmax", "LogSoftmax", "Hardmax"])
def test_softmax_family_opset11_negative_axis_semantics(op_name: str):
    verify_unary(op_name, [2, 3, 4], attrs={"axis": -2}, opset=11)


@pytest.mark.parametrize("op_name", ["Softmax", "LogSoftmax", "Hardmax"])
def test_softmax_family_opset11_positive_axis_semantics(op_name: str):
    verify_unary(op_name, [2, 3, 4], attrs={"axis": 0}, opset=11)


@pytest.mark.parametrize("op_name", ["Softmax", "LogSoftmax", "Hardmax"])
def tes_softmax_family_opset11_axis_equals_rank_semantics(op_name: str):
    verify_unary(op_name, [2, 3, 4], attrs={"axis": 3}, opset=11)


@pytest.mark.parametrize("op_name", ["Softmax", "LogSoftmax", "Hardmax"])
def test_softmax_family_opset13_default_axis_semantics(op_name: str):
    verify_unary(op_name, [2, 3, 4], opset=13)


@pytest.mark.parametrize(
    "op_name, expected_core_op",
    [
        ("Softmax", "relax.nn.softmax"),
        ("LogSoftmax", "relax.nn.log_softmax"),
        ("Hardmax", "relax.one_hot"),
    ],
)
def test_softmax_family_opset1_legacy_ir_semantics(op_name: str, expected_core_op: str):
    node = helper.make_node(op_name, ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "softmax_family_opset1_ir_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])],
    )
    model = helper.make_model(
        graph,
        producer_name="softmax_family_opset1_ir_test",
        opset_imports=[helper.make_opsetid("", 1)],
    )
    tvm_model = from_onnx(model, opset=1, keep_params_in_input=True)
    call_ops = collect_relax_call_ops(tvm_model["main"])

    assert expected_core_op in call_ops
    assert call_ops.count("relax.reshape") >= 2


def test_round_ties_to_even():
    """ONNX Round uses ties-to-even semantics under opset 11."""
    round_node = helper.make_node("Round", ["x"], ["y"])
    graph = helper.make_graph(
        [round_node],
        "round_ties_to_even_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [6])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [6])],
    )
    model = helper.make_model(graph, producer_name="round_ties_to_even_test")
    inputs = {"x": np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5], dtype="float32")}
    check_correctness(model, inputs=inputs, opset=11)


@pytest.mark.parametrize("from_type", [TensorProto.INT32, TensorProto.FLOAT, TensorProto.FLOAT16])
@pytest.mark.parametrize("to_type", [TensorProto.INT32, TensorProto.FLOAT, TensorProto.FLOAT16])
def test_cast(from_type, to_type):
    cast_node = helper.make_node("Cast", ["a"], ["a_float"], to=to_type)

    graph = helper.make_graph(
        [cast_node],
        "cast_test",
        inputs=[
            helper.make_tensor_value_info("a", from_type, [1, 32]),
        ],
        outputs=[helper.make_tensor_value_info("a_float", to_type, [1, 32])],
    )

    model = helper.make_model(graph, producer_name="cast_test")
    check_correctness(model, opset=13)


@pytest.mark.parametrize("to_type", [TensorProto.INT64, TensorProto.UINT64])
def test_cast_float_to_64bit_int_dynamic(to_type):
    cast_node = helper.make_node("Cast", ["a"], ["b"], to=to_type)
    graph = helper.make_graph(
        [cast_node],
        "cast_float_to_64bit_int_dynamic_test",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 8])],
        outputs=[helper.make_tensor_value_info("b", to_type, [1, 8])],
    )
    model = helper.make_model(graph, producer_name="cast_float_to_64bit_int_dynamic_test")
    inputs = {"a": np.array([[0.0, 1.2, 2.8, 7.9, 15.1, 31.7, 63.4, 127.9]], dtype=np.float32)}
    check_correctness(model, inputs=inputs, opset=13, check_dtypes=True)


def test_cast_nan_inf_to_int8():
    vals = np.array([300.0, np.nan, np.inf, -np.inf, 50.0, -50.0], dtype=np.float32)
    node = helper.make_node("Cast", inputs=["a"], outputs=["b"], to=TensorProto.INT8)
    graph = helper.make_graph(
        [node],
        "cast_nan_inf_test",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, list(vals.shape))],
        outputs=[helper.make_tensor_value_info("b", TensorProto.INT8, list(vals.shape))],
    )
    model = helper.make_model(graph, producer_name="cast_nan_inf_test")
    tvm_model = check_import(model, opset=13)
    tvm_output = run_in_tvm(model, inputs={"a": vals}, opset=13)
    out_np = tvm_output.numpy()
    expected = np.array([44, 0, 0, 0, 50, -50], dtype=np.int8)
    assert out_np.dtype == np.int8
    np.testing.assert_array_equal(out_np, expected)

    assert_tensor_sinfo(tvm_model["main"].ret_ty, vals.shape, "int8")
    assert_has_relax_ops(
        tvm_model["main"],
        [
            "relax.isfinite",
            "relax.logical_not",
            "relax.where",
            "relax.astype",
            "relax.bitwise_and",
            "relax.greater_equal",
            "relax.subtract",
        ],
    )
    call_ops = collect_relax_call_ops(tvm_model["main"])
    assert call_ops.count("relax.where") >= 2
    assert call_ops.count("relax.astype") >= 2


def test_gather():
    def _verify_gather(data_shape, indices, out_shape, axis=0):
        gather_node = helper.make_node("Gather", ["data", "indices"], ["y"], axis=axis)

        if isinstance(indices, list | tuple):
            indices_shape = np.asarray(indices).shape
        else:
            indices_shape = []

        graph = helper.make_graph(
            [gather_node],
            "gather_test",
            inputs=[
                helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
                helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
        )

        model = helper.make_model(graph, producer_name="gather_test")
        input_values = {
            "data": np.random.randn(*data_shape).astype("float32"),
            "indices": np.array(indices).astype("int64"),
        }
        check_correctness(model, inputs=input_values)

    _verify_gather([5, 4, 3, 2], [0, 1, 3], [3, 4, 3, 2])
    _verify_gather([3], 0, [])
    _verify_gather([3, 3], [[0, 2]], [3, 1, 2], 1)


@pytest.mark.parametrize(
    "axis, indices, out_shape",
    [
        (0, [-1, 0], [2, 4]),
        (1, [-1, 0], [3, 2]),
        (
            1,
            [[-1, 0], [1, -2]],
            [3, 2, 2],
        ),
    ],
)
@pytest.mark.parametrize("indices_type", [TensorProto.INT64, TensorProto.INT32])
def test_gather_negative_indices(axis, indices, out_shape, indices_type):
    gather_node = helper.make_node("Gather", ["data", "indices"], ["y"], axis=axis)
    indices_shape = np.asarray(indices).shape

    graph = helper.make_graph(
        [gather_node],
        "gather_negative_indices_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info("indices", indices_type, indices_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
    )

    model = helper.make_model(graph, producer_name="gather_negative_indices_test")
    indices_np_dtype = {
        TensorProto.INT64: np.int64,
        TensorProto.INT32: np.int32,
    }[indices_type]
    input_values = {
        "data": np.random.randn(3, 4).astype("float32"),
        "indices": np.array(indices).astype(indices_np_dtype),
    }
    check_correctness(model, inputs=input_values)


@pytest.mark.parametrize("indices_type", [TensorProto.INT64, TensorProto.INT32])
def test_gather_negative_indices_ir_normalization(indices_type):
    gather_node = helper.make_node("Gather", ["data", "indices"], ["y"], axis=1)
    graph = helper.make_graph(
        [gather_node],
        "gather_negative_indices_ir_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info("indices", indices_type, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])],
    )

    model = helper.make_model(graph, producer_name="gather_negative_indices_ir_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)
    call_ops = collect_relax_call_ops(tvm_model["main"])

    assert "relax.where" in call_ops
    assert "relax.less" in call_ops
    assert "relax.add" in call_ops
    assert "relax.take" in call_ops


@pytest.mark.parametrize(
    "data_shape, indices_shape, axis",
    [
        ([3, 4, 5], [1, 4, 5], 0),
        ([3, 4, 5], [3, 2, 5], 1),
        ([3, 4, 5], [3, 4, 2], 2),
    ],
)
def test_gather_elements(data_shape, indices_shape, axis):
    gather_elements_node = helper.make_node("GatherElements", ["data", "indices"], ["y"], axis=axis)

    graph = helper.make_graph(
        [gather_elements_node],
        "gather_elements_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, indices_shape)],
    )

    model = helper.make_model(graph, producer_name="gather_elements_test")
    input_values = {
        "data": np.random.randn(*data_shape).astype("float32"),
        "indices": np.random.randint(0, data_shape[axis], indices_shape).astype("int64"),
    }
    check_correctness(model, inputs=input_values)


@pytest.mark.parametrize(
    "data_shape, indices_shape, batch_dims",
    [
        ([2, 2], [2, 2], 0),
        ([2, 2], [2, 1], 0),
        ([2, 2, 2], [1], 0),
        ([2, 2, 2], [2, 2], 0),
        ([2, 2, 2], [2, 1, 2], 0),
        ([2, 2, 2], [2, 2], 1),
        ([2, 2, 2], [2, 1], 1),
    ],
)
def test_gather_nd(data_shape, indices_shape, batch_dims):
    gather_nd_node = helper.make_node("GatherND", ["data", "indices"], ["y"], batch_dims=batch_dims)

    graph = helper.make_graph(
        [gather_nd_node],
        "gather_nd_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )

    model = helper.make_model(graph, producer_name="gather_nd_test")
    input_values = {
        "data": np.random.randn(*data_shape).astype("float32"),
        "indices": np.random.randint(0, 2, indices_shape).astype("int64"),
    }
    check_correctness(model, inputs=input_values)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize(("name", "opset"), [("Scatter", 10), ("ScatterElements", 11)])
def test_scatter(axis: int, name: str, opset: int):
    if axis != 1:
        pytest.skip("The current topi impl is wrong, which only works for axis=1")
    input_shape = [16, 16, 16]
    indices_shape = [8, 8, 8]
    updates_shape = [8, 8, 8]
    output_shape = [16, 16, 16]
    node = helper.make_node(name, ["data", "indices", "updates"], ["output"], axis=axis)
    graph = helper.make_graph(
        [node],
        "scatter_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, input_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, updates_shape),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, producer_name="scatter_test")
    indices = np.random.randint(0, 16, indices_shape)
    check_correctness(model, inputs={"indices": indices}, opset=opset)


@pytest.mark.parametrize(
    "reduction, opset, data, indices, updates",
    [
        (
            None,
            11,
            np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"),
            np.array([[2, 0, 1], [1, 2, 0]], dtype="int64"),
            np.array([[30, 10, 20], [50, 60, 40]], dtype="float32"),
        ),
        (
            "none",
            18,
            np.array([[1, 2, 3], [4, 5, 6]], dtype="float32"),
            np.array([[2, 0, 1], [1, 2, 0]], dtype="int64"),
            np.array([[30, 10, 20], [50, 60, 40]], dtype="float32"),
        ),
        (
            "add",
            16,
            np.full((2, 3), 10, dtype="float32"),
            np.array([[0, 0, 2], [1, 1, 2]], dtype="int64"),
            np.array([[2, 5, 7], [20, 3, 4]], dtype="float32"),
        ),
        (
            "mul",
            16,
            np.full((2, 3), 10, dtype="float32"),
            np.array([[0, 0, 2], [1, 1, 2]], dtype="int64"),
            np.array([[2, 5, 7], [20, 3, 4]], dtype="float32"),
        ),
        (
            "min",
            18,
            np.full((2, 3), 10, dtype="float32"),
            np.array([[0, 0, 2], [1, 1, 2]], dtype="int64"),
            np.array([[2, 5, 7], [20, 3, 4]], dtype="float32"),
        ),
        (
            "max",
            18,
            np.full((2, 3), 10, dtype="float32"),
            np.array([[0, 0, 2], [1, 1, 2]], dtype="int64"),
            np.array([[2, 5, 7], [20, 3, 4]], dtype="float32"),
        ),
    ],
)
def test_scatter_elements_reduction(reduction, opset, data, indices, updates):
    attrs = {"axis": 1}
    if reduction is not None:
        attrs["reduction"] = reduction
    scatter_elements_node = helper.make_node(
        "ScatterElements", ["data", "indices", "updates"], ["output"], **attrs
    )

    graph = helper.make_graph(
        [scatter_elements_node],
        "scatter_elements_reduction_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, list(data.shape)),
            helper.make_tensor_value_info("indices", TensorProto.INT64, list(indices.shape)),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, list(updates.shape)),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, list(data.shape))],
    )
    model = helper.make_model(graph, producer_name="scatter_elements_reduction_test")

    check_correctness(
        model,
        inputs={"data": data, "indices": indices, "updates": updates},
        opset=opset,
    )


def test_scatter_elements_invalid_reduction():
    data_shape = [2, 3]
    scatter_elements_node = helper.make_node(
        "ScatterElements",
        ["data", "indices", "updates"],
        ["output"],
        axis=1,
        reduction="unsupported",
    )

    graph = helper.make_graph(
        [scatter_elements_node],
        "scatter_elements_invalid_reduction_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, data_shape),
            helper.make_tensor_value_info("updates", TensorProto.FLOAT, data_shape),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, data_shape)],
    )
    model = helper.make_model(graph, producer_name="scatter_elements_invalid_reduction_test")

    with pytest.raises(ValueError, match="Only .* reductions are supported, but got unsupported"):
        from_onnx(model, opset=18, keep_params_in_input=True)


@pytest.mark.parametrize("reduction", ["none", "add", "mul"])
def test_scatter_nd(reduction):
    def verify_scatter_nd(data_shape, indices_shape, updates_shape):
        scatter_nd_node = helper.make_node(
            "ScatterND",
            ["data", "indices", "updates"],
            ["output"],
            reduction=reduction,
        )

        graph = helper.make_graph(
            [scatter_nd_node],
            "scatter_nd_test",
            inputs=[
                helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
                helper.make_tensor_value_info("indices", TensorProto.INT64, indices_shape),
                helper.make_tensor_value_info("updates", TensorProto.FLOAT, updates_shape),
            ],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, data_shape)],
        )

        model = helper.make_model(graph, producer_name="scatter_nd_test")
        indices = np.random.choice(data_shape[0], indices_shape)
        check_correctness(model, inputs={"indices": indices}, opset=16)

    verify_scatter_nd([8], [4, 1], [4])
    verify_scatter_nd([4, 4, 4], [2, 1], [2, 4, 4])
    verify_scatter_nd([4, 5, 6], [2, 3, 2], [2, 3, 6])
    verify_scatter_nd([10], [5, 1], [5])


@pytest.mark.parametrize("tensor_shape", [[32, 32]])
@pytest.mark.parametrize("condition_shape", [None, [8], [16]])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_compress(
    tensor_shape: list[int],
    condition_shape: list[int] | None,
    axis: int | None,
):
    if condition_shape is None and axis is None:
        pytest.skip("Either condition_shape or axis must be specified")
    if condition_shape is None:
        condition_shape = [tensor_shape[axis]]
    compress_node = helper.make_node("Compress", ["tensor", "condition"], ["output"], axis=axis)
    graph = helper.make_graph(
        [compress_node],
        "compress_test",
        inputs=[
            helper.make_tensor_value_info("tensor", TensorProto.FLOAT, tensor_shape),
            helper.make_tensor_value_info("condition", TensorProto.BOOL, condition_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [])
        ],  # shape is unknown
    )
    model = helper.make_model(graph, producer_name="compress_test")
    check_correctness(model, opset=11)


def test_size():
    test_node = helper.make_node("Size", ["x"], ["y"])
    graph = helper.make_graph(
        [test_node],
        "size_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 3, 3])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.INT64, [3])],
    )

    model = helper.make_model(graph, producer_name="size_test")
    check_correctness(model)


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_eye_like(k: int):
    verify_unary("EyeLike", [32, 32], attrs={"k": k})


@pytest.mark.parametrize("alpha", [None, 0.25, 1.0])
@pytest.mark.parametrize("beta", [None, 0.35, 1.0])
@pytest.mark.parametrize("useC", [False, True])
def test_gemm(alpha, beta, useC):
    if useC:
        gemm_node = helper.make_node(
            "Gemm", ["a", "b", "c"], ["y"], alpha=alpha, beta=beta, transA=1, transB=1
        )
    else:
        gemm_node = helper.make_node(
            "Gemm", ["a", "b"], ["y"], alpha=alpha, beta=beta, transA=1, transB=1
        )

    inputs = [
        helper.make_tensor_value_info("a", TensorProto.FLOAT, [4, 3]),
        helper.make_tensor_value_info("b", TensorProto.FLOAT, [5, 4]),
    ]
    if useC:
        inputs.append(helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, 5]))

    graph = helper.make_graph(
        [gemm_node],
        "gemm_test",
        inputs=inputs,
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 5])],
    )

    model = helper.make_model(graph, producer_name="gemm_test")
    check_correctness(model)


@pytest.mark.parametrize(
    "in_shape, shape, out_shape",
    [
        ([7, 32, 32, 8], [224, 256], [224, 256]),
        ([7, 32, 32, 8], [-1, 8192], [7, 8192]),
        ([7, 32, 32, 8], [0, 32, 32, 8], [7, 32, 32, 8]),
    ],
)
def test_reshape(in_shape, shape, out_shape):
    reshape_node = helper.make_node("Reshape", ["data", "shape"], ["reshaped"])

    graph = helper.make_graph(
        [reshape_node],
        "reshape_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, in_shape),
        ],
        initializer=[helper.make_tensor("shape", TensorProto.INT64, [len(shape)], shape)],
        outputs=[helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, out_shape)],
    )
    model = helper.make_model(graph, producer_name="reshape_test")
    input_values = {
        "data": np.random.randn(*in_shape).astype("float32"),
    }
    check_correctness(model, inputs=input_values)


@pytest.mark.parametrize(
    "target_shape, output_shape",
    [
        ([-1], [3]),
        ([1, 3], [1, 3]),
        ([3, 1], [3, 1]),
    ],
)
def test_reshape_shape_output(target_shape, output_shape):
    shape_node = helper.make_node("Shape", ["data"], ["shape_out"])
    reshape_node = helper.make_node("Reshape", ["shape_out", "target_shape"], ["reshaped"])

    data_shape = [2, 3, 4]

    graph = helper.make_graph(
        [shape_node, reshape_node],
        "reshape_shape_output",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
        ],
        initializer=[
            helper.make_tensor("target_shape", TensorProto.INT64, [len(target_shape)], target_shape)
        ],
        outputs=[helper.make_tensor_value_info("reshaped", TensorProto.INT64, output_shape)],
    )
    model = helper.make_model(graph, producer_name="reshape_shape_output")
    input_values = {
        "data": np.random.randn(*data_shape).astype("float32"),
    }
    check_correctness(model, inputs=input_values)


def test_transpose():
    verify_unary("Transpose", [32, 32, 32], attrs={"perm": [1, 2, 0]})


def test_transpose_scalar():
    """Test Transpose with scalar inputs - should return scalar unchanged."""
    # Test scalar with no perm attribute (default behavior)
    scalar_node = helper.make_node("Transpose", ["x"], ["y"])
    graph = helper.make_graph(
        [scalar_node],
        "transpose_scalar_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(graph, producer_name="transpose_scalar_test")
    check_correctness(model)

    # Test with scalar constant and transpose without perm
    scalar_constant = helper.make_node(
        "Constant",
        [],
        ["scalar"],
        value=helper.make_tensor("value", TensorProto.FLOAT, [], [5.0]),
    )

    transpose_node = helper.make_node("Transpose", ["scalar"], ["y"])
    graph = helper.make_graph(
        [scalar_constant, transpose_node],
        "transpose_scalar_constant_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(graph, producer_name="transpose_scalar_constant_test")
    check_correctness(model)


def test_transpose_axes_validation():
    """Test Transpose validation - perm axes count must match tensor dimensions"""
    # Test 1D tensor with correct perm
    transpose_1d_valid = helper.make_node("Transpose", ["x"], ["y"], perm=[0])
    graph_1d_valid = helper.make_graph(
        [transpose_1d_valid],
        "transpose_1d_valid_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [10])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [10])],
    )
    model_1d_valid = helper.make_model(graph_1d_valid, producer_name="transpose_1d_valid_test")
    check_correctness(model_1d_valid)

    # Test 2D tensor with correct perm
    transpose_2d_valid = helper.make_node("Transpose", ["x"], ["y"], perm=[1, 0])
    graph_2d_valid = helper.make_graph(
        [transpose_2d_valid],
        "transpose_2d_valid_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 3])],
    )
    model_2d_valid = helper.make_model(graph_2d_valid, producer_name="transpose_2d_valid_test")
    check_correctness(model_2d_valid)

    # Test 3D tensor with correct perm
    transpose_3d_valid = helper.make_node("Transpose", ["x"], ["y"], perm=[2, 0, 1])
    graph_3d_valid = helper.make_graph(
        [transpose_3d_valid],
        "transpose_3d_valid_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 2, 3])],
    )
    model_3d_valid = helper.make_model(graph_3d_valid, producer_name="transpose_3d_valid_test")
    check_correctness(model_3d_valid)


def test_unsqueeze():
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32])],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [3], vals=[0, 2, 3])],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 1, 1, 32])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_test")
    check_correctness(model)


def test_unsqueeze_scalar_input():
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_scalar_input",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [])],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [2], vals=[0, 1])],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 1])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_scalar_input_test")
    inputs = {"a": np.array(3.0, dtype="float32")}
    check_correctness(model, inputs, opset=13)


def test_unsqueeze_dynamic_axes():
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_dynamic_axes",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 32, 1])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_dynamic_axes_test")
    inputs = {
        "a": rg.standard_normal(size=[32, 32]).astype("float32"),
        "axes": np.array([-1, 0], dtype="int64"),
    }
    check_correctness(model, inputs, opset=13)


def test_unsqueeze_dynamic_axes_ir():
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_dynamic_axes_ir",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 32, 1])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_dynamic_axes_ir_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)
    call_ops = collect_relax_call_ops(tvm_model["main"])

    assert "relax.tensor_to_shape" in call_ops
    assert "relax.reshape" in call_ops


def test_unsqueeze_dynamic_axes_rank_validation():
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_dynamic_axes_rank_validation",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [1, 2]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 32, 1])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_dynamic_axes_rank_validation_test")
    with pytest.raises(ValueError, match="Expected a 1-D tensor"):
        from_onnx(model, opset=13, keep_params_in_input=True)


def test_unsqueeze_duplicate_axes_validation():
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_duplicate_axes_validation",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32])],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [2], vals=[0, 0])],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 1, 32, 32])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_duplicate_axes_validation_test")
    with pytest.raises(ValueError, match="axes must be unique"):
        from_onnx(model, opset=13)


def test_unsqueeze_v1():
    # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Unsqueeze-1
    unsqueeze_node = helper.make_node("Unsqueeze", ["a"], ["b"], axes=[0, 2, 3])
    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_v1",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32])],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 1, 1, 32])],
    )

    model = helper.make_model(
        graph, producer_name="unsqueeze_v1_test", opset_imports=[helper.make_opsetid("", 6)]
    )
    check_correctness(model, opset=10)


def test_gelu():
    verify_unary("Gelu", [32, 32], domain="com.microsoft")


def test_gelu_approximate():
    """Test Gelu with approximate attribute from ONNX Opset 20."""
    # Test Gelu with approximate="tanh"
    verify_unary("Gelu", [32, 32], attrs={"approximate": "tanh"}, opset=20)
    # Test Gelu with approximate="none" (default, same as standard Gelu)
    verify_unary("Gelu", [32, 32], attrs={"approximate": "none"}, opset=20)


def test_bias_gelu():
    verify_binary("BiasGelu", [32, 32], [32], [32, 32], domain="com.microsoft")


def test_fast_gelu():
    """Test FastGelu with and without bias"""
    # Test FastGelu without bias
    fast_gelu_node = helper.make_node("FastGelu", ["x"], ["y"], domain="com.microsoft")
    graph = helper.make_graph(
        [fast_gelu_node],
        "fast_gelu_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [32, 32])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="fast_gelu_test")
    check_correctness(model)

    # Test FastGelu with bias
    fast_gelu_with_bias_node = helper.make_node(
        "FastGelu", ["x", "bias"], ["y"], domain="com.microsoft"
    )
    graph_with_bias = helper.make_graph(
        [fast_gelu_with_bias_node],
        "fast_gelu_with_bias_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [32]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )
    model_with_bias = helper.make_model(graph_with_bias, producer_name="fast_gelu_with_bias_test")
    check_correctness(model_with_bias)


def test_where():
    where_node = helper.make_node("Where", ["a", "b", "c"], ["d"])

    graph = helper.make_graph(
        [where_node],
        "where_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.BOOL, [32, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [32, 32]),
        ],
        outputs=[helper.make_tensor_value_info("d", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="where_test")
    check_correctness(model)


@pytest.mark.parametrize("min", [True, False])
@pytest.mark.parametrize("max", [True, False])
def test_clip(min, max):
    if min and max:
        clip_node = helper.make_node("Clip", ["input", "min", "max"], ["output"])
    elif min:
        clip_node = helper.make_node("Clip", ["input", "min"], ["output"])
    elif max:
        clip_node = helper.make_node("Clip", ["input", "max"], ["output"])
    else:
        clip_node = helper.make_node("Clip", ["input"], ["output"])

    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 64])]
    if min:
        inputs.append(helper.make_tensor_value_info("min", TensorProto.FLOAT, ()))
    if max:
        inputs.append(helper.make_tensor_value_info("max", TensorProto.FLOAT, ()))

    graph = helper.make_graph(
        [clip_node],
        "clip_test",
        inputs=inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 64])],
    )

    model = helper.make_model(graph, producer_name="clip_test")
    check_correctness(model)


@pytest.mark.parametrize("min", [-6.0, 0.0])
@pytest.mark.parametrize("max", [6.0])
def test_clip_v6(max, min):
    # https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Clip-6
    clip_node = helper.make_node("Clip", ["input"], ["output"], max=max, min=min)
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 64])]
    graph = helper.make_graph(
        [clip_node],
        "clip_v6_test",
        inputs=inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 64])],
    )
    model = helper.make_model(
        graph, producer_name="clip_v6_test", opset_imports=[helper.make_opsetid("", 6)]
    )
    check_correctness(model, opset=10)


@pytest.mark.parametrize(
    "min,max",
    [
        pytest.param(
            np.array(0.0, dtype=np.float32),
            np.array(6.0, dtype=np.float32),
        ),
        pytest.param(
            np.array(0.0, dtype=np.float32),
            np.array(np.nan, dtype=np.float32),
        ),
        pytest.param(
            np.array(np.nan, dtype=np.float32),
            np.array(6.0, dtype=np.float32),
        ),
        pytest.param(
            np.array(np.nan, dtype=np.float32),
            np.array(np.nan, dtype=np.float32),
        ),
    ],
)
@pytest.mark.parametrize(
    "input",
    [
        np.array([0.5, -3.0, 4.5, 11.0, 7.0], dtype=np.float32),
    ],
)
def test_clip_v13(input, min, max):
    # Opset 13 accepts min/max as tensor inputs. NaN tensor bounds are treated
    # as unbounded by the ONNX frontend before lowering to max/min.
    clip_node = helper.make_node("Clip", ["input", "min", "max"], ["output"])
    graph = helper.make_graph(
        [clip_node],
        "clip_v13_tensor_bounds_import",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [5]),
            helper.make_tensor_value_info("min", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("max", TensorProto.FLOAT, []),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [5])],
    )
    model = helper.make_model(graph, producer_name="clip_v13_tensor_bounds_import")
    check_correctness(
        model,
        inputs={"input": input, "min": min, "max": max},
        opset=13,
    )


def test_equal():
    equal_node = helper.make_node("Equal", ["a", "b"], ["output"])

    graph = helper.make_graph(
        [equal_node],
        "equal_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 32]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="equal_test")
    check_correctness(
        model, {"a": np.zeros([32, 32], dtype="float32"), "b": np.zeros([32, 32], dtype="float32")}
    )
    check_correctness(
        model, {"a": np.ones([32, 32], dtype="float32"), "b": np.zeros([32, 32], dtype="float32")}
    )


def test_shape():
    shape_node = helper.make_node("Shape", ["data"], ["output"])

    graph = helper.make_graph(
        [shape_node],
        "shape_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4, 5, 6]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT64, [4])],
    )

    model = helper.make_model(graph, producer_name="shape_test")
    check_correctness(model)


@pytest.mark.parametrize("upper", [True, False])
def test_trilu(upper: bool):
    verify_unary("Trilu", [3, 5, 5], attrs={"upper": upper})


@pytest.mark.parametrize("k_value", [-1, 0, 1])
def test_trilu_with_const_k(k_value: int):
    """test_trilu_with_const_k"""

    input_shape = [2, 3, 3]

    graph = helper.make_graph(
        [
            make_constant_node("k", onnx.TensorProto.INT64, [1], [k_value]),
            helper.make_node("Trilu", inputs=["x", "k"], outputs=["y"]),
        ],
        "trilu_graph",
        inputs=[
            helper.make_tensor_value_info("x", onnx.TensorProto.DOUBLE, input_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", onnx.TensorProto.DOUBLE, input_shape)],
    )

    model = helper.make_model(graph, producer_name="trilu_graph")
    check_correctness(model)


def test_selu():
    verify_unary("Selu", [3, 32, 32])
    verify_unary("Selu", [3, 32, 32], attrs={"alpha": 0.25, "gamma": 0.3})


def test_mish():
    verify_unary("Mish", [3, 32, 32], opset=18)


def test_prelu():
    verify_binary("PRelu", [3, 32, 32], [1], [3, 32, 32])
    verify_binary("PRelu", [3, 32, 32], [1, 1], [3, 32, 32])
    verify_binary("PRelu", [3, 32, 32], [32], [3, 32, 32])
    verify_binary("PRelu", [3, 32, 32], [3, 1, 1], [3, 32, 32])


def test_thresholded_relu():
    verify_unary("ThresholdedRelu", [3, 32, 32])
    verify_unary("ThresholdedRelu", [3, 32, 32], attrs={"alpha": -0.01})


def test_leakyrelu():
    verify_unary("LeakyRelu", [32, 32])
    verify_unary("LeakyRelu", [32, 32], attrs={"alpha": 0.2})


def test_hardsigmoid():
    verify_unary("HardSigmoid", [32, 32])
    verify_unary("HardSigmoid", [32, 32], attrs={"alpha": 0.3, "beta": 0.4})
    verify_unary("HardSigmoid", [1, 3, 20, 20], attrs={"alpha": 0.5, "beta": 0.6})


def test_shrink():
    verify_unary("Shrink", [32, 32])
    verify_unary("Shrink", [32, 32], attrs={"lambd": 0.2, "bias": 0.1})


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("pad", [0, 2])
@pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER", "VALID"])
def test_conv(stride: int, dilation: int, pad: int, bias: bool, auto_pad: str):
    def _verify_conv(input_shape, weight_shape):
        nd = len(weight_shape) - 2
        if auto_pad == "VALID":
            output_shape = [input_shape[0], weight_shape[0]] + [
                (input_shape[i] - dilation * (weight_shape[i] - 1) - 1) // stride + 1
                for i in range(2, len(input_shape))
            ]
            bias_shape = [output_shape[1]]
            conv_node = helper.make_node(
                "Conv",
                inputs=["x", "w"] + (["b"] if bias else []),
                outputs=["y"],
                strides=[stride] * nd,
                dilations=[dilation] * nd,
                auto_pad=auto_pad,
                group=input_shape[1] // weight_shape[1],
            )
        elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            if dilation == 2:
                # auto_pad = "SAME" and dilation = 2 is not supported in ONNX
                return
            output_shape = [input_shape[0], weight_shape[0]] + [
                (input_shape[i] + stride - 1) // stride for i in range(2, len(input_shape))
            ]
            bias_shape = [output_shape[1]]
            conv_node = helper.make_node(
                "Conv",
                inputs=["x", "w"] + (["b"] if bias else []),
                outputs=["y"],
                strides=[stride] * nd,
                dilations=[dilation] * nd,
                auto_pad=auto_pad,
                group=input_shape[1] // weight_shape[1],
            )
        else:
            output_shape = [input_shape[0], weight_shape[0]] + [
                (input_shape[i] + 2 * pad - dilation * (weight_shape[i] - 1) - 1) // stride + 1
                for i in range(2, len(input_shape))
            ]
            bias_shape = [output_shape[1]]
            conv_node = helper.make_node(
                "Conv",
                inputs=["x", "w"] + (["b"] if bias else []),
                outputs=["y"],
                strides=[stride] * nd,
                dilations=[dilation] * nd,
                pads=[pad] * nd * 2,
                group=input_shape[1] // weight_shape[1],
            )
        graph = helper.make_graph(
            [conv_node],
            "conv_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("w", TensorProto.FLOAT, weight_shape),
            ]
            + ([helper.make_tensor_value_info("b", TensorProto.FLOAT, bias_shape)] if bias else []),
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )

        model = helper.make_model(graph, producer_name="conv_test")
        check_correctness(model, atol=1e-4)

    # Conv1D
    _verify_conv([3, 4, 32], [4, 4, 3])
    _verify_conv([3, 4, 32], [2, 4, 3])  # group=2
    # Conv2D
    _verify_conv([3, 4, 32, 32], [4, 4, 3, 3])
    _verify_conv([3, 4, 32, 32], [2, 4, 3, 3])  # group=2
    # Conv3D
    _verify_conv([3, 4, 32, 32, 32], [4, 4, 3, 3, 3])
    _verify_conv([3, 4, 32, 32, 32], [2, 4, 3, 3, 3])  # group=2


@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("dilation", [1])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("pad", [0, 2])
@pytest.mark.parametrize("output_pad", [0, 1])
def test_conv_transpose(stride: int, dilation: int, pad: int, bias: bool, output_pad: int):
    def _verify_conv_transpose(input_shape, weight_shape):
        nd = len(weight_shape) - 2
        output_shape = [input_shape[0], weight_shape[0]] + [
            (input_shape[i] - 1) * stride
            - 2 * pad
            + dilation * (weight_shape[i] - 1)
            + output_pad
            + 1
            for i in range(2, len(input_shape))
        ]
        bias_shape = [output_shape[1]]
        conv_node = helper.make_node(
            "ConvTranspose",
            inputs=["x", "w"] + (["b"] if bias else []),
            outputs=["y"],
            strides=[stride] * nd,
            dilations=[dilation] * nd,
            pads=[pad] * nd * 2,
            output_padding=[output_pad] * nd,
            group=input_shape[1] // weight_shape[1],
        )
        graph = helper.make_graph(
            [conv_node],
            "conv_transpose_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("w", TensorProto.FLOAT, weight_shape),
            ]
            + ([helper.make_tensor_value_info("b", TensorProto.FLOAT, bias_shape)] if bias else []),
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )

        model = helper.make_model(graph, producer_name="conv_transpose_test")
        check_correctness(model, atol=1e-4)

    # ConvTranspose1D
    _verify_conv_transpose([3, 4, 32], [4, 4, 3])
    _verify_conv_transpose([3, 4, 32], [4, 2, 3])  # group=2
    # ConvTranspose2D
    _verify_conv_transpose([3, 4, 32, 32], [4, 4, 3, 3])
    _verify_conv_transpose([3, 4, 32, 32], [4, 2, 3, 3])  # group=2
    # ConvTranspose3D
    _verify_conv_transpose([3, 4, 12, 12, 12], [4, 4, 3, 3, 3])
    _verify_conv_transpose([3, 4, 12, 12, 12], [4, 2, 3, 3, 3])  # group=2


@pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER", "VALID"])
@pytest.mark.parametrize("stride", [1, 2])
def test_conv_transpose_auto_pad(auto_pad: str, stride: int):
    def _verify(input_shape, weight_shape):
        nd = len(weight_shape) - 2
        conv_node = helper.make_node(
            "ConvTranspose",
            inputs=["x", "w"],
            outputs=["y"],
            kernel_shape=weight_shape[2:],
            strides=[stride] * nd,
            auto_pad=auto_pad,
        )
        graph = helper.make_graph(
            [conv_node],
            "conv_transpose_auto_pad_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("w", TensorProto.FLOAT, weight_shape),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        )
        model = helper.make_model(graph, producer_name="conv_transpose_auto_pad_test")
        check_correctness(model, atol=1e-4)

    # ConvTranspose1D / 2D / 3D
    _verify([1, 1, 8], [1, 1, 3])
    _verify([1, 1, 8, 8], [1, 1, 3, 3])
    _verify([1, 1, 4, 4, 4], [1, 1, 3, 3, 3])


def test_pow():
    verify_binary("Pow", [32, 32], [32, 32], [32, 32])


@pytest.mark.parametrize("reverse", [True, False])
@pytest.mark.parametrize("exclusive", [True, False])
def test_cumsum(reverse, exclusive):
    cumsum_node = helper.make_node(
        "CumSum", ["x", "axis"], ["y"], reverse=reverse, exclusive=exclusive
    )
    shape = [32, 32]
    graph = helper.make_graph(
        [cumsum_node],
        "cumsum_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=[helper.make_tensor("axis", TensorProto.INT64, (), [1])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(graph, producer_name="cumsum_test")
    check_correctness(model)


def test_cumsum1():
    """test_cumsum1"""

    input_shape = [2, 3]

    graph = helper.make_graph(
        [
            helper.make_node("CumSum", inputs=["X", "axis"], outputs=["Y"]),
        ],
        "cumsum_graph",
        inputs=[
            helper.make_tensor_value_info("X", onnx.TensorProto.DOUBLE, input_shape),
        ],
        initializer=[helper.make_tensor("axis", onnx.TensorProto.INT32, [1], [0])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.DOUBLE, input_shape)],
    )

    model = helper.make_model(graph, producer_name="cumsum_graph")
    check_correctness(model)


def test_cumsum_dynamic_axis_not_supported():
    input_shape = [2, 3]

    graph = helper.make_graph(
        [
            helper.make_node("CumSum", inputs=["X", "axis"], outputs=["Y"]),
        ],
        "cumsum_dynamic_axis_graph",
        inputs=[
            helper.make_tensor_value_info("X", onnx.TensorProto.DOUBLE, input_shape),
            helper.make_tensor_value_info("axis", onnx.TensorProto.INT32, [1], "axis"),
        ],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.DOUBLE, input_shape)],
    )

    model = helper.make_model(graph, producer_name="cumsum_dynamic_axis_graph")
    with pytest.raises(ValueError, match="non-constant axis input is not supported"):
        from_onnx(model, opset=14, keep_params_in_input=True)


def test_cumsum_axis_shape_validation():
    input_shape = [2, 3]

    graph = helper.make_graph(
        [
            helper.make_node("CumSum", inputs=["X", "axis"], outputs=["Y"]),
        ],
        "cumsum_invalid_axis_shape_graph",
        inputs=[
            helper.make_tensor_value_info("X", onnx.TensorProto.DOUBLE, input_shape),
        ],
        initializer=[helper.make_tensor("axis", onnx.TensorProto.INT64, [2], [0, 1])],
        outputs=[helper.make_tensor_value_info("Y", onnx.TensorProto.DOUBLE, input_shape)],
    )

    model = helper.make_model(graph, producer_name="cumsum_invalid_axis_shape_graph")
    with pytest.raises(
        ValueError,
        match=r"axis input must be a scalar \(0-D\) or a single-element 1-D tensor",
    ):
        from_onnx(model, opset=14, keep_params_in_input=True)


@pytest.mark.parametrize("axis", [[0, 2], None])
def test_squeeze(axis):
    if axis:
        squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    else:
        squeeze_node = helper.make_node("Squeeze", ["x"], ["y"])
    shape = [1, 32, 1, 32]

    initializer = (
        [helper.make_tensor("axes", TensorProto.INT64, [len(axis)], axis)] if axis else None
    )

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=initializer,
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="squeeze_test")
    check_correctness(model, opset=13)


@pytest.mark.parametrize("axis", [[0, 2], None])
def test_squeeze_constant(axis):
    shape = [1, 32, 1, 32]
    constant = make_constant_node(
        "x", onnx.TensorProto.FLOAT, shape, rg.standard_normal(size=shape).astype("float32")
    )
    if axis:
        squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    else:
        squeeze_node = helper.make_node("Squeeze", ["x"], ["y"])

    initializer = (
        [helper.make_tensor("axes", TensorProto.INT64, [len(axis)], axis)] if axis else None
    )

    graph = helper.make_graph(
        [constant, squeeze_node],
        "squeeze_test",
        inputs=[],
        initializer=initializer,
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="squeeze_test")
    check_correctness(model, opset=13)


@pytest.mark.parametrize("axis", [[0]])
@pytest.mark.parametrize("A", [8, 16, 32])
@pytest.mark.parametrize("B", [8, 16, 32])
def test_dynamic_squeeze(axis, A, B):
    squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    shape = [1, "A", "B"]

    initializer = (
        [helper.make_tensor("axes", TensorProto.INT64, [len(axis)], axis)] if axis else None
    )

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=initializer,
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, ["A", "B"])],
    )

    model = helper.make_model(graph, producer_name="squeeze_test")
    inputs = {"x": rg.standard_normal(size=[1, A, B]).astype("float32")}
    check_correctness(model, inputs, opset=13)


def test_squeeze_dynamic_axes():
    squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    shape = [1, 32, 1, 32]

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_dynamic_axes_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="squeeze_dynamic_axes_test")
    inputs = {
        "x": rg.standard_normal(size=shape).astype("float32"),
        "axes": np.array([-4, 2], dtype="int64"),
    }
    check_correctness(model, inputs, opset=13)


def test_squeeze_dynamic_axes_ir():
    squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    shape = [1, 32, 1, 32]

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_dynamic_axes_ir",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="squeeze_dynamic_axes_ir_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)
    call_ops = collect_relax_call_ops(tvm_model["main"])

    assert "relax.tensor_to_shape" in call_ops
    assert "relax.reshape" in call_ops
    assert "relax.squeeze" not in call_ops


def test_squeeze_dynamic_axes_rank_validation():
    squeeze_node = helper.make_node("Squeeze", ["x", "axes"], ["y"])
    shape = [1, 32, 1, 32]

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_dynamic_axes_rank_validation",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [1, 2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="squeeze_dynamic_axes_rank_validation_test")
    with pytest.raises(ValueError, match="Expected a 1-D tensor"):
        from_onnx(model, opset=13, keep_params_in_input=True)


@pytest.mark.parametrize("axis", [[0]])
@pytest.mark.parametrize("A", [8, 16, 32])
def test_dynamic_shape_squeeze(axis, A):
    shape_node = helper.make_node("Shape", ["x"], ["y"])
    squeeze_node = helper.make_node("Squeeze", ["y", "axes"], ["z"])
    shape = ["A"]

    initializer = (
        [helper.make_tensor("axes", TensorProto.INT64, [len(axis)], axis)] if axis else None
    )

    graph = helper.make_graph(
        [shape_node, squeeze_node],
        "squeeze_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=initializer,
        outputs=[helper.make_tensor_value_info("z", TensorProto.INT64, [])],
    )

    model = helper.make_model(graph, producer_name="squeeze_test")
    inputs = {"x": rg.standard_normal(size=[A]).astype("float32")}
    check_correctness(model, inputs, opset=13)


def test_const():
    shape = [32, 32]
    const_node = helper.make_node(
        "Constant",
        [],
        ["y"],
        value=helper.make_tensor(
            "value", TensorProto.FLOAT, shape, np.random.rand(*shape).astype(np.float32).flatten()
        ),
    )
    graph = helper.make_graph(
        [const_node],
        "const_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(graph, producer_name="const_test")
    tvm_model = check_import(model)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, shape, "float32")
    assert collect_relax_call_ops(tvm_model["main"]) == []


def test_instance_norm():
    verify_ternary(
        "InstanceNormalization", [1, 3, 32, 32], [3], [3], [1, 3, 32, 32], attrs={"epsilon": 1e-12}
    )
    verify_ternary(
        "InstanceNormalization", [1, 32, 32], [32], [32], [1, 32, 32], attrs={"epsilon": 1e-12}
    )


def test_mean_variance_norm():
    verify_unary("MeanVarianceNormalization", [1, 3, 32, 32])
    verify_unary("MeanVarianceNormalization", [1, 3, 32, 32], attrs={"axes": (1, 2, 3)})


def test_layer_norm():
    def _assert_layer_norm_import(
        model,
        expected_shape,
        expected_dtype: str | None = "float32",
        opset=14,
        expected_bias_dtype: str | None = None,
        check_numeric: bool = True,
        rtol: float = 1e-7,
        atol: float = 1e-5,
    ):
        if check_numeric:
            check_correctness(model, opset=opset, rtol=rtol, atol=atol)
            return

        tvm_model = check_import(model, opset=opset)
        assert_tensor_sinfo(tvm_model["main"].ret_ty, expected_shape, expected_dtype)
        assert_has_relax_ops(tvm_model["main"], "relax.nn.layer_norm")
        if expected_bias_dtype is not None:
            [layer_norm_call] = collect_relax_calls(tvm_model["main"], "relax.nn.layer_norm")
            bias = layer_norm_call.args[2]
            assert isinstance(bias, relax.Constant)
            assert str(bias.data.numpy().dtype) == expected_bias_dtype

    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale", "bias"], ["Y"], epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [32]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    _assert_layer_norm_import(model, [32, 32])

    # Test case with no bias that is an optional input
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    _assert_layer_norm_import(model, [32, 32], expected_bias_dtype="float32")

    # No bias with a non-square input where data.shape[1] differs from the scale
    # shape, see https://github.com/apache/tvm/issues/19691.
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], axis=-1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 3, 4, 8]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [8]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 3, 4, 8]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    _assert_layer_norm_import(model, [2, 3, 4, 8], expected_bias_dtype="float32")

    # No bias with a non-square fp16 input. The synthesized zero bias must match
    # the scale dtype, otherwise layer_norm rejects the float32 bias, see
    # https://github.com/apache/tvm/issues/19691.
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], axis=-1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT16, [2, 3, 4, 8]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT16, [8]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 3, 4, 8]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    _assert_layer_norm_import(
        model,
        [2, 3, 4, 8],
        "float16",
        opset=17,
        expected_bias_dtype="float16",
        rtol=1e-2,
        atol=1e-2,
    )

    # Same no-bias path for bf16. The importer currently represents ONNX bf16
    # tensors as float32 Relax tensors, so the synthesized zero bias must also
    # be float32.
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], axis=-1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [2, 3, 4, 8]),
            helper.make_tensor_value_info("scale", TensorProto.BFLOAT16, [8]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.BFLOAT16, [2, 3, 4, 8]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    _assert_layer_norm_import(
        model,
        [2, 3, 4, 8],
        "float32",
        opset=17,
        expected_bias_dtype="float32",
        check_numeric=False,
    )


def test_layer_norm_with_nd_gamma_beta():
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale", "bias"], ["Y"], axis=1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_with_nd_gamma_beta_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [3, 4, 4]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3, 4, 4]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 4, 4]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_with_nd_gamma_beta_test")
    check_correctness(model)

    # Test case with no bias that is an optional input
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale"], ["Y"], axis=1, epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_with_nd_gamma_beta_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [32, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_with_nd_gamma_beta_test")
    check_correctness(model)


def test_layer_norm_numerical_stability():
    """Numerical stability test for https://github.com/apache/tvm/issues/19592."""
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["input", "scale", "bias"], ["Y"], axis=-1, epsilon=1e-5
    )
    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_large_values_import",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [4]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4]),
        ],
    )
    model = helper.make_model(graph, producer_name="layer_norm_numerical_stability")

    input_array = np.array([[80000.0, 80001.0, 80002.0, 80003.0]], dtype=np.float32)
    scale_array = np.ones(4, dtype=np.float32)
    bias_array = np.zeros(4, dtype=np.float32)
    inputs = {"input": input_array, "scale": scale_array, "bias": bias_array}

    mean = input_array.mean(axis=-1, keepdims=True)
    var = ((input_array - mean) ** 2).mean(axis=-1, keepdims=True)
    expected = ((input_array - mean) / np.sqrt(var + 1e-5) * scale_array + bias_array).astype(
        np.float32
    )

    tvm_output = run_in_tvm(model, inputs=inputs, ir_version=9, opset=17)
    assert np.isfinite(tvm_output.numpy()).all()
    tvm.testing.assert_allclose(tvm_output.numpy(), expected)

    tvm_model = check_import(model, ir_version=9, opset=17)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, [1, 4], "float32")
    assert_has_relax_ops(tvm_model["main"], "relax.nn.layer_norm")


def test_rms_norm():
    def _check_rms_norm(
        model,
        input_array,
        scale_array,
        axes,
        epsilon,
        output_dtype,
        rtol=1e-5,
        atol=1e-5,
    ):
        inputs = {"input": input_array, "scale": scale_array}
        data_compute = (
            input_array.astype("float32") if input_array.dtype != np.float32 else input_array
        )
        scale_compute = (
            scale_array.astype("float32") if input_array.dtype != np.float32 else scale_array
        )
        mean_square = np.mean(np.square(data_compute), axis=tuple(axes), keepdims=True)
        expected = data_compute / np.sqrt(mean_square + epsilon) * scale_compute
        expected = expected.astype(output_dtype)

        tvm_output = run_in_tvm(model, inputs=inputs, opset=23)
        tvm.testing.assert_allclose(tvm_output.numpy(), expected, rtol=rtol, atol=atol)

        tvm_model = check_import(model, opset=23)
        assert_has_relax_ops(tvm_model["main"], "relax.nn.rms_norm")
        return tvm_model

    # Basic test: default axis=-1
    rms_norm_node = helper.make_node("RMSNormalization", ["input", "scale"], ["Y"], epsilon=1e-05)

    graph = helper.make_graph(
        [rms_norm_node],
        "rms_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 8, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2, 8, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="rms_norm_test")
    input_array = rg.standard_normal(size=[2, 8, 32]).astype("float32")
    scale_array = rg.standard_normal(size=[32]).astype("float32")
    _check_rms_norm(
        model, input_array, scale_array, axes=[-1], epsilon=1e-5, output_dtype="float32"
    )

    # Test with explicit axis=1 (normalize over last 2 dims)
    rms_norm_node = helper.make_node(
        "RMSNormalization", ["input", "scale"], ["Y"], axis=1, epsilon=1e-06
    )

    graph = helper.make_graph(
        [rms_norm_node],
        "rms_norm_axis_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 8, 16]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT, [8, 16]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4, 8, 16]),
        ],
    )

    model = helper.make_model(graph, producer_name="rms_norm_axis_test")
    input_array = rg.standard_normal(size=[4, 8, 16]).astype("float32")
    scale_array = rg.standard_normal(size=[8, 16]).astype("float32")
    _check_rms_norm(
        model, input_array, scale_array, axes=[1, 2], epsilon=1e-6, output_dtype="float32"
    )

    # Test with float16 input (stash_type=1 means compute in float32)
    rms_norm_node = helper.make_node(
        "RMSNormalization", ["input", "scale"], ["Y"], epsilon=1e-05, stash_type=1
    )

    graph = helper.make_graph(
        [rms_norm_node],
        "rms_norm_fp16_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT16, [2, 8, 32]),
            helper.make_tensor_value_info("scale", TensorProto.FLOAT16, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [2, 8, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="rms_norm_fp16_test")
    input_array = rg.standard_normal(size=[2, 8, 32]).astype("float16")
    scale_array = rg.standard_normal(size=[32]).astype("float16")
    _check_rms_norm(
        model,
        input_array,
        scale_array,
        axes=[-1],
        epsilon=1e-5,
        output_dtype="float16",
        rtol=1e-2,
        atol=1e-2,
    )


# TODO Enable dynamism
@pytest.mark.parametrize("dynamic", [False])
def test_skiplayernormalization(dynamic):
    def verify_skiplayernormalization(input_shape, skip_shape, gamma_shape, beta_shape, bias_shape):
        node = onnx.helper.make_node(
            "SkipLayerNormalization",
            inputs=["input", "skip", "gamma", "beta", "bias"],
            outputs=["output", "mean", "std_dev"],
            domain="com.microsoft",
        )

        node.attribute.append(onnx.helper.make_attribute("epsilon", 1e-4))

        input_shape = list(input_shape)
        skip_shape = list(skip_shape)
        gamma_shape = list(gamma_shape)
        beta_shape = list(beta_shape)
        bias_shape = list(bias_shape)
        output_shape = list(input_shape)
        mean_shape = list([1])
        std_dev_shape = list([1])
        if dynamic:
            input_shape = ["?" for _ in input_shape]
            skip_shape = ["?" for _ in skip_shape]
            gamma_shape = ["?" for _ in gamma_shape]
            beta_shape = ["?" for _ in beta_shape]
            bias_shape = ["?" for _ in bias_shape]
            output_shape = ["?" for _ in output_shape]

        graph = helper.make_graph(
            [node],
            "skiplayernormalization_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("skip", TensorProto.FLOAT, skip_shape),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, gamma_shape),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, beta_shape),
                helper.make_tensor_value_info("bias", TensorProto.FLOAT, bias_shape),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, mean_shape),
                helper.make_tensor_value_info("std_dev", TensorProto.FLOAT, std_dev_shape),
            ],
        )

        model = helper.make_model(graph, producer_name="skiplayernormalization_test")
        check_correctness(model, rtol=1e-4, atol=1e-4)

    hidden_size = 384
    batch_size = 4
    sequence_length = 4

    verify_skiplayernormalization(
        (batch_size, sequence_length, hidden_size),
        (batch_size, sequence_length, hidden_size),
        (hidden_size,),
        (hidden_size,),
        (hidden_size,),
    )


def test_embedlayernormalization():
    def verify_embedlayernormalization(
        input_ids_shape,
        has_segment_ids,
        word_embedding_shape,
        position_embedding_shape,
        has_segment_embedding,
        gamma_shape,
        beta_shape,
    ):
        node = onnx.helper.make_node(
            "EmbedLayerNormalization",
            inputs=[
                "input_ids",
                "segment_ids" if has_segment_ids else "",
                "word_embedding",
                "position_embedding",
                "segment_embedding" if has_segment_embedding else "",
                "gamma",
                "beta",
            ],
            outputs=["output", "mask_index"],
            domain="com.microsoft",
        )

        node.attribute.append(onnx.helper.make_attribute("epsilon", 1e-4))

        segment_ids_shape = input_ids_shape if has_segment_ids else []
        segment_embedding_shape = word_embedding_shape if has_segment_embedding else []

        graph = helper.make_graph(
            [node],
            "embedlayernormalization_test",
            inputs=[
                helper.make_tensor_value_info(
                    "input_ids", TensorProto.INT32, list(input_ids_shape)
                ),
                helper.make_tensor_value_info("segment_ids", TensorProto.INT32, segment_ids_shape),
                helper.make_tensor_value_info(
                    "word_embedding", TensorProto.FLOAT, list(word_embedding_shape)
                ),
                helper.make_tensor_value_info(
                    "position_embedding", TensorProto.FLOAT, list(position_embedding_shape)
                ),
                helper.make_tensor_value_info(
                    "segment_embedding", TensorProto.FLOAT, segment_embedding_shape
                ),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, list(gamma_shape)),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, list(beta_shape)),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, list((batch_size, sequence_length, hidden_size))
                ),
                helper.make_tensor_value_info("mask_index", TensorProto.INT32, [batch_size]),
            ],
        )

        model = helper.make_model(graph, producer_name="embedlayernormalization_test")
        input_ids = (
            np.arange(np.prod(input_ids_shape), dtype="int32").reshape(input_ids_shape)
            % word_embedding_shape[0]
        )
        inputs = {"input_ids": input_ids}
        if has_segment_ids:
            segment_vocab_size = segment_embedding_shape[0] if segment_embedding_shape else 1
            inputs["segment_ids"] = (
                np.arange(np.prod(input_ids_shape), dtype="int32").reshape(input_ids_shape)
                % segment_vocab_size
            )
        check_correctness(model, inputs=inputs, rtol=1e-4, atol=1e-4)

    hidden_size = 384
    batch_size = 4
    sequence_length = 3
    vocab_size = 5

    verify_embedlayernormalization(
        (batch_size, sequence_length),
        True,
        (vocab_size, hidden_size),
        (sequence_length, hidden_size),
        True,
        (hidden_size,),
        (hidden_size,),
    )

    # Test with undefined segment embedding
    verify_embedlayernormalization(
        (batch_size, sequence_length),
        False,
        (vocab_size, hidden_size),
        (sequence_length, hidden_size),
        False,
        (hidden_size,),
        (hidden_size,),
    )


def test_local_response_norm():
    lrn_node = helper.make_node(
        op_type="LRN",
        inputs=["input"],
        outputs=["output"],
        name="LRN_Node",
        alpha=0.0001,
        beta=0.75,
        bias=1.0,
        size=3,
    )

    graph = helper.make_graph(
        [lrn_node],
        "local_response_norm_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32]),
        ],
    )

    model = helper.make_model(graph, producer_name="local_response_norm_test")
    check_correctness(model, rtol=1e-4, atol=1e-4)


def create_reduce_test_parameters_axes_attr():
    output = []
    for value in [True, False]:
        output.append(("ReduceMax", value, 11))
        output.append(("ReduceMean", value, 13))
        output.append(("ReduceMin", value, 11))
        output.append(("ReduceProd", value, 13))
        output.append(("ReduceSum", value, 11))
        output.append(("ReduceSumSquare", value, 13))
        output.append(("ReduceLogSum", value, 13))
        output.append(("ReduceLogSumExp", value, 13))
        output.append(("ReduceL1", value, 13))
        output.append(("ReduceL2", value, 13))
        # Opset 11-12 axes-as-attr: verifies get_converter does not
        # underflow to the v18 (axes-as-input) implementation.
        output.append(("ReduceMean", value, 11))
        output.append(("ReduceProd", value, 11))
        output.append(("ReduceSumSquare", value, 11))
        output.append(("ReduceLogSum", value, 11))
        output.append(("ReduceLogSumExp", value, 11))
        output.append(("ReduceL1", value, 11))
        output.append(("ReduceL2", value, 11))
    return output


@pytest.mark.parametrize("func, dynamic, opset", create_reduce_test_parameters_axes_attr())
def test_all_reduce_funcs_axes_attr(func, dynamic, opset):
    def verify_reduce_func(func, data, axis, keepdims):
        inshape = data.shape
        outshape = np.sum(data, axis=axis, keepdims=keepdims == 1).shape

        if axis:
            node = onnx.helper.make_node(
                func, inputs=["x"], outputs=["y"], axes=axis, keepdims=keepdims
            )
        else:
            node = onnx.helper.make_node(func, inputs=["x"], outputs=["y"], keepdims=keepdims)

        if dynamic:
            in_list = ["?" for _ in range(len(inshape))]
            out_list = ["?" for _ in range(len(outshape))]
        else:
            in_list = list(inshape)
            out_list = list(outshape)
        graph = helper.make_graph(
            [node],
            "reduce_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, in_list)],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_list)],
        )

        model = helper.make_model(graph, producer_name="reduce_test")
        inputs_dict = {"x": data}
        check_correctness(model, inputs_dict, opset=opset, rtol=1e-4, atol=1e-4)

    for keepdims in [True, False]:
        verify_reduce_func(
            func, np.random.randn(3, 2, 2).astype(np.float32), axis=None, keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(3, 2, 3).astype(np.float32), axis=None, keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(3, 3, 3).astype(np.float32), axis=(1,), keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(3, 3, 3, 1).astype(np.float32), axis=(1, 2), keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(3, 3, 3, 1).astype(np.float32), axis=(1,), keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(1, 3, 4, 1).astype(np.float32), axis=(1,), keepdims=keepdims
        )


def create_reduce_test_parameters_axes_input():
    output = []
    for dynamic in [True, False]:
        output.append(("ReduceMax", dynamic, 18))
        output.append(("ReduceMean", dynamic, 18))
        output.append(("ReduceMin", dynamic, 18))
        output.append(("ReduceProd", dynamic, 18))
        output.append(("ReduceSum", dynamic, 13))
        output.append(("ReduceSumSquare", dynamic, 18))
        output.append(("ReduceLogSum", dynamic, 18))
        output.append(("ReduceLogSumExp", dynamic, 18))
        output.append(("ReduceL1", dynamic, 18))
        output.append(("ReduceL2", dynamic, 18))
    return output


@pytest.mark.parametrize("func, dynamic, opset", create_reduce_test_parameters_axes_input())
def test_all_reduce_funcs_axes_input(func, dynamic, opset):
    def verify_reduce_func(
        func, data, axes, keepdims, noop_with_empty_axes=False, check_numeric=True
    ):
        inshape = data.shape

        inputs = ["x"]
        initializers = []

        # Optional `axes` input
        if axes is not None:
            axes_name = "reduce_axes"
            axes_np = np.asarray(axes, dtype=np.int64)
            axes_init = helper.make_tensor(
                name=axes_name,
                data_type=TensorProto.INT64,
                dims=axes_np.shape,
                vals=axes_np,
            )
            initializers.append(axes_init)
            inputs.append(axes_name)

        # Determine input and output shapes
        if not axes and not noop_with_empty_axes:
            outshape = np.sum(data, axis=None, keepdims=keepdims).shape
        elif not axes and noop_with_empty_axes:
            outshape = inshape
        else:
            outshape = np.sum(data, axis=axes, keepdims=keepdims).shape

        if dynamic:
            in_list = ["?"] * len(inshape)
            out_list = ["?"] * len(outshape)
        else:
            in_list = list(inshape)
            out_list = list(outshape)

        # Make a model node
        node = helper.make_node(
            func,
            inputs=inputs,
            outputs=["y"],
            keepdims=keepdims,
            noop_with_empty_axes=noop_with_empty_axes,
        )

        # Make a model graph and a model
        graph = helper.make_graph(
            [node],
            "reduce18_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, in_list)],
            initializer=initializers,
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_list)],
        )
        model = helper.make_model(graph, producer_name="reduce18_test")

        if check_numeric:
            check_correctness(model, {"x": data}, opset=opset, rtol=1e-4, atol=1e-4)
        else:
            tvm_model = check_import(model, opset=opset)
            if not (noop_with_empty_axes and not axes):
                assert_has_relax_ops(tvm_model["main"], ONNX_RELAX_OPS.get(func))

    # Verify
    for keepdims in [True, False]:
        # no `axes` input && `noop_with_empty_axes` = 0 -> reduce over all dimensions.
        verify_reduce_func(
            func,
            np.random.randn(3, 2, 2).astype(np.float32),
            axes=[],
            keepdims=keepdims,
            noop_with_empty_axes=False,
        )

        # no `axes` input && `noop_with_empty_axes` = 0 -> reduce over all dimensions.
        verify_reduce_func(
            func,
            np.random.randn(3, 2, 2).astype(np.float32),
            axes=None,
            keepdims=keepdims,
            noop_with_empty_axes=False,
        )

        # no `axes` input && `noop_with_empty_axes` = 1 -> return the input unchanged.
        verify_reduce_func(
            func,
            np.random.randn(4, 3).astype(np.float32),
            axes=[],
            keepdims=keepdims,
            noop_with_empty_axes=True,
        )

        # no `axes` input && `noop_with_empty_axes` = 1 -> return the input unchanged.
        # (onnxruntime bug) Runtime error on the onnxruntime part
        verify_reduce_func(
            func,
            np.random.randn(4, 3).astype(np.float32),
            axes=None,
            keepdims=keepdims,
            noop_with_empty_axes=True,
            check_numeric=False,
        )

        # `axes` provided -> reduce over specified axes.
        verify_reduce_func(
            func,
            np.random.randn(3, 3, 3, 1).astype(np.float32),
            axes=(1, 2),
            keepdims=keepdims,
        )


@pytest.mark.parametrize("in_dtype", [np.float32, np.int32])
@pytest.mark.parametrize("axis", [None, 0, 1, 2])
@pytest.mark.parametrize("keepdims", [None, True, False])
def test_arg_min_max(in_dtype, axis, keepdims):
    def verify_arg_min_max(input_dim, in_dtype, op_name="ArgMax", axis=None, keepdims=None):
        a_np1 = np.random.uniform(-10, 10, input_dim).astype(in_dtype)
        out_shape = list(a_np1.shape)
        def_axis = axis if axis is not None else 0
        if keepdims == 1 or keepdims is None:
            out_shape[def_axis] = 1
        else:
            out_shape.pop(def_axis)

        node = helper.make_node(op_name, inputs=["a_np1"], outputs=["out"])

        if keepdims is not None:
            keepdims_attr = helper.make_attribute("keepdims", keepdims)
            node.attribute.append(keepdims_attr)
        if axis is not None:
            axis_attr = helper.make_attribute("axis", axis)
            node.attribute.append(axis_attr)

        graph = helper.make_graph(
            [node],
            "argreduce_test",
            inputs=[
                helper.make_tensor_value_info(
                    "a_np1", helper.np_dtype_to_tensor_dtype(np.dtype(in_dtype)), list(a_np1.shape)
                )
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.INT64, list(out_shape))],
        )

        model = helper.make_model(graph, producer_name="arg_min_max_test")
        check_correctness(model, inputs={"a_np1": a_np1})

    verify_arg_min_max([3, 4, 4], in_dtype, "ArgMax", axis, keepdims)
    verify_arg_min_max([3, 4, 4], in_dtype, "ArgMin", axis, keepdims)


@pytest.mark.parametrize("axis", [-1, 0, 1])
@pytest.mark.parametrize("largest", [True, False])
def test_topk(axis: int, largest: int):
    in_shape = [32, 32, 32]
    k_value = 4
    out_shape = in_shape
    out_shape[axis] = k_value
    k = make_constant_node("k", TensorProto.INT64, [1], [k_value])
    node = onnx.helper.make_node(
        "TopK",
        inputs=["data", "k"],
        outputs=["values", "indices"],
        axis=axis,
        largest=largest,
    )
    graph = helper.make_graph(
        [k, node],
        "topk_test",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, in_shape)],
        outputs=[
            helper.make_tensor_value_info("values", TensorProto.FLOAT, out_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT64, out_shape),
        ],
    )
    model = helper.make_model(graph, producer_name="topk_test")

    check_correctness(model)


@pytest.mark.parametrize("dynamic", [False, True])
def test_expand(dynamic):
    def _test_expand(name, data, shape, ref_data):
        shape_array = np.array(shape)
        shape_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["shape"],
            value=onnx.helper.make_tensor(
                name="const_tensor",
                data_type=onnx.TensorProto.INT64,
                dims=shape_array.shape,
                vals=shape_array.flatten().astype("int64"),
            ),
        )
        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])

        in_shape = list(data.shape)
        out_shape = list(ref_data.shape)
        if dynamic:
            in_shape = ["?" for _ in range(len(in_shape))]
            out_shape = ["?" for _ in range(len(out_shape))]
        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_teint64st",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, in_shape)],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)],
        )

        model = helper.make_model(graph, producer_name=name)
        check_correctness(model, inputs={"in": data})

    def _test_expand_dynamic_shapeexpr(name, data, shape_data, shape, ref_data):
        shape_node = onnx.helper.make_node("Shape", inputs=["in_2"], outputs=["shape"])
        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])
        in_shape = list(data.shape)
        out_shape = list(ref_data.shape)
        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_test",
            inputs=[
                helper.make_tensor_value_info("in", TensorProto.FLOAT, in_shape),
                helper.make_tensor_value_info("in_2", TensorProto.FLOAT, shape),
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)],
        )

        model = helper.make_model(graph, producer_name=name)
        check_correctness(model, inputs={"in": data, "in_2": shape_data})

    if not dynamic:
        in_shape = (3, 1)
        shape = (3, 4)
        data = np.random.uniform(size=in_shape).astype(np.float32)
        ref_data = np.tile(data, 4)
        _test_expand("expand_with_dim_unchanged_test", data, shape, ref_data)

        in_shape = (3, 1)
        shape = (1, 3, 4)
        data = np.random.uniform(size=in_shape).astype(np.float32)
        ref_data = np.tile(data, (1, 1, 4))
        _test_expand("expand_with_diff_dim", data, shape, ref_data)

        in_shape = (3, 1)
        shape = (1, 1, 3, 1)
        data = np.random.uniform(size=in_shape).astype(np.float32)
        ref_data = np.tile(data, (1, 1, 1, 1))
        _test_expand("expand_with_the_same_suffix_dims", data, shape, ref_data)
    else:
        in_shape = (1, 32, 32)
        shape = ("batch", 32, 32)
        data = np.random.uniform(size=in_shape).astype(np.float32)
        shape_data = np.random.uniform(size=(64, 32, 32)).astype(np.float32)
        ref_data = np.tile(data, (64, 1, 1))
        _test_expand_dynamic_shapeexpr("expand_with_dynamic_dim", data, shape_data, shape, ref_data)


def test_expand_incompatible_broadcasting():
    """
    This test case reproduces the error where input tensor shape at dim 1 is 25
    and target shape at dim 3 is 56, which violates ONNX broadcasting rules
    """

    def _test_expand_error_case(name, data_shape, target_shape_vals):
        shape_array = np.array(target_shape_vals, dtype=np.int64)
        shape_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["shape"],
            value=onnx.helper.make_tensor(
                name="const_tensor",
                data_type=onnx.TensorProto.INT64,
                dims=shape_array.shape,
                vals=shape_array.flatten(),
            ),
        )

        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])

        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_error_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(data_shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, target_shape_vals)],
        )

        model = helper.make_model(graph, producer_name=name)

        with pytest.raises(ValueError) as exc_info:
            from_onnx(model, keep_params_in_input=True)

        error_msg = str(exc_info.value)
        assert "broadcast" in error_msg.lower() or "incompatible" in error_msg.lower(), (
            f"Expected broadcasting error, but got: {error_msg}"
        )

    # Test case 1: Reproduce the exact error from the issue-17769
    # Input shape: (25,), target shape: (1, 1, 1, 56)
    # This should faill because input dim 1 (25) != target dim 3 (56) and neither is 1
    _test_expand_error_case(
        "expand_incompatible_25_to_56",
        data_shape=(25,),
        target_shape_vals=(1, 1, 1, 56),
    )

    # Test case 2: Another incompatible case
    # Input shape: (1, 25), target shape: (1, 1, 1, 56)
    # After right-alignment, input (1, 1, 1, 25) vs. target (1, 1, 1, 56)
    # This should fail because 25 != 56 and neither is 1
    _test_expand_error_case(
        "expand_incompatible_aligned_25_to_56",
        data_shape=(1, 25),
        target_shape_vals=(1, 1, 1, 56),
    )

    # Test case 3: Valid case for comparison - should not raise error
    def _test_expand_valid_case():
        """Test a valid expand case to ensure our fix doesn't break valid operations"""
        data_shape = (1, 25)
        target_shape_vals = [2, 25]  # Valid: input (1, 25) can broadcast to (2, 25)

        shape_array = np.array(target_shape_vals, dtype=np.int64)

        shape_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["shape"],
            value=onnx.helper.make_tensor(
                name="const_tensor",
                data_type=onnx.TensorProto.INT64,
                dims=shape_array.shape,
                vals=shape_array.flatten(),
            ),
        )

        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])

        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_valid_test",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(data_shape))],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, target_shape_vals)],
        )

        model = helper.make_model(graph, producer_name="expand_valid_test_case")

        try:
            from_onnx(model, keep_params_in_input=True)
        except Exception as e:
            pytest.fail(f"Valid expand case should not fail, but got error: {e}")

    _test_expand_valid_case()


# TODO(jwfromm) Current approach to dynamic expand is technically not well formed. Reenable once fixed.
@pytest.mark.skip("Produces ill-formed IR")
def test_constantofshape():
    def verify_constantofshape(input_dim, value, dtype):
        fill_node = helper.make_node(
            "ConstantOfShape",
            ["input"],
            ["output"],
            value=helper.make_tensor(
                "value", helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), (1,), (value,)
            ),
        )

        inputs = [helper.make_tensor_value_info("input", TensorProto.INT64, [len(input_dim)])]

        graph = helper.make_graph(
            [fill_node],
            "fill_test",
            inputs,
            initializer=[
                helper.make_tensor(
                    "input",
                    TensorProto.INT64,
                    [len(input_dim)],
                    np.asarray(input_dim).astype("int64"),
                )
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", helper.np_dtype_to_tensor_dtype(np.dtype(dtype)), input_dim
                )
            ],
        )

        model = helper.make_model(graph, producer_name="fill_test")
        tvm_model = check_import(model)
        assert_tensor_sinfo(tvm_model["main"].ret_ty, input_dim, dtype)

    verify_constantofshape((2, 3, 4, 5), 10, "float32")
    verify_constantofshape((3, 3), 0, "int32")
    verify_constantofshape((1, 2, 3), -1, "float32")


def test_constantofshape_default_value():
    # Per ONNX spec, the `value` attribute is optional and defaults to a zero
    # float32 scalar of the requested shape.
    shape_init = helper.make_tensor("shape", TensorProto.INT64, [2], [2, 3])
    node = helper.make_node("ConstantOfShape", ["shape"], ["y"])
    graph = helper.make_graph(
        [node],
        "constantofshape_default_value_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializer=[shape_init],
    )
    model = helper.make_model(graph, producer_name="constantofshape_default_value_test")

    tvm_model = from_onnx(model)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, [2, 3], "float32")
    assert "relax.broadcast_to" in collect_relax_call_ops(tvm_model["main"])
    assert 0.0 in collect_scalar_constants(tvm_model["main"])


def test_slice():
    def verify_slice(data_shape, output_shape, starts, ends, axes=None, steps=None):
        if isinstance(starts, list):
            starts = np.array(starts, "int64")
        if isinstance(ends, list):
            ends = np.array(ends, "int64")
        if isinstance(axes, list):
            axes = np.array(axes, "int64")
        if isinstance(steps, list):
            steps = np.array(steps, "int64")

        slice_inputs = ["x", "starts", "ends"]
        initializer = [
            helper.make_tensor("starts", TensorProto.INT64, starts.shape, starts),
            helper.make_tensor("ends", TensorProto.INT64, ends.shape, ends),
        ]

        if axes is not None:
            initializer.append(helper.make_tensor("axes", TensorProto.INT64, axes.shape, axes))
            slice_inputs.append("axes")
        if steps is not None:
            initializer.append(helper.make_tensor("steps", TensorProto.INT64, steps.shape, steps))
            slice_inputs.append("steps")

        slice_node = helper.make_node("Slice", inputs=slice_inputs, outputs=["y"])

        graph = helper.make_graph(
            [slice_node],
            "slice_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, data_shape),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            initializer=initializer,
        )

        model = helper.make_model(graph, producer_name="slice_test")
        check_correctness(model)

    # Test with all parameters set.
    verify_slice([20, 10, 5], [3, 10, 5], starts=[0, 0], ends=[3, 10], axes=[0, 1], steps=[1, 1])
    # Test with default axes and steps.
    verify_slice([20, 10, 5], [3, 10, 5], starts=[0, 0], ends=[3, 10])
    # Test with negative steps.
    verify_slice(
        [20, 10, 5],
        [19, 3, 2],
        starts=[20, 10, 4],  # NOTE: the start is out of bounds
        ends=[0, 0, 1],
        steps=[-1, -3, -2],
        axes=[0, 1, 2],
    )
    verify_slice([20, 10, 5], [10, 5], starts=[0, 0], ends=[3, 10], axes=[1, 2])
    verify_slice([20, 10, 5], [10, 5], starts=[0, 0], ends=[3, 10], axes=[1, 2])

    # TODO (gigiblender): Enable this test when we have a way to pass the steps but not axes.
    # verify_slice(
    #     [20, 10, 5],
    #     [19, 3, 2],
    #     starts=[20, 10, 4],
    #     ends=[0, 0, 1],
    #     steps=[-1, -3, -2],
    # )


def test_slice_dynamic_inputs():
    slice_node = helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"])

    graph = helper.make_graph(
        [slice_node],
        "slice_dynamic_inputs_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [20, 10, 5]),
            helper.make_tensor_value_info("starts", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("ends", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("steps", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 10, 5])],
    )

    model = helper.make_model(graph, producer_name="slice_dynamic_inputs_test")
    inputs = {
        "x": rg.standard_normal(size=[20, 10, 5]).astype("float32"),
        "starts": np.array([0, 0], dtype="int64"),
        "ends": np.array([3, 10], dtype="int64"),
        "axes": np.array([0, 1], dtype="int64"),
        "steps": np.array([1, 1], dtype="int64"),
    }
    check_correctness(model, inputs, opset=13)


def test_slice_dynamic_inputs_ir():
    slice_node = helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"])

    graph = helper.make_graph(
        [slice_node],
        "slice_dynamic_inputs_ir",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [20, 10, 5]),
            helper.make_tensor_value_info("starts", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("ends", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("steps", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 10, 5])],
    )

    model = helper.make_model(graph, producer_name="slice_dynamic_inputs_ir_test")
    tvm_model = from_onnx(model, opset=13, keep_params_in_input=True)
    call_ops = collect_relax_call_ops(tvm_model["main"])

    assert "relax.dynamic_strided_slice" in call_ops
    assert "relax.strided_slice" not in call_ops


def test_slice_dynamic_inputs_length_validation():
    slice_node = helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"])

    graph = helper.make_graph(
        [slice_node],
        "slice_dynamic_inputs_length_validation",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [20, 10, 5]),
            helper.make_tensor_value_info("starts", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("ends", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [2]),
            helper.make_tensor_value_info("steps", TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 10, 5])],
    )

    model = helper.make_model(graph, producer_name="slice_dynamic_inputs_length_validation_test")
    with pytest.raises(ValueError, match="starts and ends to have the same length"):
        from_onnx(model, opset=13, keep_params_in_input=True)


def test_slice_dynamic_shape_expr_input_validation():
    shape_node = helper.make_node("Shape", ["x"], ["y"])
    slice_node = helper.make_node("Slice", ["y", "starts", "ends", "axes", "steps"], ["z"])

    graph = helper.make_graph(
        [shape_node, slice_node],
        "slice_dynamic_shape_expr_input_validation",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [20, 10, 5]),
            helper.make_tensor_value_info("starts", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("ends", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("axes", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("steps", TensorProto.INT64, [1]),
        ],
        outputs=[helper.make_tensor_value_info("z", TensorProto.INT64, [1])],
    )

    model = helper.make_model(graph, producer_name="slice_dynamic_shape_expr_input_validation_test")
    with pytest.raises(ValueError, match="does not support ShapeExpr input"):
        from_onnx(model, opset=13, keep_params_in_input=True)


def test_slice_zero_step_validation():
    slice_node = helper.make_node("Slice", ["x", "starts", "ends", "axes", "steps"], ["y"])

    graph = helper.make_graph(
        [slice_node],
        "slice_zero_step_validation",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [20, 10, 5])],
        initializer=[
            helper.make_tensor("starts", TensorProto.INT64, [2], vals=[0, 0]),
            helper.make_tensor("ends", TensorProto.INT64, [2], vals=[3, 10]),
            helper.make_tensor("axes", TensorProto.INT64, [2], vals=[0, 1]),
            helper.make_tensor("steps", TensorProto.INT64, [2], vals=[1, 0]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 10, 5])],
    )

    model = helper.make_model(graph, producer_name="slice_zero_step_validation_test")
    with pytest.raises(ValueError, match="step values must be non-zero"):
        from_onnx(model, opset=13)


def test_slice_dynamic_shape():
    def verify_slice(
        data_shape, data_instance_shape, output_shape, starts, ends, axes=None, steps=None
    ):
        if isinstance(starts, list):
            starts = np.array(starts, "int64")
        if isinstance(ends, list):
            ends = np.array(ends, "int64")
        if isinstance(axes, list):
            axes = np.array(axes, "int64")
        if isinstance(steps, list):
            steps = np.array(steps, "int64")

        slice_inputs = ["y", "starts", "ends"]
        initializer = [
            helper.make_tensor("starts", TensorProto.INT64, starts.shape, starts),
            helper.make_tensor("ends", TensorProto.INT64, ends.shape, ends),
        ]

        if axes is not None:
            initializer.append(helper.make_tensor("axes", TensorProto.INT64, axes.shape, axes))
            slice_inputs.append("axes")
        if steps is not None:
            initializer.append(helper.make_tensor("steps", TensorProto.INT64, steps.shape, steps))
            slice_inputs.append("steps")

        shape_node = helper.make_node("Shape", inputs=["x"], outputs=["y"])
        slice_node = helper.make_node("Slice", inputs=slice_inputs, outputs=["z"])

        graph = helper.make_graph(
            [shape_node, slice_node],
            "slice_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, data_shape),
            ],
            outputs=[helper.make_tensor_value_info("z", TensorProto.INT64, output_shape)],
            initializer=initializer,
        )

        model = helper.make_model(graph, producer_name="slice_test")
        inputs = {"x": rg.standard_normal(size=data_instance_shape).astype("float32")}
        check_correctness(model, inputs)

    verify_slice([20, 10, 5], [20, 10, 5], [2], starts=[0], ends=[2], axes=[0])
    verify_slice(["A", 10, 5], [20, 10, 5], [2], starts=[0], ends=[2], axes=[0])
    verify_slice(["A", "B", 5], [20, 10, 5], [2], starts=[0], ends=[2], axes=[0])
    verify_slice([20, 10, "C"], [20, 10, 5], [2], starts=[0], ends=[2], axes=[0])
    verify_slice(["A", "B", "C"], [20, 10, 5], [2], starts=[0], ends=[2], axes=[0])

    verify_slice([20, 10, 5], [20, 10, 5], [1], starts=[1], ends=[2], axes=[0])
    verify_slice(["A", 10, 5], [20, 10, 5], [1], starts=[1], ends=[2], axes=[0])
    verify_slice(["A", "B", 5], [20, 10, 5], [1], starts=[1], ends=[2], axes=[0])
    verify_slice([20, 10, "C"], [20, 10, 5], [1], starts=[1], ends=[2], axes=[0])
    verify_slice(["A", "B", "C"], [20, 10, 5], [1], starts=[1], ends=[2], axes=[0])

    verify_slice([20, 10, 5], [20, 10, 5], [2], starts=[1], ends=[3], axes=[0])
    verify_slice(["A", 10, 5], [20, 10, 5], [2], starts=[1], ends=[3], axes=[0])
    verify_slice(["A", "B", 5], [20, 10, 5], [2], starts=[1], ends=[3], axes=[0])
    verify_slice([20, 10, "C"], [20, 10, 5], [2], starts=[1], ends=[3], axes=[0])
    verify_slice(["A", "B", "C"], [20, 10, 5], [2], starts=[1], ends=[3], axes=[0])


# TODO Enable dynamism
@pytest.mark.parametrize("dynamic", [False])
def test_attention(dynamic):
    def verify_attention(
        input_shape,
        weight_shape,
        bias_shape,
        mask_shape,
        num_heads,
        qkv_hidden_sizes,
        relative_position_bias_shape,
    ):
        node = onnx.helper.make_node(
            "Attention",
            inputs=["input", "weight", "bias", "mask_index", "", "relative_position_bias"],
            outputs=["output"],
            domain="com.microsoft",
            num_heads=num_heads,
            # TODO(jwfromm): Enable this attribute after importer support is clarified.
            # mask_filter_value=mask_filter_value,
            qkv_hidden_sizes=qkv_hidden_sizes,
        )

        input_shape = list(input_shape)
        weight_shape = list(weight_shape)
        bias_shape = list(bias_shape)
        mask_shape = list(mask_shape)
        relative_position_bias_shape = list(relative_position_bias_shape)
        output_shape = [
            input_shape[0],
            input_shape[1],
            qkv_hidden_sizes[2] if qkv_hidden_sizes is not None else input_shape[2],
        ]
        if dynamic:
            input_shape = ["?" for _ in input_shape]
            weight_shape = ["?" for _ in weight_shape]
            bias_shape = ["?" for _ in bias_shape]
            mask_shape = ["?" for _ in mask_shape]
            output_shape = ["?" for _ in output_shape]

        graph = helper.make_graph(
            [node],
            "attention_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("weight", TensorProto.FLOAT, weight_shape),
                helper.make_tensor_value_info("bias", TensorProto.FLOAT, bias_shape),
                helper.make_tensor_value_info("mask_index", TensorProto.INT32, mask_shape),
                helper.make_tensor_value_info(
                    "relative_position_bias", TensorProto.FLOAT, relative_position_bias_shape
                ),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape),
            ],
        )

        model = helper.make_model(graph, producer_name="attention_test")
        inputs = {
            "input": rg.standard_normal(size=input_shape).astype("float32"),
            "weight": rg.standard_normal(size=weight_shape).astype("float32"),
            "bias": rg.standard_normal(size=bias_shape).astype("float32"),
            "mask_index": np.ones(mask_shape, dtype="int32"),
            "relative_position_bias": rg.standard_normal(size=relative_position_bias_shape).astype(
                "float32"
            ),
        }
        check_correctness(model, inputs=inputs, rtol=1e-3, atol=1e-3)

    input_hidden_size = 128
    batch_size = 4
    sequence_length = 4
    num_heads = 12
    qkv_hidden_sizes = [192, 192, 96]

    verify_attention(
        (batch_size, sequence_length, input_hidden_size),
        (input_hidden_size, sum(qkv_hidden_sizes)),
        (sum(qkv_hidden_sizes),),
        (batch_size, sequence_length),
        num_heads,
        qkv_hidden_sizes,
        (batch_size, num_heads, sequence_length, sequence_length),
    )


@pytest.mark.parametrize("dynamic", [True, False])
def test_pad(dynamic):
    if dynamic:
        pytest.skip("Dynamic pad not supported")

    def verify_pad(input_shape, pads, mode="constant", value=0.0):
        len_dim = len(pads) // 2
        output_shape = [
            input_shape[i] + pads[i] + pads[i + len_dim] for i in range(len(input_shape))
        ]
        pads = np.array(pads)
        if mode in ["edge", "reflect"]:
            node = helper.make_node("Pad", inputs=["input", "pads"], outputs=["output"], mode=mode)
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
                ],
                initializer=[helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads)],
                outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
            )
        else:
            node = helper.make_node(
                "Pad",
                inputs=["input", "pads", "constant_value"],
                outputs=["output"],
                mode="constant",
            )
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
                ],
                initializer=[
                    helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads),
                    helper.make_tensor("constant_value", TensorProto.FLOAT, (1,), [value]),
                ],
                outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
            )
        model = helper.make_model(graph, producer_name="pad_test")
        tvm_model = check_import(model)
        assert_has_relax_ops(tvm_model["main"], ONNX_RELAX_OPS["Pad"])

    verify_pad((2, 2), [0, 1, 0, 0], "constant", 0.0)
    verify_pad((2, 3), [1, 0, 0, 1], "constant", 0.0)
    verify_pad((3, 2), [0, 0, 1, 0], "constant", 5.0)
    verify_pad((1, 3, 4, 5), [0, 1, 1, 1, 0, 0, 1, 1], "reflect")
    verify_pad((2, 3), [1, 1, 1, 1], "edge")
    verify_pad((1, 3, 4, 5), [0, 1, 1, 1, 0, 0, 1, 1], "edge")


@pytest.mark.parametrize("dynamic", [True, False])
def test_pad_v2(dynamic):
    if dynamic:
        pytest.skip("Dynamic pad not supported")

    def verify_pad(input_shape, pads, mode="constant", value=0.0):
        len_dim = len(pads) // 2
        output_shape = [
            input_shape[i] + pads[i] + pads[i + len_dim] for i in range(len(input_shape))
        ]
        pads = np.array(pads)
        if mode in ["edge", "reflect"]:
            node = helper.make_node(
                "Pad", inputs=["input"], outputs=["output"], mode=mode, pads=pads
            )
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
                ],
                outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
            )
        else:
            node = helper.make_node(
                "Pad",
                inputs=["input"],
                outputs=["output"],
                mode="constant",
                pads=pads,
                value=value,
            )
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(input_shape))
                ],
                outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
            )
        model = helper.make_model(graph, producer_name="pad_test")
        tvm_model = check_import(model=model, opset=10)
        assert_has_relax_ops(tvm_model["main"], ONNX_RELAX_OPS["Pad"])

    verify_pad((2, 2), [0, 1, 0, 0], "constant", 0.0)
    verify_pad((2, 3), [1, 0, 0, 1], "constant", 0.0)
    verify_pad((3, 2), [0, 0, 1, 0], "constant", 5.0)
    verify_pad((1, 3, 4, 5), [0, 1, 1, 1, 0, 0, 1, 1], "reflect")
    verify_pad((2, 3), [1, 1, 1, 1], "edge")
    verify_pad((1, 3, 4, 5), [0, 1, 1, 1, 0, 0, 1, 1], "edge")


@pytest.mark.parametrize("fp_arith", [np.float16, np.float32])
@pytest.mark.parametrize("dynamic", [True, False])
def test_split(fp_arith, dynamic):
    def verify_split(indata_shape, outdata_shapes, split, axis=0, pass_split=True, opset=11):
        input_names = ["input"]
        initializer = []

        if split:
            split_index = range(len(split))
        else:
            split_index = range(len(outdata_shapes))

        indata_shape = [indata_shape] if isinstance(indata_shape, int) else list(indata_shape)
        if dynamic:
            indata_shape = ["?" for _ in indata_shape]
            outdata_shapes = [["?" for _ in range(len(o))] for o in outdata_shapes]

        inputs = [
            helper.make_tensor_value_info(
                "input", helper.np_dtype_to_tensor_dtype(np.dtype(fp_arith)), indata_shape
            )
        ]

        split_constant = None
        if pass_split:
            if opset >= 13:
                np_split = np.array(split).astype(np.int64)
                split_constant = make_constant_node(
                    "split", onnx.TensorProto.INT64, list(np_split.shape), np_split
                )
                input_names.append("split")

        node = helper.make_node(
            "Split",
            inputs=input_names,
            outputs=[f"output_{i}" for i in range(len(split_index))],
            axis=axis,
        )

        if pass_split and opset < 13:
            split_attr = helper.make_attribute("split", split)
            node.attribute.append(split_attr)

        nodes = [split_constant, node] if split_constant else [node]

        graph = helper.make_graph(
            nodes,
            "split_test",
            inputs=inputs,
            initializer=initializer,
            outputs=[
                helper.make_tensor_value_info(
                    f"output_{i}",
                    helper.np_dtype_to_tensor_dtype(np.dtype(fp_arith)),
                    list(outdata_shapes[i]),
                )
                for i in range(len(split_index))
            ],
        )
        model = helper.make_model(graph, producer_name="split_test")
        if dynamic:
            tvm_model = check_import(model, opset=opset)
            assert_has_relax_ops(tvm_model["main"], ONNX_RELAX_OPS["Split"])
        else:
            check_correctness(model, opset=opset)

    # 1D
    verify_split(6, [[2], [2], [2]], [2, 2, 2])
    verify_split(6, [[2], [2], [2]], [2, 2, 2], pass_split=False)
    verify_split(6, [[2], [1], [3]], [2, 1, 3])
    verify_split(6, [[2], [1], [3]], [2, 1, 3], opset=13)
    # 2D
    verify_split(
        (4, 4),
        [[2, 2], [2, 2]],
        [2, 2],
        axis=1,
    )
    verify_split(
        (4, 4),
        [[2, 2], [2, 2]],
        [2, 2],
        axis=1,
        opset=13,
    )
    # Split evenly (unstack)
    verify_split(3, [[1], [1], [1]], False, pass_split=False)
    # Split a single value to a single value
    verify_split(1, [[1]], [1], pass_split=True)
    # Test that the default case modifies nothing when split list has length one
    verify_split((1, 2), [[2]], [2], axis=1)
    verify_split((1, 2), [[2]], [1])


@pytest.mark.parametrize("dynamic", [True, False])
def test_tile(dynamic):
    def verify_tile(in_shape, repeats, out_shape):
        node = helper.make_node("Tile", inputs=["input", "repeats"], outputs=["out"])

        if dynamic:
            in_shape = ["?" for _ in range(len(in_shape))]
            out_shape = ["?" for _ in range(len(out_shape))]

        graph = helper.make_graph(
            [node],
            "tile_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, in_shape),
            ],
            initializer=[
                helper.make_tensor("repeats", TensorProto.INT64, list(repeats.shape), repeats)
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)],
        )

        model = helper.make_model(graph, producer_name="tile_test")

        tvm_model = check_import(model)
        assert_has_relax_ops(tvm_model["main"], ONNX_RELAX_OPS["Tile"])

    in_shape = (2, 3, 4, 5)
    repeats = np.array([2, 3, 1, 4], dtype=np.int64)
    out_shape = tuple(dim * int(repeat) for dim, repeat in zip(in_shape, repeats))
    verify_tile(in_shape, repeats, out_shape)


@pytest.mark.parametrize("dynamic_input", [True, False])
@pytest.mark.parametrize(
    "in_shape,repeats",
    [
        ((2, 3), np.array([2, 2], dtype=np.int64)),
        ((2, 3, 4), np.array([2, 2, 1], dtype=np.int64)),
        ((2, 3, 4, 5), np.array([1, 2, 1, 2], dtype=np.int64)),
    ],
)
def test_tile_dynamic_repeats(dynamic_input, in_shape, repeats):
    out_shape = tuple(dim * int(repeat) for dim, repeat in zip(in_shape, repeats))

    input_shape = ["?" for _ in in_shape] if dynamic_input else list(in_shape)
    output_shape = ["?" for _ in out_shape]

    node = helper.make_node("Tile", inputs=["input", "repeats"], outputs=["out"])
    graph = helper.make_graph(
        [node],
        "tile_dynamic_repeats_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
            helper.make_tensor_value_info("repeats", TensorProto.INT64, [len(repeats)]),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, producer_name="tile_dynamic_repeats_test")

    tvm_model = check_import(model, opset=13)
    assert_has_relax_ops(tvm_model["main"], ONNX_RELAX_OPS["Tile"])


def _generate_roi_cases():
    # Base case when with_roi is False
    roi_list = [
        pytest.param(False, None, False, id="no_roi"),
    ]

    # Valid when with_roi is True and with_constant is True/False
    roi_cases = [
        [],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.1, 0.1, 0.9, 0.9],
        [0.2, 0.2, 0.8, 0.8],
        [0.3, 0.3, 0.7, 0.7],
        [0.4, 0.4, 0.6, 0.6],
        [0.5, 0.5, 0.5, 0.5],
        [0.1, 0.2, 0.9, 0.8],
    ]
    for roi in roi_cases:
        roi_list.append(pytest.param(True, roi, True, id=f"roi_{'_'.join(str(x) for x in roi)}"))
        roi_list.append(pytest.param(True, roi, False, id=f"roi_{'_'.join(str(x) for x in roi)}"))

    return roi_list


@pytest.mark.parametrize("with_roi, roi_list, with_constant", _generate_roi_cases())
def test_resize(with_roi, roi_list, with_constant):
    nodes = []
    resize_node = helper.make_node(
        "Resize", ["X", "roi" if with_roi else "", "scales"], ["Y"], mode="cubic"
    )

    if with_roi and with_constant:
        roi_tensor = helper.make_tensor(
            name="roi",
            data_type=TensorProto.FLOAT,
            dims=[len(roi_list)],
            vals=roi_list,
        )

        roi_const_node = helper.make_node(
            "Constant",
            inputs=[],
            outputs=["roi"],
            value=roi_tensor,
        )
        nodes.append(roi_const_node)

    nodes.append(resize_node)

    initializers = [
        helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]),
    ]

    if with_roi and not with_constant:
        initializers.append(helper.make_tensor("roi", TensorProto.FLOAT, [len(roi_list)], roi_list))

    graph = helper.make_graph(
        nodes,
        "resize_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32]),
        ],
        initializer=initializers,
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 64, 64]),
        ],
    )

    model = helper.make_model(graph, producer_name="resize_test")
    check_correctness(model)


def test_resize_dynamic_roi_tf_crop_and_resize():
    """ROI is a graph input (not initializer), lowered through TOPI dynamic-ROI path."""
    resize_node = helper.make_node(
        "Resize",
        ["X", "roi", "scales"],
        ["Y"],
        mode="linear",
        coordinate_transformation_mode="tf_crop_and_resize",
    )
    graph = helper.make_graph(
        [resize_node],
        "resize_dynamic_roi",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3, 32, 32]),
            helper.make_tensor_value_info("roi", TensorProto.FLOAT, [8]),
        ],
        initializer=[
            helper.make_tensor("scales", TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3, 64, 64]),
        ],
    )
    model = helper.make_model(graph, producer_name="resize_dynamic_roi")
    roi_np = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
    check_correctness(model, atol=1e-5, inputs={"roi": roi_np})


def test_resize_dynamic_roi_3d_tf_crop_and_resize():
    """5-D NCDHW: ROI is a graph input; covers dynamic-ROI TOPI resize3d path."""
    resize_node = helper.make_node(
        "Resize",
        ["X", "roi", "scales"],
        ["Y"],
        mode="linear",
        coordinate_transformation_mode="tf_crop_and_resize",
    )
    graph = helper.make_graph(
        [resize_node],
        "resize_dynamic_roi_3d",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 4, 5]),
            helper.make_tensor_value_info("roi", TensorProto.FLOAT, [10]),
        ],
        initializer=[
            helper.make_tensor("scales", TensorProto.FLOAT, [5], [1.0, 1.0, 2.0, 2.0, 2.0]),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 6, 8, 10]),
        ],
    )
    model = helper.make_model(graph, producer_name="resize_dynamic_roi_3d")
    # Use a valid full-tensor ROI so importer coverage is independent of
    # extrapolation differences in backend implementations.
    x_np = rg.standard_normal((1, 1, 3, 4, 5)).astype(np.float32)
    roi_np = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.float32)
    check_correctness(model, opset=18, atol=1e-5, inputs={"X": x_np, "roi": roi_np})


def test_resize_nd_sizes():
    cases = [
        ("resize1d", [1, 1, 4], [1, 1, 7]),
        ("resize2d", [1, 1, 4, 5], [1, 1, 6, 7]),
        ("resize3d", [1, 1, 3, 4, 5], [1, 1, 4, 6, 7]),
    ]

    for name, input_shape, sizes in cases:
        resize_node = helper.make_node(
            "Resize",
            ["X", "", "", "sizes"],
            ["Y"],
            mode="nearest",
            coordinate_transformation_mode="asymmetric",
            nearest_mode="floor",
        )

        graph = helper.make_graph(
            [resize_node],
            name,
            inputs=[
                helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape),
            ],
            initializer=[
                helper.make_tensor("sizes", TensorProto.INT64, [len(sizes)], sizes),
            ],
            outputs=[
                helper.make_tensor_value_info("Y", TensorProto.FLOAT, sizes),
            ],
        )

        model = helper.make_model(graph, producer_name=name)
        check_correctness(model, opset=18)


def test_resize_5d_emits_relax_resize3d():
    resize_node = helper.make_node(
        "Resize",
        ["X", "", "", "sizes"],
        ["Y"],
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )
    graph = helper.make_graph(
        [resize_node],
        "resize3d_ir_check",
        inputs=[helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 4, 5])],
        initializer=[helper.make_tensor("sizes", TensorProto.INT64, [5], [1, 1, 4, 6, 7])],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 4, 6, 7])],
    )
    model = helper.make_model(graph, producer_name="resize3d_ir_check")
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    seen_resize3d = False

    def _visit(expr):
        nonlocal seen_resize3d
        if isinstance(expr, relax.Call) and isinstance(expr.op, tvm.ir.Op):
            if expr.op.name == "relax.image.resize3d":
                seen_resize3d = True

    relax.analysis.post_order_visit(tvm_model["main"].body, _visit)
    assert seen_resize3d


def test_einsum():
    eqn = "ij->i"
    einsum_node = helper.make_node("Einsum", ["x"], ["y"], equation=eqn)

    graph = helper.make_graph(
        [einsum_node],
        "einsum_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 4]),
        ],
        outputs=[
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [3]),
        ],
    )

    model = helper.make_model(graph, producer_name="einsum_test")
    tvm_model = check_import(model)
    assert_has_relax_ops(tvm_model["main"], "relax.call_tir")


def test_range():
    range_node = helper.make_node(
        "Range",
        ["start", "limit", "delta"],
        ["output"],
    )

    graph = helper.make_graph(
        [range_node],
        "range_test",
        inputs=[],
        initializer=[
            helper.make_tensor("start", TensorProto.INT64, [], [1]),
            helper.make_tensor("limit", TensorProto.INT64, [], [5]),
            helper.make_tensor("delta", TensorProto.INT64, [], [2]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.INT64, [2]),
        ],
    )

    model = helper.make_model(graph, producer_name="range_test")
    tvm_model = check_import(model)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, [2], "int64")
    assert collect_relax_call_ops(tvm_model["main"]) == []


def test_batch_norm():
    batch_norm_node = helper.make_node(
        "BatchNormalization", ["x", "s", "bias", "mean", "var"], ["y"], epsilon=1e-2
    )
    graph = helper.make_graph(
        [batch_norm_node],
        "batch_norm_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5]),
            helper.make_tensor_value_info("s", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("var", TensorProto.FLOAT, [3]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4, 5])],
    )

    model = helper.make_model(graph, producer_name="batch_norm_test")
    check_correctness(model, opset=15, rtol=1e-5, atol=1e-5)


def test_batch_norm_defaults_to_inference_mode():
    batch_norm_node = helper.make_node(
        "BatchNormalization", ["x", "s", "bias", "mean", "var"], ["y"], epsilon=1e-2
    )
    graph = helper.make_graph(
        [batch_norm_node],
        "batch_norm_inference_attr_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5]),
            helper.make_tensor_value_info("s", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("var", TensorProto.FLOAT, [3]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4, 5])],
    )
    model = helper.make_model(graph, producer_name="batch_norm_inference_attr_test")
    model.opset_import[0].version = 15

    tvm_model = from_onnx(model, opset=15, keep_params_in_input=True)
    batch_norm_attrs = []

    def visit(expr):
        if isinstance(expr, relax.Call) and expr.op == tvm.ir.Op.get("relax.nn.batch_norm"):
            batch_norm_attrs.append(expr.attrs)

    relax.analysis.post_order_visit(tvm_model["main"], visit)

    assert len(batch_norm_attrs) == 1
    assert batch_norm_attrs[0].training is False


@pytest.mark.parametrize("pool_name", ["MaxPool", "AveragePool", "LpPool"])
@pytest.mark.parametrize(
    "shape, auto_pad, kernel_shape, strides, pads",
    [
        # Pool1D
        ([1, 1, 32], "NOTSET", [3], [1], [1, 1]),
        # Pool1D with stride
        ([1, 1, 32], "NOTSET", [3], [2], [1, 1]),
        # Pool1D with stride and autopadding
        ([1, 1, 32], "SAME_UPPER", [7], [2], None),
        ([1, 1, 32], "SAME_LOWER", [4], [4], None),
        ([1, 1, 32], "VALID", [5], [5], None),
        ([1, 1, 32], "SAME_UPPER", [3], [1], None),
        # Pool2D
        ([1, 1, 32, 32], "NOTSET", [3, 3], [1, 1], [1, 1, 1, 1]),
        # Pool2D with stride
        ([1, 1, 32, 32], "NOTSET", [3, 3], [2, 2], [1, 1, 1, 1]),
        # Pool2D with stride and autopadding
        ([1, 1, 32, 32], "SAME_UPPER", [3, 7], [3, 2], None),
        ([1, 1, 32, 32], "SAME_LOWER", [3, 3], [2, 2], None),
        ([1, 1, 32, 32], "VALID", [3, 3], [2, 2], None),
        ([1, 1, 32, 32], "SAME_UPPER", [3, 3], [1, 1], None),
        # Pool3D
        ([1, 1, 32, 32, 32], "NOTSET", [3, 3, 4], [1, 1, 1], [1, 2, 1, 1, 2, 2]),
        # Pool3D with stride
        ([1, 1, 32, 32, 32], "NOTSET", [3, 4, 3], [2, 2, 3], [1, 1, 1, 1, 1, 2]),
        # Pool3D with stride and autopadding
        ([1, 1, 32, 32, 32], "SAME_UPPER", [4, 3, 3], [3, 2, 2], None),
        ([1, 1, 32, 32, 32], "SAME_LOWER", [3, 3, 4], [2, 2, 2], None),
        ([1, 1, 32, 32, 32], "VALID", [3, 3, 5], [2, 2, 3], None),
        ([1, 1, 32, 32, 32], "SAME_UPPER", [3, 3, 5], [1, 1, 1], None),
    ],
)
def test_pool(
    pool_name: str,
    shape: list[int],
    auto_pad: str,
    kernel_shape: list[int],
    strides: list[int],
    pads: list[int],
):
    verify_unary(
        pool_name,
        shape,
        attrs={
            "kernel_shape": kernel_shape,
            "strides": strides,
            "pads": pads,
            "auto_pad": auto_pad,
        },
    )


def test_global_average_pool():
    verify_unary("GlobalAveragePool", [1, 3, 32])
    verify_unary("GlobalAveragePool", [1, 3, 32, 32])
    verify_unary("GlobalAveragePool", [1, 3, 32, 32, 32])


def test_global_max_pool():
    verify_unary("GlobalMaxPool", [1, 3, 32])
    verify_unary("GlobalMaxPool", [1, 3, 32, 32])
    verify_unary("GlobalMaxPool", [1, 3, 32, 32, 32])


@pytest.mark.parametrize("p", [1, 2, 3])
def test_global_lp_pool(p: int):
    verify_unary("GlobalLpPool", [1, 3, 32], attrs={"p": p})
    verify_unary("GlobalLpPool", [1, 3, 32, 32], attrs={"p": p})
    verify_unary("GlobalLpPool", [1, 3, 32, 32, 32], attrs={"p": p})


@pytest.mark.parametrize("kernel_shape", [[2, 2], [3, 3]])
@pytest.mark.parametrize("pads", [None, [1, 1, 1, 1]])
@pytest.mark.parametrize("strides", [None, [2, 2]])
def test_maxunpool(kernel_shape, pads, strides):
    input_shape = [16, 3, 16, 16]
    input_names = ["X", "I"]
    input_info = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape),
        helper.make_tensor_value_info("I", TensorProto.INT64, input_shape),
    ]

    attrs = {"kernel_shape": kernel_shape}
    if pads is not None:
        attrs["pads"] = pads
    if strides is not None:
        attrs["strides"] = strides

    node = helper.make_node("MaxUnpool", inputs=input_names, outputs=["y"], **attrs)

    graph = helper.make_graph(
        [node],
        "maxunpool_test",
        inputs=input_info,
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )

    model = helper.make_model(graph, producer_name="maxunpool_test")
    check_correctness(
        model, inputs={"I": np.zeros(input_shape, dtype="int64")}, rtol=1e-4, atol=1e-4
    )


def test_dropout():
    verify_unary("Dropout", [1, 3, 32, 32])
    verify_unary("Dropout", [1, 3, 32, 32], opset=11, attrs={"ratio": 0.5})

    # Opset 12+ passes ratio as an optional input; check it is captured into the relax op.
    node = helper.make_node("Dropout", ["x", "ratio"], ["y"])
    graph = helper.make_graph(
        [node],
        "dropout_ratio_input",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])],
        initializer=[helper.make_tensor("ratio", TensorProto.FLOAT, [], [0.3])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    model = helper.make_model(graph, producer_name="dropout_ratio_input")
    model.opset_import[0].version = 13
    mod = from_onnx(model, opset=13)
    rates = [
        float(b.value.attrs.rate)
        for f in mod.functions.values()
        for block in getattr(f.body, "blocks", [])
        for b in block.bindings
        if getattr(getattr(b.value, "op", None), "name", "") == "relax.nn.dropout"
    ]
    assert rates == pytest.approx([0.3])


def test_flatten():
    verify_unary("Flatten", [1, 3, 32, 32], attrs={"axis": 0})
    verify_unary("Flatten", [1, 3, 32, 32], attrs={"axis": -1})
    verify_unary("Flatten", [1, 3, 32, 32], attrs={"axis": 2})


def test_flatten_dynamic():
    verify_unary_dynamic_shape("Flatten", [1, "A", "B", 32], [1, 3, 32, 32], attrs={"axis": 0})
    verify_unary_dynamic_shape("Flatten", [1, "A", "B", 32], [1, 3, 32, 32], attrs={"axis": -1})
    verify_unary_dynamic_shape("Flatten", [1, "A", "B", 32], [1, 3, 32, 32], attrs={"axis": 2})


def test_onehot():
    one_hot_node = helper.make_node("OneHot", ["indices", "depth", "values"], ["y"], axis=1)
    graph = helper.make_graph(
        [one_hot_node],
        "one_hot_test",
        inputs=[
            helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 2]),
        ],
        initializer=[
            helper.make_tensor("depth", TensorProto.INT64, [], [10]),
            helper.make_tensor("values", TensorProto.FLOAT, [2], [3, 1]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 10, 2])],
    )

    model = helper.make_model(graph, producer_name="one_hot_test")
    check_correctness(
        model,
        inputs={"indices": np.array([[0, 3], [2, 1]], dtype="int64")},
    )


@pytest.mark.parametrize("axis", [None, 0, 1, -1])
@pytest.mark.parametrize("sorted", [0, 1])
@pytest.mark.parametrize("num_outputs", [1, 2, 3, 4])
def test_unique(axis: int | None, sorted: int, num_outputs: int):
    if num_outputs in [3, 4] and axis is None:
        pytest.xfail("RuntimeError: Check failed: input_shape.size() == size (2 vs. 1)")

    input_shape = [8, 8]
    if axis is None:
        output_shape = [-1]
    else:
        output_shape = [8, 8]
        output_shape[axis] = -1

    output_names = ["y", "indices", "inverse_indices", "counts"][:num_outputs]
    unique_node = helper.make_node("Unique", ["x"], output_names, axis=axis, sorted=sorted)

    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)]
    if num_outputs > 1:
        outputs.append(helper.make_tensor_value_info("indices", TensorProto.INT64, [-1]))
    if num_outputs > 2:
        # ONNX spec: inverse_indices is always 1D
        outputs.append(helper.make_tensor_value_info("inverse_indices", TensorProto.INT64, [-1]))
    if num_outputs > 3:
        outputs.append(helper.make_tensor_value_info("counts", TensorProto.INT64, [-1]))

    graph = helper.make_graph(
        [unique_node],
        "unique_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        outputs=outputs,
    )
    model = helper.make_model(graph, producer_name="unique_test")
    check_correctness(model)


@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (4, 5, 6), (7, 8, 9, 10)])
def test_nonzero(shape):
    verify_unary("NonZero", shape, input_dtype=TensorProto.BOOL, output_dtype=TensorProto.INT64)


@pytest.mark.parametrize("mode", ["DCR", "CRD"])
def test_depth_to_space(mode: Literal["DCR", "CRD"]):
    in_shape = [1, 8, 2, 3]
    out_shape = [1, 2, 4, 6]
    blocksize = 2
    node = onnx.helper.make_node(
        "DepthToSpace", inputs=["x"], outputs=["y"], blocksize=blocksize, mode=mode
    )
    graph = helper.make_graph(
        [node],
        "depth_to_space_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
    )
    model = helper.make_model(graph, producer_name="depth_to_space_test")

    check_correctness(model)


def test_space_to_depth():
    in_shape = [1, 2, 4, 6]
    out_shape = [1, 8, 2, 3]
    blocksize = 2
    node = onnx.helper.make_node("SpaceToDepth", inputs=["x"], outputs=["y"], blocksize=blocksize)
    graph = helper.make_graph(
        [node],
        "space_to_depth_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)],
    )
    model = helper.make_model(graph, producer_name="space_to_depth_test")

    check_correctness(model)


def construct_sequence(input_shape: list[int], num_tensors: int, name: str = "sequence"):
    inputs = [f"data{i}" for i in range(num_tensors)]
    sequence_construct_node = helper.make_node("SequenceConstruct", inputs, [name])
    graph_inputs = [
        helper.make_tensor_value_info(f"data{i}", TensorProto.FLOAT, input_shape)
        for i in range(num_tensors)
    ]
    return sequence_construct_node, graph_inputs


def make_constant_node(name: str, data_type: int, dims: list[int], vals: list[int]):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=helper.make_tensor(name=name, data_type=data_type, dims=dims, vals=vals),
    )


def make_optional_tensor_value_info(name: str, elem_type: int, shape: list[int]):
    return helper.make_value_info(
        name, helper.make_optional_type_proto(helper.make_tensor_type_proto(elem_type, shape))
    )


def make_optional_sequence_value_info(name: str, elem_type: int, shape: list[int]):
    return helper.make_value_info(
        name,
        helper.make_optional_type_proto(
            helper.make_sequence_type_proto(helper.make_tensor_type_proto(elem_type, shape))
        ),
    )


def test_sequence_construct():
    node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=2)
    graph = helper.make_graph(
        [node],
        "test_sequence_construct",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_sequence_value_info("sequence", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="test_sequence_construct")
    tvm_model = check_import(model)
    assert_tuple_tensor_sinfo(tvm_model["main"].ret_ty, [([32, 32], "float32")] * 2)
    assert collect_relax_call_ops(tvm_model["main"]) == []


def test_sequence_empty():
    sequence_empty_node = helper.make_node("SequenceEmpty", [], ["sequence"])
    graph = helper.make_graph(
        [sequence_empty_node],
        "test_sequence_empty",
        inputs=[],
        outputs=[helper.make_tensor_sequence_value_info("sequence", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(graph, producer_name="test_sequence_empty")
    tvm_model = check_import(model)
    assert_tuple_tensor_sinfo(tvm_model["main"].ret_ty, [])
    assert collect_relax_call_ops(tvm_model["main"]) == []


@pytest.mark.parametrize("explicit_position", [True, False])
def test_sequence_erase(explicit_position: bool):
    seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=4)
    index = make_constant_node("index", TensorProto.INT64, (), [1])
    node_input = ["sequence", "index"] if explicit_position else ["sequence"]
    sequence_erase_node = helper.make_node("SequenceErase", node_input, ["output"])
    graph = helper.make_graph(
        [index, seq_node, sequence_erase_node],
        "test_sequence_erase",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="test_sequence_erase")
    tvm_model = check_import(model)
    assert_tuple_tensor_sinfo(tvm_model["main"].ret_ty, [([32, 32], "float32")] * 3)
    assert collect_relax_call_ops(tvm_model["main"]) == []


@pytest.mark.parametrize("explicit_position", [True, False])
def test_sequence_insert(explicit_position: bool):
    seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=4)
    index = make_constant_node("index", TensorProto.INT64, (), [0])
    node_input = ["sequence", "value", "index"] if explicit_position else ["sequence", "value"]
    sequence_insert_node = helper.make_node("SequenceInsert", node_input, ["output"])
    graph = helper.make_graph(
        [index, seq_node, sequence_insert_node],
        "test_sequence_insert",
        inputs=[*graph_inputs, helper.make_tensor_value_info("value", TensorProto.FLOAT, [32, 32])],
        outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="test_sequence_insert")
    tvm_model = check_import(model)
    assert_tuple_tensor_sinfo(tvm_model["main"].ret_ty, [([32, 32], "float32")] * 5)
    assert collect_relax_call_ops(tvm_model["main"]) == []


@pytest.mark.parametrize(
    "new_axis,axis,expected_shape",
    [
        (0, 0, [64, 32]),
        (0, 1, [32, 64]),
        (1, 0, [2, 32, 32]),
        (1, 1, [32, 2, 32]),
        (1, -1, [32, 32, 2]),
    ],
)
def test_concat_from_sequence(new_axis: int, axis: int, expected_shape: list[int]):
    seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=2)
    concat_from_sequence_node = helper.make_node(
        "ConcatFromSequence", ["sequence"], ["output"], axis=axis, new_axis=new_axis
    )
    graph = helper.make_graph(
        [seq_node, concat_from_sequence_node],
        "test_concat_from_sequence",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, expected_shape)],
    )
    model = helper.make_model(graph, producer_name="test_concat_from_sequence")
    tvm_model = check_import(model)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, expected_shape, "float32")
    assert_has_relax_ops(tvm_model["main"], "relax.concat")
    if new_axis:
        assert_has_relax_ops(tvm_model["main"], "relax.expand_dims")


def test_concat_from_sequence_new_axis_three_tensors():
    """new_axis=1 with three sequence elements (stack then concat along axis)."""
    seq_node, graph_inputs = construct_sequence(input_shape=[16, 8], num_tensors=3)
    concat_node = helper.make_node(
        "ConcatFromSequence", ["sequence"], ["output"], axis=0, new_axis=1
    )
    graph = helper.make_graph(
        [seq_node, concat_node],
        "test_concat_from_sequence_new_axis_three",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 16, 8])],
    )
    model = helper.make_model(graph, producer_name="test_concat_from_sequence_new_axis_three")
    tvm_model = check_import(model)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, [3, 16, 8], "float32")
    assert_has_relax_ops(tvm_model["main"], ["relax.expand_dims", "relax.concat"])


def test_concat_from_sequence_invalid_new_axis():
    """Verify that new_axis values other than 0 or 1 raise a ValueError."""
    seq_node, graph_inputs = construct_sequence(input_shape=[16, 8], num_tensors=2)
    concat_node = helper.make_node(
        "ConcatFromSequence", ["sequence"], ["output"], axis=0, new_axis=2
    )
    graph = helper.make_graph(
        [seq_node, concat_node],
        "test_concat_from_sequence_invalid_new_axis",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 8])],
    )
    model = helper.make_model(graph, producer_name="test_concat_from_sequence_invalid_new_axis")

    with pytest.raises(ValueError, match="ConcatFromSequence only supports new_axis in"):
        from_onnx(model, opset=11)


@pytest.mark.parametrize("split", [2, [16, 48]])
def test_split_to_sequence(split):
    split_to_sequence_node = helper.make_node(
        "SplitToSequence",
        ["data", "split"],
        ["output"],
        axis=0,
    )
    split_shape = [len(split)] if isinstance(split, list) else ()
    split_node = make_constant_node(
        "split", TensorProto.INT64, split_shape, [split] if isinstance(split, int) else split
    )
    graph = helper.make_graph(
        [split_node, split_to_sequence_node],
        "test_split_to_sequence",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, [64, 32])],
        outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="test_split_to_sequence")
    tvm_model = check_import(model)
    expected_fields = (
        [([2, 32], "float32")] * 32
        if isinstance(split, int)
        else [([16, 32], "float32"), ([48, 32], "float32")]
    )
    assert_tuple_tensor_sinfo(tvm_model["main"].ret_ty, expected_fields)
    assert_has_relax_ops(tvm_model["main"], "relax.split")


def test_sequence_at():
    seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=4)
    index = make_constant_node("index", TensorProto.INT64, (), [1])
    node_input = ["sequence", "index"]
    sequence_at_node = helper.make_node("SequenceAt", node_input, ["output"])
    graph = helper.make_graph(
        [index, seq_node, sequence_at_node],
        "test_sequence_at",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="test_sequence_at")
    tvm_model = check_import(model)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, [32, 32], "float32")
    assert collect_relax_call_ops(tvm_model["main"]) == []


def test_optional_get_element_tensor():
    x_shape = [2, 3]
    optional_node = helper.make_node("Optional", ["x"], ["optional"])
    get_element_node = helper.make_node("OptionalGetElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, get_element_node],
        "test_optional_get_element_tensor",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, x_shape)],
        value_info=[make_optional_tensor_value_info("optional", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_optional_get_element_tensor")
    tvm_model = check_import(model, opset=18, ir_version=11)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, x_shape, "float32")
    assert collect_relax_call_ops(tvm_model["main"]) == []


def test_optional_has_element_tensor():
    x_shape = [2, 3]
    optional_node = helper.make_node("Optional", ["x"], ["optional"])
    has_element_node = helper.make_node("OptionalHasElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, has_element_node],
        "test_optional_has_element_tensor",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [])],
        value_info=[make_optional_tensor_value_info("optional", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_optional_has_element_tensor")
    tvm_model = check_import(model, opset=18, ir_version=11)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, [], "bool")
    assert True in collect_scalar_constants(tvm_model["main"])


def test_optional_has_element_empty():
    x_shape = [2, 3]
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, x_shape)
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_node = helper.make_node("Optional", [], ["optional"], type=tensor_type)
    has_element_node = helper.make_node("OptionalHasElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, has_element_node],
        "test_optional_has_element_empty",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [])],
        value_info=[helper.make_value_info("optional", optional_type)],
    )
    model = helper.make_model(graph, producer_name="test_optional_has_element_empty")
    tvm_model = check_import(model, opset=18, ir_version=11)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, [], "bool")
    assert False in collect_scalar_constants(tvm_model["main"])


def test_optional_has_element_empty_ir():
    x_shape = [2, 3]
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, x_shape)
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_node = helper.make_node("Optional", [], ["optional"], type=tensor_type)
    has_element_node = helper.make_node("OptionalHasElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, has_element_node],
        "test_optional_has_element_empty_ir",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [])],
        value_info=[helper.make_value_info("optional", optional_type)],
    )
    model = helper.make_model(graph, producer_name="test_optional_has_element_empty_ir")
    model.ir_version = 11
    model.opset_import[0].version = 18
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    assert collect_relax_call_ops(tvm_model["main"]) == []
    assert False in collect_scalar_constants(tvm_model["main"])


def test_optional_get_element_tensor_ir():
    x_shape = [2, 3]
    optional_node = helper.make_node("Optional", ["x"], ["optional"])
    get_element_node = helper.make_node("OptionalGetElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, get_element_node],
        "test_optional_get_element_tensor_ir",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape)],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, x_shape)],
        value_info=[make_optional_tensor_value_info("optional", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_optional_get_element_tensor_ir")
    model.ir_version = 11
    model.opset_import[0].version = 18
    tvm_model = from_onnx(model, opset=18, keep_params_in_input=True)

    assert collect_relax_call_ops(tvm_model["main"]) == []
    assert tvm_model["main"].ret_ty.dtype == "float32"


def test_optional_get_element_sequence():
    seq_node, graph_inputs = construct_sequence(input_shape=[32, 32], num_tensors=4)
    index = make_constant_node("index", TensorProto.INT64, (), [1])
    optional_node = helper.make_node("Optional", ["sequence"], ["optional"])
    get_element_node = helper.make_node("OptionalGetElement", ["optional"], ["unwrapped"])
    sequence_at_node = helper.make_node("SequenceAt", ["unwrapped", "index"], ["output"])
    graph = helper.make_graph(
        [index, seq_node, optional_node, get_element_node, sequence_at_node],
        "test_optional_get_element_sequence",
        inputs=graph_inputs,
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 32])],
        value_info=[make_optional_sequence_value_info("optional", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="test_optional_get_element_sequence")
    tvm_model = check_import(model, opset=18, ir_version=11)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, [32, 32], "float32")
    assert collect_relax_call_ops(tvm_model["main"]) == []


def test_optional_without_input_requires_type_attr():
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, [2, 3])
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_node = helper.make_node("Optional", [], ["optional"])
    graph = helper.make_graph(
        [optional_node],
        "test_optional_without_input_requires_type_attr",
        inputs=[],
        outputs=[helper.make_value_info("optional", optional_type)],
    )
    model = helper.make_model(graph, producer_name="test_optional_without_input_requires_type_attr")
    model.opset_import[0].version = 18

    with pytest.raises(ValueError, match="type attribute"):
        from_onnx(model, opset=18, keep_params_in_input=True)


def test_empty_optional_graph_output_raises():
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, [2, 3])
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_node = helper.make_node("Optional", [], ["optional"], type=tensor_type)
    graph = helper.make_graph(
        [optional_node],
        "test_empty_optional_graph_output_raises",
        inputs=[],
        outputs=[helper.make_value_info("optional", optional_type)],
    )
    model = helper.make_model(graph, producer_name="test_empty_optional_graph_output_raises")
    model.opset_import[0].version = 18

    with pytest.raises(ValueError, match="Empty optional graph outputs are not supported"):
        from_onnx(model, opset=18, keep_params_in_input=True)


def test_optional_has_element_requires_one_input():
    has_element_node = helper.make_node("OptionalHasElement", [], ["output"])
    graph = helper.make_graph(
        [has_element_node],
        "test_optional_has_element_requires_one_input",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [])],
    )
    model = helper.make_model(graph, producer_name="test_optional_has_element_requires_one_input")
    model.opset_import[0].version = 18

    with pytest.raises(ValueError, match="expects one input"):
        from_onnx(model, opset=18, keep_params_in_input=True)


def test_optional_get_element_empty_raises():
    x_shape = [2, 3]
    tensor_type = helper.make_tensor_type_proto(TensorProto.FLOAT, x_shape)
    optional_type = helper.make_optional_type_proto(tensor_type)
    optional_node = helper.make_node("Optional", [], ["optional"], type=tensor_type)
    get_element_node = helper.make_node("OptionalGetElement", ["optional"], ["output"])
    graph = helper.make_graph(
        [optional_node, get_element_node],
        "test_optional_get_element_empty_raises",
        inputs=[],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, x_shape)],
        value_info=[helper.make_value_info("optional", optional_type)],
    )
    model = helper.make_model(graph, producer_name="test_optional_get_element_empty_raises")
    model.opset_import[0].version = 18
    with pytest.raises(ValueError, match="empty optional"):
        from_onnx(model, opset=18, keep_params_in_input=True)


@pytest.mark.parametrize("with_reshape_flatten", [False, True])
def test_symbolic_shape_deduction(with_reshape_flatten):
    index_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["indices"],
        value=helper.make_tensor("indices", TensorProto.INT64, [], [0]),
    )
    shape_node = helper.make_node("Shape", ["data"], ["shape_output"])
    nodes = [index_node, shape_node]
    gather_input = "shape_output"

    if with_reshape_flatten:
        reshape_node = helper.make_node(
            "Reshape", ["shape_output", "target_shape"], ["reshaped_shape"]
        )
        nodes.append(reshape_node)
        gather_input = "reshaped_shape"

    gather_node = helper.make_node("Gather", [gather_input, "indices"], ["gather_output"])
    unsqueeze_node = helper.make_node("Unsqueeze", ["gather_output", "axes"], ["unsqueeze_output"])
    constant_of_shape_node = helper.make_node(
        "ConstantOfShape",
        ["unsqueeze_output"],
        ["output"],
        value=helper.make_tensor("value", TensorProto.FLOAT, [], [1]),
    )
    nodes.extend([gather_node, unsqueeze_node, constant_of_shape_node])

    initializers = [helper.make_tensor("axes", TensorProto.INT64, [1], vals=[0])]
    if with_reshape_flatten:
        initializers.append(helper.make_tensor("target_shape", TensorProto.INT64, [1], vals=[-1]))

    graph = helper.make_graph(
        nodes,
        "test_shape_deduction",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, ["batch", "seq"]),
        ],
        initializer=initializers,
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT64, [1])],
    )
    model = helper.make_model(graph, producer_name="test_shape_deduction")
    tvm_model = from_onnx(model, keep_params_in_input=True)

    @R.function
    def expected(data: R.Tensor(("batch", "seq"), dtype="float32")) -> R.Tensor(
        dtype="float32", ndim=1
    ):
        batch = T.int64()
        seq = T.int64()
        R.func_attr({"num_input": 1})
        with R.dataflow():
            gv: R.Tensor((batch,), dtype="float32") = R.broadcast_to(
                R.const(1, "float32"), R.shape([batch])
            )
            R.output(gv)
        return gv

    # TODO(siyuan): Enable assertion after fixing the SizeVar roundtrip issue
    # tvm.ir.assert_structural_equal(expected, tvm_model["main"])


def test_multi_inputs_with_same_symbolic_shape():
    concat_node = helper.make_node("Concat", ["data1", "data2"], ["output"], axis=1)

    graph = helper.make_graph(
        [concat_node],
        "test_multi_symbolic_shape_input",
        inputs=[
            helper.make_tensor_value_info("data1", TensorProto.FLOAT, ["batch", 1]),
            helper.make_tensor_value_info("data2", TensorProto.FLOAT, ["batch", 1]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])],
    )
    model = helper.make_model(graph, producer_name="test_multi_symbolic_shape_input")
    tvm_model = check_import(model)
    ret_ty = tvm_model["main"].ret_ty
    assert_tensor_sinfo(ret_ty, None, "float32")
    shape_values = ret_ty.shape.values
    assert len(shape_values) == 2
    assert isinstance(shape_values[1], tvm.tirx.IntImm)
    assert int(shape_values[1]) == 2
    assert_has_relax_ops(tvm_model["main"], "relax.concat")


def test_multi_ops_with_same_params():
    reshape_node_1 = helper.make_node("Reshape", ["a", "x"], ["b"])
    reshape_node_2 = helper.make_node("Reshape", ["b", "x"], ["c"])

    a_shape = [16]
    output_shape = [1, 16]

    graph = helper.make_graph(
        [reshape_node_1, reshape_node_2],
        "test_multi_ops_with_same_params",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape),
        ],
        initializer=[
            helper.make_tensor("x", TensorProto.INT64, [2], output_shape),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, producer_name="test_multi_ops_with_same_params")
    tvm_model = check_import(model)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, output_shape, "float32")
    assert collect_relax_call_ops(tvm_model["main"]).count("relax.reshape") == 2


def test_params_names_start_with_onnx():
    reshape_node = helper.make_node("Reshape", ["a", "onnx::x"], ["b"])

    a_shape = [16]
    output_shape = [1, 16]

    graph = helper.make_graph(
        [reshape_node],
        "test_params_names_start_with_onnx",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, a_shape),
        ],
        initializer=[
            helper.make_tensor("onnx::x", TensorProto.INT64, [2], output_shape),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, output_shape)],
    )
    model = helper.make_model(graph, producer_name="test_params_names_start_with_onnx")
    tvm_model = check_import(model)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, output_shape, "float32")
    assert_has_relax_ops(tvm_model["main"], "relax.reshape")


def test_shape_dim_string_expression():
    def _verify(x_shape):
        identity_node = helper.make_node("Identity", ["x"], ["y"])

        graph = helper.make_graph(
            [identity_node],
            "test_var_shape_dim_containing_expressions_onnx",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
        )
        model = helper.make_model(
            graph, producer_name="test_var_shape_dim_containing_expressions_onnx"
        )

        tvm_model = check_import(model)
        ret_ty = tvm_model["main"].ret_ty
        assert_tensor_sinfo(ret_ty, None, "float32")
        assert len(ret_ty.shape.values) == 3
        assert collect_relax_call_ops(tvm_model["main"]) == []

    _verify(["A", "B", "A + B"])
    _verify(["A", "B", "A - B"])
    _verify(["A", "B", "A * B"])
    _verify(["A", "B", "A // B"])


def test_shape_dim_string_expression_graph_add():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    x_shape = ["A", "B", "A + B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("A", "B", "A + B"), dtype="float32")) -> R.Tensor(("A", "B", "A + B"), dtype="float32"):
            A = T.int64(is_size_var=True)
            B = T.int64(is_size_var=True)
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A + B), dtype="float32") = x
                R.output(gv)
            return gv
    # fmt: on

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_dim_string_expression_graph_subtract():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    x_shape = ["A", "B", "A - B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("A", "B", "A - B"), dtype="float32")) -> R.Tensor(("A", "B", "A - B"), dtype="float32"):
            A = T.int64(is_size_var=True)
            B = T.int64(is_size_var=True)
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A - B), dtype="float32") = x
                R.output(gv)
            return gv
    # fmt: on

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_dim_string_expression_graph_mul():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    x_shape = ["A", "B", "A * B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("A", "B", "A * B"), dtype="float32")) -> R.Tensor(("A", "B", "A * B"), dtype="float32"):
            A = T.int64(is_size_var=True)
            B = T.int64(is_size_var=True)
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A * B), dtype="float32") = x
                R.output(gv)
            return gv
    # fmt: on

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_dim_string_expression_graph_div_1():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    # this will result in a floordiv despite not using // since the operands are always int
    x_shape = ["A", "B", "A / B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("A", "B", "A // B"), dtype="float32")) -> R.Tensor(("A", "B", "A // B"), dtype="float32"):
            A = T.int64(is_size_var=True)
            B = T.int64(is_size_var=True)
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A // B), dtype="float32") = x
                R.output(gv)
            return gv

    # fmt: on

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def test_shape_dim_string_expression_graph_div_2():
    identity_node = helper.make_node("Identity", ["x"], ["y"])

    x_shape = ["A", "B", "A // B"]

    graph = helper.make_graph(
        [identity_node],
        "test_var_shape_dim_containing_expressions_onnx",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, x_shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, x_shape)],
    )
    model = helper.make_model(graph, producer_name="test_var_shape_dim_containing_expressions_onnx")

    tvm_model = from_onnx(model, opset=14, keep_params_in_input=True)

    # fmt: off
    @I.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor(("A", "B", "A // B"), dtype="float32")) -> R.Tensor(("A", "B", "A // B"), dtype="float32"):
            A = T.int64(is_size_var=True)
            B = T.int64(is_size_var=True)
            R.func_attr({"num_input": 1})
            with R.dataflow():
                gv: R.Tensor((A, B, A // B), dtype="float32") = x
                R.output(gv)
            return gv
    # fmt: on

    tvm.ir.assert_structural_equal(tvm_model, Expected)


def assert_nms_import(model):
    tvm_model = check_import(model, opset=11)
    call_ops = collect_relax_call_ops(tvm_model["main"])
    assert "relax.vision.all_class_non_max_suppression" in call_ops
    assert tvm_model["main"].ret_ty.dtype == "int64"
    shape = get_static_tensor_shape(tvm_model["main"].ret_ty)
    if shape is not None:
        assert len(shape) == 2
        assert shape[1] == 3
    return tvm_model


def check_nms_correctness(model, inputs):
    onnxruntime = pytest.importorskip("onnxruntime")

    assert_nms_import(model)
    configure_model_format(model, ir_version=8, opset=11)
    inputs = generate_random_inputs(model, inputs)

    ort_session = onnxruntime.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    ort_selected = ort_session.run([], inputs)[0]

    tvm_output = run_in_tvm(model, inputs=inputs, opset=11)
    tvm_selected = (
        tvm_output[0].numpy() if isinstance(tvm_output, list | tuple) else tvm_output.numpy()
    )

    assert tvm_selected.dtype == ort_selected.dtype
    assert tvm_selected.ndim == 2
    assert tvm_selected.shape[1] == 3
    assert tvm_selected.shape[0] >= ort_selected.shape[0]
    if ort_selected.shape[0] > 0:
        tvm.testing.assert_allclose(
            tvm_selected[: ort_selected.shape[0]], ort_selected, rtol=1e-5, atol=1e-5
        )

    return tvm_selected, ort_selected


def test_nms():
    """Test NonMaxSuppression operator conversion using our AllClassNMS implementation."""
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0,
    )

    boxes_shape = [1, 5, 4]  # batch_size, num_boxes, 4
    scores_shape = [1, 2, 5]  # batch_size, num_classes, num_boxes

    graph = helper.make_graph(
        [nms_node],
        "nms_test",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ],
        initializer=[
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [3]),
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.5]),
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.1]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [0, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_test")
    model.ir_version = 8
    model.opset_import[0].version = 11

    local_bg = np.random.MT19937(0)
    local_rg = np.random.Generator(local_bg)
    inputs = {
        "boxes": local_rg.standard_normal(size=boxes_shape).astype(np.float32),
        "scores": local_rg.standard_normal(size=scores_shape).astype(np.float32),
    }
    check_nms_correctness(model, inputs)


def test_nms_scalar_shape1_constants():
    """Scalar params given as 1-D single-element constants must import (NumPy 2.x cast)."""
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
    )
    graph = helper.make_graph(
        [nms_node],
        "nms_scalar_shape1",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 5, 4]),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 5]),
        ],
        initializer=[
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [3]),
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.5]),
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.0]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [0, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    # Default import folds initializers to relax.Constant, exercising the scalar-cast path.
    from_onnx(model)


@pytest.mark.parametrize("with_explicit_max", [False, True])
def test_nms_max_output_boxes_per_class_zero(with_explicit_max: bool):
    """ONNX default for max_output_boxes_per_class is 0, yielding empty output."""
    node_inputs = ["boxes", "scores"]
    initializer = []
    if with_explicit_max:
        node_inputs.append("max_output_boxes_per_class")
        initializer.append(
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [0])
        )

    nms_node = helper.make_node(
        "NonMaxSuppression",
        node_inputs,
        ["selected_indices"],
        center_point_box=0,
    )

    boxes_shape = [1, 4, 4]
    scores_shape = [1, 1, 4]
    graph = helper.make_graph(
        [nms_node],
        "nms_max_output_boxes_per_class_zero",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ],
        initializer=initializer,
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [0, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_max_output_boxes_per_class_zero")
    model.ir_version = 8
    model.opset_import[0].version = 11

    inputs = {
        "boxes": np.array(
            [
                [
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.1, 1.0, 1.1],
                    [2.0, 2.0, 3.0, 3.0],
                    [2.0, 2.1, 3.0, 3.1],
                ]
            ],
            dtype=np.float32,
        ),
        "scores": np.array([[[0.9, 0.8, 0.7, 0.6]]], dtype=np.float32),
    }
    tvm_selected, _ = check_nms_correctness(model, inputs)
    assert tvm_selected.shape == (0, 3)


def test_nms_algorithm_correctness():
    """Test NMS algorithm correctness with fixed data to verify suppression logic."""
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0,
    )

    boxes_data = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.5, 0.5, 1.5, 1.5],
                [2.0, 2.0, 3.0, 3.0],
            ]
        ],
        dtype=np.float32,
    )
    scores_data = np.array([[[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]]], dtype=np.float32)

    boxes_shape = [1, 3, 4]  # batch_size, num_boxes, 4
    scores_shape = [1, 2, 3]  # batch_size, num_classes, num_boxes

    graph = helper.make_graph(
        [nms_node],
        "nms_test_import",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ],
        initializer=[
            helper.make_tensor(
                "max_output_boxes_per_class", TensorProto.INT64, [1], [2]
            ),  # Only 2 boxes per class
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.5]),  # IoU threshold 0.5
            helper.make_tensor(
                "score_threshold", TensorProto.FLOAT, [1], [0.1]
            ),  # Score threshold 0.1
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [4, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_test_correctness")
    inputs = {"boxes": boxes_data, "scores": scores_data}
    check_nms_correctness(model, inputs)


def test_nms_iou_suppression_correctness():
    """Test that NMS correctly suppresses overlapping boxes based on IoU threshold."""
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0,
    )

    boxes_data = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.1, 0.1, 1.1, 1.1],
                [2.0, 2.0, 3.0, 3.0],
            ]
        ],
        dtype=np.float32,
    )
    scores_data = np.array([[[0.9, 0.8, 0.7]]], dtype=np.float32)

    boxes_shape = [1, 3, 4]
    scores_shape = [1, 1, 3]

    graph = helper.make_graph(
        [nms_node],
        "nms_test_iou_suppression",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ],
        initializer=[
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [2]),
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.5]),  # IoU threshold 0.5
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.1]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [2, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_test_iou_suppression")
    model.ir_version = 8
    model.opset_import[0].version = 11
    inputs = {"boxes": boxes_data, "scores": scores_data}
    check_nms_correctness(model, inputs)


def test_nms_max_boxes_limit_correctness():
    """Test that NMS correctly limits the number of boxes per class."""
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0,
    )

    boxes_data = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [2.0, 0.0, 3.0, 1.0],
                [0.0, 2.0, 1.0, 3.0],
                [2.0, 2.0, 3.0, 3.0],
            ]
        ],
        dtype=np.float32,
    )
    scores_data = np.array([[[0.9, 0.8, 0.7, 0.6]]], dtype=np.float32)

    boxes_shape = [1, 4, 4]
    scores_shape = [1, 1, 4]

    graph = helper.make_graph(
        [nms_node],
        "nms_test_max_boxes_limit",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ],
        initializer=[
            helper.make_tensor(
                "max_output_boxes_per_class", TensorProto.INT64, [1], [2]
            ),  # Limit to 2 boxes
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.1]),  # Low IoU threshold
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.1]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [2, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_test_max_boxes_limit")
    model.ir_version = 8
    model.opset_import[0].version = 11
    inputs = {"boxes": boxes_data, "scores": scores_data}
    check_nms_correctness(model, inputs)


def test_nms_score_threshold_correctness():
    """Test that NMS correctly filters boxes based on score threshold."""
    nms_node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output_boxes_per_class", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        center_point_box=0,
    )

    boxes_data = np.array(
        [[[0.0, 0.0, 1.0, 1.0], [2.0, 0.0, 3.0, 1.0], [0.0, 2.0, 1.0, 3.0]]],
        dtype=np.float32,
    )
    scores_data = np.array([[[0.9, 0.3, 0.1]]], dtype=np.float32)

    boxes_shape = [1, 3, 4]
    scores_shape = [1, 1, 3]

    graph = helper.make_graph(
        [nms_node],
        "nms_test_score_threshold",
        inputs=[
            helper.make_tensor_value_info("boxes", TensorProto.FLOAT, boxes_shape),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, scores_shape),
        ],
        initializer=[
            helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [3]),
            helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.1]),
            helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.05]),
        ],
        outputs=[helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [3, 3])],
    )

    model = helper.make_model(graph, producer_name="nms_test_score_threshold")
    model.ir_version = 8
    model.opset_import[0].version = 11
    inputs = {"boxes": boxes_data, "scores": scores_data}
    check_nms_correctness(model, inputs)


# align_corners=None omits the attribute, exercising the ONNX default of 0.
@pytest.mark.parametrize("align_corners", [None, 0, 1])
def test_affine_grid(align_corners):
    attrs = {} if align_corners is None else {"align_corners": align_corners}
    affine_grid_node = helper.make_node("AffineGrid", ["theta", "size"], ["grid"], **attrs)

    graph = helper.make_graph(
        [affine_grid_node],
        "affine_grid_test",
        inputs=[
            helper.make_tensor_value_info("theta", TensorProto.FLOAT, [2, 2, 3]),
        ],
        initializer=[
            helper.make_tensor("size", TensorProto.INT64, [4], [2, 3, 16, 16]),
        ],
        outputs=[
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, [2, 16, 16, 2]),
        ],
    )

    model = helper.make_model(graph, producer_name="affine_grid_test")
    check_correctness(model, opset=20)


@pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [0, 1])
def test_grid_sample(mode, padding_mode, align_corners):
    x_shape = [1, 3, 4, 4]
    grid_shape = [1, 2, 2, 2]
    out_shape = [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape),
        ],
    )

    # Grid values must be in [-1, 1]: -1 is far left/top, 1 is far right/bottom
    grid_data = np.random.uniform(-1, 1, grid_shape).astype("float32")
    # Use controlled X input to avoid extreme values affecting nearest mode boundaries
    x_data = np.random.uniform(-1, 1, x_shape).astype("float32")

    model = helper.make_model(graph, producer_name="grid_sample_test")
    check_correctness(
        model,
        inputs={"grid": grid_data, "X": x_data},
        opset=16,
    )


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [0, 1])
def test_grid_sample_5d(mode, padding_mode, align_corners):
    x_shape = [1, 1, 4, 4, 4]
    grid_shape = [1, 4, 4, 4, 3]
    out_shape = [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2], grid_shape[3]]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_5d_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_5d_test")
    rng = np.random.default_rng(0)
    grid_data = rng.uniform(-1.25, 1.25, grid_shape).astype("float32")
    x_data = rng.uniform(-1, 1, x_shape).astype("float32")

    check_correctness(
        model,
        inputs={"grid": grid_data, "X": x_data},
        opset=16,
        rtol=1e-5,
        atol=1e-5,
    )


def test_grid_sample_5d_cubic_unsupported():
    x_shape = [1, 1, 4, 4, 4]
    grid_shape = [1, 2, 3, 5, 3]
    out_shape = [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2], grid_shape[3]]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="cubic",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_5d_cubic_unsupported_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_5d_cubic_unsupported_test")
    with pytest.raises(
        NotImplementedError,
        match="5D .*GridSample with mode='cubic' is not supported",
    ):
        from_onnx(model, opset=16, keep_params_in_input=True)


def test_grid_sample_4d_non_square_output_shape():
    x_shape = [1, 3, 4, 4]
    grid_shape = [1, 3, 5, 2]
    out_shape = [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="bilinear",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_4d_non_square_output_shape_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_4d_non_square_output_shape_test")
    tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)
    inferred_shape = tuple(dim.value for dim in tvm_model["main"].ret_ty.shape.values)
    assert inferred_shape == tuple(out_shape)


def test_grid_sample_unsupported_rank():
    x_shape = [1, 3, 4]
    grid_shape = [1, 4, 2]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="bilinear",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_unsupported_rank_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, x_shape),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_unsupported_rank_test")
    with pytest.raises(NotImplementedError, match="GridSample only supports 4D or 5D input"):
        from_onnx(model, opset=16, keep_params_in_input=True)


def test_grid_sample_linear_mode_translation():
    """Test that ONNX mode='linear' is correctly translated to 'bilinear'.

    The ONNX spec defines 'linear' as a valid mode for GridSample. Real ONNX
    models exported from frameworks like PyTorch may still use 'linear'. We
    verify the translation by inspecting the Relax IR directly.
    """
    x_shape = [1, 3, 4, 4]
    grid_shape = [1, 2, 2, 2]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="linear",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_linear_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]]
            ),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_linear_test")
    tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)
    # Verify 'linear' was translated to 'bilinear' in the Relax IR
    assert 'method="bilinear"' in str(tvm_model)


def test_grid_sample_cubic_mode_translation():
    """Test that ONNX mode='cubic' is correctly translated to 'bicubic'.

    The ONNX spec defines 'cubic' as a valid mode for GridSample, but
    TVM uses 'bicubic'. We verify the translation by inspecting the
    imported Relax IR directly.
    """
    x_shape = [1, 3, 4, 4]
    grid_shape = [1, 2, 2, 2]

    node = helper.make_node(
        "GridSample",
        inputs=["X", "grid"],
        outputs=["Y"],
        mode="cubic",
    )

    graph = helper.make_graph(
        [node],
        "grid_sample_cubic_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, grid_shape),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, [x_shape[0], x_shape[1], grid_shape[1], grid_shape[2]]
            ),
        ],
    )

    model = helper.make_model(graph, producer_name="grid_sample_cubic_test")
    tvm_model = from_onnx(model, opset=16, keep_params_in_input=True)
    # Verify 'cubic' was translated to 'bicubic' in the Relax IR
    assert 'method="bicubic"' in str(tvm_model)


@pytest.mark.parametrize(
    ("coordinate_transformation_mode", "rois"),
    [
        (
            "output_half_pixel",
            np.array([[1.0, 1.0, 6.0, 6.0], [2.0, 0.5, 7.0, 7.0]], dtype="float32"),
        ),
        ("half_pixel", np.array([[1.0, 1.0, 1.2, 1.2], [2.0, 0.5, 1.1, 1.1]], dtype="float32")),
    ],
)
def test_roi_align(coordinate_transformation_mode, rois):
    x_shape = [1, 4, 8, 8]
    rois_shape = [2, 4]
    batch_indices_shape = [2]
    out_shape = [2, 4, 3, 3]

    node = helper.make_node(
        "RoiAlign",
        inputs=["X", "rois", "batch_indices"],
        outputs=["Y"],
        output_height=3,
        output_width=3,
        sampling_ratio=2,
        spatial_scale=1.0,
        mode="avg",
        coordinate_transformation_mode=coordinate_transformation_mode,
    )

    graph = helper.make_graph(
        [node],
        "roi_align_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("rois", TensorProto.FLOAT, rois_shape),
            helper.make_tensor_value_info("batch_indices", TensorProto.INT64, batch_indices_shape),
        ],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)],
    )

    model = helper.make_model(graph, producer_name="roi_align_test")
    inputs = {
        "X": rg.standard_normal(size=x_shape).astype("float32"),
        "rois": rois,
        "batch_indices": np.array([0, 0], dtype="int64"),
    }
    check_correctness(model, inputs=inputs, opset=16, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "cond_info",
    [
        helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
        helper.make_tensor_value_info("cond", TensorProto.BOOL, [1]),
    ],
    ids=["scalar_condition", "tensor_condition"],
)
def test_if(cond_info):
    """Test ONNX If operator with scalar and tensor bool conditions."""

    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    result_info = helper.make_tensor_value_info("result", TensorProto.FLOAT, [3])

    # then branch: x * 2.0
    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    then_mul = helper.make_node("Mul", ["x", "two"], ["then_out"])
    then_out_info = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [3])
    then_graph = helper.make_graph([then_mul], "then_graph", [], [then_out_info], initializer=[two])

    # else branch: x * 3.0
    three = helper.make_tensor("three", TensorProto.FLOAT, [1], [3.0])
    else_mul = helper.make_node("Mul", ["x", "three"], ["else_out"])
    else_out_info = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [3])
    else_graph = helper.make_graph(
        [else_mul], "else_graph", [], [else_out_info], initializer=[three]
    )

    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["result"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    main_graph = helper.make_graph([if_node], "if_test", [cond_info, x_info], [result_info])
    model = helper.make_model(main_graph, opset_imports=[helper.make_opsetid("", 13)])

    tvm_model = check_import(model)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, [3], "float32")
    assert_has_relax_if(tvm_model["main"])


def test_if_computed_condition():
    """Test If where condition is computed from another op in the main graph."""
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    result_info = helper.make_tensor_value_info("result", TensorProto.FLOAT, [3])

    zero = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    reduce_node = helper.make_node(
        "ReduceSum", ["x"], ["x_sum"], keepdims=0, noop_with_empty_axes=0
    )
    greater_node = helper.make_node("Greater", ["x_sum", "zero"], ["cond"])

    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    then_mul = helper.make_node("Mul", ["x", "two"], ["then_out"])
    then_out_info = helper.make_tensor_value_info("then_out", TensorProto.FLOAT, [3])
    then_graph = helper.make_graph([then_mul], "then_graph", [], [then_out_info], initializer=[two])

    three = helper.make_tensor("three", TensorProto.FLOAT, [1], [3.0])
    else_mul = helper.make_node("Mul", ["x", "three"], ["else_out"])
    else_out_info = helper.make_tensor_value_info("else_out", TensorProto.FLOAT, [3])
    else_graph = helper.make_graph(
        [else_mul], "else_graph", [], [else_out_info], initializer=[three]
    )

    if_node = helper.make_node(
        "If", inputs=["cond"], outputs=["result"], then_branch=then_graph, else_branch=else_graph
    )

    main_graph = helper.make_graph(
        [reduce_node, greater_node, if_node],
        "if_computed_cond",
        [x_info],
        [result_info],
        initializer=[zero],
    )
    model = helper.make_model(main_graph, opset_imports=[helper.make_opsetid("", 13)])

    tvm_model = check_import(model)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, [3], "float32")
    assert_has_relax_ops(tvm_model["main"], ["relax.sum", "relax.greater"])
    assert_has_relax_if(tvm_model["main"])


def test_if_multiple_outputs():
    """Test If operator where branches return multiple outputs."""
    cond_info = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    out1_info = helper.make_tensor_value_info("out1", TensorProto.FLOAT, [3])
    out2_info = helper.make_tensor_value_info("out2", TensorProto.FLOAT, [3])

    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    three = helper.make_tensor("three", TensorProto.FLOAT, [1], [3.0])

    then_mul1 = helper.make_node("Mul", ["x", "two"], ["then_out1"])
    then_mul2 = helper.make_node("Mul", ["x", "three"], ["then_out2"])
    then_o1 = helper.make_tensor_value_info("then_out1", TensorProto.FLOAT, [3])
    then_o2 = helper.make_tensor_value_info("then_out2", TensorProto.FLOAT, [3])
    then_graph = helper.make_graph(
        [then_mul1, then_mul2], "then_graph", [], [then_o1, then_o2], initializer=[two, three]
    )

    four = helper.make_tensor("four", TensorProto.FLOAT, [1], [4.0])
    five = helper.make_tensor("five", TensorProto.FLOAT, [1], [5.0])
    else_mul1 = helper.make_node("Mul", ["x", "four"], ["else_out1"])
    else_mul2 = helper.make_node("Mul", ["x", "five"], ["else_out2"])
    else_o1 = helper.make_tensor_value_info("else_out1", TensorProto.FLOAT, [3])
    else_o2 = helper.make_tensor_value_info("else_out2", TensorProto.FLOAT, [3])
    else_graph = helper.make_graph(
        [else_mul1, else_mul2], "else_graph", [], [else_o1, else_o2], initializer=[four, five]
    )

    if_node = helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["out1", "out2"],
        then_branch=then_graph,
        else_branch=else_graph,
    )
    main_graph = helper.make_graph(
        [if_node], "if_multi_out", [cond_info, x_info], [out1_info, out2_info]
    )
    model = helper.make_model(main_graph, opset_imports=[helper.make_opsetid("", 13)])

    tvm_model = check_import(model)
    assert_tuple_tensor_sinfo(tvm_model["main"].ret_ty, [([3], "float32"), ([3], "float32")])
    assert_has_relax_if(tvm_model["main"])


def test_if_nested():
    """Test nested If operator inside a branch."""
    cond1_info = helper.make_tensor_value_info("cond1", TensorProto.BOOL, [])
    cond2_info = helper.make_tensor_value_info("cond2", TensorProto.BOOL, [])
    x_info = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3])
    result_info = helper.make_tensor_value_info("result", TensorProto.FLOAT, [3])

    # Inner then: x * 2
    two = helper.make_tensor("two", TensorProto.FLOAT, [1], [2.0])
    inner_then_mul = helper.make_node("Mul", ["x", "two"], ["inner_then_out"])
    inner_then_out_info = helper.make_tensor_value_info("inner_then_out", TensorProto.FLOAT, [3])
    inner_then_graph = helper.make_graph(
        [inner_then_mul], "inner_then", [], [inner_then_out_info], initializer=[two]
    )

    # Inner else: x * 3
    three = helper.make_tensor("three", TensorProto.FLOAT, [1], [3.0])
    inner_else_mul = helper.make_node("Mul", ["x", "three"], ["inner_else_out"])
    inner_else_out_info = helper.make_tensor_value_info("inner_else_out", TensorProto.FLOAT, [3])
    inner_else_graph = helper.make_graph(
        [inner_else_mul], "inner_else", [], [inner_else_out_info], initializer=[three]
    )

    # Outer then: nested If(cond2, x*2, x*3)
    inner_if = helper.make_node(
        "If",
        inputs=["cond2"],
        outputs=["outer_then_out"],
        then_branch=inner_then_graph,
        else_branch=inner_else_graph,
    )
    outer_then_out_info = helper.make_tensor_value_info("outer_then_out", TensorProto.FLOAT, [3])
    outer_then_graph = helper.make_graph([inner_if], "outer_then", [], [outer_then_out_info])

    # Outer else: x * 4
    four = helper.make_tensor("four", TensorProto.FLOAT, [1], [4.0])
    outer_else_mul = helper.make_node("Mul", ["x", "four"], ["outer_else_out"])
    outer_else_out_info = helper.make_tensor_value_info("outer_else_out", TensorProto.FLOAT, [3])
    outer_else_graph = helper.make_graph(
        [outer_else_mul], "outer_else", [], [outer_else_out_info], initializer=[four]
    )

    outer_if = helper.make_node(
        "If",
        inputs=["cond1"],
        outputs=["result"],
        then_branch=outer_then_graph,
        else_branch=outer_else_graph,
    )
    main_graph = helper.make_graph(
        [outer_if], "nested_if", [cond1_info, cond2_info, x_info], [result_info]
    )
    model = helper.make_model(main_graph, opset_imports=[helper.make_opsetid("", 13)])

    tvm_model = check_import(model)
    assert_tensor_sinfo(tvm_model["main"].ret_ty, [3], "float32")
    assert_has_relax_if(tvm_model["main"], min_count=2)


# Helper that builds the ONNX graph for MatMulInteger so the tests don't repeat boilerplate code every time
def _make_matmulinteger_model(A_shape, B_shape, A_dtype, B_dtype, a_zp_array=None, b_zp_array=None):
    """Build a minimal single-node ONNX graph for MatMulInteger."""

    def np_dtype_to_onnx(dt):
        return {np.int8: TensorProto.INT8, np.uint8: TensorProto.UINT8}[dt]

    A_info = helper.make_tensor_value_info("A", np_dtype_to_onnx(A_dtype), A_shape)
    B_info = helper.make_tensor_value_info("B", np_dtype_to_onnx(B_dtype), B_shape)
    graph_inputs = [A_info, B_info]
    node_inputs = ["A", "B"]
    initializers = []

    def _add_zp(name, arr, dtype):
        onnx_dtype = np_dtype_to_onnx(dtype)
        shape = list(arr.shape)
        initializers.append(helper.make_tensor(name, onnx_dtype, shape, arr.flatten().tolist()))
        node_inputs.append(name)

    if a_zp_array is not None:
        _add_zp("a_zero_point", a_zp_array, A_dtype)
    elif b_zp_array is not None:
        node_inputs.append("")  # placeholder only needed if b_zp is present

    if b_zp_array is not None:
        _add_zp("b_zero_point", b_zp_array, B_dtype)

    out_info = helper.make_tensor_value_info("output", TensorProto.INT32, None)
    node = helper.make_node("MatMulInteger", inputs=node_inputs, outputs=["output"])
    graph = helper.make_graph(
        [node], "matmulinteger", graph_inputs, [out_info], initializer=initializers
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    model.ir_version = 8
    return model


@pytest.mark.parametrize(
    "A_dtype,B_dtype,a_zp,b_zp",
    [
        (np.int8, np.int8, None, None),
        (np.uint8, np.uint8, None, None),
        (np.uint8, np.int8, None, None),
        pytest.param(
            np.int8,
            np.uint8,
            None,
            None,
            marks=pytest.mark.xfail(
                reason=(
                    "Some older ORT versions don't support mixed int8/uint8 "
                    "dtype combination for MatMulInteger"
                ),
                strict=False,
            ),
        ),
        (np.uint8, np.uint8, np.uint8(128), np.uint8(128)),
        (np.int8, np.int8, np.int8(1), np.int8(2)),
    ],
)
def test_matmulinteger(A_dtype, B_dtype, a_zp, b_zp):
    """2-D MatMulInteger across dtype combos and zero-point configurations."""
    np.random.seed(0)
    A = np.random.randint(-5, 5, (4, 8)).astype(A_dtype)
    B = np.random.randint(-5, 5, (8, 6)).astype(B_dtype)
    model = _make_matmulinteger_model(
        [4, 8],
        [8, 6],
        A_dtype,
        B_dtype,
        a_zp_array=np.array(a_zp, dtype=A_dtype) if a_zp is not None else None,
        b_zp_array=np.array(b_zp, dtype=B_dtype) if b_zp is not None else None,
    )
    check_correctness(model, inputs={"A": A, "B": B}, opset=10)


@pytest.mark.parametrize(
    "A_shape,B_shape,a_zp,b_zp",
    [
        ((2, 4, 8), (2, 8, 6), np.int8(1), np.int8(2)),  # 3-D batched
        ((2, 3, 4, 8), (2, 3, 8, 6), np.int8(1), np.int8(2)),  # 4-D batched
    ],
)
def test_matmulinteger_batched(A_shape, B_shape, a_zp, b_zp):
    """Batched MatMulInteger beyond 2-D."""
    np.random.seed(1)
    A = np.random.randint(-5, 5, A_shape).astype(np.int8)
    B = np.random.randint(-5, 5, B_shape).astype(np.int8)
    model = _make_matmulinteger_model(
        list(A_shape),
        list(B_shape),
        np.int8,
        np.int8,
        a_zp_array=np.array(a_zp, dtype=np.int8),
        b_zp_array=np.array(b_zp, dtype=np.int8),
    )
    check_correctness(model, inputs={"A": A, "B": B}, opset=10)


def test_matmulinteger_per_channel_zp():
    """
    1-D zero points: per-row for A ([M]) and per-col for B ([N]).
    Exercises the expand_dims path in the converter.
    The ONNX spec permits these zero-point shapes, so this frontend test verifies
    the imported Relax structure.
    """
    np.random.seed(2)
    A = np.random.randint(-5, 5, (4, 8)).astype(np.int8)
    B = np.random.randint(-5, 5, (8, 6)).astype(np.int8)
    a_zp = np.arange(4, dtype=np.int8)  # shape [M=4], per-row
    b_zp = np.arange(6, dtype=np.int8)  # shape [N=6], per-col

    expected = np.matmul(
        A.astype(np.int32) - a_zp.astype(np.int32)[:, np.newaxis],
        B.astype(np.int32) - b_zp.astype(np.int32)[np.newaxis, :],
    ).astype(np.int32)

    model = _make_matmulinteger_model(
        [4, 8], [8, 6], np.int8, np.int8, a_zp_array=a_zp, b_zp_array=b_zp
    )

    tvm_output = run_in_tvm(model, inputs={"A": A, "B": B}, opset=10)
    tvm.testing.assert_allclose(tvm_output.numpy(), expected)

    tvm_model = check_import(model, opset=10)
    call_ops = collect_relax_call_ops(tvm_model["main"])
    assert "relax.expand_dims" in call_ops
    assert "relax.subtract" in call_ops
    assert "relax.matmul" in call_ops
    assert_tensor_sinfo(tvm_model["main"].ret_ty, [4, 6], "int32")


@pytest.mark.parametrize(
    ("pooled_shape", "rois"),
    [
        ((1, 1), np.array([[0.0, 1.0, 1.0, 6.0, 6.0], [0.0, 0.0, 0.0, 7.0, 7.0]], dtype="float32")),
        (
            (2, 3),
            np.array([[0.0, 1.2, 0.5, 6.8, 7.0], [0.0, -1.0, 2.0, 3.5, 5.2]], dtype="float32"),
        ),
        (
            (2, 2),
            np.array(
                [[0.0, 100.0, 100.0, 110.0, 110.0], [0.0, 1.0, 1.0, 6.0, 6.0]], dtype="float32"
            ),
        ),
    ],
)
def test_max_roi_pool(pooled_shape, rois):
    x_shape = [1, 4, 8, 8]
    out_shape = [2, 4, pooled_shape[0], pooled_shape[1]]

    node = helper.make_node(
        "MaxRoiPool",
        inputs=["X", "rois"],
        outputs=["Y"],
        pooled_shape=pooled_shape,
        spatial_scale=1.0,
    )

    graph = helper.make_graph(
        [node],
        "max_roi_pool_test",
        inputs=[
            helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
            helper.make_tensor_value_info("rois", TensorProto.FLOAT, [2, 5]),
        ],
        outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, out_shape)],
    )

    model = helper.make_model(graph, producer_name="max_roi_pool_test")
    inputs = {
        "X": rg.standard_normal(size=x_shape).astype("float32"),
        "rois": rois,
    }
    check_correctness(model, inputs=inputs, opset=16, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("op_name", ["ArgMax", "ArgMin"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("keepdims", [True, False])
def test_arg_min_max_select_last_index(op_name, axis, keepdims):
    """select_last_index=1 must return the last occurrence of the extreme value."""
    shape = [3, 4, 5]

    node = helper.make_node(
        op_name,
        inputs=["data"],
        outputs=["out"],
        axis=axis,
        keepdims=int(keepdims),
        select_last_index=1,
    )

    out_shape = list(shape)
    if keepdims:
        out_shape[axis] = 1
    else:
        out_shape.pop(axis)

    graph = helper.make_graph(
        [node],
        "arg_select_last_index_test",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
        outputs=[helper.make_tensor_value_info("out", TensorProto.INT64, out_shape)],
    )
    model = helper.make_model(graph, producer_name="arg_select_last_index_test")
    data = rg.uniform(-10, 10, shape).astype("float32")
    slices_first = [slice(None)] * len(shape)
    slices_last = [slice(None)] * len(shape)
    slices_first[axis] = 0
    slices_last[axis] = shape[axis] - 1
    extreme = data.max() + 1.0 if op_name == "ArgMax" else data.min() - 1.0
    data[tuple(slices_first)] = extreme
    data[tuple(slices_last)] = extreme
    check_correctness(model, inputs={"data": data}, opset=12)


@pytest.mark.parametrize("op_name", ["ArgMax", "ArgMin"])
def test_arg_min_max_select_last_index_no_tie(op_name):
    """With all-unique values, both select_last_index modes should agree."""
    shape = [4, 5]
    data = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    if op_name == "ArgMin":
        data = -data

    for select_last in [0, 1]:
        node = helper.make_node(
            op_name,
            inputs=["data"],
            outputs=["out"],
            axis=1,
            keepdims=1,
            select_last_index=select_last,
        )
        graph = helper.make_graph(
            [node],
            "arg_no_tie_test",
            inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
            outputs=[helper.make_tensor_value_info("out", TensorProto.INT64, [4, 1])],
        )
        model = helper.make_model(graph, producer_name="arg_no_tie_test")
        check_correctness(model, inputs={"data": data}, opset=12)


@pytest.mark.parametrize("op_name", ["ArgMax", "ArgMin"])
def test_arg_min_max_select_last_index_ir(op_name):
    """select_last_index=1 must lower to flip + argmax/argmin + subtract in the Relax IR."""
    shape = [3, 4, 5]
    relax_op = "relax.argmax" if op_name == "ArgMax" else "relax.argmin"

    node = helper.make_node(
        op_name,
        inputs=["data"],
        outputs=["out"],
        axis=1,
        keepdims=1,
        select_last_index=1,
    )
    graph = helper.make_graph(
        [node],
        "arg_select_last_index_ir_test",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
        outputs=[helper.make_tensor_value_info("out", TensorProto.INT64, [3, 1, 5])],
    )
    model = helper.make_model(graph, producer_name="arg_select_last_index_ir_test")
    tvm_model = from_onnx(model, opset=12, keep_params_in_input=True)

    call_ops = collect_relax_call_ops(tvm_model["main"])
    assert relax_op in call_ops, f"Expected {relax_op} in IR, got {call_ops}"
    assert "relax.flip" in call_ops, f"Expected relax.flip in IR, got {call_ops}"
    assert "relax.subtract" in call_ops, f"Expected relax.subtract in IR, got {call_ops}"


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_split_to_sequence_keepdims_0(axis: int):
    """keepdims=0, no split input: each chunk of size 1 has the split axis squeezed out."""
    shape = [3, 4, 5]
    out_shape = [s for i, s in enumerate(shape) if i != axis]

    split_to_seq_node = helper.make_node(
        "SplitToSequence",
        ["data"],  # no split input — keepdims applies here
        ["output"],
        axis=axis,
        keepdims=0,
    )
    graph = helper.make_graph(
        [split_to_seq_node],
        f"test_split_to_sequence_keepdims_0_axis{axis}",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
        outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, out_shape)],
    )
    model = helper.make_model(graph, producer_name="test_split_to_sequence_keepdims_0")
    tvm_model = check_import(model)
    assert_tuple_tensor_sinfo(
        tvm_model["main"].ret_ty,
        [(out_shape, "float32") for _ in range(shape[axis])],
    )


def test_split_to_sequence_keepdims_ignored_when_split_provided():
    """Per spec: keepdims is ignored when split input is provided.
    TVM follows the spec — output keeps the split axis even with keepdims=0."""
    split_node = make_constant_node("split", TensorProto.INT64, (), [1])
    split_to_seq_node = helper.make_node(
        "SplitToSequence",
        ["data", "split"],
        ["output"],
        axis=0,
        keepdims=0,
    )
    graph = helper.make_graph(
        [split_node, split_to_seq_node],
        "test_split_to_sequence_keepdims_ignored",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, [4, 5])],
        outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, [1, 5])],
    )
    model = helper.make_model(
        graph,
        producer_name="test_split_to_sequence_keepdims_ignored",
        opset_imports=[helper.make_opsetid("", 11)],
    )
    model.ir_version = 8
    tvm_model = check_import(model, opset=11)
    assert_tuple_tensor_sinfo(tvm_model["main"].ret_ty, [([1, 5], "float32") for _ in range(4)])


@pytest.mark.parametrize("axis", [0, 1])
def test_split_to_sequence_uneven_last_chunk(axis: int):
    """Spec: last chunk may be smaller if dim is not divisible by scalar split."""
    shape = [5, 4] if axis == 0 else [3, 5]
    split_node = make_constant_node("split", TensorProto.INT64, (), [2])
    split_to_seq_node = helper.make_node(
        "SplitToSequence", ["data", "split"], ["output"], axis=axis, keepdims=1
    )
    graph = helper.make_graph(
        [split_node, split_to_seq_node],
        f"test_split_to_sequence_uneven_axis{axis}",
        inputs=[helper.make_tensor_value_info("data", TensorProto.FLOAT, shape)],
        outputs=[helper.make_tensor_sequence_value_info("output", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, producer_name="test_split_to_sequence_uneven")
    tvm_model = check_import(model)
    expected_shapes = [[2, 4], [2, 4], [1, 4]] if axis == 0 else [[3, 2], [3, 2], [3, 1]]
    assert_tuple_tensor_sinfo(
        tvm_model["main"].ret_ty,
        [(expected_shape, "float32") for expected_shape in expected_shapes],
    )


def test_quantizelinear_singleton_qparams_opset10():
    """QuantizeLinear must treat shape-[1] scale/zp as scalar in opset10."""
    node = helper.make_node("QuantizeLinear", ["x", "scale", "zero_point"], ["y"])
    graph = helper.make_graph(
        [node],
        "quantizelinear_singleton_qparams_opset10",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 3, 2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [4, 3, 2, 2])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.03125]),
            helper.make_tensor("zero_point", TensorProto.UINT8, [1], [127]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])

    x = rg.standard_normal((4, 3, 2, 2)).astype("float32")
    check_correctness(model, inputs={"x": x}, opset=10, check_dtypes=True)


def test_dequantizelinear_singleton_qparams_opset10():
    """DequantizeLinear must treat shape-[1] scale/zp as scalar in opset10."""
    node = helper.make_node("DequantizeLinear", ["x", "scale", "zero_point"], ["y"])
    graph = helper.make_graph(
        [node],
        "dequantizelinear_singleton_qparams_opset10",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [64])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [1], [0.125]),
            helper.make_tensor("zero_point", TensorProto.UINT8, [1], [1]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])

    x = rg.integers(low=0, high=255, size=(64,), dtype=np.uint8)
    check_correctness(model, inputs={"x": x}, opset=10, check_dtypes=True)


def test_quantizelinear_optional_zero_point_opset13():
    """ONNX allows missing zero_point input; importer should default it to 0 (uint8)."""
    node = helper.make_node("QuantizeLinear", ["x", "scale"], ["y"])
    graph = helper.make_graph(
        [node],
        "quantizelinear_optional_zero_point_opset13",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 5])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 5])],
        initializer=[helper.make_tensor("scale", TensorProto.FLOAT, [], [0.2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    x = rg.standard_normal((2, 5)).astype("float32")
    check_correctness(model, inputs={"x": x}, opset=13, check_dtypes=True)


def test_dynamicquantizelinear_opset11():
    """DynamicQuantizeLinear returns (y, y_scale, y_zero_point) with ORT parity."""
    node = helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "y_scale", "y_zero_point"])
    graph = helper.make_graph(
        [node],
        "dynamicquantizelinear_opset11",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [
            helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3, 4]),
            helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, []),
            helper.make_tensor_value_info("y_zero_point", TensorProto.UINT8, []),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

    x = rg.standard_normal((2, 3, 4)).astype("float32")
    check_correctness(model, inputs={"x": x}, opset=11, atol=1e-5, rtol=1e-5, check_dtypes=True)


def test_quantizelinear_default_axis_opset10():
    """opset10 QuantizeLinear should honor default axis=1 (not hardcode axis=0)."""
    node = helper.make_node("QuantizeLinear", ["x", "scale", "zero_point"], ["y"])
    graph = helper.make_graph(
        [node],
        "quantizelinear_axis_opset10",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3, 4])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [3], [0.05, 0.1, 0.2]),
            helper.make_tensor("zero_point", TensorProto.UINT8, [3], [1, 127, 250]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])

    x = rg.standard_normal((2, 3, 4)).astype("float32")
    check_correctness(model, inputs={"x": x}, opset=10, check_dtypes=True)


def test_dequantizelinear_default_axis_opset10():
    """opset10 DequantizeLinear should honor default axis=1 (not hardcode axis=0)."""
    node = helper.make_node("DequantizeLinear", ["x", "scale", "zero_point"], ["y"])
    graph = helper.make_graph(
        [node],
        "dequantizelinear_axis_opset10",
        [helper.make_tensor_value_info("x", TensorProto.UINT8, [2, 3, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])],
        initializer=[
            helper.make_tensor("scale", TensorProto.FLOAT, [3], [0.05, 0.1, 0.2]),
            helper.make_tensor("zero_point", TensorProto.UINT8, [3], [1, 127, 250]),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])

    x = rg.integers(low=0, high=255, size=(2, 3, 4), dtype=np.uint8)
    check_correctness(model, inputs={"x": x}, opset=10, check_dtypes=True)


if __name__ == "__main__":
    tvm.testing.main()
