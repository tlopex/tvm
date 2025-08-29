# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE/2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test Python printer for IRModules with Python functions."""

import pytest
import tvm
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.script.printer.python_printer import irmodule_to_python


def test_simple_relax_function_to_python():
    """Test converting a simple Relax function to Python function."""
    
    @I.ir_module
    class SimpleModule:
        @R.function
        def simple_add(x: R.Tensor((5,), "float32"), y: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
            gv = R.add(x, y)
            return gv
    
    # Convert to Python
    python_code = irmodule_to_python(SimpleModule)
    
    # Check that the generated code contains expected elements
    assert "def simple_add(" in python_code
    assert "torch.add" in python_code
    assert "return gv" in python_code


def test_relax_function_with_nn_ops():
    """Test converting Relax function with neural network operations."""
    
    @I.ir_module
    class NNModule:
        @R.function
        def nn_forward(x: R.Tensor(("n", 64), "float32"), w: R.Tensor((64, 64), "float32")) -> R.Tensor(("n", 64), "float32"):
            lv = R.add(x, w)
            lv1 = R.nn.relu(lv)
            return lv1
    
    # Convert to Python
    python_code = irmodule_to_python(NNModule)
    
    # Check that the generated code contains expected elements
    assert "def nn_forward(" in python_code
    assert "torch.add" in python_code
    assert "F.relu" in python_code
    assert "n = x.shape[0]" in python_code  # Symbolic shape handling


def test_relax_function_with_call_tir():
    """Test converting Relax function with call_tir."""
    
    @I.ir_module
    class CallTIRModule:
        @T.prim_func
        def matmul(A: T.handle, B: T.handle, C: T.handle):
            n = T.int32()
            A_buf = T.match_buffer(A, (n, 16), "float32")
            B_buf = T.match_buffer(B, (16, 20), "float32")
            C_buf = T.match_buffer(C, (n, 20), "float32")
            for i, j, k in T.grid(n, 20, 16):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C_buf[vi, vj] = T.float32(0)
                    C_buf[vi, vj] = C_buf[vi, vj] + A_buf[vi, vk] * B_buf[vk, vj]
        
        @R.function
        def main(x: R.Tensor(("n", 16), "float32"), w: R.Tensor((16, 20), "float32")) -> R.Tensor(("n", 20), "float32"):
            n = T.int64()
            lv = R.call_tir(cls.matmul, [x, w], out_sinfo=R.Tensor((n, 20), "float32"))
            return lv
    
    # Convert to Python
    python_code = irmodule_to_python(CallTIRModule)
    
    # Check that the generated code contains expected elements
    assert "def main(" in python_code
    assert "call_tir" in python_code
    assert "n = x.shape[0]" in python_code


def test_relax_function_with_call_dps_packed():
    """Test converting Relax function with call_dps_packed."""
    
    @I.ir_module
    class CallDPSModule:
        @R.function
        def main(x: R.Tensor(("n", 20), "float32")) -> R.Tensor(("n", 20), "float32"):
            n = T.int64()
            lv = R.call_dps_packed("my_softmax", [x, R.prim_value(1)], out_sinfo=R.Tensor((n, 20), "float32"))
            return lv
    
    # Convert to Python
    python_code = irmodule_to_python(CallDPSModule)
    
    # Check that the generated code contains expected elements
    assert "def main(" in python_code
    assert "call_dps_packed" in python_code
    assert "n = x.shape[0]" in python_code


def test_complex_relax_function():
    """Test converting a complex Relax function with multiple operations."""
    
    @I.ir_module
    class ComplexModule:
        @R.function
        def complex_forward(x: R.Tensor(("n", 64), "float32"), w1: R.Tensor((64, 128), "float32"), w2: R.Tensor((128, 64), "float32")) -> R.Tensor(("n", 64), "float32"):
            n = T.int64()
            lv1 = R.matmul(x, w1)
            lv2 = R.nn.relu(lv1)
            lv3 = R.matmul(lv2, w2)
            lv4 = R.nn.relu(lv3)
            return lv4
    
    # Convert to Python
    python_code = irmodule_to_python(ComplexModule)
    
    # Check that the generated code contains expected elements
    assert "def complex_forward(" in python_code
    assert "torch.matmul" in python_code
    assert "F.relu" in python_code
    assert "n = x.shape[0]" in python_code


def test_relax_function_with_shape_operations():
    """Test converting Relax function with shape operations."""
    
    @I.ir_module
    class ShapeModule:
        @R.function
        def shape_ops(x: R.Tensor(("n", "c", "h", "w"), "float32")) -> R.Tensor(("n", "c"), "float32"):
            lv = R.add(x, x)
            lv1 = R.mean(lv, axis=[1], keepdims=False)
            return lv1
    
    # Convert to Python
    python_code = irmodule_to_python(ShapeModule)
    
    # Check that the generated code contains expected elements
    assert "def shape_ops(" in python_code
    assert "torch.add" in python_code
    assert "torch.mean" in python_code or "F.mean" in python_code
    # Check that symbolic shapes are handled
    assert "n = x.shape[0]" in python_code
    assert "c = x.shape[1]" in python_code
    assert "h = x.shape[2]" in python_code
    assert "w = x.shape[3]" in python_code


def test_python_function_preserved():
    """Test that @I.pyfunc decorated functions are preserved as-is."""
    
    @I.ir_module
    class PyFuncModule:
        @I.py_func
        def my_python_func(x: torch.Tensor) -> torch.Tensor:
            return x + 1
        
        @R.function
        def relax_func(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
            return x
    
    # Convert to Python
    python_code = irmodule_to_python(PyFuncModule)
    
    # Check that Python function is preserved
    assert "@I.py_func" in python_code
    assert "def my_python_func(" in python_code
    assert "return x + 1" in python_code


def test_mixed_module():
    """Test converting a module with both Relax and Python functions."""
    
    @I.ir_module
    class MixedModule:
        @I.py_func
        def identity(x: torch.Tensor) -> torch.Tensor:
            return x
        
        @R.function
        def main(x: R.Tensor(("n", 64), "float32")) -> R.Tensor(("n", 64), "float32"):
            n = T.int64()
            lv = R.nn.relu(x)
            return lv
    
    # Convert to Python
    python_code = irmodule_to_python(MixedModule)
    
    # Check that both function types are handled correctly
    assert "@I.py_func" in python_code
    assert "def identity(" in python_code
    assert "def main(" in python_code
    assert "F.relu" in python_code
    assert "n = x.shape[0]" in python_code


def test_python_printer_imports():
    """Test that the generated Python code has the correct imports."""
    
    @I.ir_module
    class ImportTestModule:
        @R.function
        def test_func(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
            return x
    
    # Convert to Python
    python_code = irmodule_to_python(ImportTestModule)
    
    # Check that necessary imports are included
    assert "import torch" in python_code
    assert "import torch.nn.functional as F" in python_code
    assert "from tvm import relax as R" in python_code
    assert "from tvm.script import ir as I" in python_code


def test_python_printer_helper_functions():
    """Test that helper functions are generated when needed."""
    
    @I.ir_module
    class HelperTestModule:
        @R.function
        def test_func(x: R.Tensor(("n", 64), "float32")) -> R.Tensor(("n", 64), "float32"):
            n = T.int64()
            lv = R.call_tir("my_func", [x], out_sinfo=R.Tensor((n, 64), "float32"))
            return lv
    
    # Convert to Python
    python_code = irmodule_to_python(HelperTestModule)
    
    # Check that helper functions are generated
    assert "def call_tir(" in python_code
    assert "def call_dps_packed(" in python_code
    assert "import tvm" in python_code


if __name__ == "__main__":
    pytest.main([__file__])
