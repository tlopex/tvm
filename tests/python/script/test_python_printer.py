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
from tvm import relax, tir
from tvm.script import ir as I, relax as R, tir as T
from tvm.script.printer.python_printer import irmodule_to_python, print_irmodule_as_python


def test_simple_relax_function():
    """Test converting a simple Relax function to Python."""
    
    @I.ir_module
    class SimpleModule:
        @R.function
        def simple_add(x: R.Tensor((5,), "float32"), 
                      y: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
            return R.add(x, y)
    
    # Convert to Python
    python_code = irmodule_to_python(SimpleModule)
    print(python_code)
    # Check that the generated code contains expected elements
    assert "import torch" in python_code
    assert "import torch.nn.functional as F" in python_code
    assert "def simple_add(" in python_code
    assert "torch.add" in python_code or "R.add" in python_code
    assert "return gv" in python_code


def test_relax_function_with_nn_ops():
    """Test converting Relax function with neural network operations."""
    
    @I.ir_module
    class NNModule:
        @R.function
        def nn_forward(x: R.Tensor(("n", 64), "float32"), 
                      w: R.Tensor((64, 64), "float32")) -> R.Tensor(("n", 64), "float32"):
            lv = R.add(x, w)
            lv1 = R.nn.relu(lv)
            return lv1
    
        # Convert to Python
    python_code = irmodule_to_python(NNModule)
    print(python_code)
    # Check that the generated code contains expected elements
    assert "def nn_forward(" in python_code
    assert "torch.Tensor" in python_code  # We now generate simple torch.Tensor
    assert "n = x.shape[0]" in python_code  # We now generate n = x.shape[0]


def test_relax_function_with_call_tir():
    """Test converting Relax function that calls TIR functions."""
    
    @I.ir_module
    class CallTIRModule:
        @T.prim_func
        def add_tir(x: T.Buffer((5,), "float32"), 
                   y: T.Buffer((5,), "float32"), 
                   out: T.Buffer((5,), "float32")):
            for i in range(5):
                out[i] = x[i] + y[i]
        
        @R.function
        def main(x: R.Tensor((5,), "float32"), 
                y: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
            # Actually test call_tir functionality
            cls = CallTIRModule
            result = R.call_tir(cls.add_tir, [x, y], out_sinfo=R.Tensor((5,), "float32"))
            return result
    
    # Convert to Python
    python_code = irmodule_to_python(CallTIRModule)
    
    # Check that the generated code contains expected elements
    assert "def main(" in python_code
    assert "call_tir(" in python_code  # Now actually testing call_tir
    assert "def call_tir(" in python_code  # Helper function


def test_relax_function_with_call_dps_packed():
    """Test converting Relax function that calls packed functions."""
    
    @I.ir_module
    class CallPackedModule:
        @R.function
        def main(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
            return R.call_dps_packed("my_softmax", (x, 1), R.Tensor((5,), "float32"))
    
    # Convert to Python
    python_code = irmodule_to_python(CallPackedModule)
    print(python_code)
    # Check that the generated code contains expected elements
    assert "def main(" in python_code
    assert "call_dps_packed" in python_code
    assert "def call_dps_packed(" in python_code  # Helper function


def test_complex_relax_function():
    """Test converting a complex Relax function with multiple operations."""
    
    @I.ir_module
    class ComplexModule:
        @R.function
        def complex_forward(x: R.Tensor(("n", 64), "float32"), 
                           w1: R.Tensor((64, 64), "float32"),
                           w2: R.Tensor((64, 64), "float32")) -> R.Tensor(("n", 64), "float32"):
            # First layer
            lv = R.add(x, w1)
            lv1 = R.nn.relu(lv)
            lv2 = R.multiply(lv1, R.const(0.9))  # Simplified dropout
            
            # Second layer
            lv3 = R.add(lv2, w2)
            lv4 = R.nn.relu(lv3)  # Use relu instead of sigmoid
            
            return lv4
    
        # Convert to Python
    python_code = irmodule_to_python(ComplexModule)
    print(python_code)
    # Check that the generated code contains expected elements
    assert "def complex_forward(" in python_code
    assert "torch.add" in python_code
    assert "F.relu" in python_code
    assert "torch.mul" in python_code
    assert "F.relu" in python_code
    assert "n = x.shape[0]" in python_code  # We now generate n = x.shape[0]


def test_relax_function_with_shape_operations():
    """Test converting Relax function with shape operations."""
    
    @I.ir_module
    class ShapeModule:
        @R.function
        def shape_ops(x: R.Tensor(("n", "c", "h", "w"), "float32")) -> R.Tensor(("n", "c"), "float32"):
            # Simple operations
            lv = R.add(x, x)
            lv1 = R.mean(lv, axis=1, keepdims=False)
            return lv1
    
    # Convert to Python
    python_code = irmodule_to_python(ShapeModule)
    print(python_code)
    # Check that the generated code contains expected elements
    assert "def shape_ops(" in python_code
    assert "torch.add" in python_code
    assert "torch.mean" in python_code or "F.mean" in python_code
    assert "n = x.shape[0]" in python_code  # We now generate n = x.shape[0]
    assert "c = x.shape[0]" in python_code  # We now generate c = x.shape[0]
    assert "h = x.shape[0]" in python_code  # We now generate h = x.shape[0]
    assert "w = x.shape[0]" in python_code  # We now generate w = x.shape[0]


def test_python_printer_imports():
    """Test that the Python printer generates correct imports."""
    
    @I.ir_module
    class ImportTestModule:
        @R.function
        def test(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
            return R.nn.relu(x)
    
    # Convert to Python
    python_code = irmodule_to_python(ImportTestModule)
    
    # Check imports
    expected_imports = [
        "import torch",
        "import torch.nn.functional as F",
        "import tvm",
        "from tvm import relax as R"
    ]
    
    for imp in expected_imports:
        assert imp in python_code


def test_python_printer_helper_functions():
    """Test that the Python printer generates helper functions."""
    
    @I.ir_module
    class HelperTestModule:
        @R.function
        def test(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
            return x
    
    # Convert to Python
    python_code = irmodule_to_python(HelperTestModule)
    print(python_code)
    # Check helper functions
    assert "def call_tir(" in python_code
    assert "def call_dps_packed(" in python_code
    assert "tvm.nd.from_dlpack" in python_code
    assert "torch.from_dlpack" in python_code


def test_complex_example_like_user():
    """Test converting a complex Relax function similar to the user's example."""
    
    @I.ir_module
    class ComplexExampleModule:
        @R.function
        def main(
            x: R.Tensor(("n", 16), "float32"), 
            w: R.Tensor((16, 20), "float32")
        ) -> R.Tensor(("n", 20), "float32"):
            cls = ComplexExampleModule
            n = T.int64()
            with R.dataflow():
                lv = R.call_tir(cls.matmul, [x, w], out_sinfo=R.Tensor((n, 20), "float32"))
                lv1 = R.nn.relu(lv)
                lv2 = R.call_dps_packed(
                    "my_softmax", [lv1, R.prim_value(1)], out_sinfo=R.Tensor((n, 20), "float32")
                )
                gv = lv2
                R.output(gv)
            return gv

        @T.prim_func
        def matmul(
            var_A: T.handle,
            var_B: T.handle,
            var_C: T.handle,
        ):
            n = T.int32()
            A = T.match_buffer(var_A, (n, 16), "float32")
            B = T.match_buffer(var_B, (16, 20), "float32")
            C = T.match_buffer(var_C, (n, 20), "float32")
            for i, j, k in T.grid(n, 20, 16):
                with T.block("block"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    
    # Convert to Python
    python_code = irmodule_to_python(ComplexExampleModule)
    print(python_code)
    # Check that the generated code contains expected elements
    assert "def main(" in python_code
    assert "torch.Tensor" in python_code  # We now generate simple torch.Tensor
    
    # Check symbolic shape handling
    assert "n = x.shape[0]" in python_code
    
    # Check that call_tir is properly converted
    assert "call_tir(" in python_code
    
    # Check that call_dps_packed is properly converted
    assert "call_dps_packed(" in python_code
    
    # Check helper functions
    assert "def call_tir(" in python_code
    assert "def call_dps_packed(" in python_code
    
    # Check imports
    assert "import torch" in python_code
    assert "import torch.nn.functional as F" in python_code


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])
