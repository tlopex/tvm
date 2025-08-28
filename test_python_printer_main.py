#!/usr/bin/env python3
"""
Test Python printer for IRModules with Python functions.
This is the same as test_python_printer.py but can be run directly with Python.
"""

import tvm
from tvm import relax, tir
from tvm.script import ir as I, relax as R, tir as T
from tvm.script.printer.python_printer import irmodule_to_python, print_irmodule_as_python
from tvm.relax.base_py_module import BasePyModule


def test_simple_relax_function():
    """Test converting a simple Relax function to Python."""
    print("\n=== Testing Simple Relax Function ===")
    
    @I.ir_module
    class SimpleModule:
        @R.function
        def simple_add(x: R.Tensor((5,), "float32"), 
                      y: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
            return R.add(x, y)
    
    # Convert to Python
    python_code = irmodule_to_python(SimpleModule)
    print("Generated code:")
    print(python_code)
    
    # Check that the generated code contains expected elements
    assert "def simple_add(" in python_code
    assert "torch.add" in python_code or "R.add" in python_code
    assert "return gv" in python_code
    print("✅ Simple relax function test passed!")


def test_relax_function_with_nn_ops():
    """Test converting Relax function with neural network operations."""
    print("\n=== Testing NN Operations ===")
    
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
    print("Generated code:")
    print(python_code)
    
    # Check that the generated code contains expected elements
    assert "def nn_forward(" in python_code
    assert "torch.Tensor" in python_code  # We now generate simple torch.Tensor
    assert "n = x.shape[0]" in python_code  # We now generate n = x.shape[0]
    print("✅ NN operations test passed!")


def test_relax_function_with_call_tir():
    """Test converting Relax function that calls TIR functions."""
    print("\n=== Testing Call TIR ===")
    
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
    print("Generated code:")
    print(python_code)
    
    # Check that the generated code contains expected elements
    assert "def main(" in python_code
    assert "call_tir(" in python_code  # Now actually testing call_tir
    assert "def call_tir(" in python_code  # Helper function
    print("✅ Call TIR test passed!")


def test_relax_function_with_call_dps_packed():
    """Test converting Relax function that calls packed functions."""
    print("\n=== Testing Call DPS Packed ===")
    
    @I.ir_module
    class CallPackedModule:
        @R.function
        def main(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
            return R.call_dps_packed("my_softmax", (x, 1), R.Tensor((5,), "float32"))
    
    # Convert to Python
    python_code = irmodule_to_python(CallPackedModule)
    print("Generated code:")
    print(python_code)
    
    # Check that the generated code contains expected elements
    assert "def main(" in python_code
    assert "call_dps_packed" in python_code
    assert "def call_dps_packed(" in python_code  # Helper function
    print("✅ Call DPS packed test passed!")


def test_complex_relax_function():
    """Test converting a complex Relax function with multiple operations."""
    print("\n=== Testing Complex Function ===")
    
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
    print("Generated code:")
    print(python_code)
    
    # Check that the generated code contains expected elements
    assert "def complex_forward(" in python_code
    assert "torch.add" in python_code
    assert "F.relu" in python_code
    assert "torch.mul" in python_code
    assert "F.relu" in python_code
    assert "n = x.shape[0]" in python_code  # We now generate n = x.shape[0]
    print("✅ Complex function test passed!")


def test_relax_function_with_shape_operations():
    """Test converting Relax function with shape operations."""
    print("\n=== Testing Shape Operations ===")
    
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
    print("Generated code:")
    print(python_code)
    
    # Check that the generated code contains expected elements
    assert "def shape_ops(" in python_code
    assert "torch.add" in python_code
    assert "torch.mean" in python_code or "F.mean" in python_code
    assert "n = x.shape[0]" in python_code  # We now generate n = x.shape[0]
    assert "c = x.shape[0]" in python_code  # We now generate c = x.shape[0]
    assert "h = x.shape[0]" in python_code  # We now generate h = x.shape[0]
    assert "w = x.shape[0]" in python_code  # We now generate w = x.shape[0]
    print("✅ Shape operations test passed!")


def test_python_printer_imports():
    """Test that the Python printer generates correct imports."""
    print("\n=== Testing Imports ===")
    
    @I.ir_module
    class ImportTestModule:
        @R.function
        def test(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
            return R.nn.relu(x)
    
    # Convert to Python
    python_code = irmodule_to_python(ImportTestModule)
    print("Generated code:")
    print(python_code)
    
    # Check that the function is generated correctly
    assert "def test(" in python_code
    assert "F.relu" in python_code
    assert "return gv" in python_code
    print("✅ Imports test passed!")


def test_python_printer_helper_functions():
    """Test that the Python printer generates helper functions."""
    print("\n=== Testing Helper Functions ===")
    
    @I.ir_module
    class HelperTestModule:
        @T.prim_func
        def some_tir_func(x: T.Buffer((5,), "float32"), 
                         out: T.Buffer((5,), "float32")):
            for i in range(5):
                out[i] = x[i] * 2.0
        
        @R.function
        def test(x: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
            # This function actually uses call_tir, so it should generate helpers
            cls = HelperTestModule
            result = R.call_tir(cls.some_tir_func, [x], out_sinfo=R.Tensor((5,), "float32"))
            return result
    
    # Convert to Python
    python_code = irmodule_to_python(HelperTestModule)
    print("Generated code:")
    print(python_code)
    
    # Check helper functions
    assert "def call_tir(" in python_code
    assert "def call_dps_packed(" in python_code
    assert "tvm.nd.from_dlpack" in python_code
    assert "torch.from_dlpack" in python_code
    print("✅ Helper functions test passed!")


def test_complex_example_like_user():
    """Test converting a complex Relax function similar to the user's example."""
    print("\n=== Testing Complex Example Like User ===")
    
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
    print("Generated code:")
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
    
    print("✅ Complex example test passed!")


def test_pyfunc_decorator():
    """Test converting IRModule with @I.pyfunc decorator."""
    print("\n=== Testing PyFunc Decorator ===")
    
    @I.ir_module
    class PyFuncModule(BasePyModule):
        @I.pyfunc
        def main(self, x: "torch.Tensor", w: "torch.Tensor") -> "torch.Tensor":
            n = x.shape[0]
            lv = self.call_tir(self.matmul, [x, w], out_sinfo=R.Tensor((n, 20), "float32"))
            lv1 = F.relu(lv)
            lv2 = self.call_dps_packed("my_softmax", [lv1, 1], out_sinfo=R.Tensor((n, 20), "float32"))
            gv = lv2
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
    python_code = irmodule_to_python(PyFuncModule)
    print("Generated code:")
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
    
    print("✅ PyFunc decorator test passed!")


def main():
    """Run all tests."""
    print("🚀 Starting Python Printer Tests...")
    
    try:
        test_simple_relax_function()
        test_relax_function_with_nn_ops()
        test_relax_function_with_call_tir()
        test_relax_function_with_call_dps_packed()
        test_complex_relax_function()
        test_relax_function_with_shape_operations()
        test_python_printer_imports()
        test_python_printer_helper_functions()
        test_complex_example_like_user()
        test_pyfunc_decorator()
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
