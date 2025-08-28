#!/usr/bin/env python3
"""
Test helper function detection in Python printer.
"""

import tvm
from tvm import relax, tir
from tvm.script import ir as I, relax as R, tir as T
from tvm.script.printer.python_printer import irmodule_to_python


def test_simple_module_no_helpers():
    """Test a simple module that doesn't need helpers."""
    print("=== Testing Simple Module (No Helpers) ===")
    
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
    
    # Check that no helper functions are generated
    assert "def call_tir(" not in python_code
    assert "def call_dps_packed(" not in python_code
    print("✅ Simple module correctly generated without helpers!")


def test_module_with_call_tir():
    """Test a module that needs call_tir helper."""
    print("\n=== Testing Module with Call TIR ===")
    
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
            result = R.call_tir(add_tir, [x, y], out_sinfo=R.Tensor((5,), "float32"))
            return result
    
    # Convert to Python
    python_code = irmodule_to_python(CallTIRModule)
    print("Generated code:")
    print(python_code)
    
    # Check that helper functions are generated
    assert "def call_tir(" in python_code
    print("✅ Module with call_tir correctly generated with helpers!")


if __name__ == "__main__":
    test_simple_module_no_helpers()
    test_module_with_call_tir()
    print("\n🎉 All helper detection tests passed!")

