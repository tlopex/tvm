#!/usr/bin/env python3
"""
Demo: Python Printer for IRModules with Python Functions

This example demonstrates how to use the Python printer to convert TVM IRModules
containing Relax functions into executable Python code that can run directly with PyTorch.
"""

import tvm
from tvm import relax, tir
from tvm.script import ir as I, relax as R, tir as T
from tvm.script.printer.python_printer import irmodule_to_python, print_irmodule_as_python


def demo_simple_conversion():
    """Demo: Convert a simple Relax function to Python."""
    print("=== Demo: Simple Relax Function Conversion ===\n")
    
    @I.ir_module
    class SimpleModule:
        @R.function
        def simple_add(x: R.Tensor((5,), "float32"), 
                      y: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
            return R.add(x, y)
    
    print("Original TVMScript:")
    print(SimpleModule.script())
    print("\n" + "="*50 + "\n")
    
    print("Generated Python Code:")
    python_code = irmodule_to_python(SimpleModule)
    print(python_code)


def demo_neural_network_conversion():
    """Demo: Convert a neural network Relax function to Python."""
    print("\n=== Demo: Neural Network Function Conversion ===\n")
    
    @I.ir_module
    class NNModule:
        @R.function
        def nn_forward(x: R.Tensor(("n", 64), "float32"), 
                      w: R.Tensor((64, 128), "float32")) -> R.Tensor(("n", 128), "float32"):
            lv = R.nn.linear(x, w)
            lv1 = R.nn.relu(lv)
            return lv1
    
    print("Original TVMScript:")
    print(NNModule.script())
    print("\n" + "="*50 + "\n")
    
    print("Generated Python Code:")
    python_code = irmodule_to_python(NNModule)
    print(python_code)


def demo_call_tir_conversion():
    """Demo: Convert Relax function with call_tir to Python."""
    print("\n=== Demo: Call TIR Function Conversion ===\n")
    
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
            return R.call_tir(add_tir, (x, y), R.Tensor((5,), "float32"))
    
    print("Original TVMScript:")
    print(CallTIRModule.script())
    print("\n" + "="*50 + "\n")
    
    print("Generated Python Code:")
    python_code = irmodule_to_python(CallTIRModule)
    print(python_code)


def demo_complex_function_conversion():
    """Demo: Convert a complex Relax function to Python."""
    print("\n=== Demo: Complex Function Conversion ===\n")
    
    @I.ir_module
    class ComplexModule:
        @R.function
        def complex_forward(x: R.Tensor(("n", 64), "float32"), 
                           w1: R.Tensor((64, 128), "float32"),
                           w2: R.Tensor((128, 64), "float32")) -> R.Tensor(("n", 64), "float32"):
            # First layer
            lv = R.nn.linear(x, w1)
            lv1 = R.nn.relu(lv)
            lv2 = R.nn.dropout(lv1, 0.1)
            
            # Second layer
            lv3 = R.nn.linear(lv2, w2)
            lv4 = R.nn.sigmoid(lv3)
            
            return lv4
    
    print("Original TVMScript:")
    print(ComplexModule.script())
    print("\n" + "="*50 + "\n")
    
    print("Generated Python Code:")
    python_code = irmodule_to_python(ComplexModule)
    print(python_code)


def demo_execute_generated_code():
    """Demo: Execute the generated Python code."""
    print("\n=== Demo: Execute Generated Python Code ===\n")
    
    @I.ir_module
    class ExecutableModule:
        @R.function
        def simple_forward(x: R.Tensor((3, 5), "float32"), 
                          w: R.Tensor((5, 2), "float32")) -> R.Tensor((3, 2), "float32"):
            lv = R.nn.linear(x, w)
            lv1 = R.nn.relu(lv)
            return lv1
    
    # Generate Python code
    python_code = irmodule_to_python(ExecutableModule)
    print("Generated Python Code:")
    print(python_code)
    print("\n" + "="*50 + "\n")
    
    # Execute the generated code
    print("Executing generated code...")
    
    # Create a namespace for execution
    namespace = {}
    
    # Execute the generated code
    exec(python_code, namespace)
    
    # Test the generated function
    import torch
    
    # Create test data
    x = torch.randn(3, 5, dtype=torch.float32)
    w = torch.randn(5, 2, dtype=torch.float32)
    
    print(f"Input x shape: {x.shape}")
    print(f"Input w shape: {w.shape}")
    
    # Call the generated function
    result = namespace['simple_forward'](x, w)
    print(f"Output shape: {result.shape}")
    print(f"Output dtype: {result.dtype}")
    
    print("✅ Successfully executed generated Python code!")


def main():
    """Main demo function."""
    print("TVM Python Printer Demo")
    print("=" * 50)
    print("This demo shows how to convert TVM IRModules to executable Python code.")
    print()
    
    # Run all demos
    demo_simple_conversion()
    demo_neural_network_conversion()
    demo_call_tir_conversion()
    demo_complex_function_conversion()
    demo_execute_generated_code()
    
    print("\n" + "="*50)
    print("Demo completed successfully!")
    print("\nKey Features:")
    print("✅ Converts Relax functions to Python functions")
    print("✅ Maps Relax operators to PyTorch APIs")
    print("✅ Handles symbolic shapes (e.g., n = x.shape[0])")
    print("✅ Generates helper functions for call_tir and call_dps_packed")
    print("✅ Produces executable Python code")


if __name__ == "__main__":
    main()





