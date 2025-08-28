#!/usr/bin/env python3
"""
Debug test for our Python printer.
Run this directly to see the generated code.
"""

import tvm
from tvm import relax, tir
from tvm.script import ir as I, relax as R, tir as T
from tvm.script.printer.python_printer import irmodule_to_python


@I.ir_module
class DebugModule:
    @R.function
    def nn_forward(x: R.Tensor(("n", 64), "float32"), 
                  w: R.Tensor((64, 64), "float32")) -> R.Tensor(("n", 64), "float32"):
        lv = R.add(x, w)
        lv1 = R.nn.relu(lv)
        return lv1


def test_printer():
    """Test our printer and show the generated code."""
    print("=== Testing Python Printer ===")
    
    try:
        # Convert to Python
        python_code = irmodule_to_python(DebugModule)
        print("✅ Successfully converted!")
        
        print("\n=== Generated Python Code ===")
        print(python_code)
        
        # Check key elements
        print("\n=== Verification ===")
        
        # Check function signature
        if "def nn_forward(" in python_code:
            print("✅ Function definition correct")
        else:
            print("❌ Missing function definition")
        
        # Check torch.Tensor type
        if "torch.Tensor" in python_code:
            print("✅ torch.Tensor type correct")
        else:
            print("❌ Missing torch.Tensor type")
        
        # Check symbolic shape handling
        if "n = x.shape[0]" in python_code:
            print("✅ Symbolic shape handling correct")
        else:
            print("❌ Missing symbolic shape handling")
            print("   Expected: n = x.shape[0]")
            print("   Found in code:")
            lines = python_code.split('\n')
            for i, line in enumerate(lines):
                if 'n =' in line or 'shape' in line:
                    print(f"   Line {i+1}: {line}")
        
        # Check torch operations
        if "torch.add" in python_code:
            print("✅ torch.add operation correct")
        else:
            print("❌ Missing torch.add operation")
        
        if "F.relu" in python_code:
            print("✅ F.relu operation correct")
        else:
            print("❌ Missing F.relu operation")
        
        print("\n=== Summary ===")
        print("The printer should generate code like:")
        print("def nn_forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:")
        print("    n = x.shape[0]")
        print("    lv = torch.add(x, w)")
        print("    lv1 = F.relu(lv)")
        print("    gv = lv1")
        print("    return gv")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_printer()

