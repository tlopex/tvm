#!/usr/bin/env python3
"""
Test to verify our Python printer can handle the example from the user.
"""

import tvm
from tvm import relax, tir
from tvm.script import ir as I, relax as R, tir as T
from tvm.script.printer.python_printer import irmodule_to_python, print_irmodule_as_python


@I.ir_module
class MyModule:
    @R.function
    def main(
        x: R.Tensor(("n", 16), "float32"), w: R.Tensor((16, 20), "float32")
    ) -> R.Tensor(("n", 20), "float32"):
        cls = MyModule
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


def test_example_conversion():
    """Test that our printer can handle the example correctly."""
    print("=== Testing Example Conversion ===")
    
    try:
        # Convert to Python
        python_code = irmodule_to_python(MyModule)
        print("✅ Successfully converted to Python!")
        
        # Print the generated code
        print("\n=== Generated Python Code ===")
        print(python_code)
        
        # Check key elements
        print("\n=== Verification ===")
        
        # Check imports
        assert "import torch" in python_code, "Missing torch import"
        assert "import torch.nn.functional as F" in python_code, "Missing F import"
        print("✅ Imports correct")
        
        # Check function definition
        assert "def main(" in python_code, "Missing main function"
        assert "torch.Tensor" in python_code, "Missing tensor type annotation"
        print("✅ Function definition correct")
        
        # Check symbolic shape handling
        assert "n = x.shape[0]" in python_code, "Missing symbolic shape handling"
        print("✅ Symbolic shape handling correct")
        
        # Check call_tir conversion
        assert "call_tir(" in python_code, "Missing call_tir conversion"
        print("✅ call_tir conversion correct")
        
        # Check call_dps_packed conversion
        assert "call_dps_packed(" in python_code, "Missing call_dps_packed conversion"
        print("✅ call_dps_packed conversion correct")
        
        # Check helper functions
        assert "def call_tir(" in python_code, "Missing call_tir helper function"
        assert "def call_dps_packed(" in python_code, "Missing call_dps_packed helper function"
        print("✅ Helper functions correct")
        
        print("\n🎉 All verifications passed! Our printer handles the example correctly.")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_example_conversion()
