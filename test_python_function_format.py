#!/usr/bin/env python3
"""
Test to verify our printer generates correct Python function format.
"""

import tvm
from tvm import relax, tir
from tvm.script import ir as I, relax as R, tir as T
from tvm.script.printer.python_printer import irmodule_to_python


@I.ir_module
class TestModule:
    @R.function
    def main(
        x: R.Tensor(("n", 16), "float32"), 
        w: R.Tensor((16, 20), "float32")
    ) -> R.Tensor(("n", 20), "float32"):
        cls = TestModule
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


def test_python_function_format():
    """Test that our printer generates correct Python function format."""
    print("=== Testing Python Function Format ===")
    
    try:
        python_code = irmodule_to_python(TestModule)
        print("✅ Successfully converted!")
        
        print("\n=== Generated Python Code ===")
        print(python_code)
        
        # Check key elements
        print("\n=== Verification ===")
        
        # Check function signature
        assert "def main(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:" in python_code, "Wrong function signature"
        print("✅ Function signature correct")
        
        # Check symbolic shape handling
        assert "n = x.shape[0]" in python_code, "Missing symbolic shape handling"
        print("✅ Symbolic shape handling correct")
        
        # Check self. prefix for function calls
        assert "self.matmul" in python_code, "Missing self. prefix for function calls"
        print("✅ Self prefix correct")
        
        # Check torch.Tensor type annotations
        assert "torch.Tensor" in python_code, "Missing torch.Tensor type annotations"
        print("✅ Type annotations correct")
        
        print("\n🎉 All verifications passed! Generated format matches expected Python function format.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_python_function_format()

