import torch
import torch.nn.functional as F

import tvm
from tvm import relax, tir
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.relax.base_py_module import BasePyModule


@I.ir_module
class IRModuleWithPyFunc(BasePyModule):
    """Example IRModule with Python function.
    The base class BasePyModule implements the logic of cross-function calls
    and JIT compilation in Python.
    We only allow Python functions in IRModules that subclass the BasePyModule.
    """

    @I.pyfunc
    def main(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        lv = self.call_tir(self.matmul, [x, w], out_sinfo=R.Tensor((n, 20), "float32"))
        lv1 = F.relu(lv)
        lv2 = self.call_dps_packed("my_softmax", [lv1, 1], out_sinfo=R.Tensor((n, 20), "float32"))
        lv3 = self.my_identity_func(lv2)
        gv = lv3
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

    @R.function
    def my_identity_func(x: R.Tensor(("n", 20), "float32")) -> R.Tensor(("n", 20), "float32"):
        return x

    # @R.function
    # def my_relax_func(
    #     x: R.Tensor(("n", 16), "float32"), w: R.Tensor((16, 20), "float32")
    # ) -> R.Tensor(("n", 20), "float32"):
    #     cls = IRModuleWithPyFunc
    #     n = T.int64()
    #     with R.dataflow():
    #         lv = R.call_py_func(cls.main)
    #     return x


def main():
    mod = IRModuleWithPyFunc
    print(mod.script())


if __name__ == "__main__":
    main()