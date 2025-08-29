import tvm
from tvm import relax, tir
from tvm.relax.base_py_module import BasePyModule
from tvm.script import ir as I, relax as R, tir as T
from tvm.runtime import Device
import torch


@I.ir_module
class IRModuleWithPyFunc(BasePyModule):
    """Example IRModule with Python function.
    The base class BasePyModule implements the logic of cross-function calls
    and JIT compilation in Python.
    We only allow Python functions in IRModules that subclass the BasePyModule.
    """

    @I.pyfunc
    def python_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Python function that can be called from Relax functions."""
        # Convert inputs to TVM NDArrays via DLPack
        x_tvm = self._convert_pytorch_to_tvm(x)
        y_tvm = self._convert_pytorch_to_tvm(y)
        
        # Call the compiled TIR function
        result = self.call_tir(self.add_tir, [x_tvm, y_tvm], 
                             out_sinfo=R.Tensor((5,), "float32"))
        
        k = self.main_relax(x_tvm, y_tvm)
        result = k + result
        # Convert result back to original format
        return self._convert_tvm_to_pytorch(result)

    @T.prim_func
    def add_tir(
        var_x: T.handle,
        var_y: T.handle,
        var_out: T.handle,
    ):
        x = T.match_buffer(var_x, (5,), "float32")
        y = T.match_buffer(var_y, (5,), "float32")
        out = T.match_buffer(var_out, (5,), "float32")
        
        for i in range(5):
            out[i] = x[i] + y[i]

    @R.function
    def main_relax(x: R.Tensor((5,), "float32"), 
                   y: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
        return R.add(x, y)


def main():
    """Main function showing IRModule with Python function support."""
    # Create the IRModuleWithPyFunc instance
    module = IRModuleWithPyFunc()
    
    # Execute DLPack conversion
    x_torch = torch.randn(5, dtype=torch.float32)
    y_torch = torch.randn(5, dtype=torch.float32)
    
    # Convert via DLPack
    x_tvm = module._convert_pytorch_to_tvm(x_torch)
    y_tvm = module._convert_pytorch_to_tvm(y_torch)
    
    # Convert back
    x_back = module._convert_tvm_to_pytorch(x_tvm)
    y_back = module._convert_tvm_to_pytorch(y_tvm)
    
    # Execute cross-function calls
    tir_result = module.call_tir("add_tir", [x_torch, y_torch], 
                                out_sinfo=R.Tensor((5,), "float32"))
    relax_result = module.main_relax(x_torch, y_torch)
    python_result = module.python_add(x_torch, y_torch)
    
    return module, (x_torch, y_torch, x_tvm, y_tvm, x_back, y_back), (tir_result, relax_result, python_result)


if __name__ == "__main__":
    main()
    print(main())
