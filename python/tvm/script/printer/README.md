# TVM Python Printer

The TVM Python Printer is a tool that converts TVM IRModules containing Relax functions into executable Python code that can run directly with PyTorch.

## Overview

This module implements **M2: TVMScript printer for IRModules with Python functions** from the engineering plan. It provides:

1. **Relax to PyTorch mapping**: Converts Relax operators to corresponding PyTorch APIs
2. **Symbolic shape handling**: Processes symbolic shapes like `n = x.shape[0]`
3. **Cross-function call support**: Handles `call_tir` and `call_dps_packed` operations
4. **Executable output**: Generates Python code that can be run directly

## Key Features

### High-Level Operator Mapping

The printer maps Relax operators to PyTorch equivalents:

```python
# Relax operators -> PyTorch APIs
"relax.nn.relu" -> "F.relu"
"relax.nn.linear" -> "F.linear"
"relax.add" -> "torch.add"
"relax.reshape" -> "torch.reshape"
# ... and many more
```

### Symbolic Shape Support

Handles symbolic shapes in tensor types:

```python
# Input: R.Tensor(("n", 64), "float32")
# Output: torch.Tensor[n, 64, float32]
# Generated: # Handle symbolic shape: n
```

### Cross-Function Call Support

Generates helper functions for:
- `call_tir`: Calls TIR functions with PyTorch tensors
- `call_dps_packed`: Calls packed functions with PyTorch tensors

## Usage

### Basic Usage

```python
from tvm.script.printer.python_printer import irmodule_to_python

# Convert IRModule to Python code
python_code = irmodule_to_python(my_irmodule)
print(python_code)
```

### Example: Simple Conversion

```python
@I.ir_module
class SimpleModule:
    @R.function
    def simple_add(x: R.Tensor((5,), "float32"), 
                  y: R.Tensor((5,), "float32")) -> R.Tensor((5,), "float32"):
        return R.add(x, y)

# Convert to Python
python_code = irmodule_to_python(SimpleModule)
```

Generated Python code:
```python
# Generated Python code from TVM IRModule
import torch
import torch.nn.functional as F
import tvm
from tvm import relax as R

# Helper functions for cross-function calls

def call_tir(func_name: str, args: List[torch.Tensor], out_sinfo) -> torch.Tensor:
    """Call a TIR function with PyTorch tensors."""
    # Convert PyTorch tensors to TVM NDArrays via DLPack
    tvm_args = [tvm.nd.from_dlpack(torch.to_dlpack(arg)) for arg in args]
    
    # Call the TIR function
    result = tvm.get_global_func(func_name)(*tvm_args)
    
    # Convert result back to PyTorch tensor
    return torch.from_dlpack(result.to_dlpack())

def call_dps_packed(func_name: str, args: List[torch.Tensor], out_sinfo) -> torch.Tensor:
    """Call a packed function with PyTorch tensors."""
    # Get the packed function
    packed_func = tvm.get_global_func(func_name)
    
    # Convert PyTorch tensors to TVM NDArrays via DLPack
    tvm_args = [tvm.nd.from_dlpack(torch.to_dlpack(arg)) for arg in args]
    
    # Call the packed function
    result = packed_func(*tvm_args)
    
    # Convert result back to PyTorch tensor
    return torch.from_dlpack(result.to_dlpack())

def simple_add(x: torch.Tensor[5, float32], y: torch.Tensor[5, float32]) -> torch.Tensor[5, float32]:
    gv = torch.add(x, y)
    return gv
```

### Example: Neural Network Function

```python
@I.ir_module
class NNModule:
    @R.function
    def nn_forward(x: R.Tensor(("n", 64), "float32"), 
                  w: R.Tensor((64, 128), "float32")) -> R.Tensor(("n", 128), "float32"):
        lv = R.nn.linear(x, w)
        lv1 = R.nn.relu(lv)
        return lv1
```

Generated Python code:
```python
def nn_forward(x: torch.Tensor[n, 64, float32], w: torch.Tensor[64, 128, float32]) -> torch.Tensor[n, 128, float32]:
    # Handle symbolic shape: n
    gv = F.linear(x, w)
    gv = F.relu(gv)
    return gv
```

### Example: Function with call_tir

```python
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
```

Generated Python code:
```python
def main(x: torch.Tensor[5, float32], y: torch.Tensor[5, float32]) -> torch.Tensor[5, float32]:
    gv = call_tir("add_tir", [x, y], R.Tensor((5,), "float32"))
    return gv
```

## API Reference

### Main Functions

- `irmodule_to_python(mod: IRModule) -> str`: Convert IRModule to Python code string
- `print_irmodule_as_python(mod: IRModule) -> None`: Print IRModule as Python code

### Classes

- `RelaxToPyTorchMapper`: Maps Relax operators to PyTorch APIs
- `PythonPrinter`: Main printer class that converts IRModules to Python code

## Supported Operations

### Neural Network Operations
- `R.nn.relu`, `R.nn.gelu`, `R.nn.silu`, `R.nn.sigmoid`, `R.nn.tanh`
- `R.nn.linear`, `R.nn.conv2d`, `R.nn.avg_pool2d`, `R.nn.max_pool2d`
- `R.nn.batch_norm`, `R.nn.layer_norm`, `R.nn.dropout`

### Basic Operations
- `R.add`, `R.subtract`, `R.multiply`, `R.divide`
- `R.maximum`, `R.minimum`, `R.power`

### Shape Operations
- `R.reshape`, `R.transpose`, `R.permute_dims`
- `R.expand_dims`, `R.squeeze`, `R.concat`, `R.split`

### Mathematical Functions
- `R.exp`, `R.log`, `R.sqrt`, `R.sin`, `R.cos`
- `R.abs`, `R.floor`, `R.ceil`, `R.round`

### Reduction Operations
- `R.sum`, `R.mean`, `R.max`, `R.min`

## Limitations

1. **Complex control flow**: While loops and complex conditionals may not be fully supported
2. **Advanced Relax features**: Some advanced Relax features may fall back to TVM calls
3. **Custom operators**: Custom Relax operators will use fallback TVM calls

## Future Enhancements

1. **Back-tracing**: Convert Python functions back to Relax functions
2. **FX integration**: Better integration with PyTorch FX for tracing
3. **More operators**: Expand the operator mapping coverage
4. **Optimization**: Generate more optimized Python code

## Testing

Run the test suite:

```bash
python -m pytest tests/python/script/test_python_printer.py
```

## Demo

See `examples/python_printer_demo.py` for comprehensive usage examples.





