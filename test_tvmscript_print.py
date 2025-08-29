#!/usr/bin/env python3
"""Test how TVMScript prints ordinary Relax functions."""

import tvm
from tvm.script import relax as R
from tvm.script import ir as I
from tvm.script import tir as T


@I.ir_module
class OrdinaryModule:
    @R.function
    def my_relax_func(
        x: R.Tensor(("n", 16), "float32"), w: R.Tensor((16, 20), "float32")
    ) -> R.Tensor(("n", 20), "float32"):
        n = T.int64()
        with R.dataflow():
            # Use matrix multiplication instead of addition
            lv = R.matmul(x, w)
            lv1 = R.nn.relu(lv)
            R.output(lv1)
        return lv1


def test_tvmscript_print():
    """Test how TVMScript prints the module."""
    print("=== Testing TVMScript Print ===")
    print("Original IRModule:")
    print(OrdinaryModule)
    
    print("\n=== Testing Relax Function Print ===")
    relax_func = OrdinaryModule["my_relax_func"]
    print("Relax function:")
    print(relax_func)
    
    print("\n=== Testing Module Structure ===")
    print("Functions in module:")
    for name, func in OrdinaryModule.functions.items():
        print(f"  {name}: {type(func)}")
        if hasattr(func, 'attrs'):
            print(f"    attrs: {func.attrs}")
    
    print("\n=== Testing Function Body Details ===")
    # Get the Relax function body
    relax_func = OrdinaryModule["my_relax_func"]
    if hasattr(relax_func, 'body'):
        body = relax_func.body
        print(f"Function body type: {type(body)}")
        print(f"Function body: {body}")
        
        # Look for operations in the body
        if hasattr(body, 'blocks'):
            for block in body.blocks:
                if hasattr(block, 'bindings'):
                    for binding in block.bindings:
                        if hasattr(binding, 'value'):
                            value = binding.value
                            print(f"Binding value type: {type(value)}")
                            print(f"Binding value: {value}")
                            
                            # Check operation details
                            if hasattr(value, 'op') and hasattr(value.op, 'name'):
                                print(f"Operation name: {value.op.name}")
                                print(f"Arguments: {value.args}")
                                if hasattr(value, 'out_sinfo'):
                                    print(f"Output info: {value.out_sinfo}")


if __name__ == "__main__":
    test_tvmscript_print()

