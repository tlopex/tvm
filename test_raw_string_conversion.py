#!/usr/bin/env python3
"""
Test raw string conversion functionality.
"""

from tvm.script.printer.python_printer import PythonPrinter


def test_raw_string_conversion():
    """Test converting raw strings with @I.pyfunc decorator."""
    printer = PythonPrinter()
    
    # Test case 1: Raw string with @I.pyfunc decorator
    raw_string = '''    @I.pyfunc
    def main(self, x: "torch.Tensor", w: "torch.Tensor") -> "torch.Tensor":
        n = x.shape[0]
        lv = self.call_tir(self.matmul, [x, w], out_sinfo=R.Tensor((n, 20), "float32"))
        lv1 = F.relu(lv)
        gv = lv1
        return gv'''
    
    result = printer._convert_raw_string_to_python(raw_string, "main")
    print("Test case 1 - Raw string with @I.pyfunc decorator:")
    print("Input:")
    print(raw_string)
    print("\nOutput:")
    print(result)
    print("\n" + "="*50 + "\n")
    
    # Test case 2: Raw string without decorator
    raw_string2 = '''def simple_func(self, x):
        return x + 1'''
    
    result2 = printer._convert_raw_string_to_python(raw_string2, "simple_func")
    print("Test case 2 - Raw string without decorator:")
    print("Input:")
    print(raw_string2)
    print("\nOutput:")
    print(result2)
    print("\n" + "="*50 + "\n")
    
    # Test case 3: Just function body
    raw_string3 = '''n = x.shape[0]
        return x + n'''
    
    result3 = printer._convert_raw_string_to_python(raw_string3, "body_func")
    print("Test case 3 - Just function body:")
    print("Input:")
    print(raw_string3)
    print("\nOutput:")
    print(result3)


if __name__ == "__main__":
    test_raw_string_conversion()
