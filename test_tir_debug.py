#!/usr/bin/env python3
"""
简单的TIR函数测试，用于调试RelaxToPyFuncConverter
"""

import torch
import tvm
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R
from tvm.relax.relax_to_pyfunc_converter import RelaxToPyFuncConverter

@I.ir_module
class TestModule:
    @T.prim_func
    def add_tir(var_x: T.handle, var_y: T.handle, var_out: T.handle):
        x = T.match_buffer(var_x, (5,), 'float32')
        y = T.match_buffer(var_y, (5,), 'float32')
        out = T.match_buffer(var_out, (5,), 'float32')
        for i in range(5):
            out[i] = x[i] + y[i]

    @R.function
    def test_func(x: R.Tensor((5,), 'float32'), y: R.Tensor((5,), 'float32')) -> R.Tensor((5,), 'float32'):
        return R.call_tir(TestModule.add_tir, (x, y), out_sinfo=R.Tensor((5,), 'float32'))

def test_tir_function():
    print("=" * 60)
    print("测试TIR函数转换")
    print("=" * 60)
    
    # 创建转换器
    converter = RelaxToPyFuncConverter(TestModule)
    converted_ir_mod = converter.convert(['test_func'])
    
    # 准备测试数据
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    y = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    
    print(f"输入 x: {x}")
    print(f"输入 y: {y}")
    print(f"期望结果: {torch.add(x, y)}")
    
    # 测试转换后的函数
    print("\n调用转换后的函数...")
    try:
        result = converted_ir_mod.pyfuncs['test_func'](x, y)
        print(f"实际结果: {result}")
        print(f"结果匹配: {torch.allclose(result, torch.add(x, y))}")
        
        # 检查结果是否全零
        if torch.allclose(result, torch.zeros_like(result)):
            print("⚠️  警告：结果全为零，可能TIR函数执行失败")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

def test_dps_packed():
    print("\n" + "=" * 60)
    print("测试call_dps_packed")
    print("=" * 60)
    
    # 注册一个简单的packed函数
    def mock_softmax(x, axis):
        print(f"Debug: mock_softmax called with x={x}, axis={axis}")
        return x  # 简单返回输入
    
    tvm.register_global_func("mock_softmax", mock_softmax)
    
    @I.ir_module
    class TestModule2:
        @R.function
        def test_dps(x: R.Tensor((5,), 'float32')) -> R.Tensor((5,), 'float32'):
            return R.call_dps_packed(
                "mock_softmax", (x, R.prim_value(1)), out_sinfo=R.Tensor((5,), 'float32')
            )
    
    converter = RelaxToPyFuncConverter(TestModule2)
    converted_ir_mod = converter.convert(['test_dps'])
    
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    print(f"输入 x: {x}")
    print(f"期望结果: {x}")
    
    try:
        result = converted_ir_mod.pyfuncs['test_dps'](x)
        print(f"实际结果: {result}")
        print(f"结果匹配: {torch.allclose(result, x)}")
        
        # 检查结果是否全零
        if torch.allclose(result, torch.zeros_like(result)):
            print("⚠️  警告：结果全为零，可能packed函数执行失败")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tir_function()
    test_dps_packed()
