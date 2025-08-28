# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE/2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Python printer for IRModules with Python functions.

This module provides functionality to convert IRModules containing Python functions
into executable Python code that can run directly with PyTorch.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import re

from tvm import ir, relax, tir
from tvm.ir import IRModule
from tvm.relax import Expr, Function, Call, Var, Constant, GlobalVar
from tvm.relax.expr import Call as RelaxCall
from tvm.relax import ExternFunc
# TVMScriptPrinter is not directly accessible, we'll use tvm.script instead


class RelaxToPyTorchMapper:
    """Maps Relax operators to PyTorch APIs."""
    
    # High-level operator mapping from Relax to PyTorch
    RELAX_TO_PYTORCH_OPS = {
        # Activation functions
        "relax.nn.relu": "F.relu",
        "relax.nn.gelu": "F.gelu",
        "relax.nn.silu": "F.silu",
        "relax.nn.sigmoid": "F.sigmoid",
        "relax.nn.tanh": "F.tanh",
        "relax.nn.softmax": "F.softmax",
        
        # Basic operations
        "relax.add": "torch.add",
        "relax.subtract": "torch.sub",
        "relax.multiply": "torch.mul",
        "relax.divide": "torch.div",
        "relax.power": "torch.pow",
        "relax.maximum": "torch.maximum",
        "relax.minimum": "torch.minimum",
        
        # Reduction operations
        "relax.sum": "torch.sum",
        "relax.mean": "torch.mean",
        "relax.max": "torch.max",
        "relax.min": "torch.min",
        
        # Shape operations
        "relax.reshape": "torch.reshape",
        "relax.transpose": "torch.transpose",
        "relax.permute_dims": "torch.permute",
        "relax.expand_dims": "torch.unsqueeze",
        "relax.squeeze": "torch.squeeze",
        "relax.concat": "torch.cat",
        "relax.split": "torch.split",
        "relax.take": "torch.index_select",
        "relax.strided_slice": "torch.narrow",
        
        # Mathematical functions
        "relax.exp": "torch.exp",
        "relax.log": "torch.log",
        "relax.sqrt": "torch.sqrt",
        "relax.sin": "torch.sin",
        "relax.cos": "torch.cos",
        "relax.abs": "torch.abs",
        "relax.floor": "torch.floor",
        "relax.ceil": "torch.ceil",
        "relax.round": "torch.round",
        
        # Comparison operations
        "relax.equal": "torch.eq",
        "relax.greater": "torch.gt",
        "relax.greater_equal": "torch.ge",
        "relax.less": "torch.lt",
        "relax.less_equal": "torch.le",
        "relax.not_equal": "torch.ne",
        
        # Logical operations
        "relax.logical_and": "torch.logical_and",
        "relax.logical_or": "torch.logical_or",
        "relax.logical_not": "torch.logical_not",
    }
    
    @classmethod
    def get_pytorch_op(cls, relax_op: str) -> str:
        """Get the corresponding PyTorch operation for a Relax operation."""
        return cls.RELAX_TO_PYTORCH_OPS.get(relax_op, relax_op)
    
    @classmethod
    def is_supported(cls, relax_op: str) -> bool:
        """Check if a Relax operation is supported for conversion."""
        return relax_op in cls.RELAX_TO_PYTORCH_OPS


class PythonPrinter:
    """Converts IRModules with Python functions to executable Python code."""
    
    def __init__(self):
        self.mapper = RelaxToPyTorchMapper()
        self.symbolic_vars: Set[str] = set()
        self.imports: Set[str] = set()
        self.helper_functions: List[str] = []
    
    def print_irmodule(self, mod: IRModule) -> str:
        """Convert an IRModule to Python code."""
        self.symbolic_vars.clear()
        self.imports.clear()
        self.helper_functions.clear()
        
        # Don't add any imports - user will handle imports themselves
        # self.imports.add("import torch")
        # self.imports.add("import torch.nn.functional as F")
        # self.imports.add("import tvm")
        # self.imports.add("from tvm import relax as R")
        # self.imports.add("from typing import List")
        
        # Collect all functions
        functions = []
        for gv, func in mod.functions.items():
            print(f"DEBUG: Found function {gv.name_hint} of type {type(func)}")
            if isinstance(func, Function):
                functions.append((gv, func))
            elif isinstance(func, ExternFunc):
                # Handle @I.pyfunc decorated functions
                functions.append((gv, func))
            elif isinstance(func, tir.PrimFunc):
                # Skip TIR functions for now, they'll be handled by call_tir
                continue
            else:
                print(f"DEBUG: Skipping function {gv.name_hint} of type {type(func)}")
        
        # Generate Python code for each function
        generated_functions = []
        for gv, func in functions:
            if isinstance(func, ExternFunc):
                func_code = self._print_extern_function(gv, func)
            else:
                func_code = self._print_relax_function(gv, func)
            if func_code:
                generated_functions.append(func_code)
        
        # Combine all code first
        imports_code = "\n".join(sorted(self.imports))
        functions_code = "\n\n".join(generated_functions)
        
        # Check if we need helper functions
        needs_helpers = self._check_if_needs_helpers(mod)
        
        # Generate helper functions only if needed
        helper_code = ""
        if needs_helpers:
            helper_code = self._generate_helper_functions()
        
        # Also check if the generated code contains call_tir or call_dps_packed
        # This is a fallback to ensure we don't miss any cases
        if not needs_helpers:
            # Generate a temporary version to check
            temp_code = f"""# Generated Python code from TVM IRModule
{imports_code}

{functions_code}
"""
            if "call_tir(" in temp_code or "call_dps_packed(" in temp_code:
                print("DEBUG: Fallback detected call_tir/call_dps_packed, generating helpers")
                needs_helpers = True
                helper_code = self._generate_helper_functions()
        
        if helper_code:
            code = f"""# Generated Python code from TVM IRModule
{helper_code}

{functions_code}
"""
        else:
            code = f"""# Generated Python code from TVM IRModule
{functions_code}
"""
        return code
    
    def _print_extern_function(self, gv: GlobalVar, func: ExternFunc) -> str:
        """Convert an ExternFunc (Python function) to Python code."""
        func_name = gv.name_hint
        
        print(f"DEBUG: Processing ExternFunc {func_name}")
        print(f"DEBUG: ExternFunc type: {type(func)}")
        print(f"DEBUG: ExternFunc attrs: {func.attrs}")
        
        # For ExternFunc, we need to extract the raw string from attrs
        if hasattr(func, 'attrs') and hasattr(func.attrs, 'get'):
            raw_string = func.attrs.get('raw_string', '')
            print(f"DEBUG: Raw string from attrs: {raw_string}")
            if raw_string:
                # The raw string should contain the function definition
                # We need to clean it up and convert it to proper Python
                return self._convert_raw_string_to_python(raw_string, func_name)
        
        # Also check for other possible attribute names
        if hasattr(func, 'attrs'):
            for key, value in func.attrs.items():
                print(f"DEBUG: Attr {key}: {value}")
                if isinstance(value, str) and 'def ' in value:
                    print(f"DEBUG: Found function definition in attr {key}")
                    return self._convert_raw_string_to_python(value, func_name)
        
        # Check for PackedFunc attribute which might contain the function
        if hasattr(func, 'attrs'):
            for key, value in func.attrs.items():
                if hasattr(value, '__call__') and hasattr(value, '__code__'):
                    print(f"DEBUG: Found callable function in attr {key}")
                    # Try to get source from the function object
                    try:
                        import inspect
                        source = inspect.getsource(value)
                        return self._convert_raw_string_to_python(source, func_name)
                    except:
                        pass
                
                # Check for @I.pyfunc specific attributes
                if key == "pyfunc" or key == "python_function":
                    print(f"DEBUG: Found @I.pyfunc attribute: {key}")
                    if isinstance(value, str):
                        return self._convert_raw_string_to_python(value, func_name)
                    elif hasattr(value, '__call__'):
                        try:
                            import inspect
                            source = inspect.getsource(value)
                            return self._convert_raw_string_to_python(source, func_name)
                        except:
                            pass
        
        # Fallback: generate a basic function signature
        return f"""def {func_name}(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # Python function implementation would go here
    pass"""

    def _convert_raw_string_to_python(self, raw_string: str, func_name: str) -> str:
        """Convert the raw string from ExternFunc to proper Python code."""
        if not raw_string:
            return f"""def {func_name}(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # Python function implementation would go here
    pass"""
        
        # Clean up the raw string
        cleaned = raw_string.strip()
        
        # If it already looks like a function definition, return it as-is
        if cleaned.startswith('def '):
            return cleaned
        
        # If it's just the function body, wrap it in a function definition
        if 'return' in cleaned or '=' in cleaned:
            return f"""def {func_name}(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    {cleaned}"""
        
        # Fallback: return the raw string with basic function wrapper
        return f"""def {func_name}(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    {cleaned}"""

    def _print_relax_function(self, gv: GlobalVar, func: Function) -> str:
        """Convert a Relax function to Python code."""
        func_name = gv.name_hint
        
        # Extract function parameters and return type
        params = self._extract_function_params(func)
        return_type = self._extract_return_type(func)
        
        # Convert function body
        body_code = self._convert_function_body(func.body)
        
        # Generate function signature
        param_str = ", ".join(params)
        return_annotation = f" -> {return_type}" if return_type else ""
        
        # Handle symbolic shapes
        shape_code = self._generate_shape_handling()
        
        code = f"""def {func_name}({param_str}){return_annotation}:
    {shape_code}
    {body_code}
    return gv"""
        
        return code
    
    def _extract_function_params(self, func: Function) -> List[str]:
        """Extract function parameters with type annotations."""
        params = []
        for param in func.params:
            param_name = param.name_hint
            # Convert to Python-style parameters (torch.Tensor)
            params.append(f"{param_name}: torch.Tensor")
            
            # Extract symbolic variables from parameter shapes
            if hasattr(param, 'struct_info') and hasattr(param.struct_info, 'shape'):
                self._extract_symbolic_vars_from_shape(param.struct_info.shape)
            
            # Also check for symbolic variables in the parameter name or type annotations
            if hasattr(param, 'struct_info') and hasattr(param.struct_info, 'dtype'):
                # Check if there are any symbolic dimensions in the type annotation
                type_str = str(param.struct_info)
                # Look for patterns like (n, 64) or ("n", 64)
                import re
                symbolic_matches = re.findall(r'["\']?([a-zA-Z_][a-zA-Z0-9_]*)["\']?', type_str)
                for match in symbolic_matches:
                    # Only add variables that look like shape dimensions
                    if (match not in ['Tensor', 'float32', 'float64', 'int32', 'int64'] and
                        len(match) <= 3 and  # Shape dimensions are usually short
                        match.islower()):  # Shape dimensions are usually lowercase
                        self.symbolic_vars.add(match)
        return params
    
    def _extract_return_type(self, func: Function) -> str:
        """Extract function return type."""
        return "torch.Tensor"  # Always return torch.Tensor for Python functions
    
    def _get_python_type(self, struct_info) -> str:
        """Convert TVM struct info to Python type annotation."""
        if hasattr(struct_info, 'dtype') and hasattr(struct_info, 'shape'):
            # Handle tensor types
            shape = struct_info.shape
            if shape and len(shape) > 0:
                # Handle symbolic shapes
                shape_str = self._convert_shape_to_python(shape)
                return f"torch.Tensor[{shape_str}, {struct_info.dtype}]"
            return f"torch.Tensor[{struct_info.dtype}]"
        return "torch.Tensor"
    
    def _convert_shape_to_python(self, shape) -> str:
        """Convert TVM shape to Python shape string."""
        shape_parts = []
        for dim in shape:
            if isinstance(dim, tir.IntImm):
                shape_parts.append(str(dim.value))
            elif isinstance(dim, tir.Var):
                var_name = dim.name
                self.symbolic_vars.add(var_name)
                shape_parts.append(var_name)
            else:
                shape_parts.append(str(dim))
        return ", ".join(shape_parts)
    
    def _extract_symbolic_vars_from_shape(self, shape):
        """Extract symbolic variables from a shape without converting to string."""
        if shape is None:
            return
        for dim in shape:
            if isinstance(dim, tir.Var):
                var_name = dim.name
                self.symbolic_vars.add(var_name)
            elif hasattr(dim, 'name'):  # Handle other variable types
                var_name = dim.name
                self.symbolic_vars.add(var_name)
            elif isinstance(dim, str):  # Handle string dimensions
                self.symbolic_vars.add(dim)
    
    def _convert_function_body(self, body: Expr) -> str:
        """Convert function body to Python code."""
        # Debug: print the actual type and content
        print(f"DEBUG: body type: {type(body)}")
        print(f"DEBUG: body content: {body}")
        
        if isinstance(body, RelaxCall):
            result = self._convert_call(body)
            return f"gv = {result}"
        elif isinstance(body, Var):
            return f"gv = {body.name_hint}"
        elif hasattr(body, '__class__') and 'SeqExpr' in body.__class__.__name__:
            # Handle SeqExpr - generate complete function body with all operations
            lines = []
            
            # Extract symbolic variables from the function body
            if hasattr(body, 'blocks') and body.blocks:
                for block in body.blocks:
                    if hasattr(block, 'bindings'):
                        for binding in block.bindings:
                            if (hasattr(binding, 'var') and 
                                hasattr(binding, 'value')):
                                # Generate assignment for this binding
                                var_name = binding.var.name_hint
                                value_code = self._convert_expr(binding.value)
                                lines.append(f"{var_name} = {value_code}")
                                
                                # Also extract symbolic variables from the value
                                if hasattr(binding.value, 'struct_info') and hasattr(binding.value.struct_info, 'shape'):
                                    self._extract_symbolic_vars_from_shape(binding.value.struct_info.shape)
                                
                                # Check for symbolic variables in the value code itself
                                if isinstance(value_code, str):
                                    import re
                                    # Look for variable names that might be symbolic
                                    var_matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', value_code)
                                    for match in var_matches:
                                        # Only add variables that look like shape dimensions
                                        if (match not in ['torch', 'F', 'self', 'x', 'w', 'gv', 'result', 'add', 'mean', 'dtype', 'R', 'T', 'axis', 'keepdims', 'False', 'True'] and
                                            len(match) <= 3 and  # Shape dimensions are usually short
                                            match.islower()):  # Shape dimensions are usually lowercase
                                            self.symbolic_vars.add(match)
            
            # Handle the final expression
            if hasattr(body, 'body'):
                final_expr = body.body
                if isinstance(final_expr, Var):
                    # If it's a variable reference, use it as the final result
                    final_var = final_expr.name_hint
                    if final_var != "gv":  # Avoid duplicate assignment
                        lines.append(f"gv = {final_var}")
                elif isinstance(final_expr, RelaxCall):
                    # If it's a function call, convert it
                    result = self._convert_call(final_expr)
                    lines.append(f"gv = {result}")
                else:
                    # For other expressions, try to convert
                    result = self._convert_expr(final_expr)
                    lines.append(f"gv = {result}")
            else:
                lines.append("gv = result")
            
            # Join all lines with proper indentation
            if lines:
                return "\n    ".join(lines)
            else:
                return "gv = result"
        else:
            # For complex expressions, try to get a simple representation
            try:
                # Handle return statements and other expressions
                if hasattr(body, 'value'):
                    # This might be a return statement or similar
                    result = self._convert_expr(body.value)
                    return f"gv = {result}"
                else:
                    expr_str = str(body).split('\n')[0]  # Take first line only
                    return f"gv = {expr_str}"
            except:
                return "gv = expr"
    
    def _convert_call(self, call: RelaxCall) -> str:
        """Convert a Relax call to Python code."""
        op = call.op
        
        if isinstance(op, ir.Op):
            op_name = op.name
            
            # Special handling for call_tir
            if op_name == "relax.call_tir":
                # Extract function name and arguments
                if len(call.args) >= 3:
                    func_name = self._convert_expr(call.args[0])
                    args = self._convert_expr(call.args[1])
                    out_sinfo = self._convert_expr(call.args[2])
                    return f"call_tir({func_name}, {args}, {out_sinfo})"
                else:
                    return "call_tir(...)"
            
            # Special handling for call_dps_packed
            elif op_name == "relax.call_dps_packed":
                # Extract function name and arguments
                if len(call.args) >= 3:
                    func_name = self._convert_expr(call.args[0])
                    args = self._convert_expr(call.args[1])
                    out_sinfo = self._convert_expr(call.args[2])
                    return f"call_dps_packed({func_name}, {args}, {out_sinfo})"
                else:
                    return "call_dps_packed(...)"
            
            # Handle other operations
            else:
                pytorch_op = self.mapper.get_pytorch_op(op_name)
                
                if self.mapper.is_supported(op_name):
                    args = [self._convert_expr(arg) for arg in call.args]
                    return f"{pytorch_op}({', '.join(args)})"
                else:
                    # Fallback to TVM call
                    args = [self._convert_expr(arg) for arg in call.args]
                    return f"R.call_packed('{op_name}', {', '.join(args)})"
        
        elif isinstance(op, GlobalVar):
            # Function call - should be self.func_name for Python functions
            args = [self._convert_expr(arg) for arg in call.args]
            return f"self.{op.name_hint}({', '.join(args)})"
        
        elif isinstance(op, Var):
            # Variable function call
            args = [self._convert_expr(arg) for arg in call.args]
            return f"{op.name_hint}({', '.join(args)})"
        
        else:
            # Unknown operation
            args = [self._convert_expr(arg) for arg in call.args]
            return f"R.call_packed('unknown_op', {', '.join(args)})"
    
    def _convert_expr(self, expr: Expr) -> str:
        """Convert a Relax expression to Python code."""
        if isinstance(expr, Var):
            return expr.name_hint
        elif isinstance(expr, Constant):
            if hasattr(expr.data, 'numpy'):
                return str(expr.data.numpy().item())
            return str(expr.data)
        elif isinstance(expr, RelaxCall):
            return self._convert_call(expr)
        elif hasattr(expr, '__class__') and 'Tuple' in expr.__class__.__name__:
            # Handle Tuple expressions
            if hasattr(expr, 'fields'):
                fields = [self._convert_expr(field) for field in expr.fields]
                return f"({', '.join(fields)})"
            else:
                return "()"
        elif hasattr(expr, '__class__') and 'PrimValue' in expr.__class__.__name__:
            # Handle PrimValue expressions
            if hasattr(expr, 'value'):
                return self._convert_expr(expr.value)
            else:
                return "prim_value"
        elif hasattr(expr, '__class__') and 'GlobalVar' in expr.__class__.__name__:
            # Handle GlobalVar expressions - should be self.func_name for Python functions
            if hasattr(expr, 'name_hint'):
                return f"self.{expr.name_hint}"
            else:
                return "self.global_var"
        else:
            # For complex expressions, try to get a simple representation
            try:
                return str(expr).split('\n')[0]  # Take first line only
            except:
                return "expr"
    
    def _generate_shape_handling(self) -> str:
        """Generate code to handle symbolic shapes."""
        if not self.symbolic_vars:
            return ""
        
        shape_code = []
        for var in sorted(self.symbolic_vars):
            # Generate Python code like: n = x.shape[0], c = x.shape[1], etc.
            # We'll use the first parameter (x) as the reference for shape
            if var in ['n', 'batch_size', 'batch']:
                shape_code.append(f"{var} = x.shape[0]")
            elif var in ['c', 'channels', 'features']:
                shape_code.append(f"{var} = x.shape[1]")
            elif var in ['h', 'height']:
                shape_code.append(f"{var} = x.shape[2]")
            elif var in ['w', 'width']:
                shape_code.append(f"{var} = x.shape[3]")
            else:
                # For other variables, try to infer from context or use a default
                shape_code.append(f"{var} = x.shape[0]  # TODO: Infer correct dimension")
        
        if shape_code:
            return "\n    ".join(shape_code)
        return ""
    
    def _check_if_needs_helpers(self, mod: IRModule) -> bool:
        """Check if the IRModule needs helper functions."""
        for gv, func in mod.functions.items():
            if isinstance(func, Function):
                # Check if the function body contains call_tir or call_dps_packed
                if self._contains_helper_calls(func.body):
                    return True
        return False
    
    def _contains_helper_calls(self, body: Expr) -> bool:
        """Check if the function body contains calls that need helpers."""
        # Debug: print the body type and content
        print(f"DEBUG: _contains_helper_calls - body type: {type(body)}")
        print(f"DEBUG: _contains_helper_calls - body content: {body}")
        
        if isinstance(body, RelaxCall):
            op = body.op
            if isinstance(op, ir.Op):
                op_name = op.name
                print(f"DEBUG: Found RelaxCall with op: {op_name}")
                if op_name in ["relax.call_tir", "relax.call_dps_packed"]:
                    return True
        elif hasattr(body, 'bindings'):
            # Check SeqExpr bindings
            print(f"DEBUG: Found SeqExpr with {len(body.bindings)} bindings")
            for i, binding in enumerate(body.bindings):
                print(f"DEBUG: Binding {i}: {binding}")
                if hasattr(binding, 'value') and isinstance(binding.value, RelaxCall):
                    op = binding.value.op
                    if isinstance(op, ir.Op):
                        op_name = op.name
                        print(f"DEBUG: Binding {i} has RelaxCall with op: {op_name}")
                        if op_name in ["relax.call_tir", "relax.call_dps_packed"]:
                            return True
                # Also check if the binding value is a string that contains call_tir
                if hasattr(binding, 'value'):
                    binding_str = str(binding.value)
                    print(f"DEBUG: Binding {i} value string: {binding_str}")
                    if "call_tir" in binding_str or "call_dps_packed" in binding_str:
                        print(f"DEBUG: Found call_tir/call_dps_packed in binding {i}")
                        return True
            # Also check the final value in SeqExpr
            if hasattr(body, 'body'):
                print(f"DEBUG: SeqExpr final body: {body.body}")
                if isinstance(body.body, RelaxCall):
                    op = body.body.op
                    if isinstance(op, ir.Op):
                        op_name = op.name
                        print(f"DEBUG: Final body has RelaxCall with op: {op_name}")
                        if op_name in ["relax.call_tir", "relax.call_dps_packed"]:
                            return True
        else:
            # Fallback: check the string representation
            body_str = str(body)
            print(f"DEBUG: Fallback check - body string: {body_str}")
            if "call_tir" in body_str or "call_dps_packed" in body_str:
                print(f"DEBUG: Found call_tir/call_dps_packed in fallback check")
                return True
        return False
    
    def _generate_helper_functions(self) -> str:
        """Generate helper functions for cross-function calls."""
        helper_code = """# Helper functions for cross-function calls
# Note: These functions require the following imports to be available:
# import torch
# import tvm
# from typing import List

def call_tir(func_name: str, args: List[torch.Tensor], out_sinfo) -> torch.Tensor:
    \"\"\"Call a TIR function with PyTorch tensors.\"\"\"
    # Convert PyTorch tensors to TVM NDArrays via DLPack
    tvm_args = [tvm.nd.from_dlpack(torch.to_dlpack(arg)) for arg in args]
    
    # Call the TIR function
    result = tvm.get_global_func(func_name)(*tvm_args)
    
    # Convert result back to PyTorch tensor
    return torch.from_dlpack(result.to_dlpack())

def call_dps_packed(func_name: str, args: List[torch.Tensor], out_sinfo) -> torch.Tensor:
    \"\"\"Call a packed function with PyTorch tensors.\"\"\"
    # Get the packed function
    packed_func = tvm.get_global_func(func_name)
    
    # Convert PyTorch tensors to TVM NDArrays via DLPack
    tvm_args = [tvm.nd.from_dlpack(torch.to_dlpack(arg)) for arg in args]
    
    # Call the packed function
    result = packed_func(*tvm_args)
    
    # Convert result back to PyTorch tensor
    return torch.from_dlpack(result.to_dlpack())
"""
        return helper_code


def irmodule_to_python(mod: IRModule) -> str:
    """Convert an IRModule to Python code.
    
    Parameters
    ----------
    mod : IRModule
        The IRModule to convert
        
    Returns
    -------
    str
        Generated Python code that can be executed directly
    """
    printer = PythonPrinter()
    return printer.print_irmodule(mod)


def print_irmodule_as_python(mod: IRModule) -> None:
    """Print an IRModule as Python code.
    
    Parameters
    ----------
    mod : IRModule
        The IRModule to print
    """
    code = irmodule_to_python(mod)
    print(code)
