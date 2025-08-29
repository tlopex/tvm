#!/usr/bin/env python3
"""Test the code optimization functionality."""

def test_optimization():
    """Test removing unused variables."""
    
    # Simulate the optimization logic
    def _optimize_generated_code(code: str) -> str:
        """Remove unused variables and optimize the generated code."""
        lines = code.split('\n')
        optimized_lines = []
        
        # Track variable definitions and usage
        defined_vars = set()
        used_vars = set()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                optimized_lines.append(line)
                continue
                
            # Check for variable definitions (e.g., "n = x.shape[0]")
            if ' = ' in line and 'x.shape[' in line:
                var_name = line.split(' = ')[0].strip()
                defined_vars.add(var_name)
                # Check if this variable is used in subsequent lines
                is_used = False
                for future_line in lines[i + 1:]:
                    if var_name in future_line and not future_line.startswith(var_name + ' ='):
                        is_used = True
                        used_vars.add(var_name)
                        break
                
                # Only add the line if the variable is actually used
                if is_used:
                    optimized_lines.append(line)
                else:
                    # Add a comment explaining why we're not generating this
                    optimized_lines.append(f"    # {var_name} = x.shape[0]  # Not used in function body")
            else:
                # Check for variable usage in this line
                for var in defined_vars:
                    if var in line and not line.startswith(var + ' ='):
                        used_vars.add(var)
                optimized_lines.append(line)
        
        return '\n'.join(optimized_lines)
    
    # Test case 1: Variable n is not used
    test_code_1 = """# Generated Python code from TVM IRModule
def nn_forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    lv = torch.add(x, w)
    lv1 = F.relu(lv)
    gv = lv1
    return gv"""
    
    print("=== Test Case 1: Variable n is not used ===")
    print("Original code:")
    print(test_code_1)
    print("\nOptimized code:")
    optimized_1 = _optimize_generated_code(test_code_1)
    print(optimized_1)
    
    # Test case 2: Variable n is used
    test_code_2 = """# Generated Python code from TVM IRModule
def nn_forward(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    lv = torch.add(x, w)
    lv1 = F.relu(lv)
    gv = lv1.reshape(n, -1)  # n is used here
    return gv"""
    
    print("\n=== Test Case 2: Variable n is used ===")
    print("Original code:")
    print(test_code_2)
    print("\nOptimized code:")
    optimized_2 = _optimize_generated_code(test_code_2)
    print(optimized_2)

if __name__ == "__main__":
    test_optimization()

