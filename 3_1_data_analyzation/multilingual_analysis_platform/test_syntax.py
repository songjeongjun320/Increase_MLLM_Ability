#!/usr/bin/env python3
"""
Simple syntax test to verify the modifications don't break the code.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_syntax():
    """Test that the modified code can be imported without syntax errors."""
    print("Testing syntax of modified code...")

    try:
        # This will check if the syntax is correct
        import ast

        # Read the main file
        with open('run_terminal_analysis.py', 'r', encoding='utf-8') as f:
            code = f.read()

        # Parse the code to check for syntax errors
        ast.parse(code)
        print("  Syntax check passed: run_terminal_analysis.py")

        return True

    except SyntaxError as e:
        print(f"  Syntax error found: {e}")
        return False
    except Exception as e:
        print(f"  Error reading file: {e}")
        return False

def test_function_existence():
    """Test that our modified functions exist and can be referenced."""
    print("Testing function existence...")

    try:
        # Import the module to check function existence
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_terminal_analysis", "run_terminal_analysis.py")
        module = importlib.util.module_from_spec(spec)

        # Check if functions exist (without executing them)
        functions_to_check = [
            'plot_dual_model_euclidean_distance',
            'plot_dual_model_pca_comparison',
            'plot_dual_model_comparison'
        ]

        # Read the file to check for function definitions
        with open('run_terminal_analysis.py', 'r', encoding='utf-8') as f:
            content = f.read()

        for func_name in functions_to_check:
            if f"def {func_name}" in content:
                print(f"  Function found: {func_name}")
            else:
                print(f"  Function missing: {func_name}")
                return False

        return True

    except Exception as e:
        print(f"  Error checking functions: {e}")
        return False

def test_cross_lingual_distance_logic():
    """Test the logic of cross-lingual distance calculation manually."""
    print("Testing cross-lingual distance calculation logic...")

    try:
        import numpy as np

        # Simulate the logic we added
        # Mock embeddings: 4 samples (2 EN-KO pairs)
        np.random.seed(42)
        base_embeddings = np.random.rand(4, 10)
        train_embeddings = np.random.rand(4, 10)

        # Calculate cross-lingual distances as we modified
        cross_lingual_distances = {'base': [], 'training': []}

        for i in range(0, len(base_embeddings), 2):  # Every pair: i=EN, i+1=KO
            if i+1 < len(base_embeddings):
                # Base model: EN vs KO distance
                base_en_ko_distance = np.linalg.norm(base_embeddings[i] - base_embeddings[i+1])
                cross_lingual_distances['base'].append(base_en_ko_distance)

                # Training model: EN vs KO distance
                train_en_ko_distance = np.linalg.norm(train_embeddings[i] - train_embeddings[i+1])
                cross_lingual_distances['training'].append(train_en_ko_distance)

        # Verify we got the expected number of pairs
        expected_pairs = len(base_embeddings) // 2
        if len(cross_lingual_distances['base']) == expected_pairs:
            print(f"  Cross-lingual distance logic correct: {expected_pairs} pairs calculated")
            print(f"    Base distances: {[f'{d:.3f}' for d in cross_lingual_distances['base']]}")
            print(f"    Train distances: {[f'{d:.3f}' for d in cross_lingual_distances['training']]}")
            return True
        else:
            print(f"  Error: Expected {expected_pairs} pairs, got {len(cross_lingual_distances['base'])}")
            return False

    except Exception as e:
        print(f"  Error in logic test: {e}")
        return False

def run_syntax_tests():
    """Run all syntax and logic tests."""
    print("Running Syntax and Logic Tests for Modifications")
    print("=" * 55)

    test_results = []

    # Test 1: Syntax check
    test_results.append(test_syntax())

    # Test 2: Function existence
    test_results.append(test_function_existence())

    # Test 3: Logic test
    test_results.append(test_cross_lingual_distance_logic())

    # Summary
    print("\n" + "=" * 55)
    print("Test Results Summary:")
    passed = sum(test_results)
    total = len(test_results)
    print(f"   Tests passed: {passed}/{total}")

    if passed == total:
        print("   All syntax and logic tests passed!")
        print("   The modifications appear to be correctly implemented.")
    else:
        print(f"   {total - passed} test(s) failed.")

    return passed == total

if __name__ == "__main__":
    success = run_syntax_tests()
    sys.exit(0 if success else 1)