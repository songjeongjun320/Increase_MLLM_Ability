#!/usr/bin/env python3
"""
Test script for the updated re_extract_answers.py with STRICT validation
"""

import sys
import os
sys.path.append('.')

# Import the extract_answer_robust function from re_extract_answers.py
from re_extract_answers import extract_answer_robust

def test_strict_re_extraction():
    """
    Test the strict re-extraction logic
    """
    print("=" * 80)
    print("TESTING STRICT RE-EXTRACTION LOGIC")
    print("=" * 80)

    # Test cases: (model_output, expected_result, description)
    test_cases = [
        # Should PASS - correct formats
        ("Reasoning... {A}", "A", "Box format - should pass"),
        ("#### Answer: B", "B", "#### Answer format - should pass"),
        ("#### C", "C", "#### minimal format - should pass"),
        ("First {A} then {D}", "D", "Multiple boxes - should return last"),

        # Should FAIL - incorrect formats
        ("The answer is A", None, "Plain text - should fail"),
        ("A. This is correct", None, "A. format - should fail"),
        ("(A)", None, "Parentheses - should fail"),
        ("Answer: A", None, "Answer: without #### - should fail"),
        ("I choose B", None, "Choose format - should fail"),
        ("", None, "Empty - should fail"),
    ]

    passed = 0
    failed = 0

    print(f"\nRunning {len(test_cases)} test cases...\n")

    for i, (model_output, expected, description) in enumerate(test_cases, 1):
        result = extract_answer_robust(model_output)

        if result == expected:
            status = "PASS"
            passed += 1
        else:
            status = "FAIL"
            failed += 1

        print(f"Test {i:2d}: {status}")
        print(f"  Description: {description}")
        print(f"  Input: {repr(model_output)}")
        print(f"  Expected: {expected}")
        print(f"  Got: {result}")
        print()

    print("=" * 80)
    print(f"TEST RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 80)

    if failed == 0:
        print("SUCCESS: All tests passed! Strict re-extraction is working correctly.")
    else:
        print(f"WARNING: {failed} tests failed. Please check the logic.")

    return failed == 0

if __name__ == "__main__":
    success = test_strict_re_extraction()
    exit(0 if success else 1)