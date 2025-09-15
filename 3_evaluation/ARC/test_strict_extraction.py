#!/usr/bin/env python3
"""
Test script for the new STRICT answer extraction logic in ARC evaluation
"""

import sys
import os
sys.path.append('.')

# Define the extract_answer_robust function (copied from eval_arc_3shot.py to avoid import issues)
def extract_answer_robust(model_output: str) -> str:
    """
    Extract the final answer (A, B, C, D) from model output using STRICT validation.
    Returns None if no clear structured answer is found.
    STRICT MODE: Only accepts {} format or #### format like in few-shot examples.
    """
    if not model_output:
        return None

    cleaned_output = model_output.strip().upper()
    valid_answers = ['A', 'B', 'C', 'D']

    import re

    # STRICT: Only accept the exact formats shown in few-shot examples
    # Priority 1: Box format {A} - exact match to few-shot examples
    box_pattern = r'\{([A-D])\}'
    box_matches = re.findall(box_pattern, cleaned_output)
    if box_matches:
        return box_matches[-1]  # Return the last match (final answer)

    # Priority 2: #### format - exact match to few-shot examples
    structured_patterns = [
        r'####\s*(?:ANSWER)\s*:?\s*([A-D])',  # #### Answer: A
        r'####\s*([A-D])',  # #### A
    ]

    for pattern in structured_patterns:
        matches = re.findall(pattern, cleaned_output)
        if matches:
            return matches[-1]  # Return the last match (final answer)

    # STRICT: No fallback patterns - if it doesn't match few-shot format, return None
    # This forces the model to follow the exact format shown in examples

    return None

def test_answer_extraction():
    """
    Test the strict answer extraction with various model output examples
    """
    print("=" * 80)
    print("TESTING STRICT ANSWER EXTRACTION FOR ARC")
    print("=" * 80)

    # Test cases: (model_output, expected_result, description)
    test_cases = [
        # Should PASS - correct formats matching few-shot examples
        ("Let me think about this step by step. The answer is {A}.", "A", "Box format - should pass"),
        ("After analyzing the question, {C}", "C", "Box format at end - should pass"),
        ("#### Answer: B", "B", "#### format - should pass"),
        ("#### ANSWER: D", "D", "#### ANSWER format - should pass"),
        ("#### A", "A", "#### minimal format - should pass"),
        ("Some reasoning here.\n{B}\nMore text after.", "B", "Box format with newlines - should pass"),
        ("First I {A} then I think more and finally {C}", "C", "Multiple boxes - should return last"),

        # Should FAIL - formats that DON'T match few-shot examples
        ("The answer is A", None, "Plain text A - should fail"),
        ("A. This is the correct option", None, "A. format - should fail"),
        ("(A)", None, "Parentheses format - should fail"),
        ("A)", None, "Partial parentheses - should fail"),
        ("So the answer is A.", None, "Sentence with A - should fail"),
        ("Answer: A", None, "Answer: format without #### - should fail"),
        ("I choose A because...", None, "Choose A format - should fail"),
        ("The correct answer must be C", None, "Must be C format - should fail"),
        ("Based on analysis, B is correct", None, "B is correct format - should fail"),
        ("Option A seems right", None, "Option A format - should fail"),
        ("A B C D", None, "Multiple letters - should fail"),
        ("ABC", None, "Letters together - should fail"),
        ("", None, "Empty string - should fail"),
        ("No answer provided", None, "No answer - should fail"),
        ("The question is unclear", None, "No letter at all - should fail"),

        # Edge cases
        ("{}", None, "Empty box - should fail"),
        ("{E}", None, "Invalid letter E - should fail"),
        ("{ A }", None, "Box with spaces - should fail (invalid format)"),
        ("{A} and {B}", "B", "Multiple boxes - should return last"),
        ("#### Answer: A\n#### Answer: B", "B", "Multiple #### answers - should return last"),
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
        print("SUCCESS: All tests passed! The strict extraction is working correctly.")
    else:
        print(f"WARNING: {failed} tests failed. Please check the extraction logic.")

    return failed == 0

if __name__ == "__main__":
    success = test_answer_extraction()
    exit(0 if success else 1)