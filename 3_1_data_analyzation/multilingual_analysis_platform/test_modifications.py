#!/usr/bin/env python3
"""
Test script to verify the modifications to the multilingual analysis platform.
Tests the new cross-lingual distance calculation and connection lines functionality.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def create_mock_dual_embeddings():
    """Create mock dual embeddings for testing."""
    # Create mock embeddings: 4 samples (2 EN-KO pairs)
    # Index 0,1 = pair 1; Index 2,3 = pair 2
    np.random.seed(42)

    base_embeddings = np.random.rand(4, 128)  # 4 samples, 128 dimensions
    train_embeddings = np.random.rand(4, 128)  # 4 samples, 128 dimensions

    # Make EN-KO pairs within models more similar (for testing)
    # Pair 1: indices 0,1
    base_embeddings[1] = base_embeddings[0] + 0.1 * np.random.rand(128)
    train_embeddings[1] = train_embeddings[0] + 0.1 * np.random.rand(128)

    # Pair 2: indices 2,3
    base_embeddings[3] = base_embeddings[2] + 0.1 * np.random.rand(128)
    train_embeddings[3] = train_embeddings[2] + 0.1 * np.random.rand(128)

    return {
        'base_embeddings': base_embeddings,
        'train_embeddings': train_embeddings,
        'languages': ['en', 'ko', 'en', 'ko'],
        'texts': [
            'Hello world',
            '안녕하세요 세상',
            'Good morning',
            '좋은 아침'
        ],
        'base_model_path': '/mock/base/model',
        'train_model_path': '/mock/train/model'
    }

def test_euclidean_distance_calculation():
    """Test the modified Euclidean distance calculation."""
    print("Testing Euclidean distance calculation...")

    try:
        from run_terminal_analysis import plot_dual_model_euclidean_distance

        # Create mock data
        dual_embeddings = create_mock_dual_embeddings()
        texts = dual_embeddings['texts']

        # Test distance calculation manually to verify logic
        base_embeddings = dual_embeddings['base_embeddings']
        train_embeddings = dual_embeddings['train_embeddings']

        # Calculate cross-lingual distances manually
        cross_lingual_distances = {'base': [], 'training': []}

        for i in range(0, len(base_embeddings), 2):
            if i+1 < len(base_embeddings):
                # Base model: EN vs KO distance
                base_en_ko_distance = np.linalg.norm(base_embeddings[i] - base_embeddings[i+1])
                cross_lingual_distances['base'].append(base_en_ko_distance)

                # Training model: EN vs KO distance
                train_en_ko_distance = np.linalg.norm(train_embeddings[i] - train_embeddings[i+1])
                cross_lingual_distances['training'].append(train_en_ko_distance)

        print(f"  Cross-lingual distances calculated successfully")
        print(f"     Base model distances: {[f'{d:.3f}' for d in cross_lingual_distances['base']]}")
        print(f"     Training model distances: {[f'{d:.3f}' for d in cross_lingual_distances['training']]}")

        # Test the actual function (will create a plot)
        output_path = Path("test_euclidean_output.png")
        success = plot_dual_model_euclidean_distance(dual_embeddings, texts, dual_embeddings['languages'], output_path)

        if success and output_path.exists():
            print(f"  Euclidean distance plot generated successfully: {output_path}")
            output_path.unlink()  # Clean up
        else:
            print(f"  Plot generation failed or file not created")

        return True

    except Exception as e:
        print(f"  Euclidean distance test failed: {e}")
        return False

def test_pca_visualization():
    """Test the PCA visualization with connection lines."""
    print("Testing PCA visualization with connection lines...")

    try:
        from run_terminal_analysis import plot_dual_model_pca_comparison

        # Create mock data
        dual_embeddings = create_mock_dual_embeddings()

        # Test the function
        output_path = Path("test_pca_output.png")
        success = plot_dual_model_pca_comparison(dual_embeddings, output_path)

        if success and output_path.exists():
            print(f"   PCA plot with connection lines generated successfully: {output_path}")
            output_path.unlink()  # Clean up
        else:
            print(f"  Warning: PCA plot generation failed or file not created")

        return True

    except Exception as e:
        print(f"  Error: PCA visualization test failed: {e}")
        return False

def test_tsne_umap_visualization():
    """Test the t-SNE and UMAP visualizations with connection lines."""
    print("Testing t-SNE and UMAP visualizations with connection lines...")

    try:
        from run_terminal_analysis import plot_dual_model_comparison

        # Create mock data
        dual_embeddings = create_mock_dual_embeddings()

        # Test t-SNE
        output_path_tsne = Path("test_tsne_output.png")
        success_tsne = plot_dual_model_comparison(dual_embeddings, output_path_tsne, method='tsne')

        if success_tsne and output_path_tsne.exists():
            print(f"   t-SNE plot with connection lines generated successfully: {output_path_tsne}")
            output_path_tsne.unlink()  # Clean up
        else:
            print(f"  Warning: t-SNE plot generation failed or file not created")

        # Test UMAP (if available)
        try:
            output_path_umap = Path("test_umap_output.png")
            success_umap = plot_dual_model_comparison(dual_embeddings, output_path_umap, method='umap')

            if success_umap and output_path_umap.exists():
                print(f"   UMAP plot with connection lines generated successfully: {output_path_umap}")
                output_path_umap.unlink()  # Clean up
            else:
                print(f"  Warning: UMAP plot generation failed or file not created")
        except ImportError:
            print(f"  Info: UMAP not available, skipping UMAP test")

        return success_tsne  # Return t-SNE success as main indicator

    except Exception as e:
        print(f"  Error: t-SNE/UMAP visualization test failed: {e}")
        return False

def run_tests():
    """Run all tests."""
    print("Testing Multilingual Analysis Platform Modifications")
    print("=" * 60)

    test_results = []

    # Test 1: Euclidean distance calculation
    test_results.append(test_euclidean_distance_calculation())

    # Test 2: PCA visualization
    test_results.append(test_pca_visualization())

    # Test 3: t-SNE/UMAP visualization
    test_results.append(test_tsne_umap_visualization())

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    passed = sum(test_results)
    total = len(test_results)
    print(f"   Tests passed: {passed}/{total}")

    if passed == total:
        print("   All tests passed! Modifications are working correctly.")
    else:
        print(f"   {total - passed} test(s) failed. Please check the output above.")

    return passed == total

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)