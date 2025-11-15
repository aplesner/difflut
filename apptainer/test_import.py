#!/usr/bin/env python3
"""
Test 0: Import Test (CPU-only, no GPU required)

This test verifies that the difflut package and its dependencies can be
imported successfully. It does not require a GPU and can be run during
container build to catch import errors early.

Usage:
    python test_import.py

Exit codes:
    0 - All imports successful
    1 - Import failed
"""

import sys


def test_0_imports():
    """Test 0: Basic Imports (CPU-only)"""
    print("=" * 80)
    print("  TEST 0: Import Test (CPU-only)")
    print("=" * 80)

    # Test basic imports
    try:
        print("\n[1/5] Importing torch...")
        import torch

        print(f"      ✓ PyTorch {torch.__version__}")

        print("\n[2/5] Importing difflut...")
        import difflut

        print(f"      ✓ difflut {difflut.__version__}")

        print("\n[3/5] Importing difflut.nodes...")
        from difflut.nodes import DWNNode, LinearLUTNode

        print("      ✓ DWNNode, LinearLUTNode")

        print("\n[4/5] Importing difflut.layers...")
        from difflut.layers import ConvolutionalLayer, RandomLayer

        print("      ✓ RandomLayer, ConvolutionalLayer")

        print("\n[5/5] Checking CUDA extensions...")
        try:
            import efd_cuda

            print("      ✓ efd_cuda (standard) available")
        except ImportError:
            print("      ⚠ efd_cuda (standard) NOT available")
            print(
                "        This is expected if CUDA extensions haven't been compiled yet."
            )

        try:
            import efd_fused_cuda

            print("      ✓ efd_fused_cuda (fused) available")
        except ImportError:
            print("      ⚠ efd_fused_cuda (fused) NOT available")
            print(
                "        This is expected if fused CUDA extensions haven't been compiled yet."
            )

        print("\n" + "=" * 80)
        print("  ✓ ALL IMPORTS SUCCESSFUL")
        print("=" * 80)
        print("\n✓ Test 0 passed: Package can be imported successfully.")
        print("  You can now proceed with GPU-based tests.\n")
        return 0

    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        import traceback

        traceback.print_exc()
        print("\n" + "=" * 80)
        print("  ✗ IMPORT TEST FAILED")
        print("=" * 80)
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        print("\n" + "=" * 80)
        print("  ✗ IMPORT TEST FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(test_0_imports())
