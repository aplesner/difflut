"""Pytest configuration for test suite."""



import sys
import warnings
from pathlib import Path

# Add tests directory to Python path so testing_utils can be imported
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))


from difflut.utils.warnings import CUDAWarning, DefaultValueWarning
# Suppress specific warnings during tests
warnings.filterwarnings("ignore", category=CUDAWarning)
warnings.filterwarnings("ignore", category=DefaultValueWarning)