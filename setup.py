
import os
import setuptools
from setuptools import setup, find_packages
import multiprocessing
import shutil

ext_modules = []

# Enable parallel compilation via environment variable
num_jobs = min(multiprocessing.cpu_count(), 8)  # Cap at 8 to avoid resource exhaustion
os.environ['MAX_JOBS'] = str(num_jobs)  # For Ninja/build system

# Enable ccache if available for faster rebuilds
if shutil.which('ccache'):
    os.environ['CC'] = 'ccache gcc'
    os.environ['CXX'] = 'ccache g++'
    os.environ['NVCC'] = 'ccache nvcc'
    print(f"✅ ccache enabled for faster rebuilds")

print(f"Parallel compilation enabled: MAX_JOBS={num_jobs}")

# Try to import torch and CUDAExtension, and check for CUDA
cuda_available = False
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    # Check if CUDA_HOME is set (don't rely on torch.cuda.is_available() during build)
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home:
        print(f"CUDA_HOME detected: {cuda_home}")
        cuda_available = True
    else:
        print("CUDA_HOME not set. Checking torch.cuda.is_available()...")
        cuda_available = torch.cuda.is_available()
except ImportError:
    print("Warning: PyTorch not found, CUDA extensions will not be built.")
except Exception as e:
    print(f"Warning: Could not check CUDA availability: {e}")

cuda_dirs = [
    os.path.join('difflut', 'nodes', 'cuda'),
    os.path.join('difflut', 'layers', 'cuda')
]

if cuda_available:
    for cuda_dir in cuda_dirs:
        if os.path.exists(cuda_dir):
            for filename in os.listdir(cuda_dir):
                if filename.endswith('.cpp'):
                    module_name = filename[:-4]
                    kernel_filename = module_name + '_kernel.cu'
                    kernel_path = os.path.join(cuda_dir, kernel_filename)
                    if os.path.exists(kernel_path):
                        print(f"Building CUDA extension: {module_name} from {cuda_dir}")
                        ext_modules.append(CUDAExtension(
                            module_name,
                            [os.path.join(cuda_dir, filename), kernel_path],
                            extra_compile_args={
                                'cxx': ['-O3', '-march=native'],
                                'nvcc': ['-O3', '--use_fast_math']
                            }
                        ))
                    else:
                        print(f"Warning: Found {filename} but missing {kernel_filename}")
        else:
            print(f"CUDA directory not found: {cuda_dir}")
    print(f"Total CUDA extensions to build: {len(ext_modules)}")
else:
    print("CUDA not available or not detected. Skipping CUDA extension build.")

setup_args = dict(
    name='difflut',
    version="1.1.2",
    packages=find_packages(),
    install_requires=[
        'torch>=1.9.0',
        'numpy',
    ],
    python_requires='>=3.7',
    description='Differentiable LUT-based neural networks',
    author='Simon Buehrer',
    license='MIT',
)

if ext_modules and cuda_available:
    setup_args['ext_modules'] = ext_modules
    setup_args['cmdclass'] = {
        'build_ext': BuildExtension.with_options(use_ninja=True)
    }

setup(**setup_args)