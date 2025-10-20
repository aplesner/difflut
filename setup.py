
import os
import setuptools
from setuptools import setup, find_packages

ext_modules = []

# Try to import torch and CUDAExtension, and check for CUDA
cuda_available = False
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    cuda_available = torch.cuda.is_available() and (os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH'))
except ImportError:
    print("Warning: PyTorch not found, CUDA extensions will not be built.")
except Exception as e:
    print(f"Warning: Could not check CUDA availability: {e}")

cuda_dir = os.path.join('difflut', 'nodes', 'cuda')

if cuda_available and os.path.exists(cuda_dir):
    for filename in os.listdir(cuda_dir):
        if filename.endswith('.cpp'):
            module_name = filename[:-4]
            kernel_filename = module_name + '_kernel.cu'
            kernel_path = os.path.join(cuda_dir, kernel_filename)
            if os.path.exists(kernel_path):
                print(f"Building CUDA extension: {module_name}")
                ext_modules.append(CUDAExtension(
                    module_name,
                    [os.path.join(cuda_dir, filename), kernel_path],
                    extra_compile_args={
                        'cxx': ['-O3'],
                        'nvcc': ['-O3', '--use_fast_math']
                    }
                ))
            else:
                print(f"Warning: Found {filename} but missing {kernel_filename}")
    print(f"Total CUDA extensions to build: {len(ext_modules)}")
else:
    print("CUDA not available or not detected. Skipping CUDA extension build.")

setup_args = dict(
    name='difflut',
    version="1.0.10",
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
    setup_args['cmdclass'] = {'build_ext': BuildExtension}

setup(**setup_args)