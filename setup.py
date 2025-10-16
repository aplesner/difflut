import os
import torch
import setuptools
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# Updated paths for new structure
cuda_dir = os.path.join('difflut', 'nodes', 'cuda')

ext_modules = []

# Collect CUDA extensions from cuda directory
if os.path.exists(cuda_dir):
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

setup(
    name='difflut',
    version="1.0.10",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch>=1.9.0',
        'numpy',
    ],
    python_requires='>=3.7',
    description='Differentiable LUT-based neural networks',
    author='Simon Buehrer',
    license='MIT',
)