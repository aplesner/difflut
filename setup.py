
"""
DiffLUT Setup Script
Supports GPU, CPU-only, and development installations
"""
import os
import sys
from setuptools import setup, find_packages

# Check if user explicitly requested cpu-only build
build_cpu_only = any('.[cpu]' in arg for arg in sys.argv)

# Default to GPU build unless cpu-only is requested
build_cuda = not build_cpu_only

# Try to build CUDA extensions
ext_modules = []
cmdclass = {}

if build_cuda:
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension
        
        # Check for CUDA availability
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        if cuda_home or torch.cuda.is_available():
            print("🔧 Building with CUDA support...")
            
            # Auto-discover CUDA extensions in specified directories
            cuda_dirs = ['difflut/nodes/cuda', 'difflut/layers/cuda']
            
            for cuda_dir in cuda_dirs:
                if not os.path.exists(cuda_dir):
                    continue
                
                # Find all .cpp files in the directory
                for filename in sorted(os.listdir(cuda_dir)):
                    if filename.endswith('.cpp'):
                        module_name = filename[:-4]  # Remove .cpp extension
                        cpp_file = os.path.join(cuda_dir, filename)
                        cu_file = os.path.join(cuda_dir, f'{module_name}_kernel.cu')
                        
                        # Check if corresponding .cu kernel file exists
                        if os.path.exists(cu_file):
                            ext_modules.append(CUDAExtension(
                                module_name,
                                [cpp_file, cu_file],
                                extra_compile_args={
                                    'cxx': ['-O3'],
                                    'nvcc': ['-O3', '--use_fast_math']
                                }
                            ))
                            print(f"  ✓ {module_name} (from {cuda_dir})")
                        else:
                            print(f"  ⚠️  Skipping {module_name}: missing {cu_file}")
            
            if ext_modules:
                cmdclass['build_ext'] = BuildExtension
                print(f"✅ {len(ext_modules)} CUDA extensions configured")
            else:
                print("⚠️  No CUDA extensions found, building CPU-only version")
        else:
            print("⚠️  CUDA not available, building CPU-only version")
    
    except ImportError:
        print("⚠️  PyTorch not found, building CPU-only version")
    except Exception as e:
        print(f"⚠️  CUDA build failed: {e}, building CPU-only version")
else:
    print("🔧 Building CPU-only version (CUDA disabled)")

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='difflut',
    version='1.1.2',
    description='Differentiable LUT-based neural networks for efficient FPGA deployment',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Simon Buehrer',
    author_email='sbuehrer@ethz.ch',
    url='https://github.com/aplesner/difflut',
    project_urls={
        'Documentation': 'https://github.com/aplesner/difflut/tree/main/docs',
        'Source': 'https://github.com/aplesner/difflut',
        'Issues': 'https://github.com/aplesner/difflut/issues',
    },
    license='MIT',
    packages=find_packages(exclude=['tests*', 'examples*', 'docs*', 'build*']),
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
    ],
    extras_require={
        'gpu': [],  # Dummy extra to trigger GPU build
        'cpu': [],  # Dummy extra to trigger CPU build
        'dev': [
            'bump2version>=1.0.0',
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'isort>=5.10.0',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'jupyter>=1.0.0',
            'matplotlib>=3.3.0',
            'torchvision>=0.10.0',
        ],
        'all': [
            'bump2version>=1.0.0',
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'isort>=5.10.0',
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
            'jupyter>=1.0.0',
            'matplotlib>=3.3.0',
            'torchvision>=0.10.0',
        ],
    },
    python_requires='>=3.7',
    ext_modules=ext_modules if ext_modules else [],
    cmdclass=cmdclass,
    keywords='deep-learning neural-networks lut fpga pytorch hardware-acceleration',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    include_package_data=True,
)