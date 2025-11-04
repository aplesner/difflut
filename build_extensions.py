#!/usr/bin/env python3
"""
Fast incremental CUDA extension builder for DiffLUT.

This script rebuilds only modified CUDA extensions, using timestamps to detect changes.
Much faster than full pip reinstalls for iterative development.

Usage:
    # Full rebuild
    python build_extensions.py --clean

    # Fast incremental rebuild (detects changed files)
    python build_extensions.py

    # Rebuild specific extension
    python build_extensions.py --extension fourier
"""

import os
import sys
import argparse
import subprocess
import glob
from pathlib import Path


def get_extension_info():
    """Get list of all CUDA extensions and their source files."""
    extensions = {}
    cuda_dirs = [
        os.path.join('difflut', 'nodes', 'cuda'),
        os.path.join('difflut', 'layers', 'cuda')
    ]
    
    for cuda_dir in cuda_dirs:
        if os.path.exists(cuda_dir):
            for cpp_file in glob.glob(os.path.join(cuda_dir, '*.cpp')):
                basename = os.path.basename(cpp_file)
                module_name = basename[:-4]  # Remove .cpp
                kernel_file = os.path.join(cuda_dir, f'{module_name}_kernel.cu')
                
                if os.path.exists(kernel_file):
                    extensions[module_name] = {
                        'cpp': cpp_file,
                        'cu': kernel_file,
                        'cuda_dir': cuda_dir
                    }
    
    return extensions


def get_build_artifact_path(module_name):
    """Get path where compiled .so file should be (searches multiple locations)."""
    # Check in root first (most common location after build_ext --inplace)
    root_so = f'{module_name}_cuda.cpython-311-x86_64-linux-gnu.so'
    if os.path.exists(root_so):
        return root_so
    
    # Also try other Python versions
    for pyver in ['310', '39', '312']:
        alt_so = f'{module_name}_cuda.cpython-{pyver}-x86_64-linux-gnu.so'
        if os.path.exists(alt_so):
            return alt_so
    
    # Fallback: return expected location
    return root_so


def find_so_file(module_name):
    """Find the .so file for a module (handles different Python versions)."""
    import glob
    
    # Try common Python version patterns
    patterns = [
        f'{module_name}_cuda.cpython-*-x86_64-linux-gnu.so',
        f'{module_name}_cuda.so',
        os.path.join('build/lib*', f'{module_name}_cuda*.so'),
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    # Also check in build directory
    build_dir = 'build/lib.linux-x86_64-cpython-311'
    if os.path.exists(build_dir):
        for f in os.listdir(build_dir):
            if module_name in f and f.endswith('.so'):
                return os.path.join(build_dir, f)
    
    return None


def has_changed(extension_info):
    """Check if any source files are newer than the compiled extension."""
    module_name = list(extension_info.keys())[0]
    so_path = get_build_artifact_path(module_name)
    
    if not os.path.exists(so_path):
        return True  # Never built
    
    so_mtime = os.path.getmtime(so_path)
    
    for ext_name, files in extension_info.items():
        cpp_mtime = os.path.getmtime(files['cpp'])
        cu_mtime = os.path.getmtime(files['cu'])
        
        if cpp_mtime > so_mtime or cu_mtime > so_mtime:
            return True
    
    return False


def run_command(cmd, description):
    """Run a shell command with error handling."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    print(f"  Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} - FAILED")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Fast incremental CUDA extension builder for DiffLUT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_extensions.py              # Incremental rebuild (fast)
  python build_extensions.py --clean      # Full rebuild (slow, cleans cache)
  python build_extensions.py --extension fourier  # Rebuild specific extension
        """
    )
    
    parser.add_argument('--clean', action='store_true', 
                       help='Clean build artifacts and rebuild from scratch')
    parser.add_argument('--extension', type=str,
                       help='Rebuild only specific extension (e.g., fourier)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    extensions = get_extension_info()
    
    if not extensions:
        print("❌ No CUDA extensions found!")
        return 1
    
    print(f"\n📦 Found {len(extensions)} CUDA extensions:")
    for name in sorted(extensions.keys()):
        print(f"   - {name}")
    
    # Filter by extension name if specified
    if args.extension:
        if args.extension not in extensions:
            print(f"\n❌ Extension '{args.extension}' not found!")
            print(f"Available: {', '.join(sorted(extensions.keys()))}")
            return 1
        extensions = {args.extension: extensions[args.extension]}
    
    # Clean build if requested
    if args.clean:
        print("\n🧹 Cleaning build artifacts...")
        subprocess.run(['rm', '-rf', 'build'], capture_output=True)
        for ext_name in extensions:
            so_path = get_build_artifact_path(ext_name)
            if os.path.exists(so_path):
                os.remove(so_path)
                print(f"   Removed {so_path}")
    
    # Check what needs rebuilding
    print("\n📋 Checking which extensions need rebuilding...")
    to_rebuild = []
    
    for ext_name, ext_info in extensions.items():
        so_path = get_build_artifact_path(ext_name)
        
        if not os.path.exists(so_path):
            print(f"   {ext_name}: NOT BUILT (will build)")
            to_rebuild.append(ext_name)
        else:
            so_mtime = os.path.getmtime(so_path)
            cpp_mtime = os.path.getmtime(ext_info['cpp'])
            cu_mtime = os.path.getmtime(ext_info['cu'])
            
            if cpp_mtime > so_mtime or cu_mtime > so_mtime:
                print(f"   {ext_name}: CHANGED (will rebuild)")
                to_rebuild.append(ext_name)
            else:
                print(f"   {ext_name}: UP-TO-DATE (skipping)")
    
    if not to_rebuild:
        print("\n✨ All extensions are up-to-date! Nothing to build.")
        return 0
    
    print(f"\n🔨 Building {len(to_rebuild)}/{len(extensions)} extensions...")
    
    # Run setup.py build_ext --inplace
    cmd = [sys.executable, 'setup.py', 'build_ext', '--inplace']
    if args.verbose:
        cmd.append('--verbose')
    
    if not run_command(cmd, f"Building {len(to_rebuild)} extension(s)"):
        return 1
    
    print("\n" + "="*70)
    print("  Build Summary")
    print("="*70)
    
    # Verify builds - search for .so files in all possible locations
    success_count = 0
    for ext_name in to_rebuild:
        so_file = find_so_file(ext_name)
        if so_file and os.path.exists(so_file):
            size_mb = os.path.getsize(so_file) / (1024*1024)
            print(f"✅ {ext_name:20} -> {so_file} ({size_mb:.1f} MB)")
            success_count += 1
        else:
            # Try to find in root directory
            root_so = f'{ext_name}_cuda.cpython-311-x86_64-linux-gnu.so'
            if os.path.exists(root_so):
                size_mb = os.path.getsize(root_so) / (1024*1024)
                print(f"✅ {ext_name:20} -> {root_so} ({size_mb:.1f} MB)")
                success_count += 1
            else:
                print(f"❌ {ext_name:20} - BUILD FAILED (could not find .so file)")
                # Debug: show what files exist
                so_matches = glob.glob(f'{ext_name}_cuda*.so')
                if so_matches:
                    print(f"   (Found similar files: {so_matches})")
    
    print("\n" + "="*70)
    if success_count == len(to_rebuild):
        print(f"✨ All {success_count} extension(s) built successfully!")
        print("\n📝 Next steps:")
        print("   - Run: python -c \"import sys; sys.path.insert(0, '.'); import fourier_cuda\"")
        print("   - Or: pip install --no-build-isolation --no-deps -e .")
        return 0
    else:
        print(f"⚠️  Only {success_count}/{len(to_rebuild)} extensions built successfully")
        return 1


if __name__ == '__main__':
    sys.exit(main())
