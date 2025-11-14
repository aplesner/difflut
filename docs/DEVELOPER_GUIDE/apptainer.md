# DiffLUT Apptainer Container

This directory contains everything needed to build and deploy the DiffLUT library in an Apptainer/Apptainer container for HPC cluster environments.

## 📁 Contents

- **`difflut.def`** - Container definition file (builds from PyTorch base image)
- **`build_container.sh`** - Simple build script with CUDA architecture support
- **`test_import.py`** - Import validation script (runs inside container during build)
- **`requirements.txt`** - Python dependencies (symlinked from parent directory)
- **`difflut.sif`** - Built container image (~4GB, gitignored)

## 🚀 Quick Start

### Building the Container

**IMPORTANT:** The build script must be run from the project root directory, not from inside `apptainer/`.

```bash
# Navigate to project root (if not already there)
cd /path/to/difflut

# Default build (RTX 3090, A6000 - CUDA 8.6)
bash apptainer/build_container.sh

# Build for specific architecture
bash apptainer/build_container.sh --arch 8.0      # A100
bash apptainer/build_container.sh --arch 7.5      # RTX 2080 Ti, Titan RTX
bash apptainer/build_container.sh --arch 8.6      # RTX 3090, A6000 (default)

# Build for multiple architectures (slower, but works on more GPUs)
bash apptainer/build_container.sh --arch "7.5;8.0;8.6"

# Clean build (removes old container first)
bash apptainer/build_container.sh --clean
```

**Build time:** 30-60 minutes depending on system and CUDA architectures selected.

### Testing the Container

```bash
# Test imports (validation that runs after build)
apptainer exec apptainer/difflut.sif python3 -c "import difflut; from difflut.nodes import LinearLUTNode; from difflut.layers import ConvolutionalLayer; print('Success')"

# Check DiffLUT version
apptainer exec apptainer/difflut.sif python3 -c "import difflut; print(difflut.__version__)"

# Interactive shell
apptainer shell apptainer/difflut.sif
```

## 🔄 Remote Sync Workflow

### Prerequisites

```bash
# Source project variables (from project root)
cd ..
source helper_scripts/project_variables.sh
```

### Sync Container to Remote Server

```bash
# After building, sync to remote cluster
bash helper_scripts/sync_container_to_remote.sh
```

This will:
1. Upload `difflut.sif` to the remote project storage
2. Use scratch space as temporary directory for faster transfers
3. Verify upload with progress bar


## 🏗️ Container Architecture

### Base Image
- **PyTorch:** 2.9.0
- **CUDA:** 12.8
- **cuDNN:** 9
- **Python:** 3.11

### CUDA Extensions
The container builds custom CUDA extensions for:
- **DWN nodes** - Differentiable Winner-take-all Networks
- **Fused kernels** - Memory-efficient forward passes (reduces 20GB → 3-4GB)

### Supported GPU Architectures

| Architecture | Compute Capability | GPUs |
|--------------|-------------------|------|
| `6.1` | sm_61 | Titan XP, GTX 1080 Ti |
| `7.0` | sm_70 | Tesla V100 |
| `7.5` | sm_75 | RTX 2080 Ti, Titan RTX |
| `8.0` | sm_80 | A100 |
| `8.6` | sm_86 | RTX 3090, A6000 |

**Note:** Building for fewer architectures is faster but limits GPU compatibility.

## 🔧 Container Structure

Inside the container:
```
/opt/difflut/
├── difflut/           # Main library package
├── setup.py           # Installation script
├── test_import.py     # Validation test
└── requirements.txt   # Python dependencies
```

## 📝 Build Logs

Build logs are saved to `build_logs/build_YYYYMMDD_HHMMSS.log` for debugging.

## 🐛 Troubleshooting

### Build Fails

1. **Check build log** in `build_logs/` directory
2. **Verify CUDA_HOME** is set: `echo $CUDA_HOME`
3. **Try clean build:** `./build_container.sh --clean`

### Import Test Fails

```bash
# Check what went wrong
apptainer exec difflut.sif python3 /opt/difflut/test_import.py

# Verify PyTorch and CUDA
apptainer exec difflut.sif python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### Sync Issues

```bash
# Verify environment variables are set
echo $PROJECT_NAME        # Should be: difflut
echo $CONTAINER_LOCAL     # Should be: ./apptainer/difflut.sif
echo $REMOTE_SERVER       # Should be your cluster hostname

# Test SSH connection
ssh $REMOTE_SERVER "echo Connection successful"
```

## 📦 Remote Paths

When synced to the remote cluster:

| Location | Path |
|----------|------|
| **Project Storage** | `/itet-stor/${USERNAME}/net_scratch/projects_storage/difflut/apptainer/difflut.sif` |
| **Scratch Storage** | `/scratch/${USERNAME}/difflut/apptainer/difflut.sif` |

**Workflow on cluster:**
1. Container is uploaded to project storage (permanent)
2. `remote_sync_container.sh` copies to scratch space (fast access)
3. Jobs use container from scratch space

## 🎯 Usage Examples

### Run Training Script

```bash
apptainer exec --nv difflut.sif python3 train.py --config config.yaml
```

### With Bind Mounts

```bash
apptainer exec --nv \
    --bind /data:/mnt/data \
    --bind /results:/mnt/results \
    difflut.sif python3 train.py \
    --data /mnt/data \
    --output /mnt/results
```

### Interactive Jupyter

```bash
apptainer exec --nv difflut.sif jupyter lab --ip 0.0.0.0 --port 8888
```

## 🔗 Related Files

- **Project variables:** `../helper_scripts/project_variables.sh`
- **Code sync:** `../helper_scripts/sync_code_to_remote.sh`
- **Data sync:** `../helper_scripts/sync_data_to_remote.sh`
- **Requirements:** `requirements.txt` (symlink to `../requirements.txt`)

## 📚 Additional Resources

- [Apptainer Documentation](https://apptainer.org/docs/)
- [DiffLUT Main README](../README.md)
- [Installation Guide](../docs/INSTALLATION.md)
