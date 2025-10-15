# DiffLUT - Differentiable Look-Up Table Networks

A PyTorch library for LUT-based neural networks with custom forward/backward passes and gradient calculations. This research project enables efficient FPGA deployment through automatic SystemVerilog generation.

## Features

- **Modular LUT Nodes**: Customizable forward passes (training/inference) and gradient calculations
- **Flexible Layers**: Build networks using different LUT node types
- **CUDA Acceleration**: Optimized kernels for training speed
- **FPGA Deployment**: Automatic translation to SystemVerilog for FPGA implementation
- **Research Ready**: Clean separation between core library and experiments

## Project Structure
```
.
├── LICENSE
├── README.md
├── pyproject.toml
├── setup.py
├── build/                      # Build artifacts
├── containers/                 # Singularity containers for experiments
│   ├── pytorch_universal_minimal.def
│   └── pytorch_universal_minimal.sif
├── data/                       # Training datasets (MNIST, etc.)
├── difflut/                    # Main library
│   ├── __init__.py
│   ├── registry.py             # Component registration system
│   ├── encoder/                # Input encoding modules
│   │   ├── __init__.py
│   │   ├── base_encoder.py
│   │   └── thermometer.py
│   ├── layers/                 # LUT layer implementations
│   │   ├── __init__.py
│   │   ├── base_layer.py
│   │   ├── random_layer.py
│   │   ├── cyclic_layer.py
│   │   └── learnable_layer.py
│   ├── nodes/                  # LUT node implementations
│   │   ├── __init__.py
│   │   ├── base_node.py
│   │   ├── dwn_node.py
│   │   ├── linear_lut_node.py
│   │   ├── neurallut_node.py
│   │   ├── polylut_node.py
│   │   ├── probabilistic_node.py
│   │   └── cuda/               # CUDA acceleration kernels
│   │       ├── __init__.py
│   │       ├── efd_cuda.cpp
│   │       └── efd_cuda_kernel.cu
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── fpga_export.py
│       ├── modules.py
│       └── regularizers.py
├── docs/                       # Documentation
├── examples/                   # Tutorial notebooks and examples
│   └── mnist_difflut_tutorial.ipynb
├── experiments/                # Modular experiment pipeline (Hydra + DB)
│   ├── run_experiment.py       # Main entry point (Hydra composed config)
│   ├── trainer.py              # Training loop + checkpointing + DB logging
│   ├── query_results.py        # CLI helper to inspect results in DB
│   ├── configs/                # Hydra configuration tree
│   │   ├── experiment1.yaml    # Base experiment (composes model + dataloader)
│   │   ├── dataloaders/        # Dataset configs (mnist, fashionmnist, cifar10)
│   │   └── model/              # Model configs (e.g. layered_feedforward)
│   ├── dataloaders/            # Dataset loader implementations (encoded)
│   ├── models/                 # Experiment model definitions (DiffLUT wrappers)
│   ├── utils/db_manager.py     # SQLAlchemy SQLite manager (experiments.db)
│   ├── PIPELINE_README.md      # Extended pipeline documentation
│   └── requirements.txt        # Optional extra deps (hydra-core, sqlalchemy)
├── logs/                       # Experiment logs
├── scripts/                    # Helper scripts
│   ├── run_experiment.sh       # Single job (Singularity + SLURM)
│   ├── sweep_node_types.sh     # Array job over node types
│   ├── sweep_layer_types.sh    # Array job over layer types
│   ├── sweep_datasets.sh       # Array job over datasets
│   ├── sweep_hyperparams.sh    # Grid search (lr × hidden size)
│   └── README.md               # SLURM usage & examples
└── tests/                      # Unit tests
```

## Installation

```bash
git clone https://gitlab.ethz.ch/disco-students/hs25/difflut
cd difflut
pip install -e .
```

## Quick Start
```python
import torch
from difflut.nodes import LinearLUTNode
from difflut.layers import RandomLayer
from difflut.encoder import ThermometerEncoder

# Create an encoder for input quantization
encoder = ThermometerEncoder(num_bits=4)

# Build a layer with LUT nodes
layer = RandomLayer(
    input_size=28*28,  # MNIST image size
    output_size=10,    # Number of classes
    node_type='LinearLUTNode',
    node_config={'k': 4}  # 4-input LUTs
)

# Use like a regular PyTorch module
x = torch.randn(32, 28*28)
output = layer(x)
loss = output.mean()
loss.backward()
```

## Usage

The library supports three main abstractions:
1. **Encoders**: Transform continuous inputs into discrete representations suitable for LUTs
2. **Nodes**: Define LUT behavior, weight parametrization, and gradient calculations
3. **Layers**: Define how nodes are connected to form complete neural layers

### Available Components

#### Nodes
- `LinearLUTNode`: Linear interpolation-based LUT
- `DWNNode`: Deep Weight Networks node
- `PolyLUTNode`: Polynomial-based LUT approximation
- `NeuralLUTNode`: Neural network-based LUT learning
- `ProbabilisticNode`: Probabilistic LUT with uncertainty

#### Layers
- `BaseLUTLayer`: Base layer for LUT-based networks
- `RandomLayer`: Random input-to-node mapping
- `CyclicMappingLayer`: Cyclic pattern mapping
- `LearnableMappingLayer`: Learnable input connections

#### Encoders
- `ThermometerEncoder`: Standard thermometer encoding
- `GaussianThermometerEncoder`: Gaussian-weighted encoding
- `DistributiveThermometerEncoder`: Distribution-aware encoding

### Custom Nodes
```python
from difflut.nodes import BaseNode
from difflut import register_node

@register_node('MyCustomNode')
class YourCustomNode(BaseNode):
    def forward_train(self, x):
        # Forward pass during training
        pass

    def forward_eval(self, x):
        # Forward pass during evaluation
        pass
    
    def backward_input(self, grad_output):
        # Custom input gradient
        pass
    
    def backward_weights(self, grad_output):
        # Custom weight gradient  
        pass

    def regularization(self):
        # Compute node-specific regularization term for weights
        pass

    def export_bitstream(self):
        # Generate LUT bitstream/configuration for FPGA
        pass
```

### FPGA Deployment

```python
from difflut.utils.fpga_export import export_to_rtl

# Export trained model to SystemVerilog
export_to_rtl(model, "output_design.sv")
```

### Examples

Check out the MNIST tutorial notebook for a complete example:
```bash
jupyter notebook examples/mnist_difflut_tutorial.ipynb
```

Or run a simple test:
```bash
python examples/simple_mnist_test.py
```

## Experiments Pipeline (Hydra + SQLite Tracking)

The new modular pipeline in `experiments/` replaces ad‑hoc single scripts and provides:

| Capability | Description |
|------------|-------------|
| Configuration | Hydra hierarchical configs (`configs/`) with command‑line overrides |
| Datasets | MNIST, FashionMNIST, CIFAR10 (thermometer encoded) via pluggable dataloaders |
| Models | DiffLUT feed‑forward architecture (configurable layers & nodes) |
| Tracking | SQLite database (`results/experiments.db`) with experiments, measurement points, metrics & checkpoints |
| Checkpoints | Best validation model saved automatically (validation accuracy) |
| Reproducibility | Full resolved Hydra config stored per run (`experiments/outputs/.../.hydra`) |
| Automation | SLURM + Singularity scripts for sweeps and batch runs |

### Quick Run (Local)
```bash
cd experiments
python run_experiment.py            # Uses defaults: model=layered_feedforward, dataloaders=mnist
```

### Common Hydra Overrides
```bash
# Switch dataset
python experiments/run_experiment.py dataloaders=fashionmnist

# Change node type (flattened params interface)
python experiments/run_experiment.py model.params.node_type=neurallut

# Change layer mapping strategy
python experiments/run_experiment.py model.params.layer_type=learnable

# Adjust hidden sizes & LUT inputs (n)
python experiments/run_experiment.py 'model.params.hidden_sizes=[1500,1500]' model.params.num_inputs=8

# Training hyperparameters
python experiments/run_experiment.py training.epochs=15 training.lr=0.001 training.batch_size=256

# Combine multiple overrides + custom experiment name
python experiments/run_experiment.py \
    experiment_name=mnist_neurallut_learnable \
    dataloaders=mnist \
    model.params.node_type=neurallut \
    model.params.layer_type=learnable \
    training.epochs=20 training.lr=0.005
```

### What Gets Logged
The SQLite database (`results/experiments.db`) has three tables:
1. `experiments` – full config snapshot + high‑level metadata (device, status)
2. `measurement_points` – per epoch (and phase) timing & indexing
3. `metrics` – metric name/value pairs plus checkpoint info (is_best, path)

### Query Results
```bash
cd experiments
python query_results.py                 # Lists experiments + best metrics
```
Programmatic example:
```python
from experiments.utils.db_manager import DBManager
db = DBManager()                     # Automatically finds results/experiments.db
exps = db.list_experiments(limit=5)
metrics = db.get_experiment_metrics(exps[0].id, phase='val')
best_ckpt = db.get_best_checkpoint(exps[0].id, metric_name='accuracy')
```

### SLURM + Singularity
Submit a single experiment (inside project root):
```bash
sbatch scripts/run_experiment.sh                       # defaults
sbatch scripts/run_experiment.sh model.params.node_type=polylut training.epochs=12
```
Node type sweep (array job):
```bash
sbatch scripts/sweep_node_types.sh
```
Inspect status & logs:
```bash
squeue -u $USER
tail -f logs/experiment_<job_id>.log
```

### Configuration Notes
Flattened model parameter keys (preferred for overrides):
| Key | Meaning |
|-----|---------|
| `model.params.node_type` | DiffLUT node variant (linear_lut, neurallut, polylut, dwn, probabilistic, unbound_probabilistic) |
| `model.params.layer_type` | Layer mapping strategy (random, cyclic, learnable) |
| `model.params.hidden_sizes` | List of hidden layer widths (quoted when overriding) |
| `model.params.num_inputs` | LUT input arity (n) |
| `dataloaders` | Dataset choice (mnist, fashionmnist, cifar10) |
| `dataloaders.subset_size` | Training subset for quick iteration |
| `training.*` | Standard training hyperparameters |

Legacy nested form (`model.params.layers[0].node.type`) is still supported internally but not recommended because Hydra's override grammar is stricter with list indexing.

### Migration from `examples/simple_mnist_test.py`
| Old | New |
|-----|-----|
| Monolithic training script | Modular pipeline (data, model, trainer) |
| Manual hyperparameter edits | Hydra overrides (no code changes) |
| No tracking | Structured DB logging + checkpoints |
| Hard to sweep | SLURM array scripts & overrides |

If you still need the old script for reference it remains in `examples/`, but new experiments should use the pipeline.

### Troubleshooting
| Symptom | Fix |
|---------|-----|
| `Could not find 'dataloaders/XYZ'` | Ensure `dataloaders=xyz` matches a file in `configs/dataloaders/` |
| `LexerNoViableAltException` | Quote list overrides: `'model.params.hidden_sizes=[1500,1500]'` |
| No rows in DB | Verify write permissions to `results/` and that run wasn't aborted early |
| CUDA not used | Check `torch.cuda.is_available()` inside log; ensure `--gres=gpu:1` in SBATCH |

### Minimal End‑to‑End Example
```bash
# 1. Quick sanity run (CPU ok)
python experiments/run_experiment.py training.epochs=1 dataloaders.subset_size=500

# 2. Inspect DB
python experiments/query_results.py

# 3. GPU SLURM job
sbatch scripts/run_experiment.sh training.epochs=5 model.params.node_type=neurallut
```

### Experiments
Master thesis & research experiments are now fully managed via the `experiments/` pipeline (see section above). Use singularity + SLURM scripts in `scripts/` for cluster execution.

Example (manual singularity execution):
```bash
singularity exec --nv \
    --bind $(pwd):/workspace \
    containers/pytorch_universal_minimal.sif \
    bash -c "cd /workspace/experiments && python run_experiment.py model.params.node_type=neurallut"
```

## Citation
If you use DiffLUT in your research, please cite:

```bibtex
@mastersthesis{buehrer2025difflut,
  title={DiffLUT: Differentiable LUT Networks for Efficient FPGA Deployment},
  author={Simon Jonas Bührer, Andreas Plesner, Aczel Till, Roger Wattenhofer},
  school={ETH Zurich},
  year={2026}
}
```

# THE COMAND YOU MUST NOW
srun  --mem=25GB --gres=gpu:01 --exclude=tikgpu[06-10] --pty bash -i

singularity exec --writable-tmpfs --nv --bind /usr/itetnas04:/usr/itetnas04 --pwd /usr/itetnas04/data-scratch-01/sbuehrer/data/difflut containers/pytorch_universal_minimal.sif bash -c "export CUDA_HOME=/usr/local/cuda && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:\$LD_LIBRARY_PATH && pip install --no-build-isolation -e ."


python notebooks/visualize_weights.py --checkpoints results/checkpoints --param_name all --bins 256 --output outputs/weights_over_time.png

## License
This project is licensed under the MIT License - see the LICENSE file for details.