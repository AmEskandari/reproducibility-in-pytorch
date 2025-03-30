# PyTorch Reproducibility Test Suite

This repository contains tests that demonstrate reproducibility challenges in PyTorch, specifically focused on cuDNN algorithms and GNN scatter operations.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Test Descriptions](#test-descriptions)
- [Running Tests](#running-tests)
- [License](#license)

## Overview

While PyTorch is a powerful deep learning framework, achieving perfect reproducibility across runs can be challenging. This repository demonstrates two common reproducibility issues:

1. cuDNN algorithm selection variations
2. GNN scatter operations non-determinism

## Setup

### Prerequisites

- Python 3.9+
- CUDA-enabled GPU (required for both tests)
- PyTorch with CUDA support
- PyTorch Geometric (for GNN test)
- NetworkX (for graph generation)

### Installation

We recommend using conda to set up the environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/reproducibility-in-pytorch.git
cd reproducibility-in-pytorch

# Create and activate the conda environment
conda env create -f environment.yml
conda activate pytorch-reproducibility
```

## Test Descriptions

### cuDNN Algorithm Selection Test (`test_cudnn_algorithms.py`)

This test shows how cuDNN's algorithm selection affects reproducibility, especially when `torch.backends.cudnn.benchmark` is enabled. It demonstrates:

- How enabling `cudnn.benchmark` can lead to different results between runs
- Performance differences between deterministic and non-deterministic settings
- How changing batch sizes affects algorithm selection and results

### GNN Reproducibility Test (`test_gnn_reproducibility.py`)

Tests the reproducibility of Graph Neural Networks, specifically focusing on:

- Non-deterministic behavior in torch_scatter operations used by GNN layers
- How identical inputs and seeds can still produce different results due to internal parallelism
- Measurements of the magnitude of differences between runs

## Running Tests

You can run the individual tests directly:

```bash
# Run cuDNN algorithm test
python test_cudnn_algorithms.py

# Run GNN reproducibility test
python test_gnn_reproducibility.py
```

For best reproducibility results in your PyTorch code, use the following settings:

```python
def set_seed(seed=42): 
    import random
    import numpy as np
    import torch
    
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    # For PyTorch 1.8+
    torch.use_deterministic_algorithms(True)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 