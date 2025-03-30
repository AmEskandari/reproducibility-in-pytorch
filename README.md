# PyTorch Reproducibility Test Suite

This repository contains tests that demonstrate reproducibility challenges in PyTorch, as discussed in the blog post "Reproducibility in PyTorch: Myth or Reality?".

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Test Descriptions](#test-descriptions)
- [Running Tests](#running-tests)
- [License](#license)

## Overview

While PyTorch is a powerful deep learning framework, achieving perfect reproducibility across runs and platforms can be challenging. This test suite demonstrates common reproducibility issues including:

1. Floating-point non-associativity
2. cuDNN algorithm selection variations
3. GNN scatter operations non-determinism
4. Training divergence over time

## Setup

### Prerequisites

- Python 3.9+
- CUDA-enabled GPU (optional, but recommended)

### Installation

We recommend using conda to set up the environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/reproducibility-in-pytorch.git
cd reproducibility-in-pytorch

# Create and activate the conda environment
conda env create -f environment.yml
conda activate pytorch-reproducibility

# Check if all dependencies are installed
python run_tests.py --check-deps
```

## Test Descriptions

### Floating-Point Non-Associativity Test
Demonstrates how the non-associative nature of floating-point arithmetic can lead to different results depending on computation order, particularly in parallel processing environments like GPUs.

### cuDNN Algorithm Selection Test
Shows how cuDNN's algorithm selection affects reproducibility, especially when `torch.backends.cudnn.benchmark` is enabled and input shapes change.

### GNN Reproducibility Test
Tests the reproducibility of Graph Neural Networks, which often use non-deterministic operations like scatter_add for message passing between nodes.

### Training Divergence Test
Demonstrates how small floating-point differences can accumulate over training iterations, leading to significant divergence in model behavior despite using the same seeds.

## Running Tests

You can run individual tests or all tests at once:

```bash
# Run all tests
python run_tests.py --all

# Run individual tests
python run_tests.py --fp      # Floating-point tests
python run_tests.py --cudnn   # cuDNN algorithm tests
python run_tests.py --gnn     # GNN tests
python run_tests.py --training  # Training divergence tests

# Check dependencies
python run_tests.py --check-deps
```

For best reproducibility results in your PyTorch code, use the following settings:

```python
def set_seed(seed=42): 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 