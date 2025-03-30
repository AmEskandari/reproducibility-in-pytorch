import torch
import torch.nn as nn
import numpy as np
import random
import time

def set_seed(seed=42):
    """Set all seeds to make results reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Don't set cudnn.deterministic/benchmark here as we'll modify these in the test

class SimpleCNN(nn.Module):
    """A simple CNN for testing."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 8 * 8, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x

def test_cudnn_benchmark(num_runs=3):
    """Test how cuDNN benchmark affects reproducibility and performance."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping cuDNN benchmark test")
        return
    
    # Create input data
    x = torch.randn(4, 3, 32, 32).cuda()
    
    results_benchmark_on = []
    times_benchmark_on = []
    results_benchmark_off = []
    times_benchmark_off = []
    
    # Test with benchmark=True
    print("Testing with cudnn.benchmark=True")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    for i in range(num_runs):
        set_seed(42)
        model = SimpleCNN().cuda()
        
        # First forward pass might trigger algorithm selection
        start_time = time.time()
        out = model(x)
        end_time = time.time()
        
        times_benchmark_on.append(end_time - start_time)
        results_benchmark_on.append(out.detach().cpu().sum().item())
        
        print(f"  Run {i+1}: output sum = {results_benchmark_on[-1]}, time = {times_benchmark_on[-1]:.6f}s")
    
    # Test with benchmark=False and deterministic=True
    print("\nTesting with cudnn.benchmark=False, cudnn.deterministic=True")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    for i in range(num_runs):
        set_seed(42)
        model = SimpleCNN().cuda()
        
        start_time = time.time()
        out = model(x)
        end_time = time.time()
        
        times_benchmark_off.append(end_time - start_time)
        results_benchmark_off.append(out.detach().cpu().sum().item())
        
        print(f"  Run {i+1}: output sum = {results_benchmark_off[-1]}, time = {times_benchmark_off[-1]:.6f}s")
    
    # Compare results
    print("\nResults Summary:")
    print(f"  Benchmark ON - All equal: {all(r == results_benchmark_on[0] for r in results_benchmark_on)}")
    print(f"  Benchmark OFF - All equal: {all(r == results_benchmark_off[0] for r in results_benchmark_off)}")
    print(f"  Are benchmark ON and OFF equal: {results_benchmark_on[0] == results_benchmark_off[0]}")
    print(f"  Average time with benchmark ON: {sum(times_benchmark_on) / len(times_benchmark_on):.6f}s")
    print(f"  Average time with benchmark OFF: {sum(times_benchmark_off) / len(times_benchmark_off):.6f}s")

def test_cudnn_for_different_batch_sizes():
    """Test how changing input shapes affects algorithm selection."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping batch size test")
        return
    
    # Create a model with benchmark enabled to see algorithm selection effects
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    batch_sizes = [1, 2, 4, 8, 16]
    results = []
    
    print("\nTesting cuDNN algorithm selection with different batch sizes")
    for batch_size in batch_sizes:
        set_seed(42)
        model = SimpleCNN().cuda()
        
        # Create different sized inputs
        x = torch.randn(batch_size, 3, 32, 32).cuda()
        
        # Forward pass
        out = model(x)
        
        # Record the output sum normalized by batch size for comparison
        normalized_output = out.detach().cpu().sum().item() / batch_size
        results.append(normalized_output)
        
        print(f"  Batch size {batch_size}: normalized output sum = {normalized_output}")
    
    # Check if outputs differ
    all_equal = all(abs(r - results[0]) < 1e-5 for r in results)
    print(f"\n  All normalized outputs equal: {all_equal}")
    if not all_equal:
        print("  This indicates cuDNN selected different algorithms based on batch size")

if __name__ == "__main__":
    print("=== Testing cuDNN Algorithm Selection ===")
    test_cudnn_benchmark()
    test_cudnn_for_different_batch_sizes() 