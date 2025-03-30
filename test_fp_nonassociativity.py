import torch
import numpy as np
import random

def set_seed(seed=42):
    """Set all seeds to make results reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_matrix_multiplication():
    """Test reproducibility of matrix multiplication on different devices."""
    # Create matrices with values that will highlight floating-point issues
    a = torch.tensor([1.0000001, 2.0000002, 3.0000003, 4.0000004], dtype=torch.float32)
    b = torch.tensor([5.0000005, 6.0000006, 7.0000007, 8.0000008], dtype=torch.float32)
    
    # Compute dot product in different ways
    # Left-to-right addition (sequential)
    dot_product_sequential = 0.0
    for i in range(len(a)):
        dot_product_sequential += a[i] * b[i]
    
    # Balanced tree addition
    products = a * b
    dot_product_tree = (products[0] + products[1]) + (products[2] + products[3])
    
    # PyTorch's built-in dot product
    dot_product_torch = torch.dot(a, b)
    
    print(f"Sequential addition: {dot_product_sequential}")
    print(f"Tree-based addition: {dot_product_tree}")
    print(f"PyTorch dot product: {dot_product_torch}")
    print(f"All equal? {dot_product_sequential == dot_product_tree == dot_product_torch}")
    
    # Check if GPU computation gives the same result
    if torch.cuda.is_available():
        a_cuda = a.cuda()
        b_cuda = b.cuda()
        dot_product_cuda = torch.dot(a_cuda, b_cuda).cpu()
        print(f"CUDA dot product: {dot_product_cuda}")
        print(f"CPU and CUDA equal? {dot_product_torch == dot_product_cuda}")

def test_linear_layer_reproducibility():
    """Test whether a simple linear layer produces the same results across runs."""
    set_seed(42)
    
    # Create a simple model
    model = torch.nn.Linear(10, 5)
    
    # Create input data
    x = torch.randn(8, 10)
    
    # Get output on CPU
    output_cpu = model(x)
    
    # Reset seed and rerun
    set_seed(42)
    model_reset = torch.nn.Linear(10, 5)  # Weights should be identical due to same seed
    output_reset = model_reset(x)
    
    print(f"Linear layer CPU outputs are identical: {torch.allclose(output_cpu, output_reset)}")
    
    # Test GPU if available
    if torch.cuda.is_available():
        set_seed(42)
        model_cuda = torch.nn.Linear(10, 5).cuda()
        x_cuda = x.cuda()
        output_cuda = model_cuda(x_cuda)
        
        # Reset and rerun on GPU
        set_seed(42)
        model_cuda_reset = torch.nn.Linear(10, 5).cuda()
        output_cuda_reset = model_cuda_reset(x_cuda)
        
        print(f"Linear layer GPU outputs are identical: {torch.allclose(output_cuda, output_cuda_reset)}")
        
        # Compare CPU and GPU
        print(f"CPU and GPU outputs are identical: {torch.allclose(output_cpu, output_cuda.cpu())}")

if __name__ == "__main__":
    set_seed(42)
    print("\n=== Testing Floating-Point Non-Associativity ===")
    test_matrix_multiplication()
    
    print("\n=== Testing Linear Layer Reproducibility ===")
    test_linear_layer_reproducibility() 