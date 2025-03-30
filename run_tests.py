#!/usr/bin/env python3
import argparse
import sys
import importlib
import subprocess

def main():
    """Main entry point for running reproducibility tests."""
    parser = argparse.ArgumentParser(description="Run reproducibility tests for PyTorch")
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--fp', action='store_true', help='Run floating point non-associativity tests')
    parser.add_argument('--cudnn', action='store_true', help='Run cuDNN algorithm selection tests')
    parser.add_argument('--gnn', action='store_true', help='Run GNN reproducibility tests')
    parser.add_argument('--training', action='store_true', help='Run training divergence tests')
    parser.add_argument('--check-deps', action='store_true', help='Check if all dependencies are installed')
    
    args = parser.parse_args()
    
    # If no arguments provided, print help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Check dependencies
    if args.check_deps:
        check_dependencies()
        return
    
    # Run all tests or individual tests
    if args.all or args.fp:
        print("\n" + "="*50)
        print("Running Floating Point Non-Associativity Tests")
        print("="*50)
        import test_fp_nonassociativity
        test_fp_nonassociativity.test_matrix_multiplication()
        test_fp_nonassociativity.test_linear_layer_reproducibility()
    
    if args.all or args.cudnn:
        print("\n" + "="*50)
        print("Running cuDNN Algorithm Selection Tests")
        print("="*50)
        import test_cudnn_algorithms
        test_cudnn_algorithms.test_cudnn_benchmark()
        test_cudnn_algorithms.test_cudnn_for_different_batch_sizes()
    
    if args.all or args.gnn:
        print("\n" + "="*50)
        print("Running GNN Reproducibility Tests")
        print("="*50)
        try:
            import test_gnn_reproducibility
            test_gnn_reproducibility.test_scatter_operations()
            test_gnn_reproducibility.test_manual_scatter_add()
        except ImportError as e:
            print(f"Error importing GNN test modules: {e}")
            print("Make sure PyTorch Geometric is installed")
    
    if args.all or args.training:
        print("\n" + "="*50)
        print("Running Training Divergence Tests")
        print("="*50)
        import test_training_divergence
        test_training_divergence.test_training_reproducibility()
        
        try:
            import matplotlib
            if args.all:  # Only run long training if --all was specified (takes time)
                test_training_divergence.test_long_training_divergence()
        except ImportError:
            print("Matplotlib not available, skipping visualization")

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("\nChecking dependencies:")
    
    # List of required packages and their import names
    requirements = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib (optional, for visualization)"),
        ("torch_geometric", "PyTorch Geometric (optional, for GNN tests)")
    ]
    
    all_required_installed = True
    
    for package, description in requirements:
        try:
            importlib.import_module(package)
            print(f"✓ {description} is installed")
        except ImportError:
            if package in ["matplotlib", "torch_geometric"]:
                print(f"✗ {description} is not installed (optional)")
            else:
                print(f"✗ {description} is not installed")
                all_required_installed = False
    
    # Check for CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available (device: {torch.cuda.get_device_name(0)})")
        else:
            print("✗ CUDA is not available (some tests will be limited)")
    except:
        print("✗ Unable to check CUDA availability")
    
    return all_required_installed

if __name__ == "__main__":
    main() 