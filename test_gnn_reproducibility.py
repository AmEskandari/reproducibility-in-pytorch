import torch
import networkx as nx
import numpy as np
import random

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False
    print("PyTorch Geometric not installed. GNN test will be skipped.")

def set_seed(seed=42):
    """Set all seeds to make results reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_scatter_operations():
    """Test reproducibility of scatter operations commonly used in GNNs."""
    if not HAS_PYGEOMETRIC:
        return
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GNN reproducibility test")
        return
        
    print("\n=== Testing GNN (scatter operations) reproducibility ===")
    
    # Create a random graph
    set_seed(42)
    g = nx.erdos_renyi_graph(1000, 0.01)
    edge_list = list(g.edges())
    edge_index = torch.tensor(edge_list).t().contiguous()
    
    # Create bidirectional edges (required for GCN)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to("cuda")
    
    # Random node features
    x = torch.rand(1000, 64, device="cuda")
    
    # GCN layer that uses scatter operations internally
    conv = GCNConv(64, 32).to("cuda")
    
    # Run multiple times and collect results
    results = []
    for i in range(5):
        # We keep the same seed, but results may still vary due to scatter operations
        set_seed(42)
        output = conv(x, edge_index)
        results.append(output.sum().item())
        print(f"  Run {i+1}: output sum = {results[-1]}")
    
    # Check if all results are the same
    print(f"  All equal: {all(abs(r - results[0]) < 1e-5 for r in results)}")
    
    # Test on CPU for comparison
    if not all(abs(r - results[0]) < 1e-5 for r in results):
        print("\n  Testing on CPU for comparison...")
        edge_index_cpu = edge_index.cpu()
        x_cpu = x.cpu()
        conv_cpu = GCNConv(64, 32)
        
        cpu_results = []
        for i in range(3):
            set_seed(42)
            output = conv_cpu(x_cpu, edge_index_cpu)
            cpu_results.append(output.sum().item())
            print(f"  CPU Run {i+1}: output sum = {cpu_results[-1]}")
        
        print(f"  CPU all equal: {all(abs(r - cpu_results[0]) < 1e-5 for r in cpu_results)}")

def test_manual_scatter_add():
    """Demonstrate the issue with scatter_add operations more directly."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping scatter_add test")
        return
        
    print("\n=== Testing scatter_add directly ===")
    
    # Create a source tensor
    src = torch.ones(10, 5, device="cuda")
    
    # Create an index tensor where multiple source elements map to the same target
    # This creates a race condition when multiple source elements update the same target
    index = torch.tensor([0, 1, 0, 2, 1, 3, 0, 4, 2, 1], device="cuda")
    index = index.view(-1, 1).expand(-1, 5)
    
    # Output tensor
    results = []
    for i in range(5):
        out = torch.zeros(5, 5, device="cuda")
        # Perform scatter_add operation
        out.scatter_add_(0, index, src)
        results.append(out.cpu().numpy().copy())
        print(f"  Run {i+1}: First row sum = {out[0].sum().item()}")
    
    # Check if all results are the same
    all_equal = all(np.array_equal(results[0], results[i]) for i in range(1, 5))
    print(f"  All scatter_add results equal: {all_equal}")
    
    if not all_equal:
        print("\n  Non-determinism detected in scatter_add operation")
        print("  This is expected due to race conditions when multiple source elements")
        print("  update the same target position in parallel on the GPU.")
    
    # Compare with CPU
    print("\n  Testing on CPU for comparison...")
    src_cpu = src.cpu()
    index_cpu = index.cpu()
    
    cpu_results = []
    for i in range(3):
        out_cpu = torch.zeros(5, 5)
        out_cpu.scatter_add_(0, index_cpu, src_cpu)
        cpu_results.append(out_cpu.numpy().copy())
        print(f"  CPU Run {i+1}: First row sum = {out_cpu[0].sum().item()}")
    
    cpu_all_equal = all(np.array_equal(cpu_results[0], cpu_results[i]) for i in range(1, 3))
    print(f"  CPU all scatter_add results equal: {cpu_all_equal}")

if __name__ == "__main__":
    set_seed(42)
    test_scatter_operations()
    test_manual_scatter_add() 