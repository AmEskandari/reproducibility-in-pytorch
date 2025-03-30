import torch
import torch_geometric
from torch_geometric.nn import GCNConv
import networkx as nx
import time

print("=== Testing GNN Scatter Operations Reproducibility ===")

# Set seeds for reproducibility
print("Setting seeds for reproducibility...")
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Create a random graph
print("Creating random graph...")
g = nx.erdos_renyi_graph(1000, 0.01)
edge_list = list(g.edges())
edge_index = torch.tensor(edge_list).t().contiguous()
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).to("cuda")
print(f"Graph created with {edge_index.shape[1]} edges")

# Random node features
print("Generating random node features...")
x = torch.rand(1000, 64, device="cuda")

# GCN layer that uses scatter operations internally
print("Initializing GCN layer that uses scatter operations internally...")
conv = GCNConv(64, 32).to("cuda")

# Run multiple times and collect results
print("\nRunning inference multiple times with identical inputs:")
results = []
times = []

for i in range(5):
    start_time = time.time()
    output = conv(x, edge_index)
    end_time = time.time()
    
    results.append(output.sum().item())
    times.append(end_time - start_time)
    
    print(f"  Run {i+1}: output sum = {results[-1]:.8f}, time = {times[-1]:.6f}s")

print("\nResults Summary:")
print(f"  All values: {[f'{r:.8f}' for r in results]}")
print(f"  All equal: {all(r == results[0] for r in results)}")

if not all(r == results[0] for r in results):
    print("  IMPORTANT: Different results detected despite identical inputs and seeds!")
    print("  This demonstrates non-deterministic behavior in torch_scatter operations.")
    max_diff = max([abs(r - results[0]) for r in results])
    print(f"  Maximum difference between runs: {max_diff:.8f}")