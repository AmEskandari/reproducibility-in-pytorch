import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path

def set_seed(seed=42):
    """Set all seeds to make results reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

def train_model(device, epochs=100, plot=False):
    """Train a simple model and return loss history."""
    # Generate consistent data
    set_seed(42)
    X = torch.randn(1000, 20)
    y = torch.randn(1000, 1)
    
    # Move to target device
    X = X.to(device)
    y = y.to(device)
    
    # Create model, optimizer, and loss function
    set_seed(42)
    model = SimpleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Train the model
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    
    return loss_history

def test_training_reproducibility():
    """Compare training outcomes across runs and devices."""
    print("\n=== Testing Training Reproducibility ===")
    
    # Check CPU training reproducibility
    print("Testing CPU training reproducibility...")
    cpu_histories = []
    for i in range(2):
        set_seed(42)  # Reset seed before each run
        history = train_model('cpu', epochs=100)
        cpu_histories.append(history)
        print(f"  Run {i+1} final loss: {history[-1]:.10f}")
    
    # Check if CPU training is reproducible (it should be)
    cpu_reproducible = all(abs(cpu_histories[0][i] - cpu_histories[1][i]) < 1e-6 for i in range(len(cpu_histories[0])))
    print(f"  CPU training reproducible: {cpu_reproducible}")
    
    # Test GPU reproducibility if available
    if torch.cuda.is_available():
        print("\nTesting GPU training reproducibility...")
        gpu_histories = []
        for i in range(2):
            set_seed(42)  # Reset seed before each run
            history = train_model('cuda', epochs=100)
            gpu_histories.append(history)
            print(f"  Run {i+1} final loss: {history[-1]:.10f}")
        
        # Check if GPU training is reproducible
        gpu_reproducible = all(abs(gpu_histories[0][i] - gpu_histories[1][i]) < 1e-6 for i in range(len(gpu_histories[0])))
        print(f"  GPU training reproducible: {gpu_reproducible}")
        
        # Compare CPU and GPU results
        print("\nComparing CPU and GPU results...")
        print(f"  CPU final loss: {cpu_histories[0][-1]:.10f}")
        print(f"  GPU final loss: {gpu_histories[0][-1]:.10f}")
        print(f"  CPU and GPU equal: {abs(cpu_histories[0][-1] - gpu_histories[0][-1]) < 1e-6}")
        
        # Plot the results if we need divergence visualization
        if not gpu_reproducible:
            print("\nDetected divergence in GPU training. Check 'training_divergence.png' for visualization.")
            generate_divergence_plot(gpu_histories, cpu_histories)

def test_long_training_divergence():
    """Test how longer training affects divergence."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping long training test")
        return
    
    print("\n=== Testing Long Training Divergence ===")
    print("This test will take longer to run...")
    
    # Train two models for longer periods
    set_seed(42)
    history1 = train_model('cuda', epochs=500)
    
    set_seed(42)
    history2 = train_model('cuda', epochs=500)
    
    # Calculate absolute differences at different points
    diff_50 = abs(history1[49] - history2[49])
    diff_200 = abs(history1[199] - history2[199])
    diff_500 = abs(history1[499] - history2[499])
    
    print(f"  Difference after 50 epochs: {diff_50:.10f}")
    print(f"  Difference after 200 epochs: {diff_200:.10f}")
    print(f"  Difference after 500 epochs: {diff_500:.10f}")
    print(f"  Is divergence increasing? {diff_50 < diff_200 < diff_500}")
    
    # Generate plot showing the divergence
    generate_divergence_plot([history1, history2], [], long_training=True)

def generate_divergence_plot(gpu_histories, cpu_histories=None, long_training=False):
    """Generate plot showing training divergence."""
    plt.figure(figsize=(10, 6))
    
    # Plot GPU histories
    for i, history in enumerate(gpu_histories):
        plt.plot(history, label=f'GPU Run {i+1}')
    
    # Plot CPU histories if provided
    if cpu_histories:
        for i, history in enumerate(cpu_histories):
            plt.plot(history, '--', label=f'CPU Run {i+1}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    if long_training:
        plt.title('Training Divergence over Extended Training')
        plt.savefig('long_training_divergence.png')
    else:
        plt.title('Training Divergence between Runs')
        plt.savefig('training_divergence.png')
    
    plt.close()

if __name__ == "__main__":
    # Check if matplotlib is available (for plotting)
    try:
        import matplotlib
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        print("Matplotlib not installed. Visualization will be skipped.")
    
    set_seed(42)
    test_training_reproducibility()
    
    if HAS_MATPLOTLIB and torch.cuda.is_available():
        # Only run the long training test if we have matplotlib and CUDA
        test_long_training_divergence() 