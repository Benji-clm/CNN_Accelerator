import torch
import torch.nn as nn
import numpy as np

# Define the model architecture (must match the saved model)
class MinimalCNN(nn.Module):
    def __init__(self):
        super(MinimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 12 * 12, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 12 * 12)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the saved model
model = MinimalCNN()
model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()

# Function to format and save weights
def save_weights_to_file(model, filename='model_weights.txt'):
    with open(filename, 'w') as f:
        # Iterate through model parameters
        for name, param in model.named_parameters():
            f.write(f"\nLayer: {name}\n")
            f.write(f"Shape: {param.shape}\n")
            f.write("Values:\n")
            
            # Convert tensor to numpy for readable output
            param_np = param.detach().cpu().numpy()
            
            if 'conv1.weight' in name:
                # For conv1 weights: [out_channels, in_channels, height, width]
                for out_ch in range(param_np.shape[0]):
                    f.write(f"Filter {out_ch + 1}:\n")
                    for in_ch in range(param_np.shape[1]):
                        f.write(f"  Input Channel {in_ch + 1}:\n")
                        # Format 5x5 kernel as a matrix
                        kernel = param_np[out_ch, in_ch]
                        for row in kernel:
                            f.write("    " + " ".join(f"{x:.6f}" for x in row) + "\n")
            
            elif 'conv1.bias' in name:
                # For conv1 bias: one value per filter
                for i, bias in enumerate(param_np):
                    f.write(f"Bias for Filter {i + 1}: {bias:.6f}\n")
            
            elif 'fc1.weight' in name:
                # For fc1 weights: [output_units, input_units]
                f.write(f"{param_np.shape[0]} by {param_np.shape[1]} weight matrix:\n")
                for i in range(min(100, param_np.shape[0])):
                    f.write(f"{' '.join(f'{x:.6f}' for x in param_np[i, :3137])}\n")
            
            elif 'fc1.bias' in name:
                # For fc1 bias: one value per output unit
                f.write(f"First 5 biases (of {param_np.shape[0]}):\n")
                for i in range(min(100, param_np.shape[0])):
                    f.write(f"  Bias for Unit {i + 1}: {param_np[i]:.6f}\n")
            
            elif 'fc2.weight' in name:
                # For fc2 weights: [output_classes, input_units]
                for i in range(param_np.shape[0]):
                    f.write(f"Output Class {i} (Digit {i}): {' '.join(f'{x:.6f}' for x in param_np[i, :65])} \n")
            
            elif 'fc2.bias' in name:
                # For fc2 bias: one value per output class
                for i, bias in enumerate(param_np):
                    f.write(f"Bias for Class {i} (Digit {i}): {bias:.6f}\n")
            
            f.write("-" * 80 + "\n")

# Extract and save weights
print("Extracting and saving model weights to 'model_weights.txt'...")
save_weights_to_file(model)
print("Weights saved to 'model_weights.txt'")

# Extract linear layers' weights
linear_state_dict = {
    key: value for key, value in model.state_dict().items()
    if key.startswith('fc1') or key.startswith('fc2')
}

# Save to a new .pth file
output_file = 'linear_layers.pth'
torch.save(linear_state_dict, output_file)
print(f"Linear layers' weights saved to '{output_file}'")

# Verify saved weights
print("\nSaved weights:")
for key, value in linear_state_dict.items():
    print(f"{key}: shape {value.shape}")