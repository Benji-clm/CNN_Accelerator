import torch
import torch.nn as nn
import numpy as np

# Define the model architecture (must match the saved model)
class MinimalCNN(nn.Module):
    def __init__(self):
        super(MinimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
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
                f.write(f"First 5 rows (of {param_np.shape[0]} output units):\n")
                for i in range(min(5, param_np.shape[0])):
                    f.write(f"  Output Unit {i + 1}: {' '.join(f'{x:.6f}' for x in param_np[i, :10])} ... (first 10 of {param_np.shape[1]} inputs)\n")
            
            elif 'fc1.bias' in name:
                # For fc1 bias: one value per output unit
                f.write(f"First 5 biases (of {param_np.shape[0]}):\n")
                for i in range(min(5, param_np.shape[0])):
                    f.write(f"  Bias for Unit {i + 1}: {param_np[i]:.6f}\n")
            
            elif 'fc2.weight' in name:
                # For fc2 weights: [output_classes, input_units]
                for i in range(param_np.shape[0]):
                    f.write(f"Output Class {i} (Digit {i}): {' '.join(f'{x:.6f}' for x in param_np[i, :10])} ... (first 10 of {param_np.shape[1]} inputs)\n")
            
            elif 'fc2.bias' in name:
                # For fc2 bias: one value per output class
                for i, bias in enumerate(param_np):
                    f.write(f"Bias for Class {i} (Digit {i}): {bias:.6f}\n")
            
            f.write("-" * 80 + "\n")

# Extract and save weights
print("Extracting and saving model weights to 'model_weights.txt'...")
save_weights_to_file(model)
print("Weights saved to 'model_weights.txt'")

# Verify GPU availability (optional, to ensure model context)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)