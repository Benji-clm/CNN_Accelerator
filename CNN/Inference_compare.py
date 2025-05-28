import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time

# Define the model architecture (must match the saved model)
class MinimalCNN(nn.Module):
    def __init__(self):
        super(MinimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)
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

# Load the test dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Load the saved model
model = MinimalCNN()
model.load_state_dict(torch.load('mnist_cnn.pth'))

# Function to measure inference time and average per sample
def measure_inference_time(model, data_loader, device):
    model.to(device)
    model.eval()
    num_samples = len(data_loader.dataset)  # Total number of samples (10,000 for MNIST test)
    start_time = time.time()
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_sample = (total_time / num_samples) * 1000  # Convert to milliseconds
    return total_time, avg_time_per_sample

# Measure GPU inference time if available
if torch.cuda.is_available():
    gpu_total_time, gpu_avg_time = measure_inference_time(model, test_loader, 'cuda')
    print(f"GPU total inference time: {gpu_total_time:.4f} seconds")
    print(f"GPU average inference time per sample: {gpu_avg_time:.4f} ms")
else:
    print("GPU not available")

# Measure CPU inference time
cpu_total_time, cpu_avg_time = measure_inference_time(model, test_loader, 'cpu')
print(f"CPU total inference time: {cpu_total_time:.4f} seconds")
print(f"CPU average inference time per sample: {cpu_avg_time:.4f} ms")

with open('performance.txt', 'w') as f:
    f.write(f"GPU total inference time: {gpu_total_time:.4f} seconds\n")
    f.write(f"GPU average inference time per sample: {gpu_avg_time:.4f} ms\n")
    f.write(f"CPU total inference time: {cpu_total_time:.4f} seconds\n")
    f.write(f"CPU average inference time per sample: {cpu_avg_time:.4f} ms\n")
