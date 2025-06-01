import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchsummary import summary

class InstanceNormGlobal(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        mean = x.mean(dim=tuple(range(1, x.dim())), keepdim=True)
        std = x.std(dim=tuple(range(1, x.dim())), keepdim=True)
        return (x - mean) / (std + 1e-5) * self.scale + self.shift

class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DWBlock(nn.Module):
    def __init__(self, shared_sep_conv):
        super().__init__()
        self.shared_sep_conv = shared_sep_conv
        self.relu = nn.ReLU()
        self.inst_norm = InstanceNormGlobal()
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, x):
        x = self.shared_sep_conv(x)
        x = self.relu(x)
        x = self.inst_norm(x)
        x = self.dropout(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.inst_norm1 = InstanceNormGlobal()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=0.1)
        
        self.sep_conv1 = SeparableConv2D(8, 26, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.inst_norm2 = InstanceNormGlobal()
        self.dropout2 = nn.Dropout(p=0.1)
        
        self.shared_sep_conv = SeparableConv2D(26, 26, kernel_size=3, padding=1)
        self.dw_blocks = nn.ModuleList([DWBlock(self.shared_sep_conv) for _ in range(3)])
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.dense1 = nn.Linear(26, 16)
        self.relu3 = nn.ReLU()
        self.inst_norm3 = InstanceNormGlobal()
        self.dropout3 = nn.Dropout(p=0.1)
        self.dense2 = nn.Linear(16, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.inst_norm1(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        
        x = self.sep_conv1(x)
        x = self.relu2(x)
        x = self.inst_norm2(x)
        x = self.dropout2(x)
        
        for dw_block in self.dw_blocks:
            x = dw_block(x)
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.dense1(x)
        x = self.relu3(x)
        x = self.inst_norm3(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        x = self.softmax(x)
        return x

# Example usage
model = Model()
# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(str(device))
# Initialize model, loss function, and optimizer
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy: {100 * correct / total}%')

# Save the model
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("Model saved as 'mnist_cnn.pth'")
summary(model, input_size=(1, 28, 28))