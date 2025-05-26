import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_FC(nn.Module):
    def __init__(self, model_file=None):
        super(CNN_FC, self).__init__()
        
        # Load pre-trained model from file if provided
        if model_file is None:
            raise ValueError("A model file name must be provided.")
        try:
            self.model = torch.load(model_file)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_file}: {str(e)}")
        self.model.eval()  # Set model to evaluation mode
        
        # Extract weights and biases from the pre-trained model
        try:
            self.fc1_weight = self.model.fc1.weight.data
            self.fc1_bias = self.model.fc1.bias.data
            self.fc2_weight = self.model.fc2.weight.data  # Corrected from fc2h to fc2
            self.fc2_bias = self.model.fc2.bias.data
        except AttributeError:
            raise AttributeError("Pre-trained model must have 'fc1' and 'fc2' layers.")
        
        # Store convolutional output        
        # Define pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, conv_out):
        # Apply pooling to each feature map
        pooled = []
        for i in range(conv_out.shape[1]):  # Assuming conv_out is [16, 28, 28]
            pooled.append(self.pool(conv_out[i:i+1, :, :]))  # Pool each channel
        pooled = torch.cat(pooled, dim=1)  # Concatenate along channel dimension
        
        # Vectorize the pooled output
        vectorized = pooled.view(pooled.size(0), -1)  # Shape: [batch_size, 16*14*14]
        
        # Apply fully connected layers
        fc1_out = F.linear(vectorized, self.fc1_weight, self.fc1_bias)
        fc1_out = F.relu(fc1_out)
        output = F.linear(fc1_out, self.fc2_weight, self.fc2_bias)
        
        # Get decision (class with highest score)
        decision = torch.argmax(output, dim=1)
        
        return output, decision

# Example usage (assuming a pre-trained model and conv_output are available)
# model = SomePretrainedCNN()  # Replace with actual pre-trained model
# conv_output = torch.randn(1, 16, 14, 14)  # Example input
# cnn_fc = CNN_FC(conv_output, model)
# output, decision = cnn_fc()
# print("Output:", output)
# print("Decision:", decision)

