import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class CNN(nn.Module):
    """A simple CNN model in PyTorch."""
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        
        # Input x: (Batch, 3, 64, 64) - PyTorch uses NCHW
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # JAX Flax Conv 'SAME' padding usually? 
            # Flax default padding is 'SAME'? No, default is 'VALID'. 
            # But code said `kernel_size=(3, 3)`. 
            # Let's check `cnn_model.py` again. `nn.Conv` default padding is 0.
            # If shapes are 64->32->16->8.
            # 64 - 3 + 1 = 62 / 2 = 31. 
            # If padding='valid', 64->62->31 -> 29 -> 14 -> 12 -> 6.
            # Let's stick to standard practices or check original output shapes if possible.
            # Assuming 'SAME' or compatible padding to keep dimensions simple is safer for a port unless I want exact architectural match.
            # JAX Flax Conv default padding is 'SAME' if not specified? No, it's 0 ('VALID').
            # Wait, Flax: "padding: PaddingType = 'SAME'". 
            # Let's check documentation or assume SAME for modern CNNs.
            # Actually, let's use padding=1 to emulate SAME for 3x3 kernels.
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Flatten size: 64 * (64/8)*(64/8) = 64 * 8 * 8 = 4096
        self.flatten_size = 64 * 8 * 8 
        
        self.embedding_layer = nn.Linear(self.flatten_size, 64)
        self.output_layer = nn.Linear(64, num_classes)
        
    def forward(self, x, return_embeddings=False):
        # x comes in as NHWC (Batch, 64, 64, 3) from 3dshapes
        # Permute to NCHW
        if x.shape[-1] == 3:
            x = x.permute(0, 3, 1, 2)
            
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        
        x = self.embedding_layer(x)
        embeddings = torch.relu(x)
        
        logits = self.output_layer(embeddings)
        
        if return_embeddings:
            return logits, embeddings
        return logits

def train_model(X_train, y_train, X_val, y_val, num_classes, num_epochs=10, batch_size=64, seed=0, verbose=True):
    """Full training loop in PyTorch."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create Datasets (keep on CPU, move batch to GPU)
    # X_train is numpy (N, 64, 64, 3). y_train is (N,).
    # Convert to tensors? If too large, keep as numpy and custom dataset?
    # 3dshapes is big. TensorDataset might copy.
    # But `X_train` passed here is likely already a subset or loaded array.
    
    # Normalize images here or in loop? Original code normalized in `train_step`.
    
    tensor_x = torch.tensor(X_train) # Copy?
    tensor_y = torch.tensor(y_train).long()
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = CNN(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Normalize
            inputs = inputs.float() / 255.0
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Process val in batches manually to avoid creating full DataLoader if just array
        val_batch_size = 1024
        with torch.no_grad():
            for i in range(0, len(X_val), val_batch_size):
                batch_x = torch.tensor(X_val[i:i+val_batch_size]).to(device)
                batch_y = torch.tensor(y_val[i:i+val_batch_size]).to(device)
                
                batch_x = batch_x.float() / 255.0
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.long())
                
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
    return model

def get_embeddings(model, batch_images):
    """Get embeddings for a batch."""
    device = next(model.parameters()).device
    model.eval()
    
    if not isinstance(batch_images, torch.Tensor):
        batch_images = torch.tensor(batch_images)
        
    batch_images = batch_images.to(device).float() / 255.0
    
    with torch.no_grad():
        _, embeddings = model(batch_images, return_embeddings=True)
        
    return embeddings.cpu().numpy()

