#!/usr/bin/env python
# coding: utf-8

# Import Dependencies

# In[ ]:


import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


# Define Hyperparamters

# In[ ]:


# Hyperparametears
batch = 32
EPOCHS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Setup ViT omponenents to include:<br>
# Patch Embedding, Positional Encoding, Transformer Encoder, Vision Transformer, Layer Normalization, Classification Head

# In[ ]:


# Vision Transformer model components
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, num_patches^(1/2), num_patches^(1/2)]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim] -- necessary for applying to multi-head self-attention, which looks for the sequence (or patch) dimension first.
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        return x + self.pos_embedding

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x))[0]
        x = x + self.mlp(self.layer_norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, mlp_dim=3072, num_layers=12, num_classes=40, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoding = PositionalEncoding(self.patch_embed.num_patches, embed_dim)
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
        for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        for encoder in self.transformer_encoders:
            x = encoder(x)
        x = self.layer_norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits

# Custom Dataset Class for CSV labels
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = pd.read_csv(csv_file)
        self.attributes = self.img_labels.columns[1:]  # Exclude the first column which is the image filename-- this is accessing and then slicing to the second column

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        labels = self.img_labels.iloc[idx, 1:].values.astype('float32')  # Convert labels to float32
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(labels)

# Define transformations for the training set -- necessary for applying to multi-head self-attention, which looks for the sequence (or patch) dimension first.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Setup Datasets and DataLoader

# In[ ]:


# Load the CSV file and determine the number of unique labels
#csv_file = '/mnt/d/deep_learning/datasets/celeb_a/archive/list_attr_celeba.csv'
csv_file = '/mnt/d/deep_learning/datasets/celeb_a/subset_1500img/subset_20240530_092758.csv'

df = pd.read_csv(csv_file)
attributes = df.columns[1:]  # List of attribute columns (excluding the image filename)
num_classes = len(attributes)

print(f"Number of classes: {num_classes}")
print(f"Attributes: {attributes}")

# Load the dataset
#dataset = CustomImageDataset(img_dir='/mnt/d/deep_learning/datasets/celeb_a/archive//img_align_celeba/class/', csv_file=csv_file, transform=transform)
dataset = CustomImageDataset(img_dir='/mnt/d/deep_learning/datasets/celeb_a/subset_1500img/', csv_file=csv_file, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)


# Setup Training Loop

# In[ ]:


# Training Loop
#def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cpu'):
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device=device):
    model.to(device)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / (len(train_loader.dataset) * num_classes)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Evaluate the model on the validation set
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, device)
        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val Precision: {val_precision:.4f} Val Recall: {val_recall:.4f} Val F1: {val_f1:.4f}')

    return model


def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    # Ensure the labels and predictions are in the correct format
    all_labels = all_labels.astype(int)
    all_preds = all_preds.astype(int)

    # Use multilabel indicators
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, precision, recall, f1






# def evaluate_model(model, dataloader, criterion, device):
#     model.eval()  # Set model to evaluation mode

#     running_loss = 0.0
#     all_labels = []
#     all_preds = []

#     with torch.no_grad():
#         for inputs, labels in tqdm(dataloader, desc="Evaluating"):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * inputs.size(0)
#             preds = (outputs > 0.5).float()

#             all_labels.append(labels.cpu().numpy())
#             all_preds.append(preds.cpu().numpy())

#     avg_loss = running_loss / len(dataloader.dataset)
#     all_labels = np.vstack(all_labels)
#     all_preds = np.vstack(all_preds)

#     # Convert to integer type for sklearn metrics
#     all_labels = all_labels.astype(int)
#     all_preds = all_preds.astype(int)

#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='macro')
#     recall = recall_score(all_labels, all_preds, average='macro')
#     f1 = f1_score(all_labels, all_preds, average='macro')

#     return avg_loss, accuracy, precision, recall, f1


# def evaluate_model(model, dataloader, criterion, device):
#     model.eval()  # Set model to evaluation mode

#     running_loss = 0.0
#     all_labels = []
#     all_preds = []

#     with torch.no_grad():
#         for inputs, labels in tqdm(dataloader, desc="Evaluating"):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * inputs.size(0)
#             preds = (outputs > 0.5).float()

#             all_labels.append(labels.cpu().numpy())
#             all_preds.append(preds.cpu().numpy())

#     avg_loss = running_loss / len(dataloader.dataset)
#     all_labels = np.vstack(all_labels)
#     all_preds = np.vstack(all_preds)

#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='macro')
#     recall = recall_score(all_labels, all_preds, average='macro')
#     f1 = f1_score(all_labels, all_preds, average='macro')

#     return avg_loss, accuracy, precision, recall, f1

if __name__ == "__main__":
    model = VisionTransformer(num_classes=num_classes)  # Adjust the number of classes based on the CSV
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


# Train Model

# In[ ]:


# Train the model
model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, device=device)


# Save the model

# In[ ]:


# Save the trained model
torch.save(model.state_dict(), 'vision_transformer_model.pth')
print("Model saved to vision_transformer_model.pth")


# In[ ]:


Entire ViT Script


# In[1]:


import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Hyperparametears
batch = 32
EPOCHS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Vision Transformer model components
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, num_patches^(1/2), num_patches^(1/2)]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        return x + self.pos_embedding

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x))[0]
        x = x + self.mlp(self.layer_norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, mlp_dim=3072, num_layers=12, num_classes=40, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoding = PositionalEncoding(self.patch_embed.num_patches, embed_dim)
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
        for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        for encoder in self.transformer_encoders:
            x = encoder(x)
        x = self.layer_norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits

# Custom Dataset Class for CSV labels
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = pd.read_csv(csv_file)
        self.attributes = self.img_labels.columns[1:]  # Exclude the first column which is the image filename-- this is accessing and then slicing to the second column

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        labels = self.img_labels.iloc[idx, 1:].values.astype('float32')  # Convert labels to float32
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(labels)

# Define transformations for the training set
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the CSV file and determine the number of unique labels
#csv_file = '/mnt/d/deep_learning/datasets/celeb_a/archive/list_attr_celeba.csv'
csv_file = '/mnt/d/deep_learning/datasets/celeb_a/subset_1500img/subset_20240530_092758.csv'

df = pd.read_csv(csv_file)
attributes = df.columns[1:]  # List of attribute columns (excluding the image filename)
num_classes = len(attributes)

print(f"Number of classes: {num_classes}")
print(f"Attributes: {attributes}")

# Load the dataset
#dataset = CustomImageDataset(img_dir='/mnt/d/deep_learning/datasets/celeb_a/archive/img_align_celeba/class/', csv_file=csv_file, transform=transform)
dataset = CustomImageDataset(img_dir='/mnt/d/deep_learning/datasets/celeb_a/subset_1500img/', csv_file=csv_file, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

# Training Loop
#def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cpu'):
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device=device):
    model.to(device)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / (len(train_loader.dataset) * num_classes)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Evaluate the model on the validation set
        #val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        #print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val Precision: {val_precision:.4f} Val Recall: {val_recall:.4f} Val F1: {val_f1:.4f}')
        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        return model

# def evaluate_model(model, dataloader, criterion, device):
#     model.eval()  # Set model to evaluation mode

#     running_loss = 0.0
#     all_labels = []
#     all_preds = []

#     with torch.no_grad():
#         for inputs, labels in tqdm(dataloader, desc="Evaluating"):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * inputs.size(0)
#             preds = (outputs > 0.5).float()

#             all_labels.append(labels.cpu().numpy())
#             all_preds.append(preds.cpu().numpy())

#     avg_loss = running_loss / len(dataloader.dataset)
#     all_labels = np.vstack(all_labels)
#     all_preds = np.vstack(all_preds)

#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='macro')
#     recall = recall_score(all_labels, all_preds, average='macro')
#     f1 = f1_score(all_labels, all_preds, average='macro')

#     return avg_loss, accuracy, precision, recall, f1

# def evaluate_model(model, dataloader, criterion, device):
#     model.eval()  # Set model to evaluation mode

#     running_loss = 0.0
#     all_labels = []
#     all_preds = []

#     with torch.no_grad():
#         for inputs, labels in tqdm(dataloader, desc="Evaluating"):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * inputs.size(0)
#             preds = (outputs > 0.5).int()  # Convert predictions to 0/1 for multi-label

#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())

#     avg_loss = running_loss / len(dataloader.dataset)
#     all_labels = np.array(all_labels)
#     all_preds = np.array(all_preds)

#     accuracy = accuracy_score(all_labels, all_preds, average='samples')
#     precision = precision_score(all_labels, all_preds, average='macro', zero_division='raise')
#     recall = recall_score(all_labels, all_preds, average='macro', zero_division='raise')
#     f1 = f1_score(all_labels, all_preds, average='macro', zero_division='raise')

#     return avg_loss, accuracy, precision, recall, f1

# def evaluate_model(model, dataloader, criterion, device):
#     model.eval()  # Set model to evaluation mode

#     running_loss = 0.0
#     all_labels = []
#     all_preds = []

#     with torch.no_grad():
#         for inputs, labels in tqdm(dataloader, desc="Evaluating"):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * inputs.size(0)
#             preds = (outputs > 0.5).int()  # Convert predictions to 0/1 for multi-label

#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())

#     avg_loss = running_loss / len(dataloader.dataset)
#     all_labels = np.array(all_labels)
#     all_preds = np.array(all_preds)

#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='macro', zero_division='raise')
#     recall = recall_score(all_labels, all_preds, average='macro', zero_division='raise')
#     f1 = f1_score(all_labels, all_preds, average='macro', zero_division='raise')

#     return avg_loss, accuracy, precision, recall, f1

# import torch
# from tqdm import tqdm
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# def evaluate_model(model, dataloader, criterion, device):
#     model.eval()  # Set model to evaluation mode

#     running_loss = 0.0
#     all_labels = []
#     all_preds = []

#     with torch.no_grad():
#         for inputs, labels in tqdm(dataloader, desc="Evaluating"):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * inputs.size(0)
#             preds = (outputs > 0.5).float()

#             all_labels.append(labels.cpu().numpy())
#             all_preds.append(preds.cpu().numpy())

#     avg_loss = running_loss / len(dataloader.dataset)
#     all_labels = np.concatenate(all_labels, axis=0)
#     all_preds = np.concatenate(all_preds, axis=0)

#     # Ensure the labels and predictions are in the correct format
#     all_labels = all_labels.astype(int)
#     all_preds = all_preds.astype(int)

#     # Use multilabel indicators
#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='macro')
#     recall = recall_score(all_labels, all_preds, average='macro')
#     f1 = f1_score(all_labels, all_preds, average='macro')

#    return avg_loss, accuracy, precision, recall, f1
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# def evaluate_model(model, dataloader, criterion, device):
#     model.eval()  # Set model to evaluation mode

#     running_loss = 0.0
#     all_labels = []
#     all_preds = []

#     with torch.no_grad():
#         for inputs, labels in tqdm(dataloader, desc="Evaluating"):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * inputs.size(0)
#             preds = (outputs > 0.5).float()

#             all_labels.append(labels.cpu().numpy())
#             all_preds.append(preds.cpu().numpy())

#     avg_loss = running_loss / len(dataloader.dataset)
#     all_labels = np.concatenate(all_labels, axis=0)
#     all_preds = np.concatenate(all_preds, axis=0)

#     # Ensure the labels and predictions are in the correct format
#     all_labels = all_labels.astype(int)
#     all_preds = all_preds.astype(int)

#     # Print shapes and types for debugging
#     print(f"all_labels shape: {all_labels.shape}, all_labels dtype: {all_labels.dtype}")
#     print(f"all_preds shape: {all_preds.shape}, all_preds dtype: {all_preds.dtype}")

#     # Use multilabel indicators
#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='macro')
#     recall = recall_score(all_labels, all_preds, average='macro')
#     f1 = f1_score(all_labels, all_preds, average='macro')

#     return avg_loss, accuracy, precision, recall, f1

#def evaluate_model(model, dataloader, criterion, device):
def evaluate_model(model, val_loader, criterion, device):
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        #for inputs, labels in tqdm(dataloader, desc="Evaluating"):
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    #avg_loss = running_loss / len(dataloader.dataset)
    avg_loss = running_loss / len(val_loader.dataset)
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)
    
#     all_labels = np.ravel(all_labels)  
#     all_preds = np.ravel(all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    #recall = recall_score(all_labels, all_preds, average='macro')
    #f1 = f1_score(all_labels, all_preds, average='macro')

    #return avg_loss, accuracy, precision, recall, f1  
    return avg_loss, accuracy, precision



# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device=device):
#     model.to(device)
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         model.train()  # Set model to training mode

#         running_loss = 0.0
#         running_corrects = 0

#         for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * inputs.size(0)
#             preds = (outputs > 0.5).float()
#             running_corrects += torch.sum(preds == labels.data)

#         epoch_loss = running_loss / len(train_loader.dataset)
#         epoch_acc = running_corrects.double() / (len(train_loader.dataset) * num_classes)

#         print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

#         # Evaluate the model on the validation set
#         #val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, device)
#         val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
#         #print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val Precision: {val_precision:.4f} Val Recall: {val_recall:.4f} Val F1: {val_f1:.4f}')
#         print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

#         return model









# def evaluate_model(model, dataloader, criterion, device):
#     model.eval()  # Set model to evaluation mode

#     running_loss = 0.0
#     all_labels = []
#     all_preds = []

#     with torch.no_grad():
#         for inputs, labels in tqdm(dataloader, desc="Evaluating"):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * inputs.size(0)
#             preds = (outputs > 0.5).float()

#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(preds.cpu().numpy())

#     avg_loss = running_loss / len(dataloader.dataset)
#     all_labels = np.array(all_labels)
#     all_preds = np.array(all_preds)

#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='macro')
#     recall = recall_score(all_labels, all_preds, average='macro')
#     f1 = f1_score(all_labels, all_preds, average='macro')

#     return avg_loss, accuracy, precision, recall, f1




if __name__ == "__main__":
    model = VisionTransformer(num_classes=num_classes)  # Adjust the number of classes based on the CSV
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
   # model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, device=device)
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, device=device)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# Save the trained model
torch.save(model.state_dict(), 'vision_transformer_model.pth')
print("Model saved to vision_transformer_model.pth")


# In[6]:


print(evaluate_model)


# In[ ]:


import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Ensure the labels and predictions are in the correct format
    all_labels = all_labels.astype(int)
    all_preds = all_preds.astype(int)

    # Print shapes and types for debugging
    print(f"all_labels shape: {all_labels.shape}, all_labels dtype: {all_labels.dtype}")
    print(f"all_preds shape: {all_preds.shape}, all_preds dtype: {all_preds.dtype}")

    # Use multilabel indicators
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, precision, recall, f1


# In[ ]:


import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Hyperparametears
batch = 32
EPOCHS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Vision Transformer model components
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, num_patches^(1/2), num_patches^(1/2)]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        return x + self.pos_embedding

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x))[0]
        x = x + self.mlp(self.layer_norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, mlp_dim=3072, num_layers=12, num_classes=40, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoding = PositionalEncoding(self.patch_embed.num_patches, embed_dim)
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
        for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        for encoder in self.transformer_encoders:
            x = encoder(x)
        x = self.layer_norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits

# Custom Dataset Class for CSV labels
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = pd.read_csv(csv_file)
        self.attributes = self.img_labels.columns[1:]  # Exclude the first column which is the image filename-- this is accessing and then slicing to the second column

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        labels = self.img_labels.iloc[idx, 1:].values.astype('float32')  # Convert labels to float32
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(labels)

# Define transformations for the training set
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the CSV file and determine the number of unique labels
csv_file = '/mnt/d/deep_learning/datasets/celeb_a/archive/list_attr_celeba.csv'
df = pd.read_csv(csv_file)
attributes = df.columns[1:]  # List of attribute columns (excluding the image filename)
num_classes = len(attributes)

print(f"Number of classes: {num_classes}")
print(f"Attributes: {attributes}")

# Load the dataset
dataset = CustomImageDataset(img_dir='/mnt/d/deep_learning/datasets/celeb_a/archive/img_align_celeba/class/', csv_file=csv_file, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

# Training Loop
#def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cpu'):
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device=device):
    model.to(device)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / (len(train_loader.dataset) * num_classes)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Evaluate the model on the validation set
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, device)
        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val Precision: {val_precision:.4f} Val Recall: {val_recall:.4f} Val F1: {val_f1:.4f}')

    return model

# def evaluate_model(model, dataloader, criterion, device):
#     model.eval()  # Set model to evaluation mode

#     running_loss = 0.0
#     all_labels = []
#     all_preds = []

#     with torch.no_grad():
#         for inputs, labels in tqdm(dataloader, desc="Evaluating"):
#             inputs = inputs.to(device)
#             labels = labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels)

#             running_loss += loss.item() * inputs.size(0)
#             preds = (outputs > 0.5).float()

#             all_labels.append(labels.cpu().numpy())
#             all_preds.append(preds.cpu().numpy())

#     avg_loss = running_loss / len(dataloader.dataset)
#     all_labels = np.vstack(all_labels)
#     all_preds = np.vstack(all_preds)

#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='macro')
#     recall = recall_score(all_labels, all_preds, average='macro')
#     f1 = f1_score(all_labels, all_preds, average='macro')

#     return avg_loss, accuracy, precision, recall, f1

def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).int()  # Convert predictions to 0/1 for multi-label

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division='raise')
    recall = recall_score(all_labels, all_preds, average='macro', zero_division='raise')
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division='raise')

    return avg_loss, accuracy, precision, recall, f1

if __name__ == "__main__":
    model = VisionTransformer(num_classes=num_classes)  # Adjust the number of classes based on the CSV
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, device=device)


# In[ ]:


import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Hyperparametears
batch = 32
EPOCHS = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Vision Transformer model components
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, num_patches^(1/2), num_patches^(1/2)]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        return x + self.pos_embedding

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x), self.layer_norm1(x), self.layer_norm1(x))[0]
        x = x + self.mlp(self.layer_norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, num_heads=12, mlp_dim=3072, num_layers=12, num_classes=40, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoding = PositionalEncoding(self.patch_embed.num_patches, embed_dim)
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
        for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_encoding(x)
        for encoder in self.transformer_encoders:
            x = encoder(x)
        x = self.layer_norm(x)
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        return logits

# Custom Dataset Class for CSV labels
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = pd.read_csv(csv_file)
        self.attributes = self.img_labels.columns[1:]  # Exclude the first column which is the image filename-- this is accessing and then slicing to the second column

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        labels = self.img_labels.iloc[idx, 1:].values.astype('float32')  # Convert labels to float32
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(labels)

# Define transformations for the training set
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the CSV file and determine the number of unique labels
csv_file = '/mnt/d/deep_learning/datasets/celeb_a/archive/list_attr_celeba.csv'
df = pd.read_csv(csv_file)
attributes = df.columns[1:]  # List of attribute columns (excluding the image filename)
num_classes = len(attributes)

print(f"Number of classes: {num_classes}")
print(f"Attributes: {attributes}")

# Load the dataset
dataset = CustomImageDataset(img_dir='/mnt/d/deep_learning/datasets/celeb_a/archive/img_align_celeba/class/', csv_file=csv_file, transform=transform)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

# Training Loop
#def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cpu'):
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device=device):
    model.to(device)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / (len(train_loader.dataset) * num_classes)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Evaluate the model on the validation set
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion, device)
        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} Val Precision: {val_precision:.4f} Val Recall: {val_recall:.4f} Val F1: {val_f1:.4f}')

    return model

def evaluate_model(model, dataloader, criterion, device):
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)
    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, accuracy, precision, recall, f1

if __name__ == "__main__":
    model = VisionTransformer(num_classes=num_classes)  # Adjust the number of classes based on the CSV
    criterion = nn.BCEWithLogitsLoss()  # For multi-label classification
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=EPOCHS, device=device)

    # Save the trained model
    torch.save(model.state_dict(), 'vision_transformer_model.pth')
    print("Model saved to vision_transformer_model.pth")


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Define the shapes at each stage
shapes = [
    "[B, embed_dim, sqrt(num_patches), sqrt(num_patches)]",
    "[B, embed_dim, num_patches]",
    "[B, num_patches, embed_dim]"
]

# Plotting the shapes
fig, ax = plt.subplots(figsize=(10, 5))

# Create positions for the shapes
y_positions = np.arange(len(shapes))

# Plot each shape as a rectangle with text
for i, shape in enumerate(shapes):
    ax.text(0.5, y_positions[i], shape, ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))

# Setting up the axes
ax.set_yticks(y_positions)
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_xlim(0, 1)
ax.set_ylim(-1, len(shapes))

# Adding arrows to show transformations
for i in range(len(shapes) - 1):
    ax.annotate('', xy=(0.5, y_positions[i+1] + 0.2), xytext=(0.5, y_positions[i] - 0.2),
                arrowprops=dict(facecolor='black', shrink=0.05))

ax.set_title('Forward Method Shape Transformations')

plt.show()


# In[ ]:


# Define the specific shapes at each stage
shapes_specific = [
    "[B, 3, 224, 224]",
    "[B, 768, 14, 14]",
    "[B, 768, 196]",
    "[B, 196, 768]"
]

# Plotting the shapes
fig, ax = plt.subplots(figsize=(10, 6))

# Create positions for the shapes
y_positions_specific = np.arange(len(shapes_specific))

# Plot each shape as a rectangle with text
for i, shape in enumerate(shapes_specific):
    ax.text(0.5, y_positions_specific[i], shape, ha='center', va='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='lightblue'))

# Setting up the axes
ax.set_yticks(y_positions_specific)
ax.set_yticklabels([])
ax.set_xticks([])
ax.set_xlim(0, 1)
ax.set_ylim(-1, len(shapes_specific))

# Adding arrows to show transformations
for i in range(len(shapes_specific) - 1):
    ax.annotate('', xy=(0.5, y_positions_specific[i+1] + 0.2), xytext=(0.5, y_positions_specific[i] - 0.2),
                arrowprops=dict(facecolor='black', shrink=0.05))

ax.set_title('Forward Method Shape Transformations with Specific Dimensions')

plt.show()


# In[ ]:




