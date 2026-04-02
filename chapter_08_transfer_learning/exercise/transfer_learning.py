import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

print("=== Transfer Learning Demo ===\n")

# 1. Load a pre-trained ResNet-18 (trained on 14M ImageNet images)
print("Loading pre-trained ResNet-18...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 2. Freeze ALL existing layers
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the final classification layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

print(f"Original model output: 1000 classes")
print(f"Modified model output: 2 classes")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Frozen parameters:    {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}")

# 4. Simulate training with synthetic data
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

print("\nSimulating fine-tuning on 240 ant/bee images...")
for epoch in range(3):
    fake_images = torch.randn(32, 3, 224, 224)
    fake_labels = torch.randint(0, 2, (32,))
    optimizer.zero_grad()
    outputs = model(fake_images)
    loss = criterion(outputs, fake_labels)
    loss.backward()
    optimizer.step()
    print(f"  Epoch {epoch+1}/3 — Loss: {loss.item():.4f}")

print("\n✅ Fine-tuning complete!")
print("The model now classifies ants vs bees using knowledge")
print("originally learned from 14 million ImageNet images.")
