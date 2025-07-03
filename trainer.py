import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from safetensors.torch import save_file  # ✅ Import safetensors save
from models.pix_safe_cnn import PixSafeCNN  # ✅ Corrected module and class name

# --- Configuration ---
data_dir = 'dataset'
model_save_path = 'saved-models/pixsafe_cnn.safetensors'  # ✅ Save as .safetensors
batch_size = 32
epochs = 40
learning_rate = 0.001
img_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Class Labels (for reference/logging) ---
CLASS_NAMES = {
    0: "nude",
    1: "suggestive",
    2: "safe"
}

# --- Image Transforms ---
train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# --- Load Dataset ---
train_dataset = datasets.ImageFolder(root=data_dir, transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Optional: Print class index mapping
print(f"Class to index mapping: {train_dataset.class_to_idx}")

# --- Initialize Model ---
model = PixSafeCNN(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# --- Training Loop ---
print("\n🟢 Starting Training...\n")
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

# --- Save the Model in .safetensors format ---
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
save_file(model.state_dict(), model_save_path)  # ✅ Save as .safetensors
print(f"\n✅ Model saved to: {model_save_path}")
