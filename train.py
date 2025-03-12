import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm

from dataset import *


# Define transformations
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
mask_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Define the dataset and data loader
sar_dir = 'path_to_sar_images'
optical_dir = 'path_to_optical_images'
mask_dir = 'path_to_masks'

dataset = CustomSegmentationDatasetPIL(sar_dir, optical_dir, mask_dir,
                                        transform=image_transforms,
                                        mask_transform=mask_transforms)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Load pretrained DeepLabV3 with ResNet backbone
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
# Update the classifier to match the number of classes in the masks
num_classes = 8  # For example, background and object
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for inputs, masks in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs = inputs.to(device)
        masks = masks.squeeze(1).long().to(device)  # Convert masks to correct shape and type

        # Forward pass
        outputs = model(inputs)['out']  # DeepLabv3 outputs logits
        loss = criterion(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(data_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), 'deeplabv3_custom_dataset.pth')

# Evaluation (Example)
model.eval()
with torch.no_grad():
    for inputs, masks in data_loader:
        inputs = inputs.to(device)
        masks = masks.to(device)

        outputs = model(inputs)['out']
        preds = torch.argmax(outputs, dim=1)  # Get the class with max probability
        # Visualization or further processing of `preds` can be done here