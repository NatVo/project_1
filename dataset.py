import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2

alb_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Rotate(limit=40, p=0.5),
        A.GaussNoise(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05,
                           scale_limit=0.05,
                           rotate_limit=15, p=0.5),
        # A.Normalize(mean=(0.485, 0.456, 0.406),
        #             std=(0.229, 0.224, 0.225)),
        #ToTensorV2()
    ],
    additional_targets = {
        'mask_2': 'mask'
    }
)


class CustomSegmentationDatasetPIL(Dataset):
    """
    A custom dataset for semantic segmentation.
    SAR images, optical images, and masks must have corresponding filenames.
    """

    def __init__(self, sar_dir, optical_dir, mask_dir, transform=None, mask_transform=None):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.mask_dir = mask_dir
        self.sar_images = os.listdir(sar_dir)
        self.optical_images = os.listdir(optical_dir)
        self.mask_images = os.listdir(mask_dir)
        self.transform = transform  # For SAR and optical images
        self.mask_transform = mask_transform  # For masks

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        sar_path = os.path.join(self.sar_dir, self.sar_images[idx])
        sar_image = Image.open(sar_path).convert('L')  # Ensure grayscale

        optical_path = os.path.join(self.optical_dir, self.optical_images[idx])
        optical_image = Image.open(optical_path).convert('RGB')  # Ensure RGB

        mask_path = os.path.join(self.mask_dir, self.mask_images[idx])
        mask = Image.open(mask_path).convert('L')  # Ensure grayscale

        # Apply transformations
        if self.transform:
            sar_image = self.transform(sar_image)
            optical_image = self.transform(optical_image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Combine SAR and optical images into one input tensor
        # Input Tensor: Channel 0 = SAR, Channels 1-3 = Optical RGB
        combined_input = torch.cat([sar_image, optical_image], dim=0)

        return combined_input, mask


class CustomSegmentationDatasetOPenCV(Dataset):
    """
    A custom dataset for semantic segmentation.
    SAR images, optical images, and masks must have corresponding filenames.
    """

    def __init__(self, sar_dir, optical_dir, mask_dir, transform=None, mask_transform=None):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.mask_dir = mask_dir
        self.sar_images = os.listdir(sar_dir)
        self.optical_images = os.listdir(optical_dir)
        self.mask_images = os.listdir(mask_dir)
        self.transform = transform  # For SAR and optical images
        self.mask_transform = mask_transform  # For masks

    def __len__(self):
        return len(self.sar_images)

    def __getitem__(self, idx):
        # Load SAR image
        sar_path = os.path.join(self.sar_dir, self.sar_images[idx])
        sar_image = Image.open(sar_path).convert('L')  # Ensure grayscale

        # Load optical image
        optical_path = os.path.join(self.optical_dir, self.optical_images[idx])
        optical_image = Image.open(optical_path).convert('RGB')  # Ensure RGB

        # Load mask
        mask_path = os.path.join(self.mask_dir, self.mask_images[idx])
        mask = Image.open(mask_path).convert('L')  # Ensure grayscale

        # Apply transformations
        if self.transform:
            sar_image = self.transform(sar_image)
            optical_image = self.transform(optical_image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Combine SAR and optical images into one input tensor
        # Input Tensor: Channel 0 = SAR, Channels 1-3 = Optical RGB
        combined_input = torch.cat([sar_image, optical_image], dim=0)

        return combined_input, mask