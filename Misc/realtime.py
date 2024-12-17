# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class DoubleConv3d(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.double_conv = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

# class Down3d(nn.Module):
#     """Downscaling with maxpool then double conv"""
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool3d(2),
#             DoubleConv3d(in_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up3d(nn.Module):
#     """Upscaling then double conv"""
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
#         self.conv = DoubleConv3d(in_channels, out_channels)


#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffZ = x2.size()[2] - x1.size()[2]
#         diffY = x2.size()[3] - x1.size()[3]
#         diffX = x2.size()[4] - x1.size()[4]

#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2,
#                         diffZ // 2, diffZ - diffZ // 2])
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)


# class OutConv3d(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv3d, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         return self.conv(x)
    
# class UNet3D(nn.Module):
#     def __init__(self, in_channels, out_channels, base_channels=32):
#         super(UNet3D, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.base_channels = base_channels

#         self.inc = DoubleConv3d(in_channels, base_channels)
#         self.down1 = Down3d(base_channels, 2*base_channels)
#         self.down2 = Down3d(2*base_channels, 4*base_channels)
#         self.down3 = Down3d(4*base_channels, 8*base_channels)
#         self.down4 = Down3d(8*base_channels, 8*base_channels)
        
#         self.up1 = Up3d(16*base_channels, 4*base_channels)
#         self.up2 = Up3d(8*base_channels, 2*base_channels)
#         self.up3 = Up3d(4*base_channels, base_channels)
#         self.up4 = Up3d(2*base_channels, base_channels)
#         self.outc = OutConv3d(base_channels, out_channels)
        
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
        
#         logits = self.outc(x)
#         return torch.sigmoid(logits)

# if __name__ == '__main__':
#   # Example Usage
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Assume input images are 3 channel RGB, and we will concatenate the front and lateral images
#     # along the channel dimension
#     in_channels = 3*2 
#     out_channels = 1 # output the 3D volume, in a single channel for each voxel
    
#     # Create the generator
#     generator = UNet3D(in_channels, out_channels).to(device)
    
#     # Create dummy data
#     dummy_input = torch.randn(1, in_channels, 64, 64, 64).to(device) # Batch_size, channel, width, height, depth
    
#     # Perform forward pass
#     output = generator(dummy_input)
    
#     print("Generator output shape:", output.shape)


# import torch
# import torch.nn as nn

# class Discriminator(nn.Module):
#     def __init__(self, in_channels=6, base_channels=64):
#         """
#         Discriminator model for distinguishing between real and fake image pairs.

#         Args:
#             in_channels (int): Number of input channels for the discriminator (e.g. RGB concatenated front and lateral views).
#             base_channels (int): Number of channels for the first layer of discriminator
#         """
#         super(Discriminator, self).__init__()
#         self.in_channels = in_channels
#         self.base_channels = base_channels
        
#         self.main = nn.Sequential(
#             # Layer 1
#             nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             # Layer 2
#             nn.Conv2d(base_channels, 2*base_channels, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(2*base_channels),
#             nn.LeakyReLU(0.2, inplace=True),
#             # Layer 3
#             nn.Conv2d(2*base_channels, 4*base_channels, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(4*base_channels),
#             nn.LeakyReLU(0.2, inplace=True),
#             # Layer 4
#             nn.Conv2d(4*base_channels, 8*base_channels, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(8*base_channels),
#             nn.LeakyReLU(0.2, inplace=True),
#             # Layer 5
#             nn.Conv2d(8*base_channels, 1, kernel_size=4, stride=1, padding=0),
#             nn.Sigmoid() # Binary classification
#         )
        
#     def forward(self, x):
#         return self.main(x)

# if __name__ == '__main__':
#   # Example Usage
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Assuming input image pairs (frontal + lateral concatenated) have 6 channels (3 for each RGB image).
#     in_channels = 3 * 2
    
#     # Create the discriminator
#     discriminator = Discriminator(in_channels=in_channels).to(device)
    
#     # Create dummy data (batch size of 1)
#     dummy_input = torch.randn(1, in_channels, 256, 256).to(device)
    
#     # Perform forward pass
#     output = discriminator(dummy_input)
    
#     print("Discriminator output shape:", output.shape)


# import torch

# def project_3d_volume(volume):
#     """
#     Projects a 3D volume to frontal and lateral 2D images.

#     Args:
#       volume (torch.Tensor): 3D tensor of shape (batch_size, channels, depth, height, width)

#     Returns:
#       tuple: Tuple containing the frontal (batch_size, channels, height, width) and lateral
#              (batch_size, channels, depth, width) projections
#     """
#     batch_size, channels, depth, height, width = volume.shape

#     # Frontal projection (sum along depth dimension)
#     frontal_projection = torch.sum(volume, dim=2) # output is (batch_size, channels, height, width)

#     # Lateral projection (sum along height dimension)
#     lateral_projection = torch.sum(volume, dim=3) # output is (batch_size, channels, depth, width)

#     return frontal_projection, lateral_projection

# if __name__ == '__main__':
#     # Example Usage
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Create a dummy 3D volume for testing
#     dummy_volume = torch.randn(1, 1, 64, 64, 64).to(device)

#     # Project the volume
#     frontal_proj, lateral_proj = project_3d_volume(dummy_volume)

#     # Print shapes of the projections
#     print("Frontal projection shape:", frontal_proj.shape)
#     print("Lateral projection shape:", lateral_proj.shape)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.multiprocessing as mp
# --------------------------------------------------
#  Dataset Class
# --------------------------------------------------
class XRayDataset(Dataset):
    def __init__(self, csv_file, img_folder, transform=None, device=None):
        """
        Initializes dataset with CSV file and image folder.
        """
        self.csv = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.transform = transform
        self.pairs = self.csv.groupby('uid')
        self.valid_uids = self.check_integrity()
        self.device = device

    def check_integrity(self):
        """
        Ensure we have both 'Frontal' and 'Lateral' images for each UID.
        """
        valid_uids = []
        for uid, group in self.pairs:
            frontal_data = group[group['projection'] == 'Frontal']
            lateral_data = group[group['projection'] == 'Lateral']
            if not frontal_data.empty and not lateral_data.empty:
                valid_uids.append(uid)
        return valid_uids

    def __len__(self):
        return len(self.valid_uids)

    def __getitem__(self, idx):
        """
        Fetch the images for a specific UID.
        """
        group_key = self.valid_uids[idx]
        uid = self.pairs.get_group(group_key)
        
        # Get file paths for both Frontal and Lateral images
        frontal_img_path = os.path.join(self.img_folder, uid[uid['projection'] == 'Frontal']['filename'].values[0])
        lateral_img_path = os.path.join(self.img_folder, uid[uid['projection'] == 'Lateral']['filename'].values[0])

        # Read images
        frontal_img = cv2.imread(frontal_img_path)
        lateral_img = cv2.imread(lateral_img_path)

        if frontal_img is None or lateral_img is None:
            print(f"Error reading image: {frontal_img_path if frontal_img is None else lateral_img_path}")
            return None, None  # Return None if there's an issue reading the image

        # Resize images
        frontal_img_resized = cv2.resize(frontal_img, (256, 256))
        lateral_img_resized = cv2.resize(lateral_img, (256, 256))

        # Convert BGR to RGB and then to tensor
        frontal_img_tensor = torch.tensor(cv2.cvtColor(frontal_img_resized, cv2.COLOR_BGR2RGB), dtype=torch.float32)
        lateral_img_tensor = torch.tensor(cv2.cvtColor(lateral_img_resized, cv2.COLOR_BGR2RGB), dtype=torch.float32)

        # Permute to CxHxW format
        frontal_img_tensor = frontal_img_tensor.permute(2, 0, 1)
        lateral_img_tensor = lateral_img_tensor.permute(2, 0, 1)
        
        if self.transform:
            frontal_img_tensor = self.transform(frontal_img_tensor)
            lateral_img_tensor = self.transform(lateral_img_tensor)

        if self.device:
           frontal_img_tensor = frontal_img_tensor.to(self.device)
           lateral_img_tensor = lateral_img_tensor.to(self.device)


        return frontal_img_tensor, lateral_img_tensor
    
# --------------------------------------------------
# 3D UNET Generator Class
# --------------------------------------------------
class DoubleConv3d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down3d(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3d(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = DoubleConv3d(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        self.inc = DoubleConv3d(in_channels, base_channels)
        self.down1 = Down3d(base_channels, 2*base_channels)
        self.down2 = Down3d(2*base_channels, 4*base_channels)
        self.down3 = Down3d(4*base_channels, 8*base_channels)
        self.down4 = Down3d(8*base_channels, 8*base_channels)
        
        self.up1 = Up3d(16*base_channels, 4*base_channels)
        self.up2 = Up3d(8*base_channels, 2*base_channels)
        self.up3 = Up3d(4*base_channels, base_channels)
        self.up4 = Up3d(2*base_channels, base_channels)
        self.outc = OutConv3d(base_channels, out_channels)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)
    
# --------------------------------------------------
# Discriminator Class
# --------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=6, base_channels=64):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        self.main = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(base_channels, 2*base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(2*base_channels, 4*base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4*base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4
            nn.Conv2d(4*base_channels, 8*base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8*base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 5
            nn.Conv2d(8*base_channels, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid() # Binary classification
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1) # Adaptive average pooling
        
    def forward(self, x):
         x = self.main(x)
         x = self.avgpool(x) # Apply average pooling
         return x.view(x.size(0), -1)
    
# --------------------------------------------------
# Projection Function
# --------------------------------------------------
def project_3d_volume(volume):
    """
    Projects a 3D volume to frontal and lateral 2D images.

    Args:
      volume (torch.Tensor): 3D tensor of shape (batch_size, channels, depth, height, width)

    Returns:
      tuple: Tuple containing the frontal (batch_size, channels, height, width) and lateral
             (batch_size, channels, depth, width) projections
    """
    batch_size, channels, depth, height, width = volume.shape

    # Frontal projection (sum along depth dimension)
    frontal_projection = torch.sum(volume, dim=2) # output is (batch_size, channels, height, width)

    # Lateral projection (sum along depth dimension)
    lateral_projection = torch.sum(volume, dim=2) # output is (batch_size, channels, height, width)

    return frontal_projection, lateral_projection
# --------------------------------------------------
# Training Script
# --------------------------------------------------
def train_gan(
    csv_file, 
    img_folder,
    device, 
    num_epochs=100, 
    batch_size=16, 
    lr=0.0002,
    base_channels_gen=32,
    base_channels_disc=64
):
    """
    Main training function for the GAN
    """

    # --------------------
    # Dataset and DataLoader
    # --------------------
    dataset = XRayDataset(csv_file=csv_file, img_folder=img_folder, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    generator = UNet3D(in_channels=3, out_channels=3, base_channels=base_channels_gen).to(device)
    discriminator = Discriminator(in_channels=6, base_channels=base_channels_disc).to(device) # input is concatenated front and lateral projections

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)

    # --------------------
    # Optimizers
    # --------------------
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # --------------------
    # Loss Functions
    # --------------------
    criterion = nn.BCELoss() # Binary cross entropy loss

    # --------------------
    # Training Loop
    # --------------------
    for epoch in range(num_epochs):
        for i, (frontal_real, lateral_real) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            
            batch_size = frontal_real.size(0)
            
            # --------------------
            # Train Discriminator
            # --------------------
            discriminator.zero_grad()

            # Real images
            real_labels = torch.ones(batch_size, 1, device=device)
            real_pairs = torch.cat((frontal_real, lateral_real), dim=1)
            output_real = discriminator(real_pairs)
            loss_d_real = criterion(output_real, real_labels)

            # Fake images
            fake_labels = torch.zeros(batch_size, 1, device=device)
            noise = torch.randn(batch_size, 3, 32, 256, 256, device=device) # assuming 32 depth (adjust this based on your need)
            fake_volume = generator(noise) # Generator outputs a 3D volume
            frontal_fake, lateral_fake = project_3d_volume(fake_volume)
            fake_pairs = torch.cat((frontal_fake, lateral_fake), dim=1) # 6-channel input for the discriminator
            output_fake = discriminator(fake_pairs.detach())  # detach to not train generator here
            loss_d_fake = criterion(output_fake, fake_labels)

            # Total discriminator loss
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            # --------------------
            # Train Generator
            # --------------------
            generator.zero_grad()

            # Generate fake images again
            noise = torch.randn(batch_size, 3, 32, 256, 256, device=device)
            fake_volume = generator(noise)
            frontal_fake, lateral_fake = project_3d_volume(fake_volume)
            fake_pairs = torch.cat((frontal_fake, lateral_fake), dim=1)

            # Generator tries to make discriminator predict real
            output_fake = discriminator(fake_pairs) # No detach here, training generator
            loss_g = criterion(output_fake, real_labels) # try to predict real

            loss_g.backward()
            optimizer_g.step()
            
            if (i + 1) % 10 == 0:
               print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")


    print("Training Finished")
if __name__ == "__main__":

    # --- Settings (Change these to your dataset)---
    img_folder = '/home/adwait/Desktop/ecprj/images/images_normalized/'
    csv_file = '/home/adwait/Desktop/ecprj/indiana_projections.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 100
    batch_size = 16
    lr = 0.0002
    base_channels_gen = 32
    base_channels_disc = 64
    # --- End Settings ----
    mp.set_start_method('spawn')
    # Call the train_gan Function
    train_gan(csv_file, img_folder, device, num_epochs, batch_size, lr, base_channels_gen, base_channels_disc)