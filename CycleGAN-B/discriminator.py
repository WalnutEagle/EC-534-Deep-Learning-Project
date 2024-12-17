class PatchGANDiscriminator3D(nn.Module):
    def __init__(self, input_channels=1):
        super(PatchGANDiscriminator3D, self).__init__()
        
        # Convolutional layers
        self.model = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # (64 -> 32)
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # (32 -> 16)
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),  # (16 -> 8)
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.3),
            
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),  # (8 -> 4)
            nn.InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv3d(512, 1, kernel_size=3, stride=1, padding=0)  # (4 -> 2)
        )

    def forward(self, x):
        return self.model(x)

# instantiating the discriminator
discriminator_3d = PatchGANDiscriminator3D(input_channels=1)

# TESTING THE DISCRIMINATOR with a dummy input of size 64x64x64
dummy_input_3d = torch.randn(1, 1, 64, 64, 64)  # Batch size of 1
output = discriminator_3d(dummy_input_3d)
print(f"Input shape: {dummy_input_3d.shape}")
print(f"Output shape: {output.shape}")
