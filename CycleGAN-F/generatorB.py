class Generator3DTo2D(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(Generator3DTo2D, self).__init__()

        # 3D feature extraction layers
        self.feature_extraction = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # (64, 64, 64) -> (32, 32, 32)
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),  # (32, 32, 32) -> (16, 16, 16)
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),  # (16, 16, 16) -> (8, 8, 8)
            nn.InstanceNorm3d(256),
            nn.ReLU(inplace=True),
        )

        self.output_channels = output_channels

        # 2D upsampling layer
        self.upsample = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)

    def forward(self, x):
        # feature extraction from 3D
        x = self.feature_extraction(x)

        # flattenning the 3D volume into 2D
        batch_size, channels, depth, height, width = x.shape
        x = x.view(batch_size, channels * depth, height, width)  # Merge depth dimension into channels

        # reducing the output channels
        final_conv = nn.Conv2d(
            in_channels=channels * depth, out_channels=self.output_channels,
            kernel_size=1, stride=1, padding=0
        ).to(x.device)  

        x = final_conv(x)  

        # upsampling to restore original 2D dimensions
        x = self.upsample(x)  
        return x

# TESTING THE GENERATOR with dummy input
dummy_input_3d = torch.randn(1, 1, 64, 64, 64).to(device)  
generator_3d_to_2d = Generator3DTo2D(input_channels=1, output_channels=1).to(device)  
output_2d = generator_3d_to_2d(dummy_input_3d)

print(f"Input shape (3D): {dummy_input_3d.shape}")
print(f"Output shape (2D): {output_2d.shape}")
