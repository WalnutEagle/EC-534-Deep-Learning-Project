# 3D residual block
class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class CycleGANGenerator3D(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, num_residual_blocks=6, target_depth=64):
        super(CycleGANGenerator3D, self).__init__()
        self.target_depth = target_depth

        # initial convolution clock
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3)
        )

        # downsampling layers
        self.downsampling = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (64, 64) -> (32, 32)
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 32, 32) -> (16, 16)
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # reshaping to 3D
        self.reshape_to_3d = nn.Conv3d(256, 256, kernel_size=(1, 1, 1))

        # residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock3D(256) for _ in range(num_residual_blocks)]
        )

        # upsampling layers
        self.upsampling = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(0, 0, 0)),  # (16, 16, 16) -> (32, 32, 32)
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.ConvTranspose3d(128, 64, kernel_size=(4, 4, 4), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(0, 0, 0)),  # (32, 32, 32) -> (64, 64, 64)
            nn.InstanceNorm3d(64),
            nn.ReLU(inplace=True)
        )

        # output layer
        self.output_layer = nn.Sequential(
            nn.Conv3d(64, output_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        # processing 2D input
        x = self.initial(x)

        x = self.downsampling(x)

        # reshape to add a depth dimension
        x = x.unsqueeze(2)  

        x = self.reshape_to_3d(x)

        x = self.residual_blocks(x)

        x = self.upsampling(x)

        # resizing to final 3D volume
        x = nn.functional.interpolate(x, size=(self.target_depth, 64, 64), mode='trilinear', align_corners=True)

        return self.output_layer(x)

# instantiating the generator for 64x64 inputs and 64x64x64 outputs
generator_3d = CycleGANGenerator3D(input_channels=1, output_channels=1, target_depth=64)

# TESTING THE GENERATOR with a dummy input of size 64x64
dummy_input_2d = torch.randn(1, 1, 64, 64)  
output_3d = generator_3d(dummy_input_2d)
print(f"Input shape: {dummy_input_2d.shape}")
print(f"Output shape: {output_3d.shape}")
