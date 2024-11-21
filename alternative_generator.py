class XRayTo3DGenerator(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super(XRayTo3DGenerator, self).__init__()
        
        # Input channels for frontal and lateral images (2 channels)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        
        # Upsampling to form the 3D image using deconvolution layers
        self.deconv1 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose3d(64, out_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, frontal_img, lateral_img):
        # Concatenate the frontal and lateral X-ray images along the channel dimension
        x = torch.cat([frontal_img, lateral_img], dim=1)
        
        # Apply convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Add a channel dimension for 3D deconvolution
        x = x.unsqueeze(2)  # Add a dummy dimension to simulate a 3D input (B, C, D, H, W)
        
        # Apply deconvolution (upsampling)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)  # Output the 3D image
        
        return x

# Testing the Generator
if __name__ == '__main__':
    # Example input (batch_size=1, channels=2, height=128, width=128)
    frontal_img = torch.randn(1, 1, 128, 128)  # Single frontal image
    lateral_img = torch.randn(1, 1, 128, 128)  # Single lateral image
    
    # Instantiate the generator model
    generator = XRayTo3DGenerator()
    
    # Forward pass through the generator
    output = generator(frontal_img, lateral_img)
    
    print(f"Output shape: {output.shape}")
