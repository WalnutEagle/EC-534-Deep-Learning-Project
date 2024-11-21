# Define a Dense Block for 2D operations
class DenseBlock2D(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock2D, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.InstanceNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
            ))
            in_channels += growth_rate

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)  # Concatenate input and output along channel dimension
        return x



# Define Basic 2D/3D Convolution Block
class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type="2D"):
        super(BasicConvBlock, self).__init__()
        if conv_type == "2D":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.norm = nn.InstanceNorm2d(out_channels)
        elif conv_type == "3D":
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
            self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))



# Define Up-Convolution Block]
class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_type="3D"):
        super(UpConvBlock, self).__init__()
        if conv_type == "3D":
            self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        elif conv_type == "2D":
            self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = nn.InstanceNorm3d(out_channels) if conv_type == "3D" else nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.upconv(x)))



class GANGenerator(nn.Module):
    def __init__(self, input_channels=1, growth_rate=32, num_dense_layers=4, num_upsample=4):
        super(GANGenerator, self).__init__()

        # Frontal branch (2D)
        self.frontal_branch = nn.Sequential(
            BasicConvBlock(input_channels, growth_rate, conv_type="2D"),
            DenseBlock2D(growth_rate, growth_rate, num_dense_layers),
            nn.Conv2d(growth_rate * (num_dense_layers + 1), growth_rate, kernel_size=1),
        )

        # Lateral branch (2D)
        self.lateral_branch = nn.Sequential(
            BasicConvBlock(input_channels, growth_rate, conv_type="2D"),
            DenseBlock2D(growth_rate, growth_rate, num_dense_layers),
            nn.Conv2d(growth_rate * (num_dense_layers + 1), growth_rate, kernel_size=1),
        )

        # Shared dense encoder (2D)
        self.shared_encoder = nn.Sequential(
            nn.MaxPool2d(2),
            DenseBlock2D(growth_rate, growth_rate, num_dense_layers),
            nn.Conv2d(growth_rate * (num_dense_layers + 1), growth_rate, kernel_size=1),
        )

        # Channel reduction before passing to the decoder
        self.channel_reduction = nn.Conv2d(growth_rate, 32, kernel_size=1)

        # Decoder with 3D Up-Convolutions
        self.decoder = nn.ModuleList()
        for _ in range(num_upsample):
            self.decoder.append(nn.Sequential(
                UpConvBlock(32, 32, conv_type="3D"),
                BasicConvBlock(32, 32, conv_type="3D"),
            ))

        # Final 3D output layer (generating depth/volume)
        self.final_layer = nn.Conv3d(32, 1, kernel_size=1)  # 1 for depth or voxel

        # Final activation 
        self.final_activation = nn.Sigmoid()  

    def forward(self, frontal, lateral):
        # Process the frontal and lateral branches
        frontal_feat = self.frontal_branch(frontal)
        lateral_feat = self.lateral_branch(lateral)

        # Combine features from both branches
        combined_feat = frontal_feat + lateral_feat  
        #combined_feat = torch.cat([frontal_feat, lateral_feat], dim=1)

        # Pass through the shared encoder
        encoded_feat = self.shared_encoder(combined_feat)

        # Reduce channels before passing to decoder
        encoded_feat = self.channel_reduction(encoded_feat)

        # Reshape for decoding (adding an extra dimension for 3D)
        encoded_feat = encoded_feat.unsqueeze(2)  # Convert to 3D format

        # Decode with 3D up-convolutions
        x = encoded_feat
        for upconv in self.decoder:
            x = upconv(x)

        # Final 3D output layer
        output = self.final_layer(x)

        # Apply final activation (sigmoid/tanh)
        output = self.final_activation(output)
        
        return output
