import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm2d(in_channels + i * growth_rate),
                    nn.ReLU(),
                    nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1)
                )
            )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, dim=1))
            features.append(new_feature)
        return torch.cat(features, dim=1)

class EncoderDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            DenseBlock(64, 32, 4),  # Growth rate 32, 4 layers.
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        features = self.encoder(x)
        features_3d = features.unsqueeze(-1).repeat(1, 1, 1, 1, 128)  # Expand to 3D.
        return self.decoder(features_3d)

class X2CTGenerator(nn.Module):
    def __init__(self):
        super(X2CTGenerator, self).__init__()
        self.pa_encoder_decoder = EncoderDecoder(1, 1)  # PA View
        self.lat_encoder_decoder = EncoderDecoder(1, 1)  # Lateral View
        self.fusion_layer = nn.Conv3d(2, 1, kernel_size=3, padding=1)  # Fuse both.
    
    def forward(self, pa_xray, lat_xray):
        pa_features = self.pa_encoder_decoder(pa_xray)
        lat_features = self.lat_encoder_decoder(lat_xray)
        fused_features = torch.cat([pa_features, lat_features], dim=1)  # Concatenate channels.
        return self.fusion_layer(fused_features)
    