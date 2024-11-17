class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(PatchGANDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.net(x)

def lsgan_loss(pred, target):
    return torch.mean((pred - target) ** 2)

def discriminator_loss(real_output, fake_output):
    real_loss = lsgan_loss(real_output, torch.ones_like(real_output))
    fake_loss = lsgan_loss(fake_output, torch.zeros_like(fake_output))
    return (real_loss + fake_loss) / 2

def generator_loss(fake_output, recon_loss, proj_loss, recon_weight=10, proj_weight=10):
    adv_loss = lsgan_loss(fake_output, torch.ones_like(fake_output))
    return adv_loss + recon_weight * recon_loss + proj_weight * proj_loss
