import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    """Residual Block for CycleGAN"""
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)

class GeneratorCycleGAN(nn.Module):
    """Generator: U-Net with residual blocks"""
    def __init__(self, in_channels, out_channels, n_res_blocks=9):
        super(GeneratorCycleGAN, self).__init__()
        model = [
            nn.Conv2d(in_channels, 64, kernel_size=7, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
        ]
        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True),
            ]
            in_features = out_features
            out_features *= 2
        # Residual blocks
        for _ in range(n_res_blocks):
            model += [ResNetBlock(in_features)]
        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(True),
            ]
            in_features = out_features
            out_features //= 2
        # Final layer
        model += [nn.Conv2d(64, out_channels, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class DiscriminatorCycleGAN(nn.Module):
    """PatchGAN Discriminator for CycleGAN"""
    def __init__(self, in_channels):
        super(DiscriminatorCycleGAN, self).__init__()
        model = [
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
        ]
        in_features = 64
        out_features = in_features * 2
        for _ in range(3):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.LeakyReLU(0.2, True),
            ]
            in_features = out_features
            out_features *= 2
        model += [nn.Conv2d(in_features, 1, kernel_size=4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


#######################################################
#################### TRAINING #########################
#######################################################

def cycle_consistency_loss(real, reconstructed):
    return torch.mean(torch.abs(real - reconstructed))

def identity_loss(real, same):
    return torch.mean(torch.abs(real - same))
G_AB = GeneratorCycleGAN(1, 1)
G_BA = GeneratorCycleGAN(1, 1)
D_A = DiscriminatorCycleGAN(1)
D_B = DiscriminatorCycleGAN(1)

optim_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999))
optim_D_A = torch.optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optim_D_B = torch.optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100
batch_size = 1

generator = X2CTGenerator().to(device)
discriminator = PatchGANDiscriminator(1).to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

def adversarial_loss(pred, target):
    return F.mse_loss(pred, target)

# Training loop
def train_cyclegan(G_AB, G_BA, D_A, D_B, train_loader, epochs, device):
    G_AB.to(device)
    G_BA.to(device)
    D_A.to(device)
    D_B.to(device)
    
    for epoch in range(epochs):
        for real_a, real_b in train_loader:  # real_a: real X-rays, real_b: synthetic X-rays
            real_a, real_b = real_a.to(device), real_b.to(device)

            # === Train Generators ===
            optimizer_G.zero_grad()
            
            # Forward pass
            fake_b = G_AB(real_a)
            fake_a = G_BA(real_b)
            
            rec_a = G_BA(fake_b)  # Cycle back to domain A
            rec_b = G_AB(fake_a)  # Cycle back to domain B
            
            # Generator adversarial loss
            loss_G_AB = adversarial_loss(D_B(fake_b), torch.ones_like(D_B(fake_b)))
            loss_G_BA = adversarial_loss(D_A(fake_a), torch.ones_like(D_A(fake_a)))
            
            # Cycle-consistency loss
            loss_cycle_a = cycle_consistency_loss(real_a, rec_a)
            loss_cycle_b = cycle_consistency_loss(real_b, rec_b)
            
            # Identity loss
            id_a = G_BA(real_a)  # Identity mapping A -> A
            id_b = G_AB(real_b)  # Identity mapping B -> B
            loss_identity_a = identity_loss(real_a, id_a)
            loss_identity_b = identity_loss(real_b, id_b)
            
            # Total generator loss
            loss_G = (loss_G_AB + loss_G_BA + 10.0 * (loss_cycle_a + loss_cycle_b) + 5.0 * (loss_identity_a + loss_identity_b))
            loss_G.backward()
            optimizer_G.step()
            
            # === Train Discriminators ===
            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()
            
            # Discriminator A
            real_pred_A = D_A(real_a)
            fake_pred_A = D_A(fake_a.detach())
            loss_D_A = 0.5 * (adversarial_loss(real_pred_A, torch.ones_like(real_pred_A)) +
                              adversarial_loss(fake_pred_A, torch.zeros_like(fake_pred_A)))
            
            # Discriminator B
            real_pred_B = D_B(real_b)
            fake_pred_B = D_B(fake_b.detach())
            loss_D_B = 0.5 * (adversarial_loss(real_pred_B, torch.ones_like(real_pred_B)) +
                              adversarial_loss(fake_pred_B, torch.zeros_like(fake_pred_B)))
            
            # Update discriminators
            loss_D_A.backward()
            optimizer_D_A.step()
            
            loss_D_B.backward()
            optimizer_D_B.step()

        print(f"Epoch [{epoch+1}/{epochs}] | Loss G: {loss_G.item():.4f} | Loss D_A: {loss_D_A.item():.4f} | Loss D_B: {loss_D_B.item():.4f}")
