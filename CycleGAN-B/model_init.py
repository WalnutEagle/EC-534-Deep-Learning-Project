# loss functions
adversarial_loss = nn.BCEWithLogitsLoss()  
cycle_consistency_loss = nn.L1Loss()  # for reconstruction consistency

# optimizers
lr = 1e-5
beta1, beta2 = 0.5, 0.999

optimizer_G = Adam(generator_3d.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = Adam(discriminator_3d.parameters(), lr=lr, betas=(beta1, beta2))

# moving to device
generator_3d.to(device)
discriminator_3d.to(device)

# instantiating generators 
input_channels_2d = 2  # 1 for frontal + 1 for lateral
input_channels_3d = 1  
output_channels_2d = 2  # 1 channel each for frontal and lateral
output_channels_3d = 1  

generator_2d_to_3d = CycleGANGenerator3D(input_channels=input_channels_2d, output_channels=output_channels_3d).to(device)
generator_3d_to_2d = Generator3DTo2D(input_channels=input_channels_3d, output_channels=output_channels_2d).to(device)

# optimizers
optimizer_G_A = Adam(generator_2d_to_3d.parameters(), lr=1e-5, betas=(0.5, 0.999))
optimizer_G_B = Adam(generator_3d_to_2d.parameters(), lr=1e-5, betas=(0.5, 0.999))

# weight initialization
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        elif isinstance(m, (nn.InstanceNorm2d, nn.InstanceNorm3d)) and m.weight is not None:
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

# initializing weights
initialize_weights(generator_2d_to_3d)
initialize_weights(generator_3d_to_2d)
initialize_weights(discriminator_3d)

