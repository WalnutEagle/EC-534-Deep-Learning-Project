# Create the dataset
dataset = FrontalLateralDataset(frontal, lateral)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize the generator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = GANGenerator(input_channels=3).to(device)
generator.eval()  # Set to evaluation mode

# Test the generator
with torch.no_grad():  # No need to compute gradients during testing
    for frontal_images, lateral_images in dataloader:
        frontal_images = frontal_images.to(device)
        lateral_images = lateral_images.to(device)
        
        # Generate lateral images
        generated_images = generator(frontal_images, lateral_images)
