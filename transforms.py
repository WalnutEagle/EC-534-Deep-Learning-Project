# transforms
transforms = v2.Compose([
    v2.Resize((128,128)),
    v2.ToTensor(),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

csv_file = 'modified_csv_file.csv'

# loading dataset
dataset = CustomDataset(csv_file, image_folder, transforms)
