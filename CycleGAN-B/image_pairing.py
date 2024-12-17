# mapping frontal and lateral images based on their image IDs
frontal_mapping = {}
lateral_mapping = {}

# iterating through the dataset and populate mappings
for item in dataset:
    if item is not None:
        image_type = item['image_type']
        image = item['image']
        uid = int(item['filename'].split('_')[0][3:])  # extracting the UID from the filename

        if image_type == 'Frontal':
            frontal_mapping[uid] = image
        elif image_type == 'Lateral':
            lateral_mapping[uid] = image
        else:
            print('Unexpected type')

# extracting only the images that have both frontal and lateral biplanar views
frontal, lateral = [], []
for uid, lateral_image in lateral_mapping.items():
    if uid in frontal_mapping:
        frontal.append(frontal_mapping[uid])
        lateral.append(lateral_image)

print(f"Final counts -> Frontal: {len(frontal)}, Lateral: {len(lateral)}")


class PairedXRayDataset(Dataset):
    def __init__(self, frontal_images, lateral_images):
        self.frontal_images = frontal_images
        self.lateral_images = lateral_images
        
    def __len__(self):
        return len(self.frontal_images)
    
    def __getitem__(self, idx):
        frontal_image = self.frontal_images[idx]
        lateral_image = self.lateral_images[idx]
        
        return {'frontal': frontal_image, 'lateral': lateral_image}


# forming the dataset
paired_dataset = PairedXRayDataset(frontal, lateral)

dataset_size = len(paired_dataset)

# defining train-test split sizes
train_size = int(0.9 * dataset_size)  # 90% for training
test_size = dataset_size - train_size  # 10% for testing

# performing the split
train_dataset, test_dataset = random_split(paired_dataset, [train_size, test_size])

# creating dataLoaders for train and test sets
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# verifying the splits
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Testing samples: {len(test_loader.dataset)}")
