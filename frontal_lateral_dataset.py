class FrontalLateralDataset(Dataset):
    def __init__(self, frontal, lateral, transforms=None):
        self.frontal = frontal
        self.lateral = lateral
        self.transforms = transforms

    def __len__(self):
        return len(self.frontal)

    def __getitem__(self, idx):
        frontal_image = self.frontal[idx]
        lateral_image = self.lateral[idx]

        # Apply transformations if defined
        if self.transforms:
            frontal_image = self.transforms(frontal_image)
            lateral_image = self.transforms(lateral_image)

        return frontal_image, lateral_image
