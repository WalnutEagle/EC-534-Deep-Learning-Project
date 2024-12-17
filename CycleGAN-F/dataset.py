# paths
csv_file = '<fill path to .csv file here>'
image_folder = '<fill path to dataset>'

data = pd.read_csv(csv_file)

# matching the images and their projection type
# adding the 'CXR' prefix to the UID and modifying the 'filename' column
data['filename'] = data['filename'].apply(lambda x: x.replace('.dcm', ''))  # remove .dcm from the filename
data['filename'] = data.apply(lambda row: f"CXR{row['filename']}", axis=1)  # Add 'CXR' before the UID

# saving the modified CSV file
modified_csv_file = "modified_csv_file.csv"  # Output CSV file
data.to_csv(modified_csv_file, index=False)
print(f"Modified CSV saved to: {modified_csv_file}")
df = pd.read_csv('modified_csv_file.csv')

class CustomDataset:
    def __init__(self, csv_file, image_folder, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transforms = transforms
        self.image_files = os.listdir(image_folder)

        # preprocessing data to filter entries based on occurrence count
        self.data = self.filter_data()

    def filter_data(self):
        # extracting CXR identifiers from filenames
        self.data['cxr_id'] = self.data['filename'].apply(lambda x: re.match(r'CXR\d+', x).group())

        # counting occurrences of each identifier
        cxr_counts = Counter(self.data['cxr_id'])

        # filtering out rows where cxr_id occurs only once or more than twice
        filtered_data = self.data[self.data['cxr_id'].map(cxr_counts).between(2, 2, inclusive="both")]

        # dropping the temporary column
        filtered_data = filtered_data.drop(columns=['cxr_id'])
        return filtered_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # getting the row from the filtered .csv file
        row = self.data.iloc[idx]
        csv_filename = row['filename']
        projection = row['projection']

        # finding the image file that matches the CSV filename
        image_path = os.path.join(self.image_folder, csv_filename)
        if not os.path.exists(image_path):
            print(f"Image not found for {csv_filename}")
            return None

#         try:
#             image = Image.open(image_path)
#         except FileNotFoundError:
#             print(f"Image not found at {image_path}")
#             return None
        
        try:
            image = Image.open(image_path).convert("L")  
        except FileNotFoundError:
            print(f"Image not found at {image_path}")
            return None

        # applying transformations if any
        if self.transforms:
            image = self.transforms(image)

        return {'image_type': projection, 'image': image, 'filename': csv_filename}

  # transforms
transforms = v2.Compose([
    v2.Resize((128, 128)),
    #v2.RandomHorizontalFlip(p=0.5),  
    v2.RandomRotation(degrees=10), 
    v2.ColorJitter(brightness=0.2, contrast=0.2),  
    v2.ToTensor(),
    v2.Normalize(mean=[0.0], std=[1.0]) 
])

csv_file = 'modified_csv_file.csv'

# loading dataset
dataset = CustomDataset(csv_file, image_folder, transforms)
