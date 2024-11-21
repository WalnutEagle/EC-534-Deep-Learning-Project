class CustomDataset:
    def __init__(self, csv_file, image_folder, transforms=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transforms = transforms
        self.image_files = os.listdir(image_folder)

        # Preprocess data to filter entries based on occurrence count
        self.data = self.filter_data()

    def filter_data(self):
        # Extract CXR identifiers from filenames
        self.data['cxr_id'] = self.data['filename'].apply(lambda x: re.match(r'CXR\d+', x).group())

        # Count occurrences of each identifier
        cxr_counts = Counter(self.data['cxr_id'])

        # Filter out rows where cxr_id occurs only once or more than twice
        filtered_data = self.data[self.data['cxr_id'].map(cxr_counts).between(2, 2, inclusive="both")]

        # Drop the temporary column
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

        try:
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"Image not found at {image_path}")
            return None

        # applying transformations if any
        if self.transforms:
            image = self.transforms(image)

        return {'image_type': projection, 'image': image, 'filename': csv_filename}
