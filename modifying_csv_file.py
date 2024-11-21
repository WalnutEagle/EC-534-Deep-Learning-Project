data = pd.read_csv(csv_file)

# matching the images and their projection type
# adding the 'CXR' prefix to the UID and modifying the 'filename' column
data['filename'] = data['filename'].apply(lambda x: x.replace('.dcm', ''))  # Remove .dcm from the filename
data['filename'] = data.apply(lambda row: f"CXR{row['filename']}", axis=1)  # Add 'CXR' before the UID


# saving the modified CSV file
modified_csv_file = "modified_csv_file.csv"  # Output CSV file
data.to_csv(modified_csv_file, index=False)

print(f"Modified CSV saved to: {modified_csv_file}")
