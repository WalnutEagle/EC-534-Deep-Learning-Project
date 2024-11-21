# mapping frontal and lateral images based on their image IDs
frontal_mapping = {}
lateral_mapping = {}

# iterating through the dataset and populate mappings
for item in dataset:
    if item is not None:
        image_type = item['image_type']
        image = item['image']
        uid = int(item['filename'].split('_')[0][3:])  # Extract the UID from the filename

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
