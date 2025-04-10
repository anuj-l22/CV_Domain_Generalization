import os
import torch
import numpy as np
from skimage import io, transform
from skimage.feature import hog

DATA_DIR = "/home/rishabh/Anuj_Sem6/CV_S6/DomainBed/domainbed/data/PACS"       # Root directory containing PACS dataset (with subfolders per domain)
OUTPUT_DIR = "/home/rishabh/Anuj_Sem6/CV_S6/DomainBed/domainbed/data/PACS_HOG" # Output directory for HOG feature files

# Domain name mapping: short labels to actual folder names (if needed)
domains = {
    "A": "art_painting",
    "C": "cartoon",
    "P": "photo",
    "S": "sketch"
}

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Determine class label mapping using one domain (they should all have the same classes)
sample_domain = list(domains.values())[0]
classes = sorted(os.listdir(os.path.join(DATA_DIR, sample_domain)))
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
print(f"Classes detected: {classes}")

# Loop over each domain folder and compute HOG features
for env_short, env_folder in domains.items():
    features_list = []
    labels_list = []
    env_path = os.path.join(DATA_DIR, env_folder)
    for class_name in sorted(os.listdir(env_path)):
        class_idx = class_to_idx[class_name]
        class_path = os.path.join(env_path, class_name)
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue  # skip non-image files
            img_path = os.path.join(class_path, img_name)
            # Load image (as RGB array) and resize to 224x224
            image = io.imread(img_path)
            image_resized = transform.resize(image, (224, 224), preserve_range=True)
            # Compute HOG feature vector
            feature_vector = hog(image_resized, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), block_norm='L2-Hys',
                                  channel_axis=-1)  # use channel_axis for color images
            feature_vector = feature_vector.astype('float32')  # convert to float32
            features_list.append(torch.from_numpy(feature_vector))
            labels_list.append(class_idx)
    # Stack features and labels for this domain and save to .pt file
    features_tensor = torch.stack(features_list)  # shape: [N_images, feature_dim]
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    out_file = os.path.join(OUTPUT_DIR, f"{env_short}.pt")
    torch.save((features_tensor, labels_tensor), out_file)
    print(f"Saved {env_short} domain features: {features_tensor.shape} -> {out_file}")
