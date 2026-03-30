# Since PyTorch's random_split does not guarantee stratified splits, we will implement our own stratified splitting function
# using sklearn's train_test_split function, which allows for stratified sampling based on class labels
# ensuring the ratio of cats and dogs is consistent across all sets.

import os
import glob
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

IMG_DIR = "./cat_dog_dataset/images"
ANNOTATION_DIR = './cat_dog_dataset/annotations'

# Gather all image and annotation file paths
img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
ann_files = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*.xml")))

# Extract labels from annotations to use for stratification
image_labels = []
for ann_path in ann_files:
    tree = ET.parse(ann_path)
    root = tree.getroot()

    # Grab first object label for stratification (assuming one object per image)
    first_obj = root.find("object")
    if first_obj is not None:
        image_labels.append(first_obj.find("name").text)
    else:
        image_labels.append("unknown")  # Handle case with no objects

# Firt split: 10% test set, 90% remaining for train + validation
trainval_imgs, test_imgs, trainval_anns, test_anns, trainval_labels, _ = train_test_split(
    img_files, ann_files, image_labels,
    test_size=0.10,
    random_state=42,
    stratify=image_labels
)

# Second split: 20% of remaining for validation, 80% for training
train_imgs, val_imgs, train_anns, val_anns = train_test_split(
    trainval_imgs, trainval_anns,
    test_size=0.20,
    random_state=42,
    stratify=trainval_labels
)

print(f"Total images: {len(img_files)}")
print(f"Training set: {len(train_imgs)} images")
print(f"Validation set: {len(val_imgs)} images")
print(f"Test set: {len(test_imgs)} images")