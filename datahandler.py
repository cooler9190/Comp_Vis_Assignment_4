# Code snippet from assignment 4

import torch
import xml.etree.ElementTree as ET
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from data_stratification import train_imgs, train_anns, val_imgs, val_anns, test_imgs, test_anns  # Importing the stratified splits from data_stratification.py

INPUT_IMG_SZ = 112

class CatDogDataset(Dataset):
    def __init__(self, img_files, ann_files, transform=None):
        self.img_files = img_files
        self.ann_files = ann_files
        self.transform = transform
        self.label_map = {"cat": 0, "dog": 1}  # Label mapping

    def parse_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        objects = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            label = self.label_map.get(name, -1)  # Default to -1 if unknown label
            objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})

        return width, height, objects

    def __len__(self):
        return len(self.img_files)

    # In the spirit of YOLOv1, the bounding box format should be [x_center, y_center, width, height] and normalized to [0, 1]
    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        ann_path = self.ann_files[idx]

        image = Image.open(img_path).convert("RGB")
        width, height, objects = self.parse_annotation(ann_path)

        # Create a blank 7x7 grid. Each cell has 7 values: [x, y, w, h, confidence, class1, class2]
        target_matrix = torch.zeros((7, 7, 7))
        
        #bboxes = []
        for obj in objects:
            xmin, ymin, xmax, ymax = obj['bbox']

            # xmin = obj['bbox'][0]
            # ymin = obj['bbox'][1]
            # xmax = obj['bbox'][2]
            # ymax = obj['bbox'][3]

            # Global YOLO format: [x_center, y_center, width, height] normalized to [0, 1]
            x_center = ((xmin + xmax) / 2.0) / width
            y_center = ((ymin + ymax) / 2.0) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            # Find the grid cell (i, j) that contains the center of the bounding box
            i = int(y_center * 7)  # Row index
            j = int(x_center * 7)  # Column index

            # Convert x and y to be offsets realtive to the bounds of specific grid cell
            x_cell = x_center * 7 - j  # Offset within the cell
            y_cell = y_center * 7 - i  # Offset within the cell

            # If no object is already assigned to this cell, assign this one
            if target_matrix[i, j, 4] == 0:  # Check confidence value to see if cell is already assigned
                # Set coordinates and dimensions in the target matrix
                target_matrix[i, j, 0:4] = torch.tensor([x_cell, y_cell, box_width, box_height])
                # Set object confidence to 1 (indicating an object is present)
                target_matrix[i, j, 4] = 1.0
                # Set one-hot encoding for the class label (0 for cat, 1 for dog)
                target_matrix[i, j, 5 + obj['label']] = 1.0  # Class label starts at index 5

        if self.transform:
            image = self.transform(image)

        #     bboxes.append([x_center, y_center, box_width, box_height])  # in your assignment 4, you need to convert bbox into [x, y, w, h] and value range [0, 1]

        # bboxes = torch.tensor(bboxes, dtype=torch.float32)
        # labels = torch.tensor([obj["label"] for obj in objects], dtype=torch.int64)

        # if self.transform:
        #     image = self.transform(image)

        return image, target_matrix


# Define transformations
transform = T.Compose([
    T.Resize((INPUT_IMG_SZ, INPUT_IMG_SZ)),
    T.ToTensor()
])

# Initialize separate datasets using splits
train_dataset = CatDogDataset(img_files=train_imgs, ann_files=train_anns, transform=transform)
val_dataset = CatDogDataset(img_files=val_imgs, ann_files=val_anns, transform=transform)
test_dataset = CatDogDataset(img_files=test_imgs, ann_files=test_anns, transform=transform)

# Initialize dataloaders for each dataset
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# # Function to visualize a batch
# def visualize_batch(dataloader):
#     images, target_matrices = next(iter(dataloader))
#     fig, axes = plt.subplots(1, len(images), figsize=(15, 5))

#     if len(images) == 1:
#         axes = [axes]

#     for i, (img, target_matrix) in enumerate(zip(images, target_matrices)):
#         img = img.permute(1, 2, 0).numpy()
#         axes[i].imshow(img)

#         for j in range(7):
#             for k in range(7):
#                 if target_matrix[j, k, 4] == 1:  # Check if object is present
#                     x_center = (target_matrix[j, k, 0] + k) / 7
#                     y_center = (target_matrix[j, k, 1] + j) / 7
#                     box_width = target_matrix[j, k, 2]
#                     box_height = target_matrix[j, k, 3]
#                     lbl = torch.argmax(target_matrix[j, k, 5:]).item()
#             x_center, y_center, box_width, box_height = box.tolist()
#             xmin = (x_center - box_width / 2) * img.shape[1]
#             ymin = (y_center - box_height / 2) * img.shape[0]
#             xmax = (x_center + box_width / 2) * img.shape[1]
#             ymax = (y_center + box_height / 2) * img.shape[0]

#             rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                      linewidth=2, edgecolor='r', facecolor='none')
#             axes[i].add_patch(rect)
#             axes[i].text(xmin, ymin - 5, f'Label: {lbl.item()}', color='red', fontsize=10,
#                          bbox=dict(facecolor='white', alpha=0.5))
#         axes[i].axis('off')

#     plt.show()


# # Visualize a batch
# visualize_batch(train_dataloader)
