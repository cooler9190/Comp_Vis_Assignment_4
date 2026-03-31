import torch
from torch import nn
from object_detector import ObjectDetector
from datahandler import train_dataloader, val_dataloader
from torch.optim import Adam
import csv

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=1, C=2):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum') # Sum over all elements for the loss calculation
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes per grid cell
        self.C = C  # Number of classes
        self.lambda_coord = 5.0  # Weight for coordinate loss, from YOLO Paper
        self.lambda_noobj = 0.5  # Weight for no-object confidence loss, from YOLO Paper

    def forward(self, predictions, target):
        # Reshape predictions from (Batch, 343) to (Batch, 7, 7, 7)
        predictions = predictions.view(-1, self.S, self.S, self.B * 5 + self.C)

        # Create boolean masks to separate cells with objects from cells without objects
        obj_mask = target[..., 4] == 1  # Mask for cells that contain objects
        noobj_mask = target[..., 4] == 0  # Mask for cells that do not contain objects

        # Coordinate loss (x, y)
        xy_pred = predictions[..., 0:2]  # Predicted x_center and y_center
        xy_target = target[..., 0:2]  # Target x_center and y_center
        xy_loss = self.mse(xy_pred[obj_mask], xy_target[obj_mask])  # Only calculate for cells with objects

        # Dimensions loss (w, h)
        wh_pred = predictions[..., 2:4]  # Predicted width and height
        wh_target = target[..., 2:4]  # Target width and height
        # We add a small epsilon to prevent taking the square root of zero, which can lead to instability in training
        wh_loss = self.mse(torch.sqrt(wh_pred[obj_mask] + 1e-6), torch.sqrt(wh_target[obj_mask] + 1e-6))  # Only calculate for cells with objects

        # Object confidence loss
        obj_pred = predictions[..., 4:5]  # Predicted confidence score
        obj_target = target[..., 4:5]  # Target confidence score
        obj_loss = self.mse(obj_pred[obj_mask], obj_target[obj_mask])  # Only calculate for cells with objects

        # No-object confidence loss
        noobj_loss = self.mse(obj_pred[noobj_mask], obj_target[noobj_mask])  # Only calculate for cells without objects

        # Class probability loss
        class_pred = predictions[..., 5:7]  # Predicted class probabilities
        class_target = target[..., 5:7]  # Target class probabilities
        class_loss = self.mse(class_pred[obj_mask], class_target[obj_mask])  # Only calculate for cells with objects

        # Total loss is a weighted sum of all components
        total_loss = (
            self.lambda_coord * (xy_loss + wh_loss) 
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + class_loss
        )

        # Loss components
        components = {
            "coordinate": xy_loss.item(),
            "dimension": wh_loss.item(),
            "object_confidence": obj_loss.item(),
            "noobject_confidence": noobj_loss.item(),
            "class_confidence": class_loss.item()
        }

        # Return the average loss per item in the batch
        return (total_loss / predictions.shape[0]), components


fields = [
    "epoch",
    "train_loss", "validation_loss",
    "train_coordinate", "validation_coordinate",
    "train_dimension", "validation_dimension",
    "train_object_confidence", "validation_object_confidence",
    "train_noobject_confidence", "validation_noobject_confidence",
    "train_class_confidence", "validation_class_confidence",
]
# Initialise csv document with correct field names.
def initialise_logs():
    with open("loss_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

# Fill csv document with loss values.
def fill_logs(epoch, train_loss, train_components, validation_loss, validation_components):
    epoch_data = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "train_coordinate": train_components["coordinate"],
        "validation_coordinate": validation_components["coordinate"],
        "train_dimension": train_components["dimension"],
        "validation_dimension": validation_components["dimension"],
        "train_object_confidence": train_components["object_confidence"],
        "validation_object_confidence": validation_components["object_confidence"],
        "train_noobject_confidence": train_components["noobject_confidence"],
        "validation_noobject_confidence": validation_components["noobject_confidence"],
        "train_class_confidence": train_components["class_confidence"],
        "validation_class_confidence": validation_components["class_confidence"],
    }
    with open("loss_log.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow(epoch_data)

def train_model():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "N/A")
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

    # Initialize model, loss function, and optimizer
    model = ObjectDetector().to(device)
    criterion = YoloLoss().to(device)

    # We use Adam optimizer instead of SGD for faster convergence, especially since YOLO can be sensitive to learning rates and benefits from adaptive learning rate adjustments.
    optimizer = Adam(model.parameters(), lr=1e-4) # 1e-4 is a good starting point for YOLO

    # Early stopping hyperparameters
    max_epochs = 100
    patience = 5
    best_val_loss = float('inf')
    impatience = 0

    # Initialise log file to store losses in.
    initialise_logs()

    # Epoch.
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        components = ["coordinate", "dimension", "object_confidence", "noobject_confidence", "class_confidence"]
        train_loss_components = {k: 0.0 for k in components}


        for images, targets in train_dataloader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()  # Clear gradients
            predictions = model(images)  # Forward pass
            loss, loss_components = criterion(predictions, targets)  # Compute loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Accumulate loss (components) for the epoch
            train_loss += loss.item()
            for k in loss_components: train_loss_components[k] += loss_components[k]

        # Average over the batch.
        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_loss_components = {k: v / len(train_dataloader) for k, v in train_loss_components.items()}

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_loss_components= {k: 0.0 for k in components}

        with torch.no_grad():  # No need to compute gradients during validation
            for images, targets in val_dataloader:
                images, targets = images.to(device), targets.to(device)

                predictions = model(images)  # Forward pass
                loss, loss_components = criterion(predictions, targets)  # Compute loss

                # Accumulate loss (components) for the epoch
                val_loss += loss.item()
                for k in loss_components: val_loss_components[k] += loss_components[k]

        # Average over the batch.
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_loss_components = {k: v / len(val_dataloader) for k, v in val_loss_components.items()}

        # Update logs, and print training/validation losses.
        fill_logs(epoch, avg_train_loss, avg_train_loss_components, avg_val_loss, avg_val_loss_components)
        print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")



        # Check for early stopping
        if avg_val_loss < best_val_loss: # Lower loss, improvement.
            # Update lowest/best loss, and reset patience.
            best_val_loss = avg_val_loss
            impatience = 0

            # Save the best model weights
            torch.save(model.state_dict(), "best_yolo_model.pth")
        else: # Equal/Higher loss, no improvement.
            impatience += 1 #
            if impatience >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    print("Training complete. Best validation loss: {:.4f}".format(best_val_loss))
    print("Best model saved as 'best_yolo_model.pth'.")

