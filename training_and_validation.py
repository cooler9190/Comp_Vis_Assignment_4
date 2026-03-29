import torch
from torch import nn
from object_detector import ObjectDetector
from datahandler import train_dataloader, val_dataloader
from torch.optim import Adam

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=1, C=2):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum') # Sum over all elements for the loss calculation
        self.S = S  # Grid size
        self.B = B  # Number of bounding boxes per grid cell
        self.C = C  # Number of classes
        self.lambda_coord = 5.0  # Weight for coordinate loss
        self.lambda_noobj = 0.5  # Weight for no-object confidence loss

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

        # Return the average loss per item in the batch
        return total_loss / predictions.shape[0]
    

def train_model():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, loss function, and optimizer
    model = ObjectDetector().to(device)
    criterion = YoloLoss()

    # We use Adam optimizer insead of SGD for faster convergence, especially since YOLO can be sensitive to learning rates and benefits from adaptive learning rate adjustments.
    optimizer = Adam(model.parameters(), lr=1e-4) # 1e-4 is a good starting point for YOLO

    # Early stopping hyperparameters
    EPOCHS = 50
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0

        for images, targets in train_dataloader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()  # Clear gradients
            predictions = model(images)  # Forward pass
            loss = criterion(predictions, targets)  # Compute loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            train_loss += loss.item() # Accumulate loss for the epoch
        
        avg_train_loss = train_loss / len(train_dataloader)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():  # No need to compute gradients during validation
            for images, targets in val_dataloader:
                images, targets = images.to(device), targets.to(device)

                predictions = model(images)  # Forward pass
                loss = criterion(predictions, targets)  # Compute loss

                val_loss += loss.item() # Accumulate validation loss

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            # Save the best model weights
            torch.save(model.state_dict(), "best_yolo_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    print("Training complete. Best validation loss: {:.4f}".format(best_val_loss))
    print("Best model saved as 'best_yolo_model.pth'.")

