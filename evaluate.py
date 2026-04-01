import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from object_detector import ObjectDetector
from datahandler import test_dataloader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Intersection over Union (IoU) Helper
# In order to evaluate the performance of our object detection model, we need to calculate the Intersection over Union (IoU) metric, 
# which measures the overlap between the predicted bounding boxes and the ground truth bounding boxes. 
# The IoU is calculated as the area of overlap divided by the area of union between the predicted and ground truth boxes.    

def calculate_iou(box1, box2):
    # Calculate IoU between two bounding boxes
    # box format: [x_center, y_center, width, height] normalized to [0, 1]

    # Convert from center format to corner format
    b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
    b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2

    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection coordinates
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    # Calculate intersection area
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Calculate areas of the bounding boxes
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # Union area
    union_area = b1_area + b2_area - inter_area

    # Add small epsilon to prevent division by zero
    return inter_area / (union_area + 1e-6)


# Calculate mAP at IoU=0.5 for a given dataloader using torchmetrics
def calculate_mAP_score(model, dataloader):
    device = next(model.parameters()).device  # Get device from model parameters

    # Initialize metric. 'cxcywh' is center x, center y, width, height format
    metric = MeanAveragePrecision(box_format='cxcywh', iou_type='bbox')

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for images, targets in dataloader:
            images = images.to(device)
            predictions = model(images)  # Get raw predictions from the model

            batch_size = predictions.shape[0]
            predictions = predictions.view(batch_size, 7, 7, 7).cpu()  # Reshape to (Batch, S, S, B*5 + C)
            targets = targets.cpu()  # Move targets to CPU

            preds_list = []
            target_list = []

            for b in range(batch_size):
                # Format targets
                target_boxes = []
                target_labels = []
                for i in range(7):
                    for j in range(7):
                        if targets[b, i, j, 4] == 1:  # Objectness score in target
                            t_box = targets[b, i, j, 0:4]  # Get target box coordinates
                            t_x = (t_box[0] + j) / 7.0  # Convert back to global x_center
                            t_y = (t_box[1] + i) / 7.0  # Convert back to global y_center
                            t_cls = torch.argmax(targets[b, i, j, 5:7]).item()  # Get class label
                            target_boxes.append([t_x, t_y, t_box[2].item(), t_box[3].item()])  # Append ground truth box
                            target_labels.append(t_cls)  # Append class label
                
                if len(target_boxes) > 0:
                    target_list.append({
                        'boxes': torch.tensor(target_boxes, dtype=torch.float32),
                        'labels': torch.tensor(target_labels, dtype=torch.int64)
                    })
                else:
                    target_list.append({
                        'boxes': torch.empty((0, 4), dtype=torch.float32),
                        'labels': torch.empty((0,), dtype=torch.int64)
                    })
                
                # Format predictions
                pred_boxes = []
                pred_labels = []
                pred_scores = []

                for i in range(7):
                    for j in range(7):
                        obj_conf = predictions[b, i, j, 4].item()  # Objectness score
                        class_probs = predictions[b, i, j, 5:7]  # Get the class with the highest probability and its confidence
                        p_cls = torch.argmax(class_probs).item()  # Get predicted class label
                        final_conf = obj_conf * class_probs[p_cls].item()  # Final confidence score for the predicted class

                        # We use a low threshold to capture all predictions for the curve
                        if final_conf > 0.001:
                            p_box = predictions[b, i, j, 0:4]  # Get predicted box coordinates
                            p_x = (p_box[0] + j) / 7.0  # Convert back to global x_center
                            p_y = (p_box[1] + i) / 7.0  # Convert back to global y_center
                            pred_boxes.append([p_x, p_y, p_box[2].item(), p_box[3].item()])  # Append predicted box
                            pred_labels.append(p_cls)  # Append predicted class label
                            pred_scores.append(final_conf)  # Append confidence score

                if len(pred_boxes) > 0:
                    preds_list.append({
                        'boxes': torch.tensor(pred_boxes, dtype=torch.float32),
                        'labels': torch.tensor(pred_labels, dtype=torch.int64),
                        'scores': torch.tensor(pred_scores, dtype=torch.float32)
                    })
                else:
                    preds_list.append({
                        'boxes': torch.empty((0, 4), dtype=torch.float32),
                        'labels': torch.empty((0,), dtype=torch.int64),
                        'scores': torch.empty((0,), dtype=torch.float32)
                    })

            # Update metric with current batch predictions and targets
            metric.update(preds_list, target_list)
    
    # Compute final mAP score
    results = metric.compute()

    # Return mAP at IoU threshold of 0.5
    return results['map_50'].item()

# Extract Predictions and Ground Truths
def get_boxes(predictions, targets, conf_threshold):
    # Decodes the 7x7 grid back into list of bounding boxes.
    pred_boxes = []
    target_boxes = []

    batch_size = predictions.shape[0]
    predictions = predictions.view(batch_size, 7, 7, 7)  # Reshape to (Batch, S, S, B*5 + C)

    for b in range(batch_size):
        for i in range(7):
            for j in range(7):
                # Ground truth extraction
                if targets[b, i, j, 4] == 1:  # Objectness score in target
                    t_box = targets[b, i, j, 0:4]  # Get target box coordinates
                    t_x = (t_box[0] + j) / 7.0  # Convert back to global x_center
                    t_y = (t_box[1] + i) / 7.0  # Convert back to global y_center
                    t_cls = torch.argmax(targets[b, i, j, 5:7]).item()  # Get class label
                    target_boxes.append([b, t_cls, t_x, t_y, t_box[2].item(), t_box[3].item()])  # Append ground truth box

                # Prediction extraction
                # In YOLO, final confidence score is the product of objectness score and class probability - P(object) * P(class)
                obj_conf = predictions[b, i, j, 4]  # Objectness score
                class_probs = predictions[b, i, j, 5:7]  # Get the class with the highest probability and its confidence
                p_cls = torch.argmax(class_probs).item()  # Get predicted class label
                final_conf = obj_conf * class_probs[p_cls].item()  # Final confidence score for the predicted class

                if final_conf >= conf_threshold:
                    p_box = predictions[b, i, j, 0:4]  # Get predicted box coordinates
                    p_x = (p_box[0] + j) / 7.0  # Convert back to global x_center
                    p_y = (p_box[1] + i) / 7.0  # Convert back to global y_center
                    pred_boxes.append([b, p_cls, final_conf, p_x, p_y, p_box[2].item(), p_box[3].item()])  # Append predicted box
    
    return pred_boxes, target_boxes

# Evaluate Model Metrics
def evaluate_model(model, dataloader, iou_threshold=0.5):
    model.eval()  # Set model to evaluation mode

    # We test threshold values from 0 to 0.95 in steps of 0.05 to see how it affects precision and recall
    thresholds = np.arange(0, 1.0, 0.05)

    best_f1 = 0
    best_threshold = 0
    best_confusion_matrix = None

    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
    print("-" * 45)

    with torch.no_grad():  # Disable gradient calculation for evaluation
        all_preds_raw = []
        all_targets_raw = []

        # Collect raw grid outputs for the whole test set to save time
        device = next(model.parameters()).device  # Get device from model parameters
        
        for images, targets in dataloader:
            images = images.to(device)
            predictions = model(images)  # Get raw predictions from the model
            all_preds_raw.append(predictions.cpu())  # Move predictions to CPU and store
            all_targets_raw.append(targets.cpu())  # Move targets to CPU and store
        
        all_preds_raw = torch.cat(all_preds_raw, dim=0)  # Concatenate all predictions
        all_targets_raw = torch.cat(all_targets_raw, dim=0)  # Concatenate all targets

        # Evaluate at each confidence thresholds
        for threshold in thresholds:
            pred_boxes, target_boxes = get_boxes(all_preds_raw, all_targets_raw, threshold)

            # Sort predicted boxes by confidence score in descending order to prioritize higher confidence predictions
            pred_boxes.sort(key=lambda x: x[2], reverse=True)  # Sort by confidence score (index 2 in pred_boxes)
            
            # Create confusion matrix: 3x3 (Cat, Dog, Background)
            # Index 0: Cat, Index 1: Dog, Index 2: Background (FP/FN)
            conf_matrix = np.zeros((3, 3), dtype=int)

            # Match predictions to ground truths based on IoU and class labels
            matched_targets = set()  # Keep track of matched ground truth boxes to avoid double counting

            for p_box in pred_boxes:
                b_idx, p_cls, p_conf, p_x, p_y, p_w, p_h = p_box
                best_iou = 0
                best_target_idx = -1

                for target_idx, target_box in enumerate(target_boxes):
                    if target_idx in matched_targets or target_box[0] != b_idx:  # Only compare boxes from the same image in the batch
                        continue

                    iou = calculate_iou([p_x, p_y, p_w, p_h], target_box[2:])  # Calculate IoU between predicted box and target box
                    if iou > best_iou:
                        best_iou = iou
                        best_target_idx = target_idx
                
                if best_iou >= iou_threshold:  # If IoU is above threshold, consider it a match
                    target_cls = target_boxes[best_target_idx][1]  # Get true class label from the matched target
                    conf_matrix[target_cls, p_cls] += 1  # True Positive or Misclassification
                    matched_targets.add(best_target_idx)  # Mark this target as matched
                else:
                    conf_matrix[2, p_cls] += 1  # False Positive (Background predicted as object)

            # Any ground truth not matched is a False Negative
            for target_idx, target_box in enumerate(target_boxes):
                if target_idx not in matched_targets:
                    conf_matrix[target_box[1], 2] += 1  # False Negative (Object missed)
            
            # Calculate metrics per class (Macro-Averaging)
            class_precisions = []
            class_recalls = []
            class_f1s = []

            for c in range(2):  # For each class (Cat and Dog)
                TP = conf_matrix[c, c]  # True Positives for class c
                FP = conf_matrix[:, c].sum() - TP  # False Positives is everything in the predicted column 'c', minus the True Positives
                FN = conf_matrix[c, :].sum() - TP  # False Negatives is everything in the true row 'c', minus the True Positives

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                class_precisions.append(precision)
                class_recalls.append(recall)
                class_f1s.append(f1_score)
            
            # Calculate Macro averages (the mean of the per-class metrics)
            macro_precision = np.mean(class_precisions)
            macro_recall = np.mean(class_recalls)
            macro_f1 = np.mean(class_f1s)

            # # Calculate metrics
            # TP = np.diag(conf_matrix)[:2].sum()  # True Positives for each class
            # FP = conf_matrix[2, :2].sum() + conf_matrix[0, 1] + conf_matrix[1, 0]  # False Positives (Predicted class but wrong)
            
            # # Add missclafications to FN (e.g., cat predicted as dog or dog predicted as cat should count as FN for the true class)
            # FN = conf_matrix[:2, 2].sum() + conf_matrix[0, 1] + conf_matrix[1, 0]

            # precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            # recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            # f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"{threshold:<10.2f} | {macro_precision:<10.4f} | {macro_recall:<10.4f} | {macro_f1:<10.4f}")

            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_threshold = threshold
                best_confusion_matrix = conf_matrix
    
    print("-" * 45)
    print(f"Best Threshold: {best_threshold:.2f} with F1 Score: {best_f1:.4f}")

    return best_threshold, best_confusion_matrix

# Plot the confusion matrix using seaborn heatmap
def plot_confusion_matrix(conf_matrix, threshold):
    labels = ['Cat', 'Dog', 'Background']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix at Threshold {threshold:.2f}')
    plt.savefig(f"confusion_matrix.png")
    plt.show()
