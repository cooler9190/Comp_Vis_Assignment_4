import torch
from training_and_validation import train_model
from evaluate import calculate_mAP_score, evaluate_model, plot_confusion_matrix
from object_detector import ObjectDetector
from datahandler import train_dataloader, val_dataloader, test_dataloader
from torchsummary import summary
import winsound

def run_pipeline():
    # Setup the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "N/A")
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

    # Print summary, using image shape from dataloader as input size.
    model = ObjectDetector().to(device)
    images, _ = next(iter(test_dataloader))
    summary(model, input_size=images.shape[1:])

    # Train the model and save the 'best_yolo_model.pth' file
    print("\nPhase 1: Training the model...")
    train_model(device)

    print("\nPhase 2: Evaluating the model...")
    # Initialize a fresh model load the best weights
    model = ObjectDetector().to(device)
    model.load_state_dict(torch.load("best_yolo_model.pth", map_location=device))

    # Calculate mAP score for all sets
    print("\nCalculating mAP score on all sets...")
    train_map = calculate_mAP_score(model, train_dataloader)
    val_map = calculate_mAP_score(model, val_dataloader)
    test_map = calculate_mAP_score(model, test_dataloader)

    # Run evaluation on the test set
    best_threshold, best_confusion_matrix = evaluate_model(model, test_dataloader)

    # Results
    print("RESULTS:")
    print(f"mAP Score - Train Set: {train_map:.4f}")
    print(f"mAP Score - Validation Set: {val_map:.4f}")
    print(f"mAP Score - Test Set: {test_map:.4f}")
    print("-" * 40)
    print(f"Best Threshold: {best_threshold}")
    print(f"Best Confusion Matrix:\n{best_confusion_matrix}")

    # Display the final confusion matrix
    plot_confusion_matrix(best_confusion_matrix, best_threshold)

if __name__ == "__main__":
    run_pipeline()
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    winsound.Beep(frequency, duration)
