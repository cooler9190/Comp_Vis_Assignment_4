import torch
from training_and_validation import train_model
from evaluate import evaluate_model, plot_confusion_matrix
from object_detector import ObjectDetector
from datahandler import test_dataloader

def run_pipeline():
    # Train the model and save the 'best_yolo_model.pth' file
    print("\nPhase 1: Training the model...")
    train_model()

    print("\nPhase 2: Evaluating the model...")
    # Setup the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize a fresh model and load the best weights
    model = ObjectDetector().to(device)
    model.load_state_dict(torch.load("best_yolo_model.pth", map_location=device))

    # Run evaluation on the test set
    best_threshold, best_confusion_matrix = evaluate_model(model, test_dataloader)

    # Results
    print("RESULTS:")
    print(f"Best Threshold: {best_threshold}")
    print(f"Best Confusion Matrix:\n{best_confusion_matrix}")

    # Display the final confusion matrix
    plot_confusion_matrix(best_confusion_matrix, best_threshold)

if __name__ == "__main__":
    run_pipeline()
