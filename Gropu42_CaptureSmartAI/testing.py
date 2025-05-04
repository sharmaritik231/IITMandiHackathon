import os
import pickle
import torch
import matplotlib.pyplot as plt
import pandas as pd
from train import EfficientNetRegressionModel
from torch.utils.data import DataLoader


def visualize_predictions(model_path, test_data_path, device, output_dir="output"):
    # Load the test dataset
    print("Loading test dataset...")
    with open(test_data_path, "rb") as f:
        test_dataset = pickle.load(f)

    # Recreate the DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load the model
    print("Loading model...")
    model = EfficientNetRegressionModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    predictions, ground_truths = [], []

    # Run inference
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            ss_gt = labels["SS_var"].float().view(-1, 1).to(device)
            iso_gt = labels["ISO_var"].float().view(-1, 1).to(device)

            ss_pred, iso_pred = model(images)

            predictions.append((ss_pred.cpu().numpy(), iso_pred.cpu().numpy()))
            ground_truths.append((ss_gt.cpu().numpy(), iso_gt.cpu().numpy()))

    # Flatten predictions and ground truths
    ss_preds, iso_preds = zip(*predictions)
    ss_truths, iso_truths = zip(*ground_truths)

    ss_preds = [item for sublist in ss_preds for item in sublist]
    iso_preds = [item for sublist in iso_preds for item in sublist]
    ss_truths = [item for sublist in ss_truths for item in sublist]
    iso_truths = [item for sublist in iso_truths for item in sublist]

    # Save predictions and ground truths as CSV files
    os.makedirs(output_dir, exist_ok=True)
    ss_csv_path = os.path.join(output_dir, "ss_predictions.csv")
    iso_csv_path = os.path.join(output_dir, "iso_predictions.csv")

    pd.DataFrame({
        "Ground Truth SS_var": ss_truths,
        "Predicted SS_var": ss_preds
    }).to_csv(ss_csv_path, index=False)

    pd.DataFrame({
        "Ground Truth ISO_var": iso_truths,
        "Predicted ISO_var": iso_preds
    }).to_csv(iso_csv_path, index=False)

    print(f"Saved SS_var predictions to {ss_csv_path}")
    print(f"Saved ISO_var predictions to {iso_csv_path}")

    # Visualize SS_var predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(ss_truths)), ss_truths, label="True SS_var", alpha=0.7)
    plt.scatter(range(len(ss_preds)), ss_preds, label="Predicted SS_var", alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("SS_var")
    plt.legend()
    plt.grid('off')
    plt.title("SS_var Predictions vs Ground Truth")
    plt.show()

    # Visualize ISO_var predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(iso_truths)), iso_truths, label="Ground Truth ISO_var", alpha=0.7)
    plt.scatter(range(len(iso_preds)), iso_preds, label="Predicted ISO_var", alpha=0.7)
    plt.xlabel("Sample Index")
    plt.ylabel("ISO_var")
    plt.legend()
    plt.title("ISO_var Predictions vs Ground Truth")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Define paths
    MODEL_PATH = "best_model_ss.pth"  # Path to the saved model
    TEST_DATA_PATH = "test_dataset.pkl"  # Path to the saved test dataset
    OUTPUT_DIR = "output"  # Directory to save the prediction CSVs and plots

    # Define device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Call the visualize_predictions function
    visualize_predictions(MODEL_PATH, TEST_DATA_PATH, DEVICE, OUTPUT_DIR)