import os
import torch
import pickle
import matplotlib.pyplot as plt
from torch import optim, nn
from tqdm import tqdm
from dataloaders import create_dataloaders
import timm
import pandas as pd

class EfficientNetRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        # EfficientNet-Lite0 as feature extractor
        self.feature_extractor = timm.create_model('efficientnet_lite0', pretrained=True, num_classes=0)

        # Output is already [B, 1280]
        self.layer_norm = nn.LayerNorm(1280)

        self.fnn = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU()
        )

        self.head_ss_var  = nn.Linear(128, 1)
        self.head_iso_var = nn.Linear(128, 1)

    def forward(self, x):
        x = self.feature_extractor(x)  # [B, 1280]
        x = self.layer_norm(x)
        x = self.fnn(x)
        return self.head_ss_var(x), self.head_iso_var(x)


def print_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())  # Count total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Count only trainable parameters
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


def plot_loss_history(epochs, train_ss, train_iso, val_ss, val_iso):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_ss, label="Train SS_var")
    plt.plot(range(1, epochs + 1), val_ss, label="Validation SS_var")
    plt.plot(range(1, epochs + 1), train_iso, label="Train ISO_var")
    plt.plot(range(1, epochs + 1), val_iso, label="Validation ISO_var")
    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def train():
    # Paths
    IMG_DIR    = "videos/frames/"
    LABEL_FILE = os.path.join(IMG_DIR, "frame_labels.csv")
    TEST_DATA_PATH = "test_dataset.pkl"
    BEST_MODEL_PATH = "best_model_ss.pth"

    # Hyperparameters
    BS     = 32
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        images_dir=IMG_DIR, labels_csv=LABEL_FILE, batch_size=BS
    )

    # Save the test dataset
    print("Saving test dataset...")
    with open(TEST_DATA_PATH, "wb") as f:
        pickle.dump(test_loader.dataset, f)

    # Model, loss, optimizer, scheduler
    model     = EfficientNetRegressionModel().to(DEVICE)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print_parameters(model)

    # Histories
    train_ss_hist, train_iso_hist = [], []
    val_ss_hist, val_iso_hist = [], []

    best_val_ss_loss = float("inf")

    for epoch in tqdm(range(1, EPOCHS + 1)):
        model.train()
        ss_loss_train, iso_loss_train = 0.0, 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            ss_gt  = labels['SS_var'].float().view(-1, 1).to(DEVICE)
            iso_gt = labels['ISO_var'].float().view(-1, 1).to(DEVICE)

            optimizer.zero_grad()
            ss_pred, iso_pred = model(images)
            loss = criterion(ss_pred, ss_gt) + criterion(iso_pred, iso_gt)
            loss.backward()
            optimizer.step()

            ss_loss_train  += criterion(ss_pred, ss_gt).item()
            iso_loss_train += criterion(iso_pred, iso_gt).item()

        train_ss_hist.append(ss_loss_train / len(train_loader))
        train_iso_hist.append(iso_loss_train / len(train_loader))

        model.eval()
        ss_loss_val, iso_loss_val = 0.0, 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                ss_gt  = labels['SS_var'].float().view(-1, 1).to(DEVICE)
                iso_gt = labels['ISO_var'].float().view(-1, 1).to(DEVICE)
                ss_pred, iso_pred = model(images)
                ss_loss_val  += criterion(ss_pred, ss_gt).item()
                iso_loss_val += criterion(iso_pred, iso_gt).item()

        avg_ss_val_loss = ss_loss_val / len(val_loader)
        val_ss_hist.append(avg_ss_val_loss)
        val_iso_hist.append(iso_loss_val / len(val_loader))
        scheduler.step()

        print(f"Epoch {epoch}/{EPOCHS}  "
              f"Train [SS: {train_ss_hist[-1]:.4f}, ISO: {train_iso_hist[-1]:.4f}]  "
              f"Validation [SS: {val_ss_hist[-1]:.4f}, ISO: {val_iso_hist[-1]:.4f}]")

        if avg_ss_val_loss < best_val_ss_loss:
            best_val_ss_loss = avg_ss_val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    plot_loss_history(epochs=EPOCHS, train_ss=train_ss_hist, train_iso=train_iso_hist, val_ss=val_ss_hist, val_iso=val_iso_hist)

    # Save the model
    print(f"Saving best model to {BEST_MODEL_PATH}")
    torch.save(model.state_dict(), BEST_MODEL_PATH)


if __name__ == "__main__":
    train()

