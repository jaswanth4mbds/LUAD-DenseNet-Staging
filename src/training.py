import argparse
import os
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""
Training script for image classification using DenseNet121.
- Pathology-safe augmentations
- Tracks metrics to TensorBoard
- Saves best checkpoint and evaluation artifacts
"""

# ------------------------------------------------------------
# Data augmentation
# ------------------------------------------------------------

def get_transforms():
    # Slight augmentations for training, minimal color shift
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Validation/Test: deterministic
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return train_transform, val_test_transform


# ------------------------------------------------------------
# Model builder
# ------------------------------------------------------------

def get_model(num_classes: int):
    """Initialize DenseNet121 with a custom classifier head."""
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


# ------------------------------------------------------------
# Plot training performance
# ------------------------------------------------------------

def plot_training_curves(history, save_path):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Validation Acc")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ------------------------------------------------------------
# Confusion matrix heatmap
# ------------------------------------------------------------

def save_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ------------------------------------------------------------
# Main training + evaluation
# ------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results/{args.model}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=logs_dir)

    # Datasets
    train_transform, val_test_transform = get_transforms()
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(os.path.join(args.data_dir, "test"), transform=val_test_transform)

    # Save class mapping
    with open(os.path.join(results_dir, "class_map.json"), "w") as f:
        json.dump(train_dataset.class_to_idx, f)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = get_model(num_classes=len(train_dataset.classes)).to(device)

    # Loss with class weights
    class_weights = torch.tensor(np.load(args.class_weights), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # --------------------
    # Training loop
    # --------------------
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Mixed precision forward
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)

        train_loss /= total
        train_acc = correct / total

        # --------------------
        # Validation
        # --------------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                correct += (outputs.argmax(1) == targets).sum().item()
                total += targets.size(0)

                y_true_val.extend(targets.cpu().numpy())
                y_pred_val.extend(outputs.argmax(1).cpu().numpy())

        val_loss /= total
        val_acc = correct / total

        # Log
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("F1/val", f1_score(y_true_val, y_pred_val, average="macro"), epoch)

        print(
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} "
            f"Train Acc: {train_acc:.4f} Val Acc: {val_acc:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))

        scheduler.step(val_loss)

    writer.close()

    # Save curves and history
    plot_training_curves(history, os.path.join(results_dir, "training_curves.png"))
    with open(os.path.join(results_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

    # --------------------
    # Test evaluation (best checkpoint)
    # --------------------
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth"), map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    macro_precision = precision_score(y_true, y_pred, average="macro")
    macro_recall = recall_score(y_true, y_pred, average="macro")
    balanced_acc = recall_score(y_true, y_pred, average="macro")

    metrics_summary = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "balanced_accuracy": balanced_acc,
    }

    with open(os.path.join(results_dir, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print("Final Test Metrics:", metrics_summary)

    # Per-class metrics
    report = classification_report(y_true, y_pred, target_names=train_dataset.classes, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(results_dir, "metrics_per_class.csv"))

    # LaTeX table
    with open(os.path.join(results_dir, "metrics_per_class.tex"), "w") as f:
        f.write(df_report.to_latex(float_format="%.3f"))

    # Confusion matrix plot
    save_confusion_matrix(y_true, y_pred, train_dataset.classes, os.path.join(results_dir, "confusion_matrix.png"))

    # Save config used
    config = vars(args)
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data", help="Path to dataset root containing train/val/test")
    parser.add_argument("--class_weights", type=str, default="class_weights.npy", help="Path to class weights .npy file")
    ")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()
    main(args)
