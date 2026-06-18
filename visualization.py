import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_training_history(history, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)
    timestamp = get_timestamp()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()

    filename = f"training_history_{timestamp}.png"

    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

    print(f"\nSaved: {filename}")


def save_confusion_matrix(y_true, y_pred, class_names, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)
    timestamp = get_timestamp()

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, ax=ax, cbar=False)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    filename = f"confusion_matrix_{timestamp}.png"

    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()

    print(f"\nSaved: {filename}")


def save_classification_report(report_text, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)
    timestamp = get_timestamp()

    filename = f"classification_report_{timestamp}.txt"

    with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nSaved: {filename}")
