import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_training_history(history, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)
    ts = get_timestamp()

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    lr = history.history.get("lr", None)

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train Acc")
    plt.plot(epochs, val_acc, label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    file_path = os.path.join(
        save_dir,
        f"training_curves_{ts}.png")

    plt.savefig(file_path, dpi=300)
    plt.close()

    print(f"\nSaved training curves: {file_path}")

    # LR plot
    if lr is not None:
        try:
            plt.figure()
            plt.plot(epochs, lr, label="Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.xlabel("Epoch")
            plt.ylabel("LR")
            plt.legend()
            plt.tight_layout()

            lr_path = os.path.join(
                save_dir,
                f"lr_{ts}.png")

            plt.savefig(lr_path, dpi=300)
            plt.close()

            print(f"Saved LR curve: {lr_path}")

        except Exception:
            pass


def save_confusion_matrix(y_true, y_pred, class_names, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)
    ts = get_timestamp()

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    file_path = os.path.join(
        save_dir,
        f"confusion_matrix_{ts}.png")

    plt.savefig(file_path, dpi=300)
    plt.close()

    print(f"\nSaved confusion matrix: {file_path}")


def save_classification_report(report_text, save_dir="results"):

    os.makedirs(save_dir, exist_ok=True)
    ts = get_timestamp()

    file_path = os.path.join(
        save_dir,
        f"classification_report_{ts}.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nSaved classification report: {file_path}")
