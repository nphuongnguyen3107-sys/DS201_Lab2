"""
Module engine: Vòng lặp huấn luyện, đánh giá và trực quan hóa kết quả cho mô hình CNN.
"""

from pathlib import Path
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Huấn luyện mô hình trong 1 epoch."""
    model.train()
    total, correct, running = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running += loss.item() * yb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    return running / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Đánh giá mô hình trên tập validation/test."""
    model.eval()
    total, correct, running = 0, 0, 0.0
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        running += criterion(logits, yb).item() * yb.size(0)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        y_true += yb.cpu().tolist()
        y_pred += pred.cpu().tolist()
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return running / total, correct / total, macro_f1, y_true, y_pred


def train_network(model, train_loader, val_loader, criterion, optimizer, epochs, device, model_name="model", out_dir="outputs"):
    """Vòng lặp huấn luyện đầy đủ qua các epoch, tự động lưu checkpoint có val Macro-F1 cao nhất."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path / f"{model_name}_best.pt"

    history = []
    best_f1 = -1.0
    start_time = time.time()

    print(f"Huấn luyện {model_name} ({epochs} epochs):")
    for ep in range(1, epochs + 1):
        ep_start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, criterion, device)
        ep_time = time.time() - ep_start

        history.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_f1": va_f1,
            "time": ep_time
        })

        mark = ""
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save(model.state_dict(), ckpt_path)
            mark = " *"

        print(f"Epoch {ep:2d}/{epochs:2d} | Train: Loss {tr_loss:.4f} Acc {tr_acc:.4f} | "
              f"Val: Loss {va_loss:.4f} Acc {va_acc:.4f} F1 {va_f1:.4f} ({ep_time:.1f}s){mark}")

    total_time = time.time() - start_time
    avg_ep_time = total_time / epochs
    print(f"--> Hoàn thành {model_name} trong {total_time/60:.2f} phút (~{avg_ep_time:.2f}s/epoch). Best Val F1: {best_f1:.4f}")

    # Nạp lại checkpoint tốt nhất trên tập val
    if ckpt_path.is_file():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    return history, avg_ep_time


def plot_loss_curves(history, title="Đường cong Loss", save_path=None):
    """Vẽ đường cong Loss huấn luyện và kiểm định."""
    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, tr_loss, label="Train Loss", color="royalblue", lw=1.8)
    plt.plot(epochs, va_loss, label="Val Loss", color="crimson", lw=1.8, linestyle="--")
    plt.title(title, fontsize=12)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, title="Ma trận nhầm lẫn (21 lớp)", save_path=None):
    """Vẽ ma trận nhầm lẫn 21x21 bằng Seaborn Heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=12)
    plt.xlabel("Dự đoán (Predicted)")
    plt.ylabel("Thực tế (True)")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def compute_metrics(y_true, y_pred, classes):
    """Tính toán và hiển thị 4 độ đo chính kèm classification report chi tiết."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print(f"Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f} (macro)")
    print(f"Recall    : {rec:.4f} (macro)")
    print(f"Macro-F1  : {f1:.4f}")
    print("Báo cáo chi tiết từng lớp:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0))
    return acc, prec, rec, f1


__all__ = [
    "train_one_epoch",
    "evaluate",
    "train_network",
    "plot_loss_curves",
    "plot_confusion_matrix",
    "compute_metrics"
]
