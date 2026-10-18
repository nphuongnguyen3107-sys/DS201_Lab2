import argparse
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from data import build_test_loader
from engine import evaluate
from model import build_model


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--data_root", type=str, default="../VinaFood21")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def save_confusion_matrix(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 64)
    model_name = args.model or ckpt.get("model_name", "lenet")
    state_dict = ckpt.get("model_state", ckpt.get("model"))

    model = build_model(model_name, num_classes=len(classes), img_size=img_size).to(device)
    model.load_state_dict(state_dict)

    test_dl, test_classes = build_test_loader(
        args.data_root, img_size=img_size, batch_size=args.batch_size, num_workers=args.num_workers)
    assert test_classes == classes, "Class mismatch between checkpoint and test folder."

    print(f"Evaluating {model_name} | Checkpoint: {ckpt_path.name} | Test set: {len(test_dl.dataset)} samples")

    criterion = nn.CrossEntropyLoss()
    loss, acc, f1, y_true, y_pred = evaluate(model, test_dl, criterion, device)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Loss: {loss:.4f} | Acc: {acc:.4f} ({acc*100:.2f}%) | Prec: {macro_prec:.4f} | Rec: {macro_rec:.4f} | F1: {f1:.4f}\n")

    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("Classification Report:")
    print(report)

    out_dir = ckpt_path.parent
    report_file = out_dir / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name} | Checkpoint: {ckpt_path}\n")
        f.write(f"Acc: {acc:.4f} | Prec: {macro_prec:.4f} | Rec: {macro_rec:.4f} | F1: {f1:.4f}\n\n")
        f.write(report)

    cm_path = out_dir / "confusion_matrix.png"
    save_confusion_matrix(confusion_matrix(y_true, y_pred), classes, cm_path)


if __name__ == "__main__":
    main()
