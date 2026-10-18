import argparse
import csv
from pathlib import Path
import time

import torch
import torch.nn as nn

from data import build_train_val_loaders
from engine import evaluate, train_one_epoch
from models import build_model, get_model_defaults, n_params


def get_args():
    p = argparse.ArgumentParser(description="Huấn luyện mô hình CNN trên bộ dữ liệu VinaFood21")
    p.add_argument("--model", type=str, default="lenet",
                   choices=["lenet", "lenet_max", "googlenet", "resnet18", "resnet50_fe", "resnet50_ft"],
                   help="Tên mô hình cần huấn luyện: lenet, lenet_max, googlenet, resnet18, resnet50_fe, resnet50_ft")
    p.add_argument("--data_root", type=str, default="../VinaFood21",
                   help="Đường dẫn tới thư mục chứa dữ liệu VinaFood21 (chứa train/ và test/)")
    p.add_argument("--img_size", type=int, default=None,
                   help="Kích thước ảnh vuông (mặc định: auto theo mô hình - 64/96/224)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=None,
                   help="Tốc độ học (mặc định: auto - 1e-3 cho lenet/googlenet/resnet18/fe, 1e-4 cho ft)")
    p.add_argument("--no_augment", action="store_true",
                   help="Tắt Data Augmentation ở tập train (dùng cho thí nghiệm so sánh của Bài 3 ResNet-18)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="outputs",
                   help="Thư mục gốc lưu trữ checkpoints và đồ thị")
    return p.parse_args()


def save_loss_curve(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    plt.figure(figsize=(8, 5))
    plt.plot(ep, [h["train_loss"] for h in history], "b-o", label="Train Loss")
    plt.plot(ep, [h["val_loss"] for h in history], "r-s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Xác định mặc định kích thước ảnh và learning rate nếu chưa truyền
    def_size, def_lr, def_epochs = get_model_defaults(args.model)
    img_size = args.img_size if args.img_size is not None else def_size
    lr = args.lr if args.lr is not None else def_lr
    augment = not args.no_augment

    # Thiết lập thư mục lưu checkpoint
    tag = f"{args.model}{'_noaug' if args.no_augment else ''}"
    model_out_dir = Path(args.out_dir) / tag
    model_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"MÔ HÌNH: {args.model.upper()} | Tag: {tag}")
    print(f"Device : {device} | Ảnh: {img_size}x{img_size} | Batch: {args.batch_size} | LR: {lr}")
    print(f"Data Augmentation: {augment} | Epochs: {args.epochs}")
    print("=" * 65)

    # Nạp dữ liệu
    train_dl, val_dl, classes = build_train_val_loaders(
        args.data_root, img_size=img_size, batch_size=args.batch_size,
        augment=augment, num_workers=args.num_workers, seed=args.seed)
    num_classes = len(classes)
    print(f"Số lớp: {num_classes} | Train samples: {len(train_dl.dataset)} | Val samples: {len(val_dl.dataset)}")

    # Khởi tạo mô hình
    model = build_model(args.model, num_classes=num_classes, img_size=img_size).to(device)
    total_params = n_params(model)
    trainable_params = n_params(model, trainable_only=True)
    print(f"Tổng tham số: {total_params:,} | Tham số cần học: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()

    # Chỉ đưa tham số cần học vào Optimizer (rất quan trọng cho Feature Extraction)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    history = []
    best_f1 = -1.0
    start_time = time.time()

    print("\nBắt đầu huấn luyện...")
    for ep in range(1, args.epochs + 1):
        ep_start = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_dl, criterion, optimizer, device)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_dl, criterion, device)
        ep_duration = time.time() - ep_start

        history.append(dict(epoch=ep, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, val_f1=va_f1,
                            duration=ep_duration))

        mark = ""
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                "model_name": args.model,
                "model_state": model.state_dict(),
                "classes": classes,
                "img_size": img_size,
                "epoch": ep,
                "best_val_f1": best_f1
            }, model_out_dir / "best.pt")
            mark = "  * (best)"

        print(f"Epoch {ep:>2}/{args.epochs} [{ep_duration:4.1f}s] "
              f"Train loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
              f"Val loss: {va_loss:.4f} acc: {va_acc:.4f} F1: {va_f1:.4f}{mark}")

    total_time = time.time() - start_time
    print(f"\nHuấn luyện hoàn tất trong {total_time/60:.2f} phút.")
    print(f"Best Val Macro-F1: {best_f1:.4f} -> Checkpoint: {model_out_dir / 'best.pt'}")

    # Lưu lịch sử ra file CSV
    with open(model_out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)

    # Lưu biểu đồ loss
    save_loss_curve(history, model_out_dir / "loss_curve.png")
    print(f"Đã lưu biểu đồ: {model_out_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
