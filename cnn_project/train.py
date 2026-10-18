import argparse
import csv
from pathlib import Path
import time

import torch
import torch.nn as nn

from data import build_train_val_loaders
from engine import evaluate, train_one_epoch
from model import build_model, get_model_defaults, n_params


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="lenet",
                   choices=["lenet", "lenet_max", "googlenet", "resnet18", "resnet50_fe", "resnet50_ft"])
    p.add_argument("--data_root", type=str, default="../VinaFood21")
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="outputs")
    return p.parse_args()


def save_loss_curve(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    plt.figure(figsize=(7, 4))
    plt.plot(ep, [h["train_loss"] for h in history], label="Train Loss")
    plt.plot(ep, [h["val_loss"] for h in history], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def_size, def_lr, _ = get_model_defaults(args.model)
    img_size = args.img_size if args.img_size is not None else def_size
    lr = args.lr if args.lr is not None else def_lr
    augment = not args.no_augment

    tag = f"{args.model}{'_noaug' if args.no_augment else ''}"
    model_out_dir = Path(args.out_dir) / tag
    model_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model} | Device: {device} | Size: {img_size}x{img_size} | Batch: {args.batch_size} | LR: {lr}")

    train_dl, val_dl, classes = build_train_val_loaders(
        args.data_root, img_size=img_size, batch_size=args.batch_size,
        augment=augment, num_workers=args.num_workers, seed=args.seed)
    num_classes = len(classes)

    model = build_model(args.model, num_classes=num_classes, img_size=img_size).to(device)
    print(f"Classes: {num_classes} | Train: {len(train_dl.dataset)} | Val: {len(val_dl.dataset)}")
    print(f"Params: total {n_params(model):,} | trainable {n_params(model, trainable_only=True):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    history = []
    best_f1 = -1.0
    start_time = time.time()

    print("\nTraining...")
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
            mark = " *"

        print(f"Epoch {ep:2d}/{args.epochs} [{ep_duration:4.1f}s] "
              f"loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
              f"val_loss: {va_loss:.4f} val_acc: {va_acc:.4f} F1: {va_f1:.4f}{mark}")

    total_time = time.time() - start_time
    print(f"\nDone in {total_time/60:.1f}m. Best Val F1: {best_f1:.4f} -> {model_out_dir / 'best.pt'}")

    with open(model_out_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)

    save_loss_curve(history, model_out_dir / "loss_curve.png")


if __name__ == "__main__":
    main()
