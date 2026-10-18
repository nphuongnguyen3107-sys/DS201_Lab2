import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from data import build_test_loader
from engine import evaluate
from models import build_model


def get_args():
    p = argparse.ArgumentParser(description="Đánh giá mô hình CNN trên tập test")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Đường dẫn tới file checkpoint best.pt")
    p.add_argument("--model", type=str, default=None,
                   help="Tên mô hình (nếu checkpoint không lưu model_name)")
    p.add_argument("--data_root", type=str, default="../VinaFood21",
                   help="Đường dẫn tới thư mục chứa dữ liệu VinaFood21")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


def save_confusion_matrix(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Nhãn dự đoán (Predicted)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Nhãn thực tế (True)", fontsize=11, fontweight="bold")
    ax.set_title("Confusion Matrix - VinaFood21", fontsize=13, fontweight="bold")
    fig.colorbar(im)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 64)
    model_name = args.model or ckpt.get("model_name", "lenet")
    state_dict = ckpt.get("model_state", ckpt.get("model"))

    print("=" * 65)
    print(f"ĐÁNH GIÁ MÔ HÌNH: {model_name.upper()} | Checkpoint: {ckpt_path.name}")
    print(f"Tập dữ liệu: {args.data_root} | Kích thước ảnh: {img_size}x{img_size} | Số lớp: {len(classes)}")
    print("=" * 65)

    # Khởi tạo mô hình và tải trọng số
    model = build_model(model_name, num_classes=len(classes), img_size=img_size).to(device)
    model.load_state_dict(state_dict)

    # Nạp tập test
    test_dl, test_classes = build_test_loader(
        args.data_root, img_size=img_size, batch_size=args.batch_size, num_workers=args.num_workers)
    assert test_classes == classes, "Thứ tự nhãn lớp ở test khác với train/val!"

    # Thực hiện đánh giá
    criterion = nn.CrossEntropyLoss()
    loss, acc, f1, y_true, y_pred = evaluate(model, test_dl, criterion, device)

    # 4 độ đo yêu cầu của đề bài
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n[KẾT QUẢ TRÊN TẬP TEST]")
    print(f"  * Test Loss       : {loss:.4f}")
    print(f"  * Test Accuracy   : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  * Macro-Precision : {macro_prec:.4f}")
    print(f"  * Macro-Recall    : {macro_rec:.4f}")
    print(f"  * Macro-F1        : {f1:.4f}")

    # Báo cáo chi tiết từng lớp
    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("\n[CHI TIẾT TỪNG LỚP]")
    print(report)

    # Lưu báo cáo và Confusion Matrix
    out_dir = ckpt_path.parent
    report_file = out_dir / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Mô hình: {model_name} | Checkpoint: {ckpt_path}\n")
        f.write(f"Accuracy: {acc:.4f} | Precision: {macro_prec:.4f} | Recall: {macro_rec:.4f} | Macro-F1: {f1:.4f}\n\n")
        f.write(report)
    print(f"Đã lưu báo cáo: {report_file}")

    cm_path = out_dir / "confusion_matrix.png"
    save_confusion_matrix(confusion_matrix(y_true, y_pred), classes, cm_path)
    print(f"Đã lưu ma trận nhầm lẫn: {cm_path}")


if __name__ == "__main__":
    main()
