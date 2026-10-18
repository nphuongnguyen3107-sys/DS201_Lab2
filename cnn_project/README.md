# CNN Unified Framework · VinaFood21

Bộ mã nguồn chuẩn hóa (Modular Deep Learning Codebase) hỗ trợ huấn luyện và đánh giá cả 4 mô hình trong **Lab 2: Mạng Neural Tích Chập (CNN)**:
1. **LeNet-5** (AvgPool vs MaxPool)
2. **GoogLeNet** (Inception Blocks + Bottleneck 1x1 + GAP)
3. **ResNet-18** (Residual Blocks, Huấn luyện từ đầu có/không có Augmentation)
4. **Pretrained ResNet-50** (Transfer Learning: Feature Extraction vs Fine-Tuning)

---

## 🗂️ Cấu Trúc Thư Mục

```text
cnn_project/
├── models/
│   ├── __init__.py        # Export build_model, get_model_defaults
│   ├── factory.py         # Model factory ánh xạ tên mô hình
│   ├── lenet.py           # LeNet-5 (339,421 tham số)
│   ├── googlenet.py       # InceptionBlock & GoogLeNetReduced
│   ├── resnet18.py        # BasicBlock & ResNet18 (11,187,285 tham số)
│   └── resnet50.py        # Pretrained ResNet-50 (HuggingFace / torchvision)
├── data.py                # Transforms (Augment/No-Augment) + DataLoaders
├── engine.py              # train_one_epoch, evaluate
├── train.py               # Huấn luyện đa mô hình qua Terminal (CLI)
├── evaluate.py            # Đánh giá trên tập test (4 độ đo + Confusion Matrix)
├── requirements.txt       # Danh sách thư viện phụ thuộc
└── README.md              # Hướng dẫn chi tiết
```

---

## 🛠️ Cài Đặt Môi Trường

```bash
cd cnn_project
pip install -r requirements.txt
```

---

## 🚀 Lệnh Huấn Luyện Từng Mô Hình (`train.py`)

Chỉ cần chọn `--model`, script sẽ tự động gán kích thước ảnh và learning rate tối ưu tương ứng:

```bash
# 1. Huấn luyện LeNet-5 (AvgPool - mặc định, ảnh 64x64, lr 1e-3)
python train.py --model lenet --data_root /path/to/VinaFood21 --epochs 20

# 1b. Huấn luyện LeNet-5 (MaxPool)
python train.py --model lenet_max --data_root /path/to/VinaFood21 --epochs 20

# 2. Huấn luyện GoogLeNet rút gọn (ảnh 64x64, 4 blocks + GAP)
python train.py --model googlenet --data_root /path/to/VinaFood21 --epochs 20

# 3a. Huấn luyện ResNet-18 KHÔNG DÙNG Data Augmentation (ảnh 96x96, khảo sát Overfitting)
python train.py --model resnet18 --no_augment --data_root /path/to/VinaFood21 --epochs 20

# 3b. Huấn luyện ResNet-18 CÓ Data Augmentation
python train.py --model resnet18 --data_root /path/to/VinaFood21 --epochs 20

# 4a. ResNet-50: Feature Extraction (đóng băng backbone, chỉ học classifier 43k tham số, lr 1e-3)
python train.py --model resnet50_fe --data_root /path/to/VinaFood21 --epochs 20

# 4b. ResNet-50: Fine-Tuning (cập nhật toàn bộ 23.5M tham số, lr 1e-4)
python train.py --model resnet50_ft --data_root /path/to/VinaFood21 --epochs 20
```

---

## 📊 Lệnh Đánh Giá Mô Hình Trên Tập Test (`evaluate.py`)

```bash
# Đánh giá checkpoint tốt nhất và tự động xuất Confusion Matrix:
python evaluate.py --ckpt outputs/lenet/best.pt --data_root /path/to/VinaFood21
python evaluate.py --ckpt outputs/googlenet/best.pt --data_root /path/to/VinaFood21
python evaluate.py --ckpt outputs/resnet18/best.pt --data_root /path/to/VinaFood21
python evaluate.py --ckpt outputs/resnet50_ft/best.pt --data_root /path/to/VinaFood21
```

### Kết quả xuất ra trong thư mục `outputs/<tên_mô_hình>/`:
* `best.pt`: Checkpoint trọng số mô hình có Validation Macro-F1 cao nhất.
* `history.csv`: Bảng lịch sử train loss, val loss, val accuracy, val F1 qua từng epoch.
* `loss_curve.png`: Biểu đồ thể hiện trực quan quá trình hội tụ của hàm mất mát.
* `confusion_matrix.png`: Ma trận nhầm lẫn $21 \times 21$ trên tập Test.
* `classification_report.txt`: Báo cáo chi tiết Precision, Recall, F1 theo từng món ăn.
