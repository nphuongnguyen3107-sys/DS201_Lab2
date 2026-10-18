# DS201 - Lab 2: Convolutional Neural Network (CNN)

Báo cáo bài thực hành số 2 môn Deep Learning (DS201) - Phân loại 21 món ăn Việt Nam trên bộ dữ liệu **VinaFood21**.

## Thông tin sinh viên
- **Họ và tên:** [Nguyễn Văn A]
- **MSSV:** [2xxxxxxx]
- **Lớp:** [DS201.xxx]

## Các mô hình thực hiện
1. **LeNet-5**: Mạng CNN cơ bản cho ảnh $64 \times 64$, khảo sát và so sánh `AvgPool2d` với `MaxPool2d`.
2. **GoogLeNet (Reduced)**: Cài đặt `InceptionBlock` với conv 1x1 giảm chiều (bottleneck) và `Global Average Pooling (GAP)` thay cho Flatten.
3. **ResNet-18 (From Scratch)**: Cài đặt khối `BasicBlock` có skip connection, phân tích hiện tượng Overfitting giữa hai chế độ có và không có Data Augmentation.
4. **ResNet-50 (Pretrained)**: Sử dụng trọng số ImageNet từ HuggingFace (`microsoft/resnet-50`), so sánh hai chiến lược Feature Extraction và Fine-Tuning.

## Cấu trúc thư mục

```text
.
├── assignments.ipynb       # Notebook bài làm chính (code, biểu đồ và nhận xét)
├── README.md               # Báo cáo tổng quan và hướng dẫn chạy
├── .gitignore              # Bỏ qua dataset và checkpoint nặng
├── outputs/                # Thư mục lưu checkpoint và đồ thị
└── cnn_project/            # Module Python
    ├── data.py             # DataLoader và Data Augmentation
    ├── model.py            # Định nghĩa các kiến trúc mạng
    ├── engine.py           # Hàm train_one_epoch và evaluate
    ├── train.py            # Script train qua dòng lệnh
    ├── evaluate.py         # Script test và tính metrics
    └── requirements.txt
```

## Cài đặt môi trường

```bash
pip install -r cnn_project/requirements.txt
```

Các thư viện chính: `torch`, `torchvision`, `transformers`, `scikit-learn`, `matplotlib`.

## Hướng dẫn chạy

### 1. Chạy trên Google Colab / Jupyter Notebook (Khuyên dùng)
- Mở file `assignments.ipynb` trên Google Colab.
- Chọn Runtime: **GPU T4** (`Runtime` -> `Change runtime type` -> `T4 GPU`).
- Chạy tuần tự các cell từ trên xuống dưới để nạp dữ liệu, huấn luyện và hiển thị biểu đồ.

### 2. Chạy qua dòng lệnh (CLI)
Có thể huấn luyện từng mô hình thông qua `train.py`:

```bash
cd cnn_project

# Huấn luyện LeNet-5
python train.py --model lenet --data_root ../VinaFood21 --epochs 20

# Huấn luyện GoogLeNet
python train.py --model googlenet --data_root ../VinaFood21 --epochs 20

# Huấn luyện ResNet-18 (có Augmentation)
python train.py --model resnet18 --data_root ../VinaFood21 --epochs 20

# Huấn luyện ResNet-50 Fine-Tuning
python train.py --model resnet50_ft --data_root ../VinaFood21 --epochs 20

# Đánh giá checkpoint
python evaluate.py --ckpt outputs/lenet/best.pt --data_root ../VinaFood21
```

## Bảng tổng hợp tham số mô hình

| Mô hình | Kích thước ảnh | Tổng tham số | Tham số cần học | Ghi chú |
| :--- | :---: | :---: | :---: | :--- |
| LeNet-5 (AvgPool) | 64x64 | 339,421 | 339,421 | F5 chiếm 95.6% tham số |
| LeNet-5 (MaxPool) | 64x64 | 339,421 | 339,421 | So sánh với AvgPool |
| GoogLeNet rút gọn | 64x64 | ~1.1M | ~1.1M | 4 Inception blocks + GAP |
| ResNet-18 (No Aug) | 96x96 | 11,187,285 | 11,187,285 | Khảo sát Overfitting |
| ResNet-18 (Aug) | 96x96 | 11,187,285 | 11,187,285 | Dùng RandomCrop, Flip, Rotation |
| ResNet-50 (Feature Extr.) | 224x224 | 23,551,061 | 43,029 | Đóng băng backbone |
| ResNet-50 (Fine-Tuning) | 224x224 | 23,551,061 | 23,551,061 | Mở toàn bộ trọng số |
