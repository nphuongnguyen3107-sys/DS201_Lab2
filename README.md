# DS201 - Lab 2: Convolutional Neural Network (CNN)

Thực nghiệm, so sánh và đánh giá các kiến trúc mạng nơ-ron tích chập (LeNet-5, GoogLeNet rút gọn, ResNet-18, ResNet-50 Pretrained) trên bộ dữ liệu **VinaFood21** (21 lớp món ăn Việt Nam).

---

## 1. Cấu trúc thư mục

```text
DS201_Lab2/
├── assignments.ipynb       # Notebook chính chứa toàn bộ mã nguồn, log và biểu đồ thực nghiệm
├── README.md               # Báo cáo tổng kết thực nghiệm
├── .gitignore              # Cấu hình bỏ qua dataset và checkpoint nặng
│
├── outputs/                # Thư mục lưu checkpoint và đồ thị khi chạy CLI
├── cnn_project.zip         # Gói mã nguồn nén hỗ trợ giải nén tự động trên Colab / Kaggle
└── cnn_project/            # Module mã nguồn Python chuẩn hóa
    ├── data.py             # Dataloader, transforms và Data Augmentation
    ├── model.py            # Cài đặt các kiến trúc (LeNet-5, GoogLeNet, ResNet-18, PretrainedResnet)
    ├── engine.py           # Hàm huấn luyện (train_one_epoch) và đánh giá (evaluate)
    ├── train.py            # Script huấn luyện dòng lệnh
    ├── evaluate.py         # Script kiểm tra checkpoint trên tập test
    └── requirements.txt    # Danh sách thư viện cần thiết
```

---

## 2. Thiết kế kiến trúc các mô hình

| STT | Mô hình | Kích thước ảnh | Tổng tham số | Tham số học | Đặc điểm thiết kế |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1a** | LeNet-5 (AvgPool) | $64 \times 64$ | 339,421 | 339,421 | 2 conv $5\times5$, AvgPool, F5 chiếm 95.6% tham số |
| **1b** | LeNet-5 (MaxPool) | $64 \times 64$ | 339,421 | 339,421 | Thay AvgPool bằng MaxPool(2, 2) |
| **2** | GoogLeNet rút gọn | $64 \times 64$ | 414,709 | 414,709 | 4 Inception blocks + bottleneck $1\times1$ + GAP |
| **3a** | ResNet-18 (No Aug) | $96 \times 96$ | 11,187,285 | 11,187,285 | 4 stage `[2, 2, 2, 2]`, huấn luyện từ đầu |
| **3b** | ResNet-18 (With Aug) | $96 \times 96$ | 11,187,285 | 11,187,285 | Thêm RandomCrop, Flip, Rotation, ColorJitter |
| **4a** | ResNet-50 (Feature Ext.) | $224 \times 224$ | 23,551,061 | 43,029 | Đóng băng backbone ImageNet, chỉ train classifier |
| **4b** | ResNet-50 (Fine-Tuning) | $224 \times 224$ | 23,551,061 | 23,551,061 | Cập nhật toàn bộ trọng số mạng với lr nhỏ ($10^{-4}$) |

---

## 3. Kết quả thực nghiệm trên tập Test (VinaFood21)

Số liệu đo lường thực tế sau khi hoàn thành huấn luyện (đánh giá trên tập test gồm 6,682 ảnh):

| STT | Mô hình | Ảnh | Tham số cần học | Thời gian / Epoch | Test Accuracy | Test Macro-F1 | Tình trạng Overfitting |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1a** | LeNet-5 (AvgPool) | $64 \times 64$ | 339,421 | ~100s | **27.10%** | **0.2581** | Overfit nặng (Train loss 0.09, Val loss 7.74) |
| **1b** | LeNet-5 (MaxPool) | $64 \times 64$ | 339,421 | ~100s | **26.28%** | **0.2376** | Overfit nặng (tương đương bản AvgPool) |
| **2** | GoogLeNet rút gọn | $64 \times 64$ | 414,709 | ~104s | **44.39%** | **0.4281** | Không overfit, loss train/val giảm đều |
| **3a** | ResNet-18 (No Aug) | $96 \times 96$ | 11,187,285 | ~106s | **39.69%** | **0.3925** | Overfit từ epoch 6 (Train acc 96%, Test 39%) |
| **3b** | ResNet-18 (With Aug)| $96 \times 96$ | 11,187,285 | ~116s | **49.49%** | **0.4758** | Giảm overfit, tăng +9.8% Acc (+8.33% F1) |
| **4a** | ResNet-50 (FE) | $224 \times 224$ | 43,029 | ~125s | **62.36%** | **0.6312** | Không overfit, học nhanh |
| **4b** | ResNet-50 (Fine-Tuning)| $224 \times 224$ | 23,551,061 | ~146s | **82.95%** | **0.8297** | Hiệu năng cao nhất, phân loại chính xác vượt trội |

---

## 4. Nhận xét & Đánh giá kỹ thuật

- **LeNet-5 (Bài 1):** Mạng CNN đời đầu quá nông (chỉ 2 lớp conv) và lớp fully-connected F5 chiếm tới 95.6% tham số mà không có Dropout/BatchNorm. Mạng nhanh chóng học thuộc lòng tập train dẫn đến hiện tượng overfitting nặng (Val loss tăng vọt lên ~7.74). Phép gộp AvgPool và MaxPool cho hiệu năng gần như nhau (~26-27%).
- **GoogLeNet rút gọn (Bài 2):** Khối Inception giúp mô hình trích xuất đặc trưng đa tỉ lệ. Việc thay thế lớp `Flatten` truyền thống bằng **Global Average Pooling (GAP)** và thêm Dropout (0.4) đã triệt tiêu hoàn toàn hiện tượng nổ loss, giúp Accuracy tăng vọt lên 44.39% (+17.29% so với LeNet).
- **Tác dụng của Data Augmentation ở ResNet-18 (Bài 3):**
  - Khi không augment, mô hình 11 triệu tham số bị overfit rõ sau vài epoch đầu (Test Acc chỉ đạt 39.69%).
  - Khi áp dụng tăng cường dữ liệu (`RandomCrop`, `Flip`, `Rotation`, `ColorJitter`), mô hình được kiểm soát overfit tốt hơn, Test Accuracy tăng thêm **+9.80%** và Macro-F1 tăng thêm **+8.33%**.
- **Hiệu quả của Transfer Learning ở ResNet-50 (Bài 4):**
  - Chế độ **Feature Extraction** chỉ huấn luyện 43,029 tham số lớp phân loại nhưng đạt 62.36% Acc, vượt trội hơn cả ResNet-18 huấn luyện từ đầu nhờ bộ lọc đặc trưng ImageNet.
  - Chế độ **Fine-Tuning** toàn bộ mạng với learning rate nhỏ ($10^{-4}$) đạt hiệu năng cao nhất toàn bài (**Accuracy 82.95%**, **Macro-F1 0.8297**).

