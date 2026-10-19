# BÀI THỰC HÀNH LAB 2: MẠNG NEURAL TÍCH CHẬP (CNN)

- **Môn học:** DS201 – Deep Learning (Học sâu trong Khoa học Dữ liệu)
- **Khoa:** Khoa Khoa học và Kỹ thuật Thông tin – Trường Đại học Công nghệ Thông tin (UIT), ĐHQG-HCM
- **Giảng viên hướng dẫn:** ThS. Nguyễn Thị Hoàng Anh
- **Bộ dữ liệu thực nghiệm:** VinaFood21 (Nhận diện 21 món ăn truyền thống Việt Nam)

---

## 1. Thông tin sinh viên
- **Họ và tên:** Nguyễn Phương Nguyên
- **MSSV:** 24521196
- **Lớp:** DS201.R11.1
- **Kho lưu trữ GitHub:** [https://github.com/nphuongnguyen3107-sys/DS201_Lab2](https://github.com/nphuongnguyen3107-sys/DS201_Lab2)

---

## 2. Tóm tắt nội dung thực hiện

Bài thực hành hoàn thành trọn vẹn cả 4 bài tập theo yêu cầu chuẩn của tài liệu hướng dẫn:

### Bài 1: LeNet-5 (Ảnh 64×64)
- Thiết kế mạng LeNet-5 điều chỉnh xử lý ảnh màu $64 \times 64 \times 3$ với 21 nhãn lớp, khớp chính xác tuyệt đối **339,421 tham số**.
- Giải thích chi tiết sự phân bố tham số: Lớp fully-connected đầu tiên (F5) chiếm tới **95.6%** tổng tham số mạng do kích thước vector duỗi phẳng lớn ($16 \times 13 \times 13 = 2,704$).
- Huấn luyện và so sánh hiệu quả giữa hai phép gộp: `AvgPool2d` và `MaxPool2d`.

### Bài 2: GoogLeNet rút gọn (Ảnh 64×64)
- Cài đặt khối `InceptionBlock` với 4 nhánh trích xuất đặc trưng đa tỉ lệ ($1\times1$, $3\times3$, $5\times5$, $3\times3 \text{ MaxPool}$), tích hợp conv $1\times1$ làm nút thắt cổ chai (*bottleneck*) để giảm chi phí tính toán:
  - Block 1 (in: 1, out: 56): Khớp chính xác **4,376 tham số**.
  - Block 2 (in: 56, out: 112): Khớp chính xác **22,104 tham số**.
- Thiết kế mạng GoogLeNet rút gọn gồm 4 Inception blocks, dùng **Global Average Pooling (GAP)** thay cho lớp `Flatten` truyền thống, giúp nén feature map về vector 352 chiều, tổng tham số **414,709**, giảm mạnh tham số lớp phân loại và hạn chế hiện tượng overfitting.

### Bài 3: ResNet-18 From Scratch (Ảnh 96×96)
- Cài đặt khối thặng dư `BasicBlock` với kết nối tắt (*skip connection / shortcut*) và chuẩn hóa theo lô (*BatchNorm2d*).
- Xây dựng ResNet-18 hoàn chỉnh gồm 4 stage `[2, 2, 2, 2]`, khớp chính xác tuyệt đối **11,187,285 tham số**.
- Phân tích và thực nghiệm hiện tượng **Overfitting** khi huấn luyện mô hình sâu trên dữ liệu nhỏ giữa hai chế độ:
  - *Không có Data Augmentation*: Khoảng cách giữa train loss và val loss lớn, mô hình bị overfitting rõ rệt sau 5–8 epochs.
  - *Có Data Augmentation* (`RandomCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`): Mở rộng không gian mẫu, thu hẹp khoảng cách train/val loss và cải thiện khả năng tổng quát hóa trên tập test.

### Bài 4: Pretrained ResNet-50 (Ảnh 224×224, ImageNet)
- Tận dụng mô hình `microsoft/resnet-50` tiền huấn luyện từ ImageNet (thông qua Hugging Face).
- Kiểm chứng và so sánh hai chiến lược học chuyển giao (Transfer Learning):
  - **Feature Extraction**: Đóng băng (*freeze*) toàn bộ backbone, chỉ huấn luyện classifier mới (**43,029 tham số học**, $\eta = 10^{-3}$).
  - **Fine-Tuning**: Cho phép cập nhật trọng số trên toàn bộ mạng (**23,551,061 tham số học**, $\eta = 10^{-4}$).

---

## 3. Bảng tổng hợp thông số kiến trúc mô hình

| STT | Mô hình | Kích thước ảnh | Tổng tham số | Tham số cần học | Điểm nổi bật trong thiết kế |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1a** | LeNet-5 (AvgPool) | $64 \times 64$ | 339,421 | 339,421 | Kiến trúc CNN cổ điển, F5 chiếm 95.6% tham số |
| **1b** | LeNet-5 (MaxPool) | $64 \times 64$ | 339,421 | 339,421 | So sánh khả năng trích xuất biên cạnh với AvgPool |
| **2** | GoogLeNet rút gọn | $64 \times 64$ | 414,709 | 414,709 | 4 Inception blocks + GAP thay cho Flatten |
| **3a** | ResNet-18 (No Aug) | $96 \times 96$ | 11,187,285 | 11,187,285 | Baseline khảo sát hiện tượng Overfitting |
| **3b** | ResNet-18 (With Aug) | $96 \times 96$ | 11,187,285 | 11,187,285 | Áp dụng 4 kỹ thuật tăng cường dữ liệu |
| **4a** | ResNet-50 (Feature Extraction) | $224 \times 224$ | 23,551,061 | 43,029 | Đóng băng backbone, huấn luyện nhanh |
| **4b** | ResNet-50 (Fine-Tuning) | $224 \times 224$ | 23,551,061 | 23,551,061 | Cập nhật toàn bộ mạng với learning rate nhỏ |

---

## 4. Bảng kết quả thực nghiệm thực tế trên tập Test (VinaFood21)

Toàn bộ kết quả thực nghiệm dưới đây được trích xuất trực tiếp từ quá trình huấn luyện thực tế trong [`assignments.ipynb`](assignments.ipynb):

| STT | Mô hình | Kích thước ảnh | Tham số cần học | Thời gian / Epoch | Test Accuracy | Test Macro-F1 | Đánh giá & Nhận xét |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1a** | LeNet-5 (AvgPool) | $64 \times 64$ | 339,421 | ~100.88s | 27.10% | 0.2581 | Overfitting nặng, Val loss tăng vọt lên ~7.74 |
| **1b** | LeNet-5 (MaxPool) | $64 \times 64$ | 339,421 | ~100.54s | 26.28% | 0.2376 | Hiệu năng tương đương AvgPool, vẫn overfitting |
| **2** | GoogLeNet rút gọn | $64 \times 64$ | 414,709 | ~104.14s | 44.39% | 0.4281 | Cải thiện vượt bậc (+17.29%), GAP giúp triệt tiêu nổ loss |
| **3a** | ResNet-18 (No Aug) | $96 \times 96$ | 11,187,285 | ~106.52s | 39.69% | 0.3925 | Mạng sâu 11M tham số bị overfitting rõ rệt sau 5–8 epoch |
| **3b** | ResNet-18 (With Aug)| $96 \times 96$ | 11,187,285 | ~115.96s | 49.49% | 0.4758 | Tăng +9.80% Acc (+8.33% F1), kiểm soát overfitting |
| **4a** | ResNet-50 (FE) | $224 \times 224$ | 43,029 | ~125.84s | 62.36% | 0.6312 | Chỉ học classifier, tận dụng trích xuất đặc trưng ImageNet |
| **4b** | ResNet-50 (Fine-Tuning)| $224 \times 224$ | 23,551,061 | ~145.99s | **82.95%** | **0.8297** | Đạt hiệu năng cao nhất, nhận diện chính xác vượt trội |

---

## 5. Phân tích hiện tượng Overfitting & Bài học rút ra

### 5.1. Hiện tượng Overfitting ở LeNet-5 (Bài 1)
- **Biểu hiện:** Train Loss giảm sâu về $0.0993$ (Train Acc đạt $96.89\%$), trong khi Val Loss bùng nổ lên tới $7.7399$ (Test Acc chỉ đạt $27.10\%$).
- **Nguyên nhân:** LeNet-5 (1998) thiết kế cho dữ liệu ảnh xám MNIST đơn giản. Khi đưa vào ảnh màu VinaFood21 phức tạp, lớp F5 chiếm tới $95.6\%$ tham số mạng nhưng không có Dropout, BatchNorm hay Data Augmentation. Mạng bị "học thuộc lòng" toàn bộ dữ liệu huấn luyện và mất hoàn toàn khả năng tổng quát hóa.

### 5.2. Giải pháp Inception & GAP ở GoogLeNet (Bài 2)
- **Biểu hiện:** Train Loss ($1.3250$) và Val Loss ($1.5369$) đồng quy đi xuống bám sát nhau, Test Accuracy tăng vọt lên **$44.39\%$** (+17.29% so với LeNet-5).
- **Nguyên nhân:** Khối Inception trích xuất đặc trưng đa tỉ lệ kết hợp nút thắt cổ chai conv $1\times1$. Đặc biệt, **Global Average Pooling (GAP)** thay thế hoàn toàn cho lớp Flatten, giúp nén feature map về vector 352 chiều mà không tạo ra thêm trọng số dư thừa, kết hợp cùng Dropout ($0.4$) triệt tiêu hoàn toàn hiện tượng nổ loss.

### 5.3. Khảo sát thực nghiệm Data Augmentation ở ResNet-18 (Bài 3)
- **Không có Data Augmentation:** Mạng sâu 11.2 triệu tham số nhanh chóng ghi nhớ vị trí điểm ảnh cố định, Train Acc lên $96.36\%$ nhưng Val Loss tăng ngược trở lại $2.8669$ và Test Acc chỉ đạt $39.69\%$.
- **Có Data Augmentation (`RandomCrop`, `Flip`, `Rotation`, `ColorJitter`):** Bằng cách mở rộng không gian mẫu, mạng bị ép phải học các đặc trưng bất biến của món ăn. Khoảng cách Train/Val loss được thu hẹp đáng kể, Test Accuracy tăng thêm **$+9.80\%$** và Macro-F1 tăng thêm **$+8.33\%$**.

### 5.4. Sức mạnh của Transfer Learning ở ResNet-50 (Bài 4)
- **Feature Extraction ($62.36\%$ Acc):** Đóng băng backbone và chỉ huấn luyện 43,029 tham số lớp phân loại, đạt hiệu năng vượt trội hơn cả ResNet-18 huấn luyện từ đầu nhờ bộ lọc đặc trưng cực mạnh được học sẵn từ hàng triệu ảnh ImageNet.
- **Fine-Tuning ($82.95\%$ Acc, $0.8297$ Macro-F1):** Cho phép toàn bộ mạng thích nghi sâu với miền dữ liệu VinaFood21 ở learning rate nhỏ ($10^{-4}$), mang lại độ chính xác cao nhất toàn bài thực hành.

---

## 6. Cấu trúc mã nguồn dự án

```text
DS201_Lab2_CNN/
│
├── assignments.ipynb       # Notebook chính: chứa toàn bộ code, output số liệu, đồ thị loss và bảng tổng kết
├── README.md               # Báo cáo tổng quan chi tiết bài thực hành
├── .gitignore              # Cấu hình bỏ qua dataset nặng và checkpoint vượt giới hạn GitHub
│
├── outputs/                # Thư mục chứa log và biểu đồ xuất ra khi huấn luyện CLI
│   └── .gitkeep
│
├── cnn_project.zip         # Gói mã nguồn nén sạch hỗ trợ giải nén tự động trên Google Colab / Kaggle
└── cnn_project/            # Module mã nguồn Python chuẩn hóa phục vụ chạy script CLI
    ├── data.py             # Pipeline tiền xử lý, Data Augmentation và nạp dữ liệu VinaFood21
    ├── model.py            # Cài đặt 4 kiến trúc mạng (LeNet5, GoogLeNet, ResNet18, PretrainedResnet)
    ├── engine.py           # Hàm huấn luyện (train_one_epoch) và đánh giá (evaluate)
    ├── train.py            # Script huấn luyện đa mô hình qua dòng lệnh
    ├── evaluate.py         # Script đánh giá checkpoint trên tập test và xuất ma trận nhầm lẫn
    └── requirements.txt    # Danh sách thư viện cần thiết
```

---

## 7. Hướng dẫn chạy và tái hiện kết quả

### Cách 1: Xem hoặc chạy trực tiếp trên Jupyter Notebook / Google Colab (Khuyên dùng)
- Mở file **`assignments.ipynb`** để xem toàn bộ kết quả thực nghiệm, đồ thị loss và ma trận nhầm lẫn đã được lưu sẵn.
- Nếu muốn chạy lại toàn bộ từ đầu trên Colab:
  1. Mở notebook qua liên kết: [Open in Colab](https://colab.research.google.com/github/nphuongnguyen3107-sys/DS201_Lab2/blob/main/assignments.ipynb).
  2. Bật GPU: `Runtime` -> `Change runtime type` -> `T4 GPU`.
  3. Chọn `Runtime` -> `Run all`. Notebook đã tích hợp cơ chế tự động clone source code và tự tải dataset `VinaFood21.zip`.

### Cách 2: Huấn luyện qua dòng lệnh (CLI)
Cài đặt thư viện:
```bash
pip install -r cnn_project/requirements.txt
```

Huấn luyện từng mô hình theo nhu cầu:
```bash
cd cnn_project

# 1. Huấn luyện LeNet-5
python train.py --model lenet --data_root ../VinaFood21 --epochs 20

# 2. Huấn luyện GoogLeNet rút gọn
python train.py --model googlenet --data_root ../VinaFood21 --epochs 20

# 3. Huấn luyện ResNet-18 (với Data Augmentation)
python train.py --model resnet18 --data_root ../VinaFood21 --epochs 20

# 4. Huấn luyện ResNet-50 Fine-Tuning
python train.py --model resnet50_ft --data_root ../VinaFood21 --epochs 20

# 5. Đánh giá checkpoint tốt nhất trên tập test
python evaluate.py --ckpt outputs/lenet/best.pt --data_root ../VinaFood21
```
