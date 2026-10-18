# BÀI THỰC HÀNH LAB 2: MẠNG NEURAL TÍCH CHẬP (CNN)

- **Môn học:** DS201 – Deep Learning (Học sâu)
- **Khoa:** Khoa Khoa học Dữ liệu – Trường ĐH Công nghệ Thông tin (UIT)
- **Bộ dữ liệu thực nghiệm:** VinaFood21 (Nhận diện 21 món ăn Việt Nam)

---

## 1. Thông tin sinh viên
- **Họ và tên:** Nguyễn Phương Nguyên
- **MSSV:** 24521196
- **Lớp:** DS201.R11.1

---

## 2. Tóm tắt nội dung thực hiện

Bài thực hành hoàn thành đầy đủ 4 bài tập theo yêu cầu của đề bài:

### Bài 1: LeNet-5 (Ảnh 64×64)
- Thiết kế mạng LeNet-5 xử lý ảnh màu $64 \times 64 \times 3$ với 21 nhãn lớp, khớp chính xác **339,421 tham số**.
- Giải thích chi tiết sự phân bố tham số: Lớp fully-connected đầu tiên (F5) chiếm tới **95.6%** tổng tham số mạng do kích thước vector duỗi phẳng lớn ($16 \times 13 \times 13 = 2,704$).
- Huấn luyện và so sánh hiệu quả giữa hai phép gộp: `AvgPool2d` và `MaxPool2d`.

### Bài 2: GoogLeNet rút gọn (Ảnh 64×64)
- Cài đặt khối `InceptionBlock` với 4 nhánh trích xuất đặc trưng đa tỉ lệ ($1\times1$, $3\times3$, $5\times5$, $3\times3 \text{ MaxPool}$), tích hợp conv $1\times1$ làm nút thắt cổ chai (*bottleneck*) để giảm chi phí tính toán.
  - Block 1 (in: 1, out: 56): 4,376 tham số.
  - Block 2 (in: 56, out: 112): 22,104 tham số.
- Thiết kế mạng GoogLeNet rút gọn gồm 4 Inception blocks, dùng **Global Average Pooling (GAP)** thay cho lớp `Flatten` truyền thống, giúp nén feature map về vector 112 chiều, giảm mạnh tham số lớp phân loại và hạn chế hiện tượng overfitting.

### Bài 3: ResNet-18 From Scratch (Ảnh 96×96)
- Cài đặt khối thặng dư `BasicBlock` với kết nối tắt (*skip connection / shortcut*) và chuẩn hóa theo lô (*BatchNorm2d*).
- Xây dựng ResNet-18 hoàn chỉnh gồm 4 stage `[2, 2, 2, 2]`, khớp chính xác **11,187,285 tham số**.
- Phân tích và thực nghiệm hiện tượng **Overfitting** khi huấn luyện mô hình sâu trên dữ liệu nhỏ giữa hai chế độ:
  - *Không có Data Augmentation*: Khoảng cách giữa train loss và val loss lớn, mô hình bị overfitting rõ rệt sau 5–8 epochs.
  - *Có Data Augmentation* (`RandomCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`): Mở rộng không gian mẫu, thu hẹp khoảng cách train/val loss và cải thiện khả năng tổng quát hóa trên tập test.

### Bài 4: Pretrained ResNet-50 (Ảnh 224×224)
- Tận dụng mô hình `microsoft/resnet-50` tiền huấn luyện từ ImageNet (thông qua Hugging Face).
- Kiểm chứng và so sánh hai chiến lược học chuyển giao:
  - **Feature Extraction**: Đóng băng (*freeze*) toàn bộ backbone, chỉ huấn luyện classifier mới (**43,029 tham số học**, $\eta = 10^{-3}$).
  - **Fine-Tuning**: Cho phép cập nhật trọng số trên toàn bộ mạng (**23,551,061 tham số học**, $\eta = 10^{-4}$).

---

## 3. Bảng tổng hợp thông số và kiến trúc mô hình

| STT | Mô hình | Kích thước ảnh | Tổng tham số | Tham số cần học | Điểm nổi bật trong thiết kế |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1a** | LeNet-5 (AvgPool) | $64 \times 64$ | 339,421 | 339,421 | Kiến trúc CNN cổ điển, F5 chiếm 95.6% tham số |
| **1b** | LeNet-5 (MaxPool) | $64 \times 64$ | 339,421 | 339,421 | So sánh khả năng trích xuất biên cạnh với AvgPool |
| **2** | GoogLeNet rút gọn | $64 \times 64$ | ~1,100,000 | ~1,100,000 | 4 Inception blocks + GAP thay cho Flatten |
| **3a** | ResNet-18 (No Aug) | $96 \times 96$ | 11,187,285 | 11,187,285 | Baseline khảo sát hiện tượng Overfitting |
| **3b** | ResNet-18 (With Aug) | $96 \times 96$ | 11,187,285 | 11,187,285 | Áp dụng 4 kỹ thuật tăng cường dữ liệu |
| **4a** | ResNet-50 (Feature Extraction) | $224 \times 224$ | 23,551,061 | 43,029 | Đóng băng backbone, huấn luyện nhanh |
| **4b** | ResNet-50 (Fine-Tuning) | $224 \times 224$ | 23,551,061 | 23,551,061 | Cập nhật toàn bộ mạng với learning rate nhỏ |

---

## 4. Cấu trúc mã nguồn

Toàn bộ bài làm được tổ chức module hóa, phục vụ cả việc chạy thực nghiệm trực quan trên Notebook lẫn chạy tự động qua dòng lệnh (CLI):

```text
DS201_Lab2_CNN/
│
├── assignments.ipynb       # Notebook chính nộp bài: chứa code, output chạy, biểu đồ và nhận xét chi tiết
├── README.md               # Báo cáo tổng quan tóm tắt nội dung bài thực hành
├── .gitignore              # Cấu hình bỏ qua dataset và checkpoint nặng
│
├── outputs/                # Thư mục lưu checkpoint (best.pt), log (history.csv) và biểu đồ (loss_curve.png)
│
└── cnn_project/            # Module mã nguồn Python chuẩn hóa
    ├── data.py             # Pipeline tiền xử lý, Data Augmentation và nạp dữ liệu VinaFood21
    ├── model.py            # Cài đặt 4 kiến trúc mạng (LeNet5, GoogLeNet, ResNet18, PretrainedResnet)
    ├── engine.py           # Hàm huấn luyện (train_one_epoch) và đánh giá (evaluate)
    ├── train.py            # Script huấn luyện đa mô hình qua dòng lệnh
    ├── evaluate.py         # Script đánh giá checkpoint trên tập test và xuất ma trận nhầm lẫn
    └── requirements.txt    # Danh sách thư viện cần thiết
```

---

## 5. Hướng dẫn chạy và kiểm tra kết quả

### Cách 1: Xem hoặc chạy trực tiếp trên Jupyter Notebook / Google Colab (Khuyên dùng)
- Mở file **`assignments.ipynb`** để xem toàn bộ kết quả thực nghiệm, đồ thị loss và bảng chỉ số đã được lưu sẵn.
- Nếu muốn chạy lại toàn bộ từ đầu:
  1. Tải notebook lên **Google Colab**.
  2. Bật GPU: `Runtime` -> `Change runtime type` -> `T4 GPU`.
  3. Chọn `Runtime` -> `Run all` để chạy tuần tự tất cả các cell.

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
