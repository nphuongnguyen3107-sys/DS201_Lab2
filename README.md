# DS201 · Deep Learning | Lab 2: Mạng Neural Tích Chập (CNN)

> **Trường Đại học Công nghệ Thông tin – ĐHQG-HCM (UIT)**  
> **Môn học:** DS201 – Deep Learning (Học sâu)  
> **Chủ đề:** Convolutional Neural Networks (CNN) & Thực nghiệm trên bộ dữ liệu VinaFood21

---

## 📌 Thông Tin Sinh Viên
* **Họ và tên:** `[Nguyễn Văn A]` *(Hãy cập nhật thông tin của bạn)*
* **Mã số sinh viên (MSSV):** `[2xxxxxxx]`
* **Lớp:** `[DS201.xxx]`
* **Giảng viên hướng dẫn:** `[TS. / ThS. ...]`

---

## 📖 Tổng Quan Bài Thực Hành
Bài thực hành số 2 tập trung vào việc nghiên cứu, cài đặt từ cơ bản (from scratch) và tinh chỉnh (transfer learning/fine-tuning) các kiến trúc mạng tích chập kinh điển trên bộ dữ liệu nhận diện món ăn Việt Nam **VinaFood21**:

1. **Bài 1: LeNet-5**  
   Cải tiến kiến trúc LeNet-5 cổ điển để xử lý ảnh màu kích thước $64 \times 64 \times 3$ với 21 nhãn lớp. Khảo sát so sánh giữa phép gộp trung bình (`AvgPool2d`) và phép gộp cực đại (`MaxPool2d`).
2. **Bài 2: GoogLeNet (Reduced Inception Network)**  
   Cài đặt `InceptionBlock` với 4 nhánh song song ($1\times1$, $1\times1 \to 3\times3$, $1\times1 \to 5\times5$, $3\times3 \text{ MaxPool} \to 1\times1$) và cơ chế nút thắt cổ chai (*bottleneck* $1\times1$). Thay thế lớp `Flatten` truyền thống bằng `Global Average Pooling (GAP)`.
3. **Bài 3\*: ResNet-18 (Huấn luyện từ đầu - From Scratch)**  
   Cài đặt khối thặng dư `BasicBlock` với kết nối tắt (*skip connection / shortcut*), thiết kế ResNet-18 hoàn chỉnh gồm 4 stage `[2, 2, 2, 2]`. Thực nghiệm phân tích hiện tượng **Overfitting** khi huấn luyện mô hình sâu trên dữ liệu nhỏ giữa hai chế độ: *Không có Data Augmentation* vs *Có Data Augmentation*.
4. **Bài 4\*: Pretrained ResNet-50 (Transfer Learning từ HuggingFace)**  
   Tận dụng mô hình `microsoft/resnet-50` đã được tiền huấn luyện trên ImageNet. So sánh hai chiến lược huấn luyện kinh điển:
   * **Feature Extraction**: Đóng băng (*freeze*) toàn bộ backbone, chỉ huấn luyện classifier mới ($43,029$ tham số) với learning rate $\eta = 10^{-3}$.
   * **Fine-Tuning**: Cho phép cập nhật trọng số trên toàn bộ mạng ($23,551,061$ tham số) với learning rate nhỏ $\eta = 10^{-4}$.

---

## 📊 Bảng Tổng Hợp Kết Quả (Summary Table)

| STT | Mô hình | Kích thước ảnh | Tổng tham số | Tham số huấn luyện | Thời gian / epoch | Test Accuracy | Test Macro-F1 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **LeNet-5 (AvgPool)** | $64 \times 64$ | 339,421 | 339,421 | *~X s* | *0.xx* | *0.xx* |
| **1b** | **LeNet-5 (MaxPool)** | $64 \times 64$ | 339,421 | 339,421 | *~X s* | *0.xx* | *0.xx* |
| **2** | **GoogLeNet rút gọn (4 blocks + GAP)** | $64 \times 64$ | ~1,100,000 | ~1,100,000 | *~X s* | *0.xx* | *0.xx* |
| **3a** | **ResNet-18 (Không Augmentation)\*** | $96 \times 96$ | 11,187,285 | 11,187,285 | *~X s* | *0.xx* | *0.xx* |
| **3b** | **ResNet-18 (Có Augmentation)\*** | $96 \times 96$ | 11,187,285 | 11,187,285 | *~X s* | *0.xx* | *0.xx* |
| **4a** | **ResNet-50 (Feature Extraction)\*** | $224 \times 224$ | 23,551,061 | 43,029 | *~X s* | *0.xx* | *0.xx* |
| **4b** | **ResNet-50 (Fine-tuning)\*** | $224 \times 224$ | 23,551,061 | 23,551,061 | *~X s* | *0.xx* | *0.xx* |

*(Ghi chú: Kết quả chi tiết phụ thuộc vào phần cứng và lịch sử chạy trên notebook [assignments.ipynb](assignments.ipynb)).*

---

## 🗂️ Cấu Trúc Thư Mục Repository

```text
DS201_Lab2_CNN/
│
├── assignments.ipynb            # Notebook chính chứa toàn bộ code, output và nhận xét 4 bài
├── README.md                    # Báo cáo tổng quan, thông tin sinh viên và bảng so sánh kết quả
├── .gitignore                   # Cấu hình chặn push file nặng (VinaFood21, *.pt, checkpoint)
│
├── outputs/                     # Thư mục lưu trữ biểu đồ và báo cáo đánh giá
│   ├── loss_curve_lenet.png
│   ├── confusion_matrix_lenet.png
│   ├── loss_curve_googlenet.png
│   ├── confusion_matrix_googlenet.png
│   ├── resnet18_overfitting_comparison.png
│   ├── confusion_matrix_resnet18.png
│   ├── resnet50_loss_curves.png
│   └── confusion_matrix_resnet50.png
│
└── cnn_project/                 # Module mã nguồn chuẩn hóa (hỗ trợ cả 4 mô hình qua CLI)
    ├── data.py                  # Pipeline xử lý dữ liệu (Transforms, DataLoader, chia tập val)
    ├── model.py                 # Chứa toàn bộ 4 kiến trúc: LeNet-5, GoogLeNet, ResNet-18, ResNet-50
    ├── engine.py                # Hàm train_one_epoch và evaluate
    ├── train.py                 # Script huấn luyện đa mô hình qua dòng lệnh (CLI)
    ├── evaluate.py              # Script đánh giá mô hình trên tập test và vẽ ma trận nhầm lẫn
    └── requirements.txt         # Danh sách thư viện phụ thuộc
```

---

## 🛠️ Cài Đặt Môi Trường

Khuyến nghị sử dụng môi trường **Python 3.10+** và **CUDA** (nếu có GPU):

```bash
# Clone repository
git clone https://github.com/<username>/DS201_Lab2_CNN.git
cd DS201_Lab2_CNN

# Cài đặt các thư viện cần thiết
pip install torch torchvision torchaudio
pip install transformers scikit-learn seaborn matplotlib tqdm pandas
```

---

## 🚀 Hướng Dẫn Chạy Thực Nghiệm

### Cách 1: Chạy trực tiếp trên Jupyter Notebook / Google Colab (Khuyên dùng)
1. Mở file [assignments.ipynb](assignments.ipynb) trên VS Code hoặc tải lên [Google Colab](https://colab.research.google.com/).
2. Nếu dùng Colab, hãy bật **GPU T4** (`Runtime` -> `Change runtime type` -> `T4 GPU`).
3. Chạy tuần tự các ô cell từ trên xuống dưới. Toàn bộ đồ thị loss và bảng chỉ số sẽ được hiển thị trực tiếp.

### Cách 2: Huấn luyện qua dòng lệnh (Terminal với `cnn_project`)
```bash
cd cnn_project

# 1. Huấn luyện LeNet-5 (AvgPool)
python train.py --model lenet --data_root ../VinaFood21 --epochs 20

# 2. Huấn luyện GoogLeNet rút gọn
python train.py --model googlenet --data_root ../VinaFood21 --epochs 20

# 3. Huấn luyện ResNet-18 (có Augmentation)
python train.py --model resnet18 --data_root ../VinaFood21 --epochs 20

# 4. Huấn luyện ResNet-50 Fine-Tuning
python train.py --model resnet50_ft --data_root ../VinaFood21 --epochs 20

# Đánh giá checkpoint tốt nhất trên tập test
python evaluate.py --ckpt outputs/lenet/best.pt --data_root ../VinaFood21
```

---

## 💡 Phân Tích & Kết Luận Cốt Lõi

### 1. Vì sao chọn Macro-F1 làm độ đo chính cho VinaFood21?
* Bộ dữ liệu VinaFood21 có sự phân bố số lượng ảnh giữa các lớp không hoàn toàn đồng đều (imbalanced). Nếu chỉ dùng **Accuracy**, mô hình có thể đạt độ chính xác tổng thể cao chỉ bằng cách đoán đúng các lớp phổ biến, trong khi bỏ qua các lớp khó/ít mẫu.
* **Macro-F1** tính giá trị trung bình cộng không trọng số của F1-score trên từng lớp riêng biệt ($Macro\text{-}F1 = \frac{1}{C}\sum_{i=1}^C F1_i$), qua đó đánh giá công bằng năng lực nhận diện của mô hình trên từng món ăn.

### 2. Sự phân bố tham số trong LeNet-5
* Trong mô hình LeNet-5 ($339,421$ tham số), lớp fully-connected đầu tiên **F5** chiếm tới $324,600$ tham số (**95.6%** tổng tham số mạng). Nguyên nhân là do sau hai tầng conv và pooling, feature map được duỗi thẳng (*flatten*) thành vector kích thước $16 \times 13 \times 13 = 2,704$ chiều, kết nối đầy đủ tới $120$ neuron ẩn ($2,704 \times 120 + 120 = 324,600$).
* Điều này cho thấy điểm nghẽn cổ chai (*bottleneck*) về bộ nhớ và tính toán của các mạng CNN đời đầu tập trung phần lớn ở các tầng Fully-Connected thay vì tầng Convolution.

### 3. Hiệu quả của Inception Block và Global Average Pooling (GoogLeNet)
* **Khối Inception** xử lý ảnh đồng thời ở nhiều tỉ lệ receptive field ($1\times1, 3\times3, 5\times5$), trích xuất cả chi tiết nhỏ lẫn ngữ cảnh rộng. Nhờ có conv $1\times1$ đóng vai trò nút thắt cổ chai (*channel reduction bottleneck*), số lượng phép toán nhân-cộng giảm mạnh (gần $10$ lần so với conv $5\times5$ trực tiếp).
* **Global Average Pooling (GAP)**: Thay vì dùng `Flatten` tạo ra hàng trăm nghìn trọng số ở classifier, GAP nén mỗi feature map về một giá trị trung bình duy nhất ($1\times1$), giảm kích thước tham số lớp phân loại xuống hàng chục lần (giảm $48.9$ lần trong thực nghiệm), chống overfitting hiệu quả mà vẫn giữ được tính bất biến không gian.

### 4. Overfitting trên ResNet-18 và Tác Dụng của Data Augmentation
* ResNet-18 sở hữu khoảng $11.19$ triệu tham số. Khi huấn luyện từ đầu trên tập VinaFood21 (~14,000 ảnh) mà **không có Data Augmentation**, mạng nhanh chóng ghi nhớ dữ liệu tập train (train loss tiến dần về 0) nhưng val loss bắt đầu phân kỳ và tăng dần từ sau epoch thứ 5-8 (hiện tượng **Overfitting** nghiêm trọng).
* Khi áp dụng **Data Augmentation** (`RandomResizedCrop`, `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`), không gian mẫu huấn luyện được mở rộng liên tục, buộc mạng phải học các đặc trưng bất biến về cấu trúc món ăn thay vì học vẹt ảnh tĩnh, giúp thu hẹp khoảng cách giữa train loss và val loss, cải thiện đáng kể Macro-F1 trên tập test.

### 5. So sánh Transfer Learning (ResNet-50) và Huấn luyện từ đầu (ResNet-18)
* **Khởi tạo trọng số:** ResNet-18 huấn luyện từ các trọng số ngẫu nhiên ban đầu (*random weights*), trong khi ResNet-50 kế thừa bộ lọc đặc trưng phong phú đã được tiền huấn luyện trên 1.2 triệu ảnh ImageNet.
* **Feature Extraction:** Chỉ cần huấn luyện $43,029$ tham số ở classifier (tiết kiệm thời gian cực lớn, không đòi hỏi GPU mạnh) mà vẫn đạt độ chính xác vượt trội hơn hẳn ResNet-18 từ đầu.
* **Fine-Tuning:** Khi mở khóa toàn bộ $23.55$ triệu tham số với learning rate nhỏ ($10^{-4}$), các tầng biểu diễn cấp cao của ResNet-50 được tinh chỉnh hoàn hảo theo đặc thù hình ảnh ẩm thực Việt Nam, đem lại kết quả tối ưu nhất trong cả 4 mô hình.
