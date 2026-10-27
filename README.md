# DS201 - LAB 2: CONVOLUTIONAL NEURAL NETWORK (CNN)

- **Môn học:** Deep Learning (DS201)
- **Sinh viên:** Nguyễn Phương Nguyên
- **MSSV:** 24521196
- **Lớp:** DS201.R11.1
- **Bộ dữ liệu:** [VinaFood21](https://arxiv.org/abs/2108.02929) (21 lớp món ăn Việt Nam, ~14.000 ảnh)

---

## 1. Cấu trúc thư mục mã nguồn

```text
DS201_Lab2/
├── assignments.ipynb       # Jupyter Notebook chính chứa mã nguồn, kết quả huấn luyện và báo cáo
├── README.md               # Báo cáo tổng hợp và hướng dẫn chi tiết
├── outputs/                # Lưu trữ checkpoint mô hình (*_best.pt), loss curves, confusion matrix
└── cnn_project/            # Source code mô-đun hóa
    ├── data.py             # Pipeline nạp dữ liệu, transform (train/val riêng biệt), split seed
    ├── model.py            # Cài đặt các kiến trúc (LeNet-5, GoogLeNet, ResNet-18, ResNet-50)
    ├── engine.py           # Vòng lặp huấn luyện train_one_epoch và đánh giá evaluate
    ├── train.py            # CLI script huấn luyện độc lập
    ├── evaluate.py         # CLI script đánh giá checkpoint và trích xuất classification report
    └── requirements.txt    # Thư viện phụ thuộc
```

---

## 2. Quy chuẩn thiết kế thực nghiệm chung

Theo quy định bắt buộc trong bài giảng:
1. **Quy trình xử lý dữ liệu 7 bước:** `ImageFolder` $\rightarrow$ `random_split` $\rightarrow$ `Resize + Normalize` $\rightarrow$ `DataLoader` $\rightarrow$ `Model` $\rightarrow$ `Loss (CE) + Optimizer (Adam)` $\rightarrow$ `Đánh giá Macro-F1`.
2. **Cố định ngẫu nhiên (Fixed Seed):** Toàn bộ 4 bài thực hành đều chạy trên cùng một phép chia dữ liệu (Train/Val/Test) với `SEED = 42`. Nếu không cố định seed, bảng so sánh cuối bài không có giá trị đối chiếu khoa học.
3. **Tách biệt Data Transform:**
   - **Tập Train:** Áp dụng Data Augmentation gồm `RandomResizedCrop(scale=(0.7, 1.0))`, `RandomHorizontalFlip()`, `RandomRotation(15)`, `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`.
   - **Tập Val và Test:** Tuyệt đối không dùng Augmentation, chỉ `Resize` và `Normalize` với chuẩn ImageNet `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.
4. **Chế độ đánh giá (Evaluation Mode):** Bắt buộc gọi `model.eval()` và `torch.no_grad()` trước khi đánh giá trên tập val/test để tắt chế độ ngẫu nhiên của Dropout và sử dụng thống kê tích lũy của BatchNorm.
5. **Hệ thống độ đo (Metrics):**
   - **Độ đo chính:** **Macro-F1** (trung bình không trọng số của F1 trên 21 lớp, phản ánh chính xác hiệu năng khi phân bố số ảnh giữa các lớp không đều nhau).
   - **Độ đo đối chiếu:** Accuracy, Precision (macro), Recall (macro), F1 từng lớp và Ma trận nhầm lẫn (Confusion Matrix $21 \times 21$).

---

## 3. Cấu hình chi tiết bốn bài thực hành

| Bài | Kiến trúc | Kích thước ảnh | Optimizer & LR | Data Augmentation | Điểm chốt riêng của bài |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Bài 1** | LeNet-5 điều chỉnh (7 layers) | $64 \times 64$ | Adam ($10^{-3}$) | Không / Flip ngang | So sánh `AvgPool2d` vs `MaxPool2d` |
| **Bài 2** | GoogLeNet rút gọn (4 Inception blocks) | $64 \times 64$ | Adam ($10^{-3}$) | Không (như Bài 1) | Dùng Global Average Pooling (GAP) thay cho Flatten |
| **Bài 3\*** | ResNet-18 [2,2,2,2] BasicBlock | $96 \times 96$ | Adam ($10^{-3}$) | **Chạy 2 lần** (Có vs Không) | So sánh đối chứng Overfitting & Tỉ lệ tham số/dữ liệu |
| **Bài 4\*** | ResNet-50 Pretrained (HuggingFace) | $224 \times 224$ | Adam (FE: $10^{-3}$, FT: $10^{-4}$) | Không (như Bài 3) | **Chạy cả 2 chế độ:** Feature Extraction vs Fine-Tuning |

---

## 4. Báo cáo chi tiết từng bài thực hành

### 4.1. Bài 1: LeNet-5 trên VinaFood21
- **Kiến trúc:** Điều chỉnh từ LeNet-5 cổ điển cho ảnh màu 3 channel, 21 lớp đầu ra:
  $$\text{Input}(3, 64, 64) \rightarrow \text{C1}(6, 5\times 5) \rightarrow \text{S2}(2\times 2) \rightarrow \text{C3}(16, 5\times 5) \rightarrow \text{S4}(2\times 2) \rightarrow \text{Flatten}(2704) \rightarrow \text{F5}(120) \rightarrow \text{F6}(84) \rightarrow \text{OUT}(21)$$
- **Đối chiếu số lượng tham số:**
  - C1: $(5 \times 5 \times 3 + 1) \times 6 = 456$
  - C3: $(5 \times 5 \times 6 + 1) \times 16 = 2,416$
  - F5: $2,704 \times 120 + 120 = 324,600$ (chiếm tới $95.6\%$ tham số cả mạng)
  - F6: $120 \times 84 + 84 = 10,164$
  - OUT: $84 \times 21 + 21 = 1,785$
  - **Tổng tham số thực tế:** **339,421** tham số.
- **Kết quả thử nghiệm trên tập Test:**
  - **LeNet-5 (AvgPool):** Accuracy = **0.2710**, Precision = 0.2682, Recall = 0.2621, **Macro-F1 = 0.2581**.
  - **LeNet-5 (MaxPool):** Accuracy = **0.2628**, Precision = 0.2574, Recall = 0.2519, **Macro-F1 = 0.2376**.
  - **So sánh AvgPool vs MaxPool:** `AvgPool2d` cho Macro-F1 nhỉnh hơn `MaxPool2d` ($\Delta = +0.0205$), do AvgPool giữ lại đặc trưng nền mịn của các món ăn tốt hơn khi kích thước ảnh bị thu nhỏ về $64 \times 64$.

---

### 4.2. Bài 2: GoogLeNet rút gọn trên VinaFood21
- **Kiến trúc:** Gồm Conv Stem ($64$ channels), 4 khối Inception, Global Average Pooling (GAP) và Classifier Head.
- **Đối chiếu số channel output từng Inception Block (Tính toán lý thuyết vs Code thực tế):**
  - Công thức: $\text{out\_channels} = f_1 + f_3 + f_5 + f_p$ ($f_{3r}$ và $f_{5r}$ là trung gian nút cổ chai $1\times 1$).
  - **Block 1 (`inc1`):** In 64 $\rightarrow$ $(f_1=32, f_3=48, f_5=16, f_p=16) \rightarrow \mathbf{112}$ channels. *(Code: 112 - KHỚP)*
  - **Block 2 (`inc2`):** In 112 $\rightarrow$ $(f_1=64, f_3=64, f_5=32, f_p=32) \rightarrow \mathbf{192}$ channels. *(Code: 192 - KHỚP)*
  - *(Downsample qua MaxPool $2\times 2$)*
  - **Block 3 (`inc3`):** In 192 $\rightarrow$ $(f_1=96, f_3=96, f_5=32, f_p=32) \rightarrow \mathbf{256}$ channels. *(Code: 256 - KHỚP)*
  - **Block 4 (`inc4`):** In 256 $\rightarrow$ $(f_1=128, f_3=128, f_5=48, f_p=48) \rightarrow \mathbf{352}$ channels. *(Code: 352 - KHỚP)*
  - Sau Block 4: Tensor $(N, 352, 16, 16) \rightarrow$ qua GAP thành $(N, 352, 1, 1) \rightarrow$ Flatten thành vector 352 $\rightarrow$ `Linear(352, 21)` ($352 \times 21 + 21 = 7,413$ tham số).
- **So sánh trực tiếp với Bài 1 (LeNet-5):**
  - **Số tham số:** GoogLeNet (**414,709**) vs LeNet-5 (**339,421**) $\rightarrow$ chỉ tăng nhẹ $22.2\%$.
  - **Thời gian/epoch:** GoogLeNet (**104.14s**) vs LeNet-5 (**100.88s**) $\rightarrow$ xấp xỉ nhau (~$3\%$ chênh lệch).
  - **Macro-F1:** GoogLeNet (**0.4281**) vs LeNet-5 (**0.2581**) $\rightarrow$ **Tăng vọt $+0.1700$ ($+17.00\%$)**!
  - **Kết luận:** Kiến trúc Inception đa nhánh kết hợp GAP mang lại hiệu quả vượt bậc so với mạng tuần tự cổ điển mà không làm tăng gánh nặng tính toán.

---

### 4.3. Bài 3\*: ResNet-18 huấn luyện từ đầu
- **Kiến trúc:** ResNet-18 chuẩn `[2, 2, 2, 2]` BasicBlock, tổng cộng **11,187,285 tham số**. Ảnh đầu vào $96 \times 96$.
- **Xác định Epoch tách đường giữa Train Loss và Val Loss (Hiện tượng Overfitting):**
  - **Trường hợp KHÔNG Augmentation:** Trong 6 epoch đầu, Train Loss và Val Loss cùng giảm đều. **Bắt đầu từ Epoch 7 - 8**, hai đường tách nhau rõ rệt: Train Loss tiếp tục giảm mạnh từ 1.46 xuống 0.08 (Train Acc đạt $97.33\%$), trong khi Val Loss đảo chiều tăng vọt từ 1.82 lên trên 3.14. Đây là biểu hiện rõ rệt của **Overfitting nghiêm trọng** (mạng học thuộc lòng ảnh train nhưng mất tính khái quát).
  - **Trường hợp CÓ Data Augmentation:** Train Loss và Val Loss giảm song song ổn định, Val Loss duy trì trong khoảng 1.42 - 1.55 suốt 20 epoch.
- **Chênh lệch Macro-F1 giữa 2 lần chạy:**
  - Có Augmentation: Macro-F1 = **0.4758** (Accuracy: **49.49%**).
  - Không Augmentation: Macro-F1 = **0.3925** (Accuracy: **39.69%**).
  - **Mức cải thiện:** $\Delta = \mathbf{+0.0833}$ ($+8.33\%$).
- **Tỉ lệ tham số trên dữ liệu:**
  - Mô hình có **11,187,285 tham số**, gấp **33 lần** LeNet-5, trên tập dữ liệu chỉ khoảng **14,000 ảnh** (tỉ lệ $\approx 800$ tham số cho mỗi ảnh). Do đó, Data Augmentation là giải pháp bắt buộc để huấn luyện mô hình sâu từ đầu.

---

### 4.4. Bài 4\*: ResNet-50 Pretrained trên VinaFood21
- **Kiến trúc:** `microsoft/resnet-50` từ HuggingFace (hoặc TorchVision fallback), kích thước ảnh $224 \times 224$.
- **Thực nghiệm cả hai chế độ:**
  1. **Feature Extraction (Đóng băng backbone, chỉ train Classifier 43,029 tham số, lr = $10^{-3}$):**
     - Test Accuracy: **0.6236** ($62.36\%$).
     - Test Macro-F1: **0.6312** ($63.12\%$).
  2. **Fine-Tuning (Mở khóa toàn bộ 23.55M tham số, lr = $10^{-4}$):**
     - Test Accuracy: **0.8295** ($82.95\%$).
     - Test Macro-F1: **0.8297** ($82.97\%$).
- **So sánh 2 chế độ:** Fine-Tuning cải thiện Macro-F1 thêm $+19.85\%$ so với Feature Extraction, đạt đỉnh $82.97\%$.

---

## 5. Bảng so sánh bốn mô hình - Tổng kết

Dưới đây là bảng tổng hợp kết quả chính thức của 4 mô hình thực nghiệm, :

| Mô hình | Resize | Tham số | Giây/epoch | Accuracy | macro-F1 | Ghi chú cấu hình |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LeNet-5 điều chỉnh** | $64 \times 64$ | 339,421 | 100.88s | 0.2710 | 0.2581 | AvgPool, 7 layers cổ điển |
| **GoogLeNet rút gọn** | $64 \times 64$ | 414,709 | 104.14s | 0.4439 | 0.4281 | 4 Inception blocks, GAP |
| **ResNet-18 từ đầu\*** | $96 \times 96$ | 11,187,285 | 115.96s | 0.4949 | 0.4758 | Có Augmentation (Không Aug: 0.3925) |
| **ResNet-50 pretrained\*** | $224 \times 224$ | 23,551,061 | 145.99s | **0.8295** | **0.8297** | Fine-Tuning toàn mạng (FE: 0.6312) |

---

## 6. Phân tích định lượng và Nhận xét chuyên sâu

### 6.1. Ảnh hưởng của Kiến trúc (Ba dòng đầu: LeNet-5 vs GoogLeNet vs ResNet-18)
- Chạy trên cùng seed 42, cùng phép chia dữ liệu:
  - **LeNet-5:** Năng lực trích xuất đặc trưng rất yếu do chỉ có 2 lớp Conv $5\times 5$. Mạng bị "nghẽn cổ chai" tham số ở tầng fully-connected (F5 chiếm $95.6\%$ tham số toàn mạng), dẫn tới Macro-F1 chỉ dừng ở mức khiêm tốn $0.2581$.
  - **GoogLeNet rút gọn:** Macro-F1 tăng vọt $+17.00\%$ (lên $0.4281$) dù tham số chỉ tăng $22\%$ và thời gian train mỗi epoch tương đương. Đó là nhờ khối Inception trích xuất đặc trưng đa độ phân giải song song và GAP loại bỏ lớp Dense cồng kềnh.
  - **ResNet-18 từ đầu:** Kiến trúc 18 layers với Skip Connection giúp dòng gradient lưu thông trực tiếp, cho phép học các biểu diễn phức tạp hơn và đạt Macro-F1 $0.4758$. Tuy nhiên, việc huấn luyện 11.18M tham số từ đầu trên tập ảnh nhỏ rất dễ overfit nếu không có Data Augmentation.

### 6.2. Ảnh hưởng của Transfer Learning (Dòng cuối so với ba dòng trên)
- **Chênh lệch kết quả:** ResNet-50 Fine-Tuning đạt Macro-F1 **0.8297**, vượt trội hoàn toàn so với mô hình tốt nhất học từ đầu là ResNet-18 (**0.4758**), tức chênh lệch tới **$+35.39\%$** ($+0.3539$).
- Ngay cả khi chỉ train classifier (Feature Extraction), Macro-F1 đã đạt **0.6312** (cao hơn ResNet-18 học từ đầu $+15.54\%$).
- **Giải thích nguyên nhân cốt lõi:**
  1. **Quy mô tập dữ liệu đích:** VinaFood21 chỉ có ~14k ảnh, không đủ lớn để một mạng sâu học hàng chục triệu tham số từ đầu mà không gặp rủi ro quá khớp.
  2. **Chuyển giao tri thức thị giác (Feature Reuse):** 23.5 triệu tham số backbone của ResNet-50 đã được tối ưu trên 1.2 triệu ảnh ImageNet, trong đó có sẵn nhiều lớp thuộc miền thực phẩm. Các đặc trưng phổ quát (cạnh, kết cấu, màu sắc món ăn) đã rất tối ưu.
  3. **Tốc độ hội tụ và độ ổn định:** Fine-tuning với learning rate nhỏ ($10^{-4}$) cho phép mô hình tinh chỉnh nhanh chóng và đạt độ chính xác cao chỉ sau 10 epoch.

### 6.3. Kiểm chứng Giả thuyết dựa trên Confusion Matrix
- **Giả thuyết:** Các món có bố cục và màu sắc tương tự nhau (nhóm món nước: phở, bún bò, bún riêu...) sẽ có tỉ lệ nhầm lẫn cao.
- **Thực tế kiểm chứng:**
  - Trên ma trận nhầm lẫn của LeNet-5 và GoogLeNet, nhóm các món nước (`bo-kho`, `ca-ri`, `canh-kho-qua`, `hoanh-thanh-nuoc`) và nhóm bánh mềm (`banh-can`, `banh-u`, `ha-cao`) có sự nhầm lẫn chéo rất lớn.
  - Ngược lại, món có đặc trưng màu sắc/hình thái riêng biệt như **`canh-khoai-mo`** (màu tím đặc trưng) đạt F1 cao nhất trong mọi mô hình (F1 = $0.6139$ ở LeNet, $>0.95$ ở ResNet-50).
  - Ở mô hình ResNet-50 Fine-Tuning, ma trận nhầm lẫn gom mạnh về đường chéo chính, tỉ lệ nhầm lẫn giữa các món nước giảm rõ rệt nhờ backbone sâu phân biệt được các chi tiết sợi bánh, nước dùng và rau ăn kèm.

---

## 7. Hướng dẫn chạy và tái hiện kết quả (CLI Scripts)

### Cài đặt môi trường
```bash
pip install -r cnn_project/requirements.txt
```

### Huấn luyện từng mô hình bằng CLI
```bash
# Bài 1: LeNet-5 (AvgPool)
python cnn_project/train.py --model lenet --img_size 64 --lr 1e-3 --epochs 20

# Bài 1: LeNet-5 (MaxPool)
python cnn_project/train.py --model lenet_max --img_size 64 --lr 1e-3 --epochs 20

# Bài 2: GoogLeNet rút gọn
python cnn_project/train.py --model googlenet --img_size 64 --lr 1e-3 --epochs 20

# Bài 3: ResNet-18 (Không Augmentation)
python cnn_project/train.py --model resnet18 --img_size 96 --lr 1e-3 --epochs 20 --no_augment

# Bài 3: ResNet-18 (Có Augmentation)
python cnn_project/train.py --model resnet18 --img_size 96 --lr 1e-3 --epochs 20

# Bài 4: ResNet-50 Feature Extraction
python cnn_project/train.py --model resnet50_fe --img_size 224 --lr 1e-3 --epochs 10

# Bài 4: ResNet-50 Fine-Tuning
python cnn_project/train.py --model resnet50_ft --img_size 224 --lr 1e-4 --epochs 10
```

### Đánh giá checkpoint
```bash
python cnn_project/evaluate.py --ckpt outputs/resnet50_ft/best.pt --model resnet50_ft
```
