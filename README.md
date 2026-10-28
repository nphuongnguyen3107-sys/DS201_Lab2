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
├── outputs/                # Bảng kết quả tổng hợp (summary_comparison_table.csv); checkpoint (*.pt) và biểu đồ được lưu cục bộ (được .gitignore quản lý để tối ưu repo)
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
   - **Tập Train:** Áp dụng `Resize` và `Normalize` theo chuẩn ImageNet (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`). Riêng Bài 3 (chế độ có augmentation) bổ sung thêm `RandomResizedCrop(scale=(0.7, 1.0))`, `RandomHorizontalFlip()`, `RandomRotation(15)`, `ColorJitter(0.2, 0.2, 0.2)`.
   - **Tập Val và Test:** Không dùng Augmentation, chỉ `Resize` và `Normalize`.
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
- **Hiện tượng Overfitting và phân tích đường loss:**
  - **Trường hợp KHÔNG Augmentation:** Trong 6 epoch đầu, Train Loss và Val Loss cùng giảm đều. Từ epoch 7 - 8, hai đường tách nhau rất rõ: Train Loss tiếp tục giảm sâu từ 1.46 xuống 0.08 (Train Acc đạt 97.33%), trong khi Val Loss đảo chiều tăng vọt từ 1.82 lên 3.14. Đây là biểu hiện quá khớp (overfitting) khi mạng học thuộc lòng tập train.
  - **Trường hợp CÓ Data Augmentation:** Train Loss và Val Loss giảm song hành ổn định hơn. Val Loss giảm liên tục từ 2.68 (epoch 1) xuống 1.43 (epoch 19), và chỉ ở các epoch cuối mới dao động quanh mức 1.42 - 1.58. Tại epoch 20, cả Train Loss (1.08) và Val Loss (1.48) vẫn đang giảm đều, cho thấy mô hình chưa hoàn toàn hội tụ ở epoch 20 và có thể tiếp tục tăng độ chính xác nếu train thêm epoch.
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

Dưới đây là bảng tổng hợp kết quả của 4 mô hình thực nghiệm:

| Mô hình | Resize | Tham số | Giây/epoch | Accuracy | macro-F1 | Ghi chú cấu hình |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LeNet-5 điều chỉnh** | $64 \times 64$ | 339,421 | 100.88s | 0.2710 | 0.2581 | AvgPool, 7 layers cổ điển |
| **GoogLeNet rút gọn** | $64 \times 64$ | 414,709 | 104.14s | 0.4439 | 0.4281 | 4 Inception blocks, GAP |
| **ResNet-18 từ đầu\*** | $96 \times 96$ | 11,187,285 | 115.96s | 0.4949 | 0.4758 | Có Augmentation (Không Aug: 0.3925) |
| **ResNet-50 pretrained\*** | $224 \times 224$ | 23,551,061 | 145.99s | **0.8295** | **0.8297** | Fine-Tuning toàn mạng (FE: 0.6312) |

---

## 6. Phân tích định lượng và Nhận xét chuyên sâu

### 6.1. Ảnh hưởng của Kiến trúc (LeNet-5 vs GoogLeNet vs ResNet-18)
- Chạy trên cùng seed 42 và cùng phép chia dữ liệu:
  - **LeNet-5:** Năng lực trích xuất đặc trưng hạn chế do chỉ có 2 lớp Conv nhỏ ($5\times 5$), hơn 95% tham số dồn vào lớp Fully-Connected đầu tiên (F5). Vì vậy Macro-F1 chỉ đạt mức khiêm tốn 0.2581.
  - **GoogLeNet rút gọn:** Macro-F1 tăng vọt +17.00% (lên 0.4281) dù tham số chỉ tăng 22% và thời gian train mỗi epoch tương đương. Khối Inception đa nhánh trích xuất đặc trưng ở nhiều thang đo và Global Average Pooling (GAP) loại bỏ lớp Dense cồng kềnh giúp mạng gọn nhẹ mà hiệu quả hơn nhiều.
  - **ResNet-18 từ đầu:** Kiến trúc 18 layer với Residual Connection (đường tắt identity) giúp dòng gradient truyền tốt hơn, học được đặc trưng phức tạp hơn và đạt Macro-F1 0.4758. Tuy nhiên với 11.18M tham số trên tập ảnh nhỏ (~14k ảnh), mô hình rất dễ overfit nếu không có Data Augmentation.

### 6.2. Ảnh hưởng của Transfer Learning (ResNet-50 so với học từ đầu)
- **Chênh lệch kết quả:** ResNet-50 Fine-Tuning đạt Macro-F1 **0.8297** (Accuracy 82.95%), vượt trội so với mô hình học từ đầu tốt nhất là ResNet-18 (0.4758, chênh lệch +35.39%).
- Ngay cả khi chỉ train classifier (Feature Extraction), Macro-F1 đã đạt **0.6312**, cao hơn ResNet-18 học từ đầu.
- **Giải thích:**
  1. **Quy mô tập dữ liệu:** VinaFood21 chỉ có ~14k ảnh, không đủ lớn để một mạng sâu học tốt hàng chục triệu tham số từ đầu mà không bị quá khớp.
  2. **Trọng số Pretrained từ ImageNet:** 23.5 triệu tham số backbone của ResNet-50 đã được tối ưu sẵn trên 1.2 triệu ảnh ImageNet, các tầng trích xuất đặc trưng (cạnh, vân, màu sắc món ăn) đã rất tối ưu. Mô hình chỉ cần tinh chỉnh nhẹ với learning rate nhỏ ($10^{-4}$) là hội tụ rất nhanh và đạt độ chính xác cao.

### 6.3. Phân tích ma trận nhầm lẫn (Confusion Matrix)
- **Về các lớp dữ liệu và giả thuyết món nước:**
  - Câu hỏi gợi ý trong đề có nhắc tới các món nước (*phở, bún bò, hủ tiếu...*). Tuy nhiên trong 21 lớp thực tế của VinaFood21 không có các món này. Nhóm món có nước dùng hoặc nước xốt trong tập dữ liệu gồm: `bo-kho`, `ca-ri`, `canh-kho-qua`, `canh-khoai-mo`, `hoanh-thanh-nuoc`, `pha-lau`.
- **Thực tế kiểm chứng nhầm lẫn:**
  - **Cặp món bị nhầm lẫn nhiều nhất:** Là **`ca-ri` và `bo-kho`**, do cả hai cùng có nước xốt vàng cam/nâu sệt, thịt cắt khối và rau củ (khoai tây, cà rốt). Vì vậy món `ca-ri` có điểm F1 thấp nhất ở hầu hết các mô hình (0.1250 ở ResNet-18, 0.3518 ở ResNet-50 FE, 0.5506 ở ResNet-50 FT). Ngoài ra món `goi-ga` (gà xé trộn bắp cải) cũng có tỷ lệ nhầm lẫn đáng kể (F1 chỉ đạt 0.7463 ở ResNet-50 FT).
  - **Món `canh-kho-qua` không bị nhầm lẫn nhiều:** Ngược lại, món này luôn nằm trong **Top 3 F1 của tất cả các mô hình** (LeNet: 0.4584, GoogLeNet: 0.5891, ResNet-18: 0.6171, ResNet-50 FE: 0.7461, ResNet-50 FT: 0.9226) do khoanh khổ qua cắt khúc nhồi thịt màu xanh lá cây là đặc trưng thị giác rất riêng biệt và dễ nhận diện.
  - **Món `canh-khoai-mo`:** Có màu tím đặc trưng nên đạt F1 cao nhất ở các mô hình học từ đầu (LeNet: 0.6139, GoogLeNet: 0.7978, ResNet-18 Aug: 0.7704). Ở ResNet-50, F1 của `canh-khoai-mo` đạt 0.7591 (bản FE, xếp sau `bap-nuong` 0.7942) và 0.9195 (bản FT, xếp sau `bap-nuong` 0.9574 và `canh-kho-qua` 0.9226). Món có F1 cao nhất ở ResNet-50 Fine-Tuning là `bap-nuong` (0.9574) nhờ cấu trúc bắp nướng mỡ hành có hạt tròn xếp hàng đặc thù.
  - Ở mô hình ResNet-50 Fine-Tuning, ma trận nhầm lẫn gom rõ về đường chéo chính, tỷ lệ nhầm lẫn giữa các món giảm rõ rệt nhờ backbone sâu phân biệt tốt chi tiết từng món.

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
