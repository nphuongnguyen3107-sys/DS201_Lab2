# DS201 - Lab 2: Convolutional Neural Network (CNN)

Bài thực hành môn Học sâu (DS201) - Cài đặt, huấn luyện và đánh giá các kiến trúc mạng nơ-ron tích chập trên bộ dữ liệu **VinaFood21**.

## 1. Cấu trúc thư mục

```text
DS201_Lab2/
├── assignments.ipynb       # Notebook chính chứa toàn bộ mã nguồn và kết quả thực nghiệm
├── README.md               # Mô tả bài thực hành
└── cnn_project/            # Thư mục module nguồn
    ├── data.py             # Dataloader và tiền xử lý dữ liệu
    ├── model.py            # Cài đặt các kiến trúc mô hình (LeNet, GoogLeNet, ResNet)
    ├── engine.py           # Hàm huấn luyện và đánh giá
    ├── train.py            # Script huấn luyện
    ├── evaluate.py         # Script đánh giá checkpoint
    └── requirements.txt    # Thư viện cần thiết
```

## 2. Các nội dung thực hiện

- **Bài 1**: Cài đặt mô hình **LeNet-5** cho ảnh kích thước $64 \times 64$, huấn luyện và đánh giá trên 4 độ đo (Accuracy, Precision, Recall, Macro-F1). Thử nghiệm so sánh giữa AvgPool và MaxPool.
- **Bài 2**: Cài đặt mô hình **GoogLeNet rút gọn** sử dụng các khối Inception và Global Average Pooling (GAP).
- **Bài 3\***: Cài đặt mạng **ResNet-18** từ đầu, đánh giá tác dụng của Data Augmentation trong việc hạn chế hiện tượng overfitting.
- **Bài 4\***: Sử dụng mô hình **ResNet-50 Pretrained** (ImageNet), so sánh giữa 2 chế độ Feature Extraction và Fine-Tuning toàn bộ mạng.
