# LeNet-5 · VinaFood21

```
lenet_project/
├── data.py        # transform + DataLoader
├── model.py       # LeNet-5
├── engine.py      # train_one_epoch, evaluate
├── train.py       # chạy huấn luyện, lưu outputs/best.pt
└── evaluate.py    # đánh giá trên test, lưu confusion matrix
```

Dữ liệu theo cấu trúc `ImageFolder`; không có `val/` thì tự tách 10% từ `train/`:

```
VinaFood21/
├── train/<tên lớp>/*.jpg
├── val/<tên lớp>/*.jpg     (tuỳ chọn)
└── test/<tên lớp>/*.jpg
```

## Chạy trên Colab (Terminal)

```bash
cd /content/lenet_project
pip install -r requirements.txt

python model.py                                   # kiểm tra shape
python train.py --data_root /content/VinaFood21 --epochs 20
python evaluate.py --data_root /content/VinaFood21 --ckpt outputs/best.pt
```

Tham số của `train.py`: `--img_size 64 --batch_size 32 --epochs 20 --lr 1e-3 --num_workers 2 --seed 42 --out_dir outputs`.

Kết quả trong `outputs/`: `best.pt`, `history.csv`, `loss_curve.png`, `confusion_matrix.png`.
