# Tiền xử lý và nạp dữ liệu 

from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Tạo bộ biến đổi ảnh: train có augment, val/test chỉ resize chuẩn hoá
def build_transforms(img_size, augment=True):
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf

# Tạo DataLoader nạp dữ liệu train & val 
def build_train_val_loaders(data_root, img_size, batch_size=32,
                            augment=True, val_ratio=0.1, num_workers=2, seed=42):
    root = Path(data_root)
    if not (root / "train").is_dir() and (root / "VinaFood21" / "train").is_dir():
        root = root / "VinaFood21"

    train_tf, eval_tf = build_transforms(img_size, augment=augment)
    if (root / "val").is_dir():
        train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
        val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
        classes = train_ds.classes
    else:
        full_train = datasets.ImageFolder(root / "train", transform=train_tf)
        full_eval = datasets.ImageFolder(root / "train", transform=eval_tf)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(full_train), generator=g).tolist()
        n_val = int(len(idx) * val_ratio)
        train_ds = Subset(full_train, idx[n_val:])
        val_ds = Subset(full_eval, idx[:n_val])
        classes = full_train.classes
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_dl, val_dl, classes

# Tạo DataLoader nạp tập test để đánh giá mô hình sau khi huấn luyện
def build_test_loader(data_root, img_size, batch_size=32, num_workers=2):
    root = Path(data_root)
    if not (root / "test").is_dir() and (root / "VinaFood21" / "test").is_dir():
        root = root / "VinaFood21"

    _, eval_tf = build_transforms(img_size, augment=False)
    test_ds = datasets.ImageFolder(root / "test", transform=eval_tf)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return test_dl, test_ds.classes
