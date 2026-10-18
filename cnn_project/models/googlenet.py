import torch
import torch.nn as nn


def n_params(module, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


class InceptionBlock(nn.Module):
    """
    Inception block tiêu chuẩn với 4 nhánh song song và nút thắt cổ chai 1x1.
    Tất cả các nhánh đều giữ nguyên H và W (stride=1, padding phù hợp).
    Tổng số channel đầu ra = f1 + f3 + f5 + f_pool.
    """
    def __init__(self, in_ch, f1, f3_reduce, f3, f5_reduce, f5, f_pool):
        super().__init__()
        # Nhánh 1: Conv 1x1
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, f1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # Nhánh 2: 1x1 giảm chiều -> 3x3
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, f3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f3_reduce, f3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Nhánh 3: 1x1 giảm chiều -> 5x5
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, f5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f5_reduce, f5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        # Nhánh 4: MaxPool 3x3 (stride=1, pad=1) -> 1x1
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, f_pool, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Nối dọc theo chiều channel (dim=1 trong PyTorch NCHW)
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogLeNetReduced(nn.Module):
    """
    Mô hình GoogLeNet rút gọn (4 Inception blocks) phù hợp cho ảnh kích thước 64x64.
    Sử dụng Global Average Pooling (GAP) trước Linear classifier để tránh phình tham số.
    """
    def __init__(self, in_ch=3, num_classes=21, dropout=0.4):
        super().__init__()
        # Stem: Conv 3x3 -> BN -> ReLU -> MaxPool (64x64 -> 32x32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Stage 1: 2 Inception blocks
        self.inc1 = InceptionBlock(64, f1=32, f3_reduce=32, f3=48, f5_reduce=8, f5=16, f_pool=16)   # out: 112
        self.inc2 = InceptionBlock(112, f1=64, f3_reduce=48, f3=64, f5_reduce=16, f5=32, f_pool=32) # out: 192

        # Downsample giữa Stage 1 và Stage 2: (32x32 -> 16x16)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2: 2 Inception blocks
        self.inc3 = InceptionBlock(192, f1=96, f3_reduce=64, f3=96, f5_reduce=16, f5=32, f_pool=32)  # out: 256
        self.inc4 = InceptionBlock(256, f1=128, f3_reduce=96, f3=128, f5_reduce=24, f5=48, f_pool=48) # out: 352

        # Classifier head: Global Average Pooling thay cho Flatten
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(352, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.maxpool(x)
        x = self.inc3(x)
        x = self.inc4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


if __name__ == "__main__":
    # Test Inception block 1 và block 2 đối chiếu với tài liệu
    b1 = InceptionBlock(1, 16, 16, 24, 4, 8, 8)
    assert n_params(b1) == 4376, f"Expected 4376, got {n_params(b1)}"

    b2 = InceptionBlock(56, 32, 32, 48, 8, 16, 16)
    assert n_params(b2) == 22104, f"Expected 22104, got {n_params(b2)}"
    print("InceptionBlock assertions passed!")

    # Test full GoogLeNet
    model = GoogLeNetReduced(num_classes=21)
    x = torch.rand(2, 3, 64, 64)
    out = model(x)
    print("GoogLeNetReduced:")
    print("  Input shape :", tuple(x.shape))
    print("  Output shape:", tuple(out.shape))
    print("  Total params:", f"{n_params(model):,}")
    assert out.shape == (2, 21)
