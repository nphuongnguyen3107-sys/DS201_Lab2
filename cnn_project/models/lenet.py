import torch
import torch.nn as nn


def n_params(module, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


class LeNet5(nn.Module):
    """
    LeNet-5 điều chỉnh cho ảnh màu kích thước 64x64x3 và phân loại 21 lớp.
    Hỗ trợ cả 2 cơ chế pooling: 'avg' (mặc định) và 'max'.
    """
    def __init__(self, in_ch=3, img_size=64, num_classes=21, pool="avg"):
        super().__init__()
        PoolLayer = nn.AvgPool2d if pool == "avg" else nn.MaxPool2d
        self.pool_type = pool

        self.c1 = nn.Conv2d(in_ch, 6, kernel_size=5)
        self.s2 = PoolLayer(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = PoolLayer(kernel_size=2, stride=2)

        n_flat = self._calculate_flat_dim(in_ch, img_size)
        self.f5 = nn.Linear(n_flat, 120)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)
        self.n_flat = n_flat

    def _calculate_flat_dim(self, in_ch, img_size):
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, img_size, img_size)
            dummy = self.s4(self.c3(self.s2(self.c1(dummy))))
            return int(dummy.flatten(1).shape[1])

    def forward(self, x):
        x = self.s2(torch.relu(self.c1(x)))
        x = self.s4(torch.relu(self.c3(x)))
        x = x.flatten(1)
        x = torch.relu(self.f5(x))
        x = torch.relu(self.f6(x))
        return self.out(x)


if __name__ == "__main__":
    m = LeNet5(in_ch=3, img_size=64, num_classes=21, pool="avg")
    x = torch.rand(2, 3, 64, 64)
    out = m(x)
    print("LeNet-5 (AvgPool):")
    print("  Input shape :", tuple(x.shape))
    print("  Output shape:", tuple(out.shape))
    print("  Total params:", f"{n_params(m):,}")
    assert m.n_flat == 2704, f"Expected 2704, got {m.n_flat}"
    assert n_params(m) == 339421, f"Expected 339,421, got {n_params(m)}"
    print("  Assertion passed!")
