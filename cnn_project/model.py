import torch
import torch.nn as nn


def n_params(module, trainable_only=False):
    """Tính tổng số tham số (hoặc chỉ tham số có requires_grad=True)."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


# ==============================================================================
# BÀI 1: LENET-5 (64x64, 21 lớp, 339,421 tham số)
# ==============================================================================
class LeNet5(nn.Module):
    """
    LeNet-5 điều chỉnh cho ảnh màu 64x64 và 21 lớp phân loại.
    Hỗ trợ 2 chế độ pooling: 'avg' (mặc định) và 'max'.
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


# ==============================================================================
# BÀI 2: GOOGLENET (Inception Block + Global Average Pooling)
# ==============================================================================
class InceptionBlock(nn.Module):
    """
    Khối Inception với 4 nhánh song song, dùng conv 1x1 giảm chiều (bottleneck).
    Tất cả các nhánh giữ nguyên chiều cao (H) và chiều rộng (W).
    Channel ra = f1 + f3 + f5 + f_pool.
    """
    def __init__(self, in_ch, f1, f3_reduce, f3, f5_reduce, f5, f_pool):
        super().__init__()
        # Nhánh 1: Conv 1x1
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, f1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # Nhánh 2: 1x1 giảm chiều -> 3x3 conv
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, f3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f3_reduce, f3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Nhánh 3: 1x1 giảm chiều -> 5x5 conv
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, f5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f5_reduce, f5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        # Nhánh 4: MaxPool 3x3 (stride=1, pad=1) -> 1x1 conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, f_pool, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogLeNetReduced(nn.Module):
    """
    GoogLeNet rút gọn (4 Inception blocks) cho ảnh 64x64.
    Dùng Global Average Pooling (GAP) trước Linear classifier.
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

        # Classifier head: Global Average Pooling (GAP)
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


# ==============================================================================
# BÀI 3: RESNET-18 (BasicBlock, 4 stages [2, 2, 2, 2], 11,187,285 tham số)
# ==============================================================================
class BasicBlock(nn.Module):
    """
    Khối thặng dư cơ bản của ResNet.
    y = ReLU(F(x) + shortcut(x)).
    Shortcut dùng Conv 1x1 + BN (projection) khi đổi kích thước hoặc channel.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity                     # Cộng trước
        return torch.relu(out)              # ReLU sau


class ResNet18(nn.Module):
    """
    ResNet-18 hoàn chỉnh gồm 4 stage [2, 2, 2, 2] BasicBlocks.
    Dùng ảnh 96x96 hoặc 112x112 do mạng giảm chiều 32 lần (2^5).
    """
    def __init__(self, in_ch=3, num_classes=21):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_stage(64, 64, stride=1)
        self.layer2 = self._make_stage(64, 128, stride=2)
        self.layer3 = self._make_stage(128, 256, stride=2)
        self.layer4 = self._make_stage(256, 512, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _make_stage(in_ch, out_ch, stride):
        return nn.Sequential(
            BasicBlock(in_ch, out_ch, stride=stride),
            BasicBlock(out_ch, out_ch, stride=1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ==============================================================================
# BÀI 4: PRETRAINED RESNET-50 (Transfer Learning từ ImageNet)
# ==============================================================================
class PretrainedResnet(nn.Module):
    """
    ResNet-50 Pretrained từ HuggingFace (microsoft/resnet-50) hoặc torchvision.
    - Feature Extraction (freeze_backbone=True): 43,029 tham số học.
    - Fine-Tuning (freeze_backbone=False): 23,551,061 tham số học.
    """
    def __init__(self, num_classes=21, freeze_backbone=False):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.source = "huggingface"

        try:
            from transformers import ResNetForImageClassification
            base = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
            self.backbone = base.resnet
            self.classifier = nn.Linear(2048, num_classes)
            self.source = "huggingface"
        except Exception:
            from torchvision.models import resnet50, ResNet50_Weights
            base = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.backbone = nn.Sequential(*list(base.children())[:-1])
            self.classifier = nn.Linear(2048, num_classes)
            self.source = "torchvision"

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        if self.source == "huggingface":
            feats = self.backbone(x).pooler_output
            feats = feats.squeeze(-1).squeeze(-1)
        else:
            feats = self.backbone(x)
            feats = torch.flatten(feats, 1)
        return self.classifier(feats)


# ==============================================================================
# FACTORY VÀ CẤU HÌNH MẶC ĐỊNH
# ==============================================================================
MODEL_DEFAULTS = {
    # model_name: (img_size, lr, epochs)
    "lenet": (64, 1e-3, 20),
    "lenet_avg": (64, 1e-3, 20),
    "lenet_max": (64, 1e-3, 20),
    "googlenet": (64, 1e-3, 20),
    "resnet18": (96, 1e-3, 20),
    "resnet50_fe": (224, 1e-3, 20),
    "resnet50_ft": (224, 1e-4, 20),
}


def get_model_defaults(model_name: str):
    name = model_name.lower().strip()
    return MODEL_DEFAULTS.get(name, (64, 1e-3, 20))


def build_model(model_name: str, num_classes: int = 21, in_ch: int = 3, img_size: int = None):
    """Factory khởi tạo mô hình tự động theo tên."""
    name = model_name.lower().strip()
    if name in ["lenet", "lenet_avg"]:
        return LeNet5(in_ch=in_ch, img_size=img_size or 64, num_classes=num_classes, pool="avg")
    elif name == "lenet_max":
        return LeNet5(in_ch=in_ch, img_size=img_size or 64, num_classes=num_classes, pool="max")
    elif name in ["googlenet", "inception"]:
        return GoogLeNetReduced(in_ch=in_ch, num_classes=num_classes)
    elif name in ["resnet18", "resnet-18"]:
        return ResNet18(in_ch=in_ch, num_classes=num_classes)
    elif name in ["resnet50_fe", "resnet50_freeze"]:
        return PretrainedResnet(num_classes=num_classes, freeze_backbone=True)
    elif name in ["resnet50_ft", "resnet50_finetune", "resnet50"]:
        return PretrainedResnet(num_classes=num_classes, freeze_backbone=False)
    else:
        raise ValueError(
            f"Không hỗ trợ mô hình '{model_name}'. Các tùy chọn: {list(MODEL_DEFAULTS.keys())}"
        )


__all__ = [
    "LeNet5",
    "InceptionBlock",
    "GoogLeNetReduced",
    "BasicBlock",
    "ResNet18",
    "PretrainedResnet",
    "build_model",
    "get_model_defaults",
    "MODEL_DEFAULTS",
    "n_params"
]


# ==============================================================================
# KIỂM ĐỊNH CẢ 4 MÔ HÌNH KHI CHẠY TRỰC TIẾP: python model.py
# ==============================================================================
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("Checking 4 CNN Model Architectures (VinaFood21 - 21 Classes)")
    print("=" * 60)

    # 1. LeNet-5
    m1 = LeNet5(in_ch=3, img_size=64, num_classes=21, pool="avg")
    x1 = torch.rand(2, 3, 64, 64)
    print(f"1. LeNet-5 (64x64)     : {n_params(m1):,} params -> Out: {tuple(m1(x1).shape)}")
    assert n_params(m1) == 339421, f"Mismatch: {n_params(m1)}"

    # 2. GoogLeNet
    b1 = InceptionBlock(1, 16, 16, 24, 4, 8, 8)
    assert n_params(b1) == 4376
    b2 = InceptionBlock(56, 32, 32, 48, 8, 16, 16)
    assert n_params(b2) == 22104
    m2 = GoogLeNetReduced(num_classes=21)
    x2 = torch.rand(2, 3, 64, 64)
    print(f"2. GoogLeNet (64x64)   : {n_params(m2):,} params -> Out: {tuple(m2(x2).shape)}")

    # 3. ResNet-18
    m3 = ResNet18(num_classes=21)
    x3 = torch.rand(2, 3, 96, 96)
    print(f"3. ResNet-18 (96x96)   : {n_params(m3):,} params -> Out: {tuple(m3(x3).shape)}")
    assert n_params(m3) == 11187285, f"Mismatch: {n_params(m3)}"

    # 4. ResNet-50 Pretrained
    m4_fe = PretrainedResnet(num_classes=21, freeze_backbone=True)
    m4_ft = PretrainedResnet(num_classes=21, freeze_backbone=False)
    p_fe = n_params(m4_fe, trainable_only=True)
    p_ft = n_params(m4_ft, trainable_only=True)
    print(f"4a. ResNet-50 FE (224) : {p_fe:,} trainable params (Total: {n_params(m4_fe):,})")
    print(f"4b. ResNet-50 FT (224) : {p_ft:,} trainable params (Total: {n_params(m4_ft):,})")
    assert p_fe == 43029
    assert p_ft == 23551061

    print("=" * 60)
    print("All 4 models passed assertions successfully!")
    print("=" * 60)
