import torch
import torch.nn as nn


def n_params(module, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


class PretrainedResnet(nn.Module):
    """
    Mô hình ResNet-50 Pretrained từ HuggingFace (microsoft/resnet-50) hoặc torchvision.
    Hỗ trợ 2 chế độ huấn luyện:
      1. Feature Extraction (freeze_backbone=True): Đóng băng backbone, chỉ học lớp classifier mới (43,029 tham số).
      2. Fine-Tuning (freeze_backbone=False): Cập nhật toàn bộ mạng (23,551,061 tham số).
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


if __name__ == "__main__":
    fe = PretrainedResnet(num_classes=21, freeze_backbone=True)
    ft = PretrainedResnet(num_classes=21, freeze_backbone=False)

    print("Pretrained ResNet-50 (source:", fe.source, "):")
    print(f"  Feature Extraction - Total: {n_params(fe):,}, Trainable: {n_params(fe, True):,}")
    print(f"  Fine-Tuning        - Total: {n_params(ft):,}, Trainable: {n_params(ft, True):,}")

    assert n_params(fe) == 23551061
    assert n_params(fe, trainable_only=True) == 43029
    assert n_params(ft, trainable_only=True) == 23551061
    print("  Assertions passed!")
