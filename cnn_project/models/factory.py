from .lenet import LeNet5, n_params
from .googlenet import GoogLeNetReduced, InceptionBlock
from .resnet18 import ResNet18, BasicBlock
from .resnet50 import PretrainedResnet

MODEL_DEFAULTS = {
    # model_key: (default_img_size, default_lr, default_epochs)
    "lenet": (64, 1e-3, 20),
    "lenet_avg": (64, 1e-3, 20),
    "lenet_max": (64, 1e-3, 20),
    "googlenet": (64, 1e-3, 20),
    "resnet18": (96, 1e-3, 20),
    "resnet50_fe": (224, 1e-3, 20),   # Feature extraction: lr=1e-3
    "resnet50_ft": (224, 1e-4, 20),   # Fine-tuning: lr nhỏ 1e-4
}


def build_model(model_name: str, num_classes: int = 21, in_ch: int = 3, img_size: int = None):
    """
    Khởi tạo mô hình dựa trên tên được truyền vào.
    Hỗ trợ:
      - 'lenet' / 'lenet_avg': LeNet-5 với AvgPool2d
      - 'lenet_max': LeNet-5 với MaxPool2d
      - 'googlenet': GoogLeNet rút gọn (4 Inception blocks + GAP)
      - 'resnet18': ResNet-18 (4 stages, [2, 2, 2, 2] BasicBlocks)
      - 'resnet50_fe': Pretrained ResNet-50 ở chế độ Feature Extraction (đóng băng backbone)
      - 'resnet50_ft': Pretrained ResNet-50 ở chế độ Fine-Tuning (cập nhật toàn bộ)
    """
    name = model_name.lower().strip()

    if name in ["lenet", "lenet_avg"]:
        size = img_size or 64
        return LeNet5(in_ch=in_ch, img_size=size, num_classes=num_classes, pool="avg")
    elif name == "lenet_max":
        size = img_size or 64
        return LeNet5(in_ch=in_ch, img_size=size, num_classes=num_classes, pool="max")
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
            f"Không hỗ trợ mô hình '{model_name}'. Các tùy chọn hợp lệ: {list(MODEL_DEFAULTS.keys())}"
        )


def get_model_defaults(model_name: str):
    name = model_name.lower().strip()
    return MODEL_DEFAULTS.get(name, (64, 1e-3, 20))
