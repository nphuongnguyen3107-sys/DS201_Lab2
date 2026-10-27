"""
Package cnn_project: Các mô-đun cho bài thực hành mạng nơ-ron tích chập (DS201 - Lab 2).
"""

from .data import build_transforms, build_train_val_loaders, build_test_loader
from .model import (
    LeNet5,
    InceptionBlock,
    GoogLeNetReduced,
    BasicBlock,
    ResNet18,
    PretrainedResnet,
    build_model,
    get_model_defaults,
    n_params,
    show,
    layer_table
)
from .engine import (
    train_one_epoch,
    evaluate,
    train_network,
    plot_loss_curves,
    plot_confusion_matrix,
    compute_metrics
)

__all__ = [
    "build_transforms",
    "build_train_val_loaders",
    "build_test_loader",
    "LeNet5",
    "InceptionBlock",
    "GoogLeNetReduced",
    "BasicBlock",
    "ResNet18",
    "PretrainedResnet",
    "build_model",
    "get_model_defaults",
    "n_params",
    "show",
    "layer_table",
    "train_one_epoch",
    "evaluate",
    "train_network",
    "plot_loss_curves",
    "plot_confusion_matrix",
    "compute_metrics"
]
