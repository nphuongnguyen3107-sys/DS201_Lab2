from models import LeNet5, build_model, n_params

__all__ = ["LeNet5", "build_model", "n_params"]

if __name__ == "__main__":
    import torch
    m = LeNet5(in_ch=3, img_size=64, num_classes=21)
    x = torch.rand(2, 3, 64, 64)
    print("LeNet-5:")
    print("  Input shape:", tuple(x.shape))
    print("  Output shape:", tuple(m(x).shape))
    print("  Total params:", f"{n_params(m):,}")
    assert n_params(m) == 339421
    print("  OK!")
