from __future__ import annotations

import argparse
import platform


parser = argparse.ArgumentParser()
parser.add_argument("--expect", choices=("cpu", "cuda"), default="cpu")
args = parser.parse_args()

try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch import failed: {type(exc).__name__}: {exc}") from exc

print(f"Python: {platform.python_version()} ({platform.machine()})")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
layer = torch.nn.Linear(2, 3)
y = layer(x)
loss = y.square().mean()
loss.backward()
print(f"CPU forward/backward test passed: loss={loss.item():.6f}")

if args.expect == "cuda":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False")
    device = torch.device("cuda")
    torch.nn.Linear(2, 3).to(device)(x.to(device)).sum().backward()
    print(f"CUDA test passed on {torch.cuda.get_device_name(0)}")

print("PyTorch verification passed")
