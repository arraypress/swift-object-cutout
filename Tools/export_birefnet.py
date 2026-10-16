# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.9.0", "torchvision==0.24.0", "transformers==4.57.1", "timm", "kornia", "einops",
#                 "numpy", "pillow", "coreai-torch==0.4.3", "coreai-core==1.0.0b3"]
# ///
"""BiRefNet general (Zheng et al., CAAI AIR 2024; MIT) → cutout-birefnet-<dtype>.aimodel.

    uv run Tools/export_birefnet.py [--dtype float16|float32] [--out Tools/exports] [--check image.jpg …]

The network is the authors' own (ZhengPeng7/BiRefNet at a pinned revision, loaded with its remote code); one
change, made for the runtime and checked against the original: torchvision's `deform_conv2d`, which Core AI does
not have, is rewritten as what it computes — each kernel tap's bilinear sample (four gathers from the zero-padded
feature map), scaled by the modulator, then a matrix product with that tap's weights, summed over the taps.
Sample positions are computed in float32 even in a float16 export: at 256 px a half float steps by 0.125, which
would move every bilinear weight.

One entry point, `main`: `image` [1,3,1024,1024] float32 RGB in 0…1 (resized by the host) → `alpha`
[1,1,1024,1024] float32 in 0…1. The ImageNet normalisation and the final sigmoid are inside. In a float16 export
the weights and arithmetic are half; input and output stay float32, so the host is the same for both.

--check runs the original network (torchvision's deform_conv2d) and the rewritten one on each image in float32 on
the CPU and prints the alpha difference, before anything is exported."""
import argparse, os, sys, time, shutil
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPOSITORY = "ZhengPeng7/BiRefNet"
REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
SIZE = 1024

ap = argparse.ArgumentParser()
ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
ap.add_argument("--out", default=str(HERE / "exports"))
ap.add_argument("--check", nargs="*", default=[], help="images to compare original vs rewritten on (float32, CPU)")
ap.add_argument("--no-export", action="store_true")
args = ap.parse_args()

from transformers import AutoModelForImageSegmentation
torch.set_num_threads(4)
model = AutoModelForImageSegmentation.from_pretrained(REPOSITORY, trust_remote_code=True, revision=REVISION).float().eval()
remote = sys.modules[type(model).__module__]
print(f"BiRefNet general @ {REVISION[:7]}: {sum(p.numel() for p in model.parameters())/1e6:.1f} M parameters", flush=True)


def deform_conv2d(x, offset, weight, bias, padding, mask):
    """torchvision.ops.deform_conv2d (stride 1, dilation 1, one offset group), from gathers and matmuls."""
    _, C, H, W = x.shape
    O, _, kh, kw = weight.shape
    pad = padding if isinstance(padding, int) else padding[0]
    Ho, Wo = H + 2 * pad - kh + 1, W + 2 * pad - kw + 1
    Hp, Wp = H + 2, W + 2
    # One row of zeros round the map: every corner that falls outside reads 0, as torchvision's does.
    table = F.pad(x, (1, 1, 1, 1)).reshape(C, Hp * Wp).t()                      # [Hp*Wp, C]
    off = offset.float().reshape(kh * kw, 2, Ho, Wo)                              # (dy, dx) per tap
    mod = mask.float().reshape(kh * kw, Ho * Wo, 1)
    ys = torch.arange(Ho, dtype=torch.float32).reshape(Ho, 1) - pad
    xs = torch.arange(Wo, dtype=torch.float32).reshape(1, Wo) - pad
    out = None
    for k in range(kh * kw):
        ky, kx = divmod(k, kw)
        py = ys + ky + off[k, 0]
        px = xs + kx + off[k, 1]
        y0, x0 = torch.floor(py), torch.floor(px)
        ly, lx = py - y0, px - x0
        y0c, y1c = y0.clamp(-1, H) + 1, (y0 + 1).clamp(-1, H) + 1
        x0c, x1c = x0.clamp(-1, W) + 1, (x0 + 1).clamp(-1, W) + 1
        sample = None
        for yc, xc, w in ((y0c, x0c, (1 - ly) * (1 - lx)), (y0c, x1c, (1 - ly) * lx),
                          (y1c, x0c, ly * (1 - lx)), (y1c, x1c, ly * lx)):
            index = (yc * Wp + xc).reshape(-1).to(torch.int64)
            term = F.embedding(index, table) * (w.reshape(-1, 1) * mod[k]).to(x.dtype)
            sample = term if sample is None else sample + term
        part = sample @ weight[:, :, ky, kx].t()                                  # [Ho*Wo, O]
        out = part if out is None else out + part
    out = out.t().reshape(1, O, Ho, Wo)
    return out if bias is None else out + bias.reshape(1, O, 1, 1)


def rewritten_forward(self, x):
    offset = self.offset_conv(x)
    modulator = 2. * torch.sigmoid(self.modulator_conv(x))
    return deform_conv2d(x, offset, self.regular_conv.weight, self.regular_conv.bias, self.padding, modulator)


original_forward = remote.DeformableConv2d.forward
MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


class Net(nn.Module):
    """0…1 RGB in, alpha out: the normalisation and the sigmoid inside, the last (full-size) prediction only."""
    def __init__(self, m, dtype):
        super().__init__(); self.m = m.to(dtype); self.dtype = dtype
        self.register_buffer("mean", MEAN.to(dtype)); self.register_buffer("std", STD.to(dtype))
    def forward(self, image):
        x = (image.to(self.dtype) - self.mean) / self.std
        return self.m(x)[-1].sigmoid().to(torch.float32)


if args.check:
    from PIL import Image
    with torch.no_grad():
        for path in args.check:
            im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
            x = torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)[None]
            remote.DeformableConv2d.forward = original_forward
            a = Net(model, torch.float32)(x)
            remote.DeformableConv2d.forward = rewritten_forward
            b = Net(model, torch.float32)(x)
            d = (a - b).abs()
            print(f"{Path(path).name}: |original − rewritten| max {d.max():.2e} mean {d.mean():.2e}", flush=True)

if args.no_export:
    sys.exit(0)

import coreai_torch
from coreai.runtime import AIModelAssetMetadata
remote.DeformableConv2d.forward = rewritten_forward
dtype = torch.float16 if args.dtype == "float16" else torch.float32
net = Net(model, dtype).eval()
t0 = time.time()
converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
with torch.no_grad():
    ex = torch.export.export(net, (torch.rand(1, 3, SIZE, SIZE),)).run_decompositions(coreai_torch.get_decomp_table())
    print(f"traced in {time.time()-t0:.0f} s", flush=True)
    converter.add_exported_program(ex, input_names=["image"], output_names=["alpha"], entrypoint_name="main")
program = converter.to_coreai()
print(f"converted in {time.time()-t0:.0f} s", flush=True)
meta = AIModelAssetMetadata()
meta.author = "Zheng, Gao, Fan, Liu, Laaksonen, Ouyang, Sebe — BiRefNet (CAAI AIR 2024), general; Core AI export by ObjectCutout"
meta.license = "MIT"
meta.model_description = (f"BiRefNet general ({REPOSITORY}@{REVISION[:7]}), {args.dtype}. Entry 'main': image [1,3,{SIZE},{SIZE}] "
                          f"float32 RGB 0…1 → alpha [1,1,{SIZE},{SIZE}] float32 0…1. Resize to {SIZE}² and back in the host.")
meta.creation_date = int(time.time())
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
asset = out / f"cutout-birefnet-{args.dtype}.aimodel"
if asset.exists(): shutil.rmtree(asset)
program.save_asset(asset, meta)
print(f"saved {asset} ({sum(f.stat().st_size for f in asset.rglob('*'))/1e6:.0f} MB) in {time.time()-t0:.0f} s", flush=True)
