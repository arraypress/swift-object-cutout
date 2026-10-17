# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["torch==2.9.0", "torchvision==0.24.0", "numpy", "pillow", "huggingface_hub", "hydra-core==1.3.2",
#                 "sam-2 @ git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4",
#                 "coreai-torch==0.4.3", "coreai-core==1.0.0b3"]
# ///
"""SAM 2.1 tiny (Meta, Apache-2.0) → cutout-sam2-tiny-<dtype>.aimodel, for the click.

    SAM2_BUILD_CUDA=0 uv run Tools/export_sam2.py [--dtype float16|float32] [--out Tools/exports] [--check image.jpg …]

The network is the authors' own: facebookresearch/sam2 at a pinned commit, with the official SAM 2.1 tiny checkpoint
(facebook/sam2.1-hiera-tiny at a pinned revision). Nothing in it is rewritten; the two entry points are the image
predictor's own steps (SAM2ImagePredictor.set_image and ._predict), traced as they are.

Two entry points, so one encoded picture serves any number of clicks:

  encode  `image` [1,3,1024,1024] float32 RGB in 0…1 (stretched to the square by the host)
          → `image_embedding` [1,256,64,64], `feats_s0` [1,32,256,256], `feats_s1` [1,64,128,128]
          The ImageNet normalisation and SAM 2's no-memory embedding are inside.
  decode  those three, `point` [1,1,2] float32 (pixels of the 1024² square, x then y) and `label` [1,1] float32
          (1 = the object, 0 = not it) → `masks` [1,3,256,256] logits and `scores` [1,3] (predicted IoU).
          The prompt encoder and the mask decoder, multimask output, SAM's padding point added inside.

In a float16 export the weights and arithmetic are half; every input and output stays float32.

--check runs the official SAM2ImagePredictor and the two traced steps on each image (a click at the centre) in
float32 on the CPU and prints the difference, before anything is exported."""
import argparse, sys, time, shutil
from pathlib import Path
import numpy as np, torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
REPOSITORY = "facebook/sam2.1-hiera-tiny"
REVISION = "de431c4043854a71d8101e17995dfe596bf101a5"
CHECKPOINT = "sam2.1_hiera_tiny.pt"
CODE = "facebookresearch/sam2@2b90b9f"
SIZE = 1024

ap = argparse.ArgumentParser()
ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
ap.add_argument("--out", default=str(HERE / "exports"))
ap.add_argument("--weights", default=None, help="a folder to keep the checkpoint in (default: the HF cache)")
ap.add_argument("--check", nargs="*", default=[], help="images to compare the predictor and the traced steps on")
ap.add_argument("--no-export", action="store_true")
args = ap.parse_args()

from huggingface_hub import hf_hub_download
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

torch.set_num_threads(4)
ckpt = hf_hub_download(REPOSITORY, CHECKPOINT, revision=REVISION, local_dir=args.weights)
sam = build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", ckpt, device="cpu", mode="eval")
print(f"SAM 2.1 tiny @ {REVISION[:7]}: {sum(p.numel() for p in sam.parameters())/1e6:.1f} M parameters", flush=True)
FEAT_SIZES = [(256, 256), (128, 128), (64, 64)]       # SAM2ImagePredictor._bb_feat_sizes at 1024²
MEAN = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)


class Encode(nn.Module):
    """SAM2ImagePredictor.set_image after its resize: normalise, backbone, no-memory embedding, reshape."""
    def __init__(self, m, dtype):
        super().__init__(); self.m = m; self.dtype = dtype
        self.register_buffer("mean", MEAN.to(dtype)); self.register_buffer("std", STD.to(dtype))
    def forward(self, image):
        x = (image.to(self.dtype) - self.mean) / self.std
        backbone_out = self.m.forward_image(x)
        _, vision_feats, _, _ = self.m._prepare_backbone_features(backbone_out)
        if self.m.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.m.no_mem_embed
        feats = [f.permute(1, 2, 0).reshape(1, -1, *s) for f, s in zip(vision_feats[::-1], FEAT_SIZES[::-1])][::-1]
        return feats[2].to(torch.float32), feats[0].to(torch.float32), feats[1].to(torch.float32)


class Decode(nn.Module):
    """SAM2ImagePredictor._predict for one point, multimask: prompt encoder, mask decoder."""
    def __init__(self, m, dtype):
        super().__init__(); self.m = m; self.dtype = dtype
    def forward(self, image_embedding, feats_s0, feats_s1, point, label):
        d = self.dtype
        # The prompt encoder stays float32 (it is tiny, and its padding point is made float32 inside).
        sparse, dense = self.m.sam_prompt_encoder(points=(point, label.to(torch.int32)), boxes=None, masks=None)
        masks, scores, _, _ = self.m.sam_mask_decoder(
            image_embeddings=image_embedding.to(d), image_pe=self.m.sam_prompt_encoder.get_dense_pe().to(d),
            sparse_prompt_embeddings=sparse.to(d), dense_prompt_embeddings=dense.to(d), multimask_output=True,
            repeat_image=False, high_res_features=[feats_s0.to(d), feats_s1.to(d)])
        return masks.to(torch.float32), scores.to(torch.float32)


def square(path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    return im, torch.from_numpy(np.asarray(im.resize((SIZE, SIZE), Image.BILINEAR), np.float32) / 255).permute(2, 0, 1)[None]


if args.check:
    predictor = SAM2ImagePredictor(sam)
    enc, dec = Encode(sam, torch.float32).eval(), Decode(sam, torch.float32).eval()
    with torch.no_grad():
        for path in args.check:
            im, x = square(path)
            predictor.set_image(np.asarray(im))
            w, h = im.size
            _, s_ref, low_ref = predictor.predict(point_coords=np.array([[w / 2, h / 2]]), point_labels=np.array([1]),
                                                  multimask_output=True, return_logits=True)
            # The predictor resizes with its own transform; feed the traced steps its tensor so only the steps differ.
            x = predictor._transforms(np.asarray(im))[None]
            x = x * STD + MEAN
            e, s0, s1 = enc(x)
            m, s = dec(e, s0, s1, torch.tensor([[[SIZE / 2, SIZE / 2]]]), torch.ones(1, 1))
            # The predictor clamps its low-resolution logits to ±32; the export does not (the host thresholds).
            dm = np.abs(m[0].clamp(-32, 32).numpy() - low_ref).max()
            print(f"{Path(path).name}: |predictor − traced| masks {dm:.2e} scores {np.abs(s[0].numpy() - s_ref).max():.2e}",
                  flush=True)

if args.no_export:
    sys.exit(0)

import coreai_torch
from coreai.runtime import AIModelAssetMetadata
dtype = torch.float16 if args.dtype == "float16" else torch.float32
sam = sam.to(dtype)
sam.sam_prompt_encoder.float()
enc, dec = Encode(sam, dtype).eval(), Decode(sam, dtype).eval()
t0 = time.time()
converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
with torch.no_grad():
    x = torch.rand(1, 3, SIZE, SIZE)
    ex = torch.export.export(enc, (x,)).run_decompositions(coreai_torch.get_decomp_table())
    converter.add_exported_program(ex, input_names=["image"],
                                   output_names=["image_embedding", "feats_s0", "feats_s1"], entrypoint_name="encode")
    e, s0, s1 = enc(x)
    ex = torch.export.export(dec, (e, s0, s1, torch.tensor([[[300.0, 400.0]]]), torch.ones(1, 1))
                             ).run_decompositions(coreai_torch.get_decomp_table())
    converter.add_exported_program(ex, input_names=["image_embedding", "feats_s0", "feats_s1", "point", "label"],
                                   output_names=["masks", "scores"], entrypoint_name="decode")
    print(f"traced in {time.time()-t0:.0f} s", flush=True)
program = converter.to_coreai()
print(f"converted in {time.time()-t0:.0f} s", flush=True)
meta = AIModelAssetMetadata()
meta.author = "Ravi, Gabeur, Hu, Hu, Ryali, Ma, Khedr, Rädle, Rolland, Gustafson, et al. (Meta) — SAM 2.1 tiny; Core AI export by ObjectCutout"
meta.license = "Apache-2.0"
meta.model_description = (f"SAM 2.1 hiera tiny ({REPOSITORY}@{REVISION[:7]}, code {CODE}), {args.dtype}. "
                          f"'encode': image [1,3,{SIZE},{SIZE}] float32 RGB 0…1 → image_embedding, feats_s0, feats_s1. "
                          f"'decode': those + point [1,1,2] (pixels of the {SIZE}² square) + label [1,1] "
                          f"→ masks [1,3,256,256] logits, scores [1,3].")
meta.creation_date = int(time.time())
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
asset = out / f"cutout-sam2-tiny-{args.dtype}.aimodel"
if asset.exists(): shutil.rmtree(asset)
program.save_asset(asset, meta)
print(f"saved {asset} ({sum(f.stat().st_size for f in asset.rglob('*'))/1e6:.0f} MB) in {time.time()-t0:.0f} s", flush=True)
