# swift-object-cutout

Tap any object in an image → a crisp, transparent-background **cutout** — on-device
and license-clean, via **SAM 2.1 tiny** (Segment Anything 2, Meta, Apache-2.0), exported to
Core AI. Unlike automatic subject-lift, you choose *which* object with a click.

- `SAM2Segmenter` — the click model (one Core AI asset, `encode` + `decode`):
  `maskWithScore(for:at:)` / `cutout(from:at:)` from one click, or `encode` once and
  `predict` per click (~5 ms each).
- `ObjectCutout` — turns a mask into a transparent-background cutout + the tight
  content bounds (crop there for a sticker).

```swift
let seg = try await SAM2Segmenter.shared()        // ClickModel.installedURL
if let r = try await seg.cutout(from: image, at: CGPoint(x: 0.5, y: 0.4)) {
    let sticker = r.image.cropped(to: r.bounds)   // transparent-bg cutout
}
```

Models (not committed; the library finds them, `cutout` downloads them on first use):
[arraypress/cutout-sam2-tiny](https://huggingface.co/arraypress/cutout-sam2-tiny) (63 MB) for
the click, [arraypress/cutout-birefnet](https://huggingface.co/arraypress/cutout-birefnet)
(463 MB) for the subject model.
Requirements: macOS 26 / iOS 26, Swift 6.2; the click and subject models need macOS 27 / iOS 27
(Core AI), and the subject falls back to Vision below that. The iOS Simulator SDK has no Core AI:
there both model paths are compiled out (`#if canImport(CoreAI)`), `SubjectModel.runsHere` is
false and `Matte` uses Vision.

## The click model (1.5) — SAM 2.1 tiny on Core AI

1.5 replaces Apple's three-part Core ML conversion (`apple/coreml-sam2.1-tiny`) with our own
Core AI export of the official SAM 2.1 tiny checkpoint (`Tools/export_sam2.py`, nothing in the
network rewritten), installed at `~/Library/Application Support/cutout-sam2-tiny/`. Measured on
60 clicks over 30 generated pictures (20 AI-generated rooms, 10 synthetic shape scenes),
against the PyTorch original in float32 on the same pixels:

| Runtime | Chosen-mask IoU, mean | worst | Clicks ≥ 0.99 | Same mask chosen | Score error, worst |
|---|---|---|---|---|---|
| Core AI float16 (1.5) | 0.9996 | 0.9959 | 60 of 60 | 60 of 60 | 0.006 |
| Apple's Core ML FLOAT16 (1.4) | 0.968 | 0.500 | 34 of 60 | 56 of 60 | 0.22 |

M3 Max, GPU: `encode` 0.15 s, `decode` 5 ms, first load (compile) well under a second, a
60 MB compiled copy. Through `cutout` a click run takes 0.6–1.0 s from start to finish
(4.5 s with Core ML, mostly loading); a 30-picture folder 0.51 s a picture (0.42 s), peak 1.2 GB
(1.4 GB). Needs macOS 27 / iOS 27; where the SDK has no Core AI (the iOS Simulator) the click
path is compiled out. `CUTOUT_SAM2_LIVE=1 swift test` runs the real model when installed.

## The subject model (1.4) — BiRefNet on Core AI

`Matte`'s subject and person mattes now come from **BiRefNet general** (Zheng et al., 2024;
MIT code and weights), exported to a Core AI `.aimodel` (float16, 463 MB) and hosted at
[arraypress/cutout-birefnet](https://huggingface.co/arraypress/cutout-birefnet) — wherever it is
installed (`~/Library/Application Support/cutout-birefnet/`) and the OS runs Core AI (macOS 27 /
iOS 27). Elsewhere they are Vision's, as before. The library finds the model; it does not download
it — `cutout` does that on first use (or `hf download`), and every tool on the Mac then shares the copy.

```swift
// Explicitly, async, loaded once per program:
let matter = try await SubjectMatter.shared()            // SubjectModel.installedURL
let mask = try await matter.mask(for: image)             // soft, the image's extent

// Or through Matte, which picks the engine (sync; waits on the model internally):
let r = try Matte.cutout(from: image, mode: .subject)                     // model if installed, else Vision
let v = try Matte.cutout(from: image, mode: .subject, engine: .vision)    // Vision only
let m = try Matte.subjectMask(for: image, engine: .model)                 // the model or MatteError
```

- **`.person`** keeps the model's alpha where Vision's person segmentation finds people (grown by
  5 % of the short side, softened): the model's edge, Vision's say over what counts as a person.
- **The edge:** the model's own alpha is kept — `.crisp` becomes `.soft` for it
  (`EdgeOptions.forModel`), the colour still decontaminated. Re-solving its alpha from colour made
  every set worse (table below); it is Vision's softer mask that crisp helps.
- **Measured** on AIM-500 (true alpha; not in this checkpoint's training data), 100 images a set,
  through this library in Swift. SAD and MSE are over the whole matte; the edge band is the strip
  round the subject where hair and fur live.

| Set | Engine | SAD | MSE×1e3 | Grad | Conn | IoU | Edge-band SAD | Edge-band MSE×1e3 |
|---|---|---|---|---|---|---|---|---|
| Objects | Vision subject, crisp (1.3.1) | 81.39 | 33.09 | 0.690 | 82.82 | 0.802 | 77.50 | 92.5 |
| Objects | BiRefNet, PyTorch fp32 (MPS) | 23.39 | 3.29 | 0.208 | 22.82 | 0.919 | 23.28 | 12.6 |
| Objects | BiRefNet .aimodel fp16, own alpha | 23.40 | 3.29 | 0.211 | 22.87 | 0.919 | 23.29 | 12.6 |
| Objects | BiRefNet .aimodel fp16 + crisp | 30.70 | 8.32 | 0.451 | 28.86 | 0.916 | 30.48 | 23.4 |
| Animals | Vision subject, crisp (1.3.1) | 17.17 | 5.79 | 0.309 | 17.15 | 0.976 | 15.64 | 47.7 |
| Animals | BiRefNet, PyTorch fp32 (MPS) | 6.10 | 0.96 | 0.070 | 5.46 | 0.991 | 6.01 | 9.1 |
| Animals | BiRefNet .aimodel fp16, own alpha | 6.12 | 0.99 | 0.074 | 5.50 | 0.991 | 6.03 | 9.4 |
| Animals | BiRefNet .aimodel fp16 + crisp | 9.56 | 1.90 | 0.144 | 9.06 | 0.988 | 9.13 | 17.1 |
| Portraits | Vision person, crisp (1.3.1 .person) | 24.91 | 10.63 | 0.224 | 24.45 | 0.962 | 11.88 | 36.0 |
| Portraits | Vision subject, crisp | 15.24 | 5.54 | 0.240 | 15.08 | 0.975 | 12.53 | 39.7 |
| Portraits | BiRefNet .aimodel fp16 (.subject) | 4.86 | 0.62 | 0.058 | 4.19 | 0.992 | 4.83 | 6.7 |
| Portraits | BiRefNet gated by Vision person (.person) | 7.52 | 1.55 | 0.067 | 7.08 | 0.989 | 6.49 | 10.5 |
| Portraits | BiRefNet .aimodel fp16 + crisp | 7.69 | 1.39 | 0.127 | 7.12 | 0.989 | 7.43 | 14.2 |

  Vision found no subject in 4 of the 100 objects (counted as empty). The model won by SAD on 99
  of 100 objects, 100 of 100 animals and 99 of 100 portraits; glass is the hardest for both (SAD
  211 for Vision, 51 for the model). The float16 asset matches PyTorch float32 (mean alpha
  difference 1.9 × 10⁻⁵ at 1024²). M3 Max: first load compiles for ~15 s and keeps a 553 MB
  compiled copy in the Core AI cache; then about 1–1.5 s a photo including the edge finish, and
  3–3.8 GB peak for a folder (PyTorch on MPS: up to 10 GB). The export script is
  `Tools/export_birefnet.py`: torchvision's deformable convolution, which Core AI lacks, is
  rewritten as gathers and matrix products (alpha within 4.6 × 10⁻⁶ of the original).

## Built-in mattes — no models, no click

`Matte` uses the segmentation models the OS ships (with `engine: .vision`, or wherever the
subject model is not installed): `personMask` /
`cutout(from:mode:.person)` mattes every human in the frame whole (the
remove-the-background case a torso click answers wrongly — it returns
the garment), and `.subject` lifts whatever the system judges salient.
Both share the click path's edge treatment, and neither needs the SAM
model.

```swift
if let r = try Matte.cutout(from: image, mode: .person) {
    let figure = r.image.cropped(to: r.bounds)
}
```

## Edges — crisp by default

A model's mask is soft at the edge, and the photo's pixels there are part
subject, part background. Kept as they are, that mix is a halo: a rim of the
old background's colour round the subject. 1.3 finishes every edge:

- **Decontaminated, always.** Near the mask's edge each pixel is
  `I = a·F + (1 − a)·B`. F and B are estimated for each pixel from the
  confidently-inside and confidently-outside pixels near it (background
  samples that do not look like their neighbours — a strand tip the mask
  missed — are left out), and the background's share is taken out of the
  colour.
- **`.crisp` (the default):** where F and B differ enough to tell apart, the
  alpha is solved from the pixel's colour instead of taken from the mask, so
  a strand of hair keeps its own width and the gaps between strands open;
  background showing through inside the mask, near its outline, joins that
  band. Elsewhere the mask is steepened to about a pixel of anti-aliasing.
- **`.soft`:** the model's own mask, decontaminated. **`.feather(N)`:** the
  mask blurred N px, decontaminated.
- **`choke`:** pulls the finished edge in N px.

```swift
let r = try Matte.cutout(from: image, mode: .subject)                          // crisp
let s = try Matte.cutout(from: image, mode: .subject, edge: EdgeOptions(edge: .soft, choke: 1))
CutoutEdge("feather:4")                                                         // from text
```

The 1.2 `feather:` arguments still compile (deprecated): `nil` now gives the
crisp edge, a number a feathered one.

**Measured** on five generated subjects with known truth (1536², Vision's
subject mask; a disc on green, a figure with 420 hair strands on a sky
gradient, a blonde curl mass on magenta, a white mug on dark red, a pale box on
off-white). *Width* is the mean 10–90 % alpha transition in px; *spill* is how
far the colour of the visible outer edge sits toward the real background (0 =
the subject's colour, 1 = the background's); *hair* is the share of strand
pixels kept opaque; *holes* the share of solid subject pixels lost.

| Subject | 1.2.1 width | crisp width | 1.2.1 spill | crisp spill | 1.2.1 hair | crisp hair | crisp holes |
|---|---|---|---|---|---|---|---|
| disc on green | 21.4 | 0.9 | 0.98 | 0.03 | – | – | 0.00 % |
| figure, sky | 27.9 | 1.0 | 0.96 | 0.09 | 0.86 | 0.95 | 0.09 % |
| curls, magenta | 29.4 | 1.1 | 0.86 | 0.18 | 0.74 | 0.84 | 0.09 % |
| mug, dark red | 20.1 | 0.5 | 0.99 | 0.01 | – | – | 0.00 % |
| pale on off-white | 19.6 | 0.8 | 0.84 | 0.11 | – | – | 0.27 % |

`soft` measured 5–6 px wide with spill 0.06–0.35; `feather:4` 12–14 px,
spill 0.03–0.15. The pale box is the hard case — subject and background
nearly the same colour (0.27 % of its pixels lost at the edge, against 0.67 %
for 1.2.1).

1.3.1: on a generated portrait (grey hair against a grey wall) 1.3.0's gap
search punched blotches into the hair where its grey matched the wall. Gaps
are now looked for only where subject and background differ by 0.2 or more,
and only thin ones (a few pixels, found by an opening) count — a wide patch of
the background's colour inside the mask is the subject. The colour solve at
the edge needs 0.15 of separation (was 0.1). Processing is on the CPU, on
plain pixel buffers: deterministic, 0.2–0.4 s at 1536² in a release build.

## License

MIT — see [LICENSE](LICENSE).

Bundled or downloaded models keep their own licences; see the notes above where a model is named.
