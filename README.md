# swift-object-cutout

Tap any object in an image → a crisp, transparent-background **cutout** — on-device
and license-clean, via **SAM 2** (Segment Anything 2, Apache-2.0; Apple's Core ML
conversion). Unlike automatic subject-lift, you choose *which* object with a click.

- `SAM2Segmenter` — wraps the three SAM 2 tiny Core ML models (image encoder,
  prompt encoder, mask decoder); `mask(for:at:)` runs the pipeline from one click.
- `ObjectCutout` — turns the mask into a transparent-background cutout + the tight
  content bounds (crop there for a sticker).

```swift
let seg = try SAM2Segmenter(encoderURL: …, promptEncoderURL: …, decoderURL: …)
if let r = ObjectCutout(segmenter: seg).cutout(from: image, at: CGPoint(x: 0.5, y: 0.4)) {
    let sticker = r.image.cropped(to: r.bounds)   // transparent-bg cutout
}
```

Models: `apple/coreml-sam2.1-tiny` on Hugging Face (~76 MB, not committed).
Requirements: macOS 26 / iOS 26, Swift 6.2.

## Built-in mattes — no models, no click

`Matte` uses the segmentation models the OS ships: `personMask` /
`cutout(from:mode:.person)` mattes every human in the frame whole (the
remove-the-background case a torso click answers wrongly — it returns
the garment), and `.subject` lifts whatever the system judges salient.
Both share the click path's edge treatment, and neither needs the SAM
models.

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
