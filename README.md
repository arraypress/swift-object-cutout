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

## License

MIT — see [LICENSE](LICENSE).

Bundled or downloaded models keep their own licences; see the notes above where a model is named.
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


