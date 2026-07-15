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
