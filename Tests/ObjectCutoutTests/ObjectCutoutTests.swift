//
//  ObjectCutoutTests.swift
//  ObjectCutoutTests
//

import CoreImage
import Foundation
import Testing
@testable import ObjectCutout

@Suite("Object cutout")
struct ObjectCutoutTests {

    @Test("The best of the three masks is the highest score, the first on a tie")
    func bestMask() {
        #expect(ClickMask.best([0.1, 0.9, 0.5]) == 1)
        #expect(ClickMask.best([0.7, 0.7, 0.2]) == 0)
        #expect(ClickMask.best([]) == nil)
    }

    @Test("A logit map lands on the picture's extent the right way up, ramped from the boundary")
    func clickMaskOrientation() throws {
        let side = 4, count = side * side
        // Channel 1: top two rows strongly on, bottom two off; channels 0 and 2 empty.
        var logits = [Float](repeating: -10, count: 3 * count)
        for i in 0..<(2 * side) { logits[count + i] = 10 }
        let extent = CGRect(x: 10, y: 20, width: 100, height: 60)
        let mask = try #require(ClickMask.image(logits: logits, index: 1, side: side, extent: extent))
        #expect(mask.extent == extent)
        var buf = [UInt8](repeating: 0, count: 100 * 60 * 4)
        CIContext().render(mask, toBitmap: &buf, rowBytes: 400, bounds: extent, format: .RGBA8, colorSpace: nil)
        // Lanczos rings a little at a 4-pixel map's edge; well inside each half it is settled.
        #expect(buf[(12 * 100 + 50) * 4] > 240, "top of the picture is the object")
        #expect(buf[(48 * 100 + 50) * 4] < 15, "bottom is not")
        // A channel that is not there is no mask.
        #expect(ClickMask.image(logits: logits, index: 3, side: side, extent: extent) == nil)
    }

    #if canImport(CoreAI)
    @Test("A missing click asset is a modelUnavailable error, not a crash")
    @available(macOS 27, iOS 27, *)
    func missingClickAsset() async {
        await #expect(throws: MatteError.self) {
            _ = try await SAM2Segmenter(contentsOf: URL(fileURLWithPath: "/nonexistent/cutout-sam2-tiny.aimodel"))
        }
    }

    @Test("The real click model lifts the disc it is clicked on",
          .enabled(if: ProcessInfo.processInfo.environment["CUTOUT_SAM2_LIVE"] != nil && ClickModel.isInstalled))
    @available(macOS 27, iOS 27, *)
    func liveClick() async throws {
        let (image, _) = EdgeTests.scene()
        let segmenter = try await SAM2Segmenter.shared()
        let found = try #require(try await segmenter.maskWithScore(for: image, at: CGPoint(x: 0.5, y: 0.5)))
        #expect(found.score > 0.8, "score \(found.score)")
        let size = EdgeTests.size
        var buf = [UInt8](repeating: 0, count: size * size * 4)
        CIContext().render(found.mask, toBitmap: &buf, rowBytes: size * 4, bounds: image.extent, format: .RGBA8, colorSpace: nil)
        var inter = 0, union = 0
        for y in 0..<size {
            for x in 0..<size {
                let dx = Double(x) + 0.5 - 128, dy = Double(y) + 0.5 - 128
                let truth = dx * dx + dy * dy <= 64 * 64, hit = buf[(y * size + x) * 4] > 127
                if truth && hit { inter += 1 }
                if truth || hit { union += 1 }
            }
        }
        #expect(Double(inter) / Double(max(union, 1)) > 0.95)
        // One encoding, two clicks: the background click is not the disc.
        let encoding = try await segmenter.encode(image)
        let disc = try await segmenter.predict(encoding, at: CGPoint(x: 0.5, y: 0.5))
        let corner = try await segmenter.predict(encoding, at: CGPoint(x: 0.05, y: 0.05))
        #expect(disc.logits[disc.best * 65536 + 128 * 256 + 128] > 0)
        #expect(corner.logits[corner.best * 65536 + 128 * 256 + 128] < 0)
        let cut = try #require(try await segmenter.cutout(from: image, at: CGPoint(x: 0.5, y: 0.5)))
        #expect(cut.bounds.width > 100 && cut.bounds.width < 160)
    }
    #endif

    /// The built-in mattes: a person photo yields a whole figure (not a
    /// garment), the subject lift finds the same figure, and a flat image
    /// yields nobody. Gated on a local stock photo; skips elsewhere.
    @Test func personAndSubjectMattesTakeWholeFigures() throws {
        let photo = URL(fileURLWithPath: ("~/Downloads/stock-people/businesswoman-walking.jpg" as NSString).expandingTildeInPath)
        guard FileManager.default.fileExists(atPath: photo.path) else { return }   // needs the local stock photo
        let image = try #require(CIImage(contentsOf: photo))
        let person = try #require(try Matte.cutout(from: image, mode: .person))
        #expect(person.bounds.height > image.extent.height * 0.5, "a whole person, not a garment")
        let mask = try #require(try Matte.personMask(for: image))
        #expect(mask.extent == image.extent)
        let subject = try #require(try Matte.cutout(from: image, mode: .subject))
        #expect(subject.bounds.height > image.extent.height * 0.4)
        let flat = CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 0.5)).cropped(to: CGRect(x: 0, y: 0, width: 256, height: 256))
        #expect(try Matte.personMask(for: flat) == nil, "a flat grey has no person in it")
    }

    /// A synthetic square cutout gains a contour outline just outside its
    /// edge, a shadow below, transparency in the corners, and lands centred
    /// on a square canvas when asked. Deterministic — no models.
    @Test func stickerDressesACutout() throws {
        let rect = CGRect(x: 64, y: 64, width: 128, height: 128)
        let full = CGRect(x: 0, y: 0, width: 256, height: 256)
        let red = CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(to: rect)
        let clear = CIImage(color: .clear).cropped(to: full)
        let cutout = ObjectCutout.Result(image: red.composited(over: clear), bounds: rect)

        let sticker = Sticker.make(from: cutout, options: Sticker.Options(border: 10, shadow: true))
        let context = CIContext()
        let w = Int(sticker.extent.width), h = Int(sticker.extent.height)
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        context.render(sticker, toBitmap: &buf, rowBytes: w * 4, bounds: sticker.extent,
                       format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
        func px(_ x: Int, _ y: Int) -> (r: UInt8, g: UInt8, b: UInt8, a: UInt8) {
            let i = ((h - 1 - y) * w + x) * 4   // bitmap rows are top-down; probe in bottom-left coords
            return (buf[i], buf[i + 1], buf[i + 2], buf[i + 3])
        }
        let margin = (sticker.extent.width - rect.width) / 2
        let cx = w / 2
        // Centre: the figure itself.
        let centre = px(cx, h / 2)
        #expect(centre.r > 200 && centre.a > 200, "the figure survives dressing")
        // Just outside the figure's left edge: the white outline, SOLID.
        let rim = px(Int(margin) - 4, h / 2)
        #expect(rim.r > 245 && rim.g > 245 && rim.b > 245 && rim.a > 245, "outline not solid: \(rim)")
        // Two pixels beyond the rim's outer edge: already clear — the
        // outline ends crisply, it does not fade.
        let past = px(Int(margin) - 10 - 4, h / 2)
        #expect(past.a < 40, "outline fades instead of ending: \(past)")
        // The very corner: transparent.
        #expect(px(1, h - 2).a < 10, "corners must stay clear")
        // Just under the outline ring: shadow alpha.
        let below = px(cx, Int(margin) - 10 - 3)
        #expect(below.a > 5 && below.r < 120, "shadow missing below: \(below)")

        let square = Sticker.make(from: cutout, options: Sticker.Options(border: 10, shadow: true, size: 512))
        #expect(square.extent == CGRect(x: 0, y: 0, width: 512, height: 512))
    }
}
