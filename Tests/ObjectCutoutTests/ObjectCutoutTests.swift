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

    private func env(_ k: String) -> String? { ProcessInfo.processInfo.environment[k] }

    @Test("Cut out the object at a click and write a PNG",
          .enabled(if: ProcessInfo.processInfo.environment["SAM2_RENDER"] != nil))
    func cutoutRender() throws {
        guard let dir = env("SAM2_DIR") else { print("[skip] set SAM2_DIR"); return }
        let base = URL(fileURLWithPath: dir)
        let seg = try SAM2Segmenter(
            encoderURL: base.appendingPathComponent("SAM2_1TinyImageEncoderFLOAT16.mlpackage"),
            promptEncoderURL: base.appendingPathComponent("SAM2_1TinyPromptEncoderFLOAT16.mlpackage"),
            decoderURL: base.appendingPathComponent("SAM2_1TinyMaskDecoderFLOAT16.mlpackage"))

        let imgPath = env("SAM2_IMAGE") ?? (NSHomeDirectory() + "/Downloads/depth_frame.png")
        guard let frame = CIImage(contentsOf: URL(fileURLWithPath: imgPath)) else { print("[skip] no image"); return }

        let cutout = ObjectCutout(segmenter: seg)
        let t0 = Date().timeIntervalSinceReferenceDate
        guard let result = cutout.cutout(from: frame, at: CGPoint(x: 0.5, y: 0.42)) else {
            Issue.record("no cutout"); return
        }
        let ms = (Date().timeIntervalSinceReferenceDate - t0) * 1000
        print("CUTOUT bounds \(result.bounds) in \(Int(ms))ms")

        let cropped = result.image.cropped(to: result.bounds)
        let ctx = CIContext()
        try ctx.writePNGRepresentation(of: cropped,
                                       to: URL(fileURLWithPath: NSHomeDirectory() + "/Downloads/sam2_cutout.png"),
                                       format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
        print("WROTE ~/Downloads/sam2_cutout.png")
    }

    @Test("Track an object across frames",
          .enabled(if: ProcessInfo.processInfo.environment["SAM2_RENDER"] != nil))
    func trackObject() async throws {
        guard let dir = env("SAM2_DIR") else { print("[skip]"); return }
        let base = URL(fileURLWithPath: dir)
        let seg = try SAM2Segmenter(
            encoderURL: base.appendingPathComponent("SAM2_1TinyImageEncoderFLOAT16.mlpackage"),
            promptEncoderURL: base.appendingPathComponent("SAM2_1TinyPromptEncoderFLOAT16.mlpackage"),
            decoderURL: base.appendingPathComponent("SAM2_1TinyMaskDecoderFLOAT16.mlpackage"))
        let video = URL(fileURLWithPath: env("SAM2_VIDEO") ?? (NSHomeDirectory() + "/Downloads/body_test.mp4"))
        guard FileManager.default.fileExists(atPath: video.path) else { print("[skip] no video"); return }

        let track = try await SAM2VideoTracker.track(videoURL: video, startPoint: CGPoint(x: 0.5, y: 0.42), using: seg, fps: 1.5)
        print("TRACK \(track.count) frames; first center \(track.first?.center ?? .zero), last \(track.last?.center ?? .zero)")
        #expect(track.count >= 2)
        // The object stays roughly centered (didn't drift off to a corner).
        if let c = track.last?.center {
            #expect(c.x > 0.2 && c.x < 0.8 && c.y > 0.1 && c.y < 0.9, "track drifted: \(c)")
        }
    }

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
        // Just outside the figure's left edge: the white outline.
        let rim = px(Int(margin) - 4, h / 2)
        #expect(rim.r > 200 && rim.g > 200 && rim.b > 200 && rim.a > 200, "outline missing: \(rim)")
        // The very corner: transparent.
        #expect(px(1, h - 2).a < 10, "corners must stay clear")
        // Just under the outline ring: shadow alpha.
        let below = px(cx, Int(margin) - 10 - 3)
        #expect(below.a > 5 && below.r < 120, "shadow missing below: \(below)")

        let square = Sticker.make(from: cutout, options: Sticker.Options(border: 10, shadow: true, size: 512))
        #expect(square.extent == CGRect(x: 0, y: 0, width: 512, height: 512))
    }
}
