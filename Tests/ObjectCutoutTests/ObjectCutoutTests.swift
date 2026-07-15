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
}
