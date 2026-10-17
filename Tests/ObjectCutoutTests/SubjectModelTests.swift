//
//  SubjectModelTests.swift
//  ObjectCutoutTests
//
//  What the subject model is given and what comes back, without the model:
//  planar RGB in the right order and orientation, the square alpha landing
//  the right way up on the picture's extent, and the engine choice. With
//  CUTOUT_BIREFNET_LIVE=1 and the model installed, the real model on a
//  generated subject with known truth.
//

import CoreGraphics
import CoreImage
import Foundation
import Testing
@testable import ObjectCutout

@Suite("Subject model")
struct SubjectModelTests {

    /// A `width`×`height` picture: red top half, blue bottom half.
    static func halves(width: Int, height: Int) -> CGImage {
        let c = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                          space: CGColorSpace(name: CGColorSpace.sRGB)!,
                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        c.setFillColor(CGColor(srgbRed: 0, green: 0, blue: 1, alpha: 1))
        c.fill(CGRect(x: 0, y: 0, width: width, height: height / 2))                  // CG: bottom half
        c.setFillColor(CGColor(srgbRed: 1, green: 0, blue: 0, alpha: 1))
        c.fill(CGRect(x: 0, y: height / 2, width: width, height: height - height / 2)) // top half
        return c.makeImage()!
    }

    @Test("The input is planar RGB in 0…1, top row first, stretched to the square")
    func planarInput() throws {
        let side = 8
        let rgb = try #require(MattePixels.planarRGB(of: Self.halves(width: 40, height: 20), side: side))
        #expect(rgb.count == 3 * side * side)
        let plane = side * side
        // Top-left pixel: red. Bottom-left: blue.
        #expect(rgb[0] == 1 && rgb[plane] == 0 && rgb[2 * plane] == 0)
        let bottom = (side - 1) * side
        #expect(rgb[bottom] == 0 && rgb[plane + bottom] == 0 && rgb[2 * plane + bottom] == 1)
        #expect(rgb.allSatisfy { (0...1).contains($0) })
    }

    @Test("The square alpha lands on the picture's extent the right way up")
    func maskOrientation() {
        let side = 4
        // Top two rows on, bottom two off.
        let alpha: [Float] = (0..<(side * side)).map { $0 < 2 * side ? 1 : 0 }
        let extent = CGRect(x: 0, y: 0, width: 100, height: 60)
        let mask = MattePixels.mask(alpha: alpha, side: side, extent: extent)
        #expect(mask.extent == extent)
        var buf = [UInt8](repeating: 0, count: 100 * 60 * 4)
        CIContext().render(mask, toBitmap: &buf, rowBytes: 400, bounds: extent, format: .RGBA8, colorSpace: nil)
        // Bitmap rows from the top: the first row is the top of the picture.
        #expect(buf[0] > 250)
        #expect(buf[(59 * 100) * 4] < 5)
        // Values past 0…1 are clamped.
        let wild = MattePixels.mask(alpha: [Float](repeating: 3, count: 16), side: 4, extent: extent)
        var one = [UInt8](repeating: 0, count: 4)
        CIContext().render(wild, toBitmap: &one, rowBytes: 4, bounds: CGRect(x: 50, y: 30, width: 1, height: 1),
                           format: .RGBA8, colorSpace: nil)
        #expect(one[0] == 255)
    }

    @Test("Vision is never the model; the model by name needs it installed")
    func engineChoice() throws {
        #expect(try Matte.usesModel(.vision) == false)
        #expect(try Matte.usesModel(.automatic) == (SubjectModel.runsHere && SubjectModel.isInstalled))
        if SubjectModel.runsHere && SubjectModel.isInstalled {
            #expect(try Matte.usesModel(.model))
        } else {
            #expect(throws: MatteError.self) { try Matte.usesModel(.model) }
        }
        #expect(SubjectModel.installedURL.lastPathComponent == SubjectModel.assetName)
        #expect(SubjectModel.installedURL.deletingLastPathComponent().lastPathComponent == SubjectModel.sharedFolder)
    }

    #if canImport(CoreAI)
    @Test("A missing asset is a modelUnavailable error, not a crash")
    @available(macOS 27, iOS 27, *)
    func missingAsset() async {
        await #expect(throws: MatteError.self) {
            _ = try await SubjectMatter(contentsOf: URL(fileURLWithPath: "/nonexistent/cutout-birefnet.aimodel"))
        }
    }

    @Test("The real model lifts a disc off green, through the sync and async paths",
          .enabled(if: ProcessInfo.processInfo.environment["CUTOUT_BIREFNET_LIVE"] != nil && SubjectModel.isInstalled))
    @available(macOS 27, iOS 27, *)
    func live() async throws {
        let (image, _) = EdgeTests.scene()
        let matter = try await SubjectMatter.shared()
        let mask = try await matter.mask(for: image)
        let size = EdgeTests.size
        var buf = [UInt8](repeating: 0, count: size * size * 4)
        CIContext().render(mask, toBitmap: &buf, rowBytes: size * 4, bounds: image.extent, format: .RGBA8, colorSpace: nil)
        // Truth: the disc at (64, 64) 128×128 — top-left rows equal bottom-left here, it is centred.
        var inter = 0, union = 0
        for y in 0..<size {
            for x in 0..<size {
                let dx = Double(x) + 0.5 - 128, dy = Double(y) + 0.5 - 128
                let truth = dx * dx + dy * dy <= 64 * 64
                let found = buf[(y * size + x) * 4] > 127
                if truth && found { inter += 1 }
                if truth || found { union += 1 }
            }
        }
        let iou = Double(inter) / Double(max(union, 1))
        #expect(iou > 0.95, "IoU \(iou)")
        // The synchronous Matte path reaches the same model.
        let viaMatte = try #require(try Matte.subjectMask(for: image, engine: .model))
        #expect(viaMatte.extent == image.extent)
    }
    #endif
}
