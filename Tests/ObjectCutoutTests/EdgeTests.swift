//
//  EdgeTests.swift
//  ObjectCutoutTests
//
//  The edge finish on a generated subject with known truth: an orange disc
//  on green, cut with a deliberately soft mask (the disc's shape blurred
//  6 px, as a model's mask is soft). No models.
//

import CoreGraphics
import CoreImage
import Foundation
import Testing
@testable import ObjectCutout

@Suite("Cutout edges")
struct EdgeTests {

    static let size = 256
    static let green: (Double, Double, Double) = (0.18, 0.76, 0.29)
    static let orange: (Double, Double, Double) = (0.95, 0.5, 0.15)

    /// The picture (orange disc on green) and a soft mask of the disc.
    static func scene() -> (image: CIImage, mask: CIImage) {
        let s = size
        func draw(_ body: (CGContext) -> Void) -> CGImage {
            let c = CGContext(data: nil, width: s, height: s, bitsPerComponent: 8, bytesPerRow: s * 4,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            body(c)
            return c.makeImage()!
        }
        let disc = CGRect(x: 64, y: 64, width: 128, height: 128)
        let picture = draw { c in
            c.setFillColor(CGColor(srgbRed: green.0, green: green.1, blue: green.2, alpha: 1))
            c.fill(CGRect(x: 0, y: 0, width: s, height: s))
            c.setFillColor(CGColor(srgbRed: orange.0, green: orange.1, blue: orange.2, alpha: 1))
            c.fillEllipse(in: disc)
        }
        let hard = draw { c in
            c.setFillColor(CGColor(gray: 0, alpha: 1)); c.fill(CGRect(x: 0, y: 0, width: s, height: s))
            c.setFillColor(CGColor(gray: 1, alpha: 1)); c.fillEllipse(in: disc.insetBy(dx: -3, dy: -3))
        }
        let extent = CGRect(x: 0, y: 0, width: s, height: s)
        let soft = CIImage(cgImage: hard).clampedToExtent()
            .applyingFilter("CIGaussianBlur", parameters: ["inputRadius": 6]).cropped(to: extent)
        return (CIImage(cgImage: picture), soft)
    }

    static func pixels(_ image: CIImage) -> [UInt8] {
        var buf = [UInt8](repeating: 0, count: size * size * 4)
        CIContext().render(image, toBitmap: &buf, rowBytes: size * 4, bounds: CGRect(x: 0, y: 0, width: size, height: size),
                           format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        return buf
    }

    /// Pixels with 0.1 < alpha < 0.9, and the mean share of green in the
    /// colour of every visible edge pixel (0 = orange, 1 = green).
    static func measure(_ buf: [UInt8]) -> (partial: Int, greenShare: Double, opaque: Int) {
        var partial = 0, opaque = 0, share = 0.0, count = 0.0
        for i in 0..<(size * size) {
            let a = Double(buf[i * 4 + 3]) / 255
            if a > 0.1 && a < 0.9 { partial += 1 }
            if a >= 0.5 { opaque += 1 }
            guard a > 0.05 && a < 0.98 else { continue }
            let c = (Double(buf[i * 4]) / 255 / a, Double(buf[i * 4 + 1]) / 255 / a, Double(buf[i * 4 + 2]) / 255 / a)
            let toOrange = abs(c.0 - orange.0) + abs(c.1 - orange.1) + abs(c.2 - orange.2)
            let toGreen = abs(c.0 - green.0) + abs(c.1 - green.1) + abs(c.2 - green.2)
            share += toOrange / max(0.001, toOrange + toGreen)
            count += 1
        }
        return (partial, count > 0 ? share / count : 0, opaque)
    }

    @Test func crispIsTightAndCarriesNoBackground() throws {
        let (image, mask) = Self.scene()
        let crisp = Self.measure(Self.pixels(try #require(Matte.cutout(from: image, mask: mask)).image))
        let feathered = Self.measure(Self.pixels(try #require(Matte.cutout(from: image, mask: mask, edge: EdgeOptions(edge: .feather(4)))).image))
        // The disc's outline is ~400 px round: crisp keeps about a pixel of
        // anti-aliasing (≤ 1.6 px a pixel of outline), the feather many.
        #expect(Double(crisp.partial) / 400 < 1.6, "crisp edge \(crisp.partial) px")
        #expect(feathered.partial > crisp.partial * 3)
        #expect(crisp.greenShare < 0.1, "green left in the edge: \(crisp.greenShare)")
        #expect(feathered.greenShare < 0.25, "a feathered edge is decontaminated too: \(feathered.greenShare)")
    }

    @Test func chokePullsTheEdgeIn() throws {
        let (image, mask) = Self.scene()
        let plain = Self.measure(Self.pixels(try #require(Matte.cutout(from: image, mask: mask)).image))
        let choked = Self.measure(Self.pixels(try #require(Matte.cutout(from: image, mask: mask, edge: EdgeOptions(choke: 3))).image))
        // 3 px round a 128 px disc: about 2·π·64·3 ≈ 1,200 px fewer.
        #expect(plain.opaque - choked.opaque > 900, "\(plain.opaque) → \(choked.opaque)")
    }

    @Test func edgesParse() {
        #expect(CutoutEdge("crisp") == .crisp)
        #expect(CutoutEdge(" Soft ") == .soft)
        #expect(CutoutEdge("feather:4") == .feather(4))
        #expect(CutoutEdge("feather:2.5")?.description == "feather:2.5")
        #expect(CutoutEdge("feather:-1") == nil)
        #expect(CutoutEdge("fuzzy") == nil)
    }

    @Test func dilateAndErodeAreSquare() {
        var marks = [Bool](repeating: false, count: 25)
        marks[12] = true
        let grown = EdgeRefiner.dilate(marks, radius: 1, width: 5, height: 5)
        #expect(grown.filter { $0 }.count == 9)
        var alpha = [Float](repeating: 1, count: 25)
        alpha[12] = 0
        #expect(EdgeRefiner.erode(alpha, radius: 1, width: 5, height: 5).filter { $0 == 0 }.count == 9)
    }
}
