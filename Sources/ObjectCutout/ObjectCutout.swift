//
//  ObjectCutout.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//

import CoreGraphics
import CoreImage
import Foundation

/// Tap a point in an image, get a transparent-background cutout of that object.
/// Wraps ``SAM2Segmenter`` and turns its mask into an alpha cutout plus the
/// tight content rect (for making a sticker).
public struct ObjectCutout: Sendable {

    public let segmenter: SAM2Segmenter
    private let ciContext = CIContext()

    public init(segmenter: SAM2Segmenter) {
        self.segmenter = segmenter
    }

    /// A cutout: the object on transparency (full extent) + its tight bounds.
    public struct Result: Sendable {
        /// The object masked onto a transparent background, at the source extent.
        public let image: CIImage
        /// Tight content rectangle in source pixels (crop here for a sticker).
        public let bounds: CGRect
    }

    /// Cuts out the object at `point` (normalized `0…1`, top-left), its
    /// edge finished as `edge` asks (crisp and decontaminated by default).
    public func cutout(from image: CIImage, at point: CGPoint, edge: EdgeOptions = .crisp) -> Result? {
        guard let mask = segmenter.mask(for: image, at: point) else { return nil }
        return cutout(from: image, mask: mask, edge: edge)
    }

    /// Cuts out using an already-computed `mask` (avoids a second segmentation
    /// when the caller also needs the mask, e.g. for a track centroid).
    public func cutout(from image: CIImage, mask maskInput: CIImage, edge: EdgeOptions = .crisp) -> Result? {
        MaskComposite.cutout(from: image, mask: maskInput, edge: edge, context: ciContext)
    }

    /// The 1.2 form: `feather` `nil` now means the crisp edge, a number a
    /// feathered one — both decontaminated.
    @available(*, deprecated, message: "use cutout(from:at:edge:)")
    public func cutout(from image: CIImage, at point: CGPoint, feather: CGFloat?) -> Result? {
        cutout(from: image, at: point, edge: EdgeOptions(feather: feather))
    }

    @available(*, deprecated, message: "use cutout(from:mask:edge:)")
    public func cutout(from image: CIImage, mask maskInput: CIImage, feather: CGFloat?) -> Result? {
        cutout(from: image, mask: maskInput, edge: EdgeOptions(feather: feather))
    }
}

extension EdgeOptions {
    /// The 1.2 `feather` argument as an edge.
    init(feather: CGFloat?) {
        self.init(edge: feather.map { .feather(Double($0)) } ?? .crisp)
    }
}

// MARK: - The shared mask-to-cutout pipeline

/// One edge treatment for every mask, wherever the mask came from — the SAM
/// click, the person matte or the subject lift. Split out so the matte path
/// needs no segmenter (and so the two paths cannot drift apart).
enum MaskComposite {

    static func cutout(from image: CIImage, mask maskInput: CIImage, edge: EdgeOptions,
                       context ciContext: CIContext) -> ObjectCutout.Result? {
        let extent = image.extent.integral
        let width = Int(extent.width), height = Int(extent.height)
        guard width > 0, height > 0, width * height <= 120_000_000 else { return nil }
        var mask = maskInput
        var radius: CGFloat = 0
        if case .feather(let feather) = edge.edge, feather > 0 {
            radius = feather
            mask = mask.clampedToExtent()
                .applyingFilter("CIGaussianBlur", parameters: ["inputRadius": feather])
                .cropped(to: extent)
        }
        // Bounds from the mask, outset by any feather (plus one 128-scan
        // cell) so a sticker cropped to them keeps its own edge.
        let scanCell = max(extent.width, extent.height) / 128
        let bounds = (contentBounds(of: maskInput, extent: extent, context: ciContext) ?? extent)
            .insetBy(dx: -(radius + scanCell), dy: -(radius + scanCell))
            .intersection(extent)

        let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
        var picture = [UInt8](repeating: 0, count: width * height * 4)
        ciContext.render(image, toBitmap: &picture, rowBytes: width * 4, bounds: extent, format: .RGBA8, colorSpace: srgb)
        // The mask's values as they are: no colour matching, or 0.5 would land at 0.74.
        var maskPixels = [UInt8](repeating: 0, count: width * height * 4)
        ciContext.render(mask, toBitmap: &maskPixels, rowBytes: width * 4, bounds: extent, format: .RGBA8, colorSpace: nil)
        let coverage = (0..<(width * height)).map { maskPixels[$0 * 4] }

        let finished = EdgeRefiner.refine(rgba: picture, mask: coverage, width: width, height: height,
                                          edge: edge.edge, choke: edge.choke)
        guard let provider = CGDataProvider(data: Data(finished) as CFData),
              let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                               space: srgb, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else { return nil }
        let cut = CIImage(cgImage: cg).transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY))
        return ObjectCutout.Result(image: cut, bounds: bounds)
    }

    /// The tight bounds of the mask's content, found by scanning a small render.
    static func contentBounds(of mask: CIImage, extent: CGRect, context ciContext: CIContext) -> CGRect? {
        let side = 128
        var buf = [UInt8](repeating: 0, count: side * side * 4)
        let scaled = mask
            .transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
            .transformed(by: CGAffineTransform(scaleX: CGFloat(side) / extent.width, y: CGFloat(side) / extent.height))
        ciContext.render(scaled, toBitmap: &buf, rowBytes: side * 4,
                         bounds: CGRect(x: 0, y: 0, width: side, height: side),
                         format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())

        var minX = side, minY = side, maxX = -1, maxY = -1
        for y in 0..<side {
            for x in 0..<side where buf[(y * side + x) * 4] > 128 {   // R channel = mask value
                minX = min(minX, x); maxX = max(maxX, x)
                minY = min(minY, y); maxY = max(maxY, y)
            }
        }
        guard maxX >= minX, maxY >= minY else { return nil }
        // The render bitmap is top-left; map back to the image's bottom-left extent.
        let fx = extent.width / CGFloat(side), fy = extent.height / CGFloat(side)
        let x = extent.minX + CGFloat(minX) * fx
        let w = CGFloat(maxX - minX + 1) * fx
        let h = CGFloat(maxY - minY + 1) * fy
        let yTop = CGFloat(minY) * fy
        let y = extent.minY + extent.height - yTop - h
        return CGRect(x: x, y: y, width: w, height: h).intersection(extent)
    }
}
