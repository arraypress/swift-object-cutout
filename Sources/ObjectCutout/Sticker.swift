//
//  Sticker.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//
//  A cutout dressed as a sticker: the classic white contour outline that
//  follows the figure's edge, a soft shadow under it, and optionally a
//  square canvas with padding. All of it is arithmetic on the alpha mask
//  the cutout already carries — dilate for the outline, blur for the
//  shadow — so it runs in the same fraction of a second, entirely on
//  device.
//

import CoreImage

/// Turns a cutout into a sticker.
public enum Sticker {

    /// A colour as plain values, so options stay `Sendable`.
    public struct Color: Sendable, Equatable {
        /// Channels `0…1`.
        public var red, green, blue, alpha: Double

        public init(red: Double, green: Double, blue: Double, alpha: Double = 1) {
            self.red = red; self.green = green; self.blue = blue; self.alpha = alpha
        }

        /// Sticker white.
        public static let white = Color(red: 1, green: 1, blue: 1)

        var ciColor: CIColor { CIColor(red: red, green: green, blue: blue, alpha: alpha) }
    }

    /// How the sticker is dressed.
    public struct Options: Sendable, Equatable {
        /// Contour outline thickness, in pixels of the source image.
        public var border: Double
        /// The outline's colour.
        public var borderColor: Color
        /// Whether a soft shadow sits under the figure.
        public var shadow: Bool
        /// Fit the sticker into a square canvas of this many pixels
        /// (with a little padding), or `nil` to keep the natural size.
        public var size: Int?

        public init(border: Double = 12, borderColor: Color = .white, shadow: Bool = true, size: Int? = nil) {
            self.border = border
            self.borderColor = borderColor
            self.shadow = shadow
            self.size = size
        }
    }

    /// The cutout as a sticker, cropped to its own canvas (origin zero).
    public static func make(from cutout: ObjectCutout.Result, options: Options = Options()) -> CIImage {
        let border = CGFloat(max(0, options.border))
        let shadowBlur = options.shadow ? max(3, border * 0.75) : 0
        let shadowDrop = options.shadow ? max(2, border * 0.5) : 0

        // Room for the outline all round and the shadow below.
        let margin = border + shadowBlur * 2 + shadowDrop
        let canvas = cutout.bounds.insetBy(dx: -margin, dy: -margin)
        let figure = cutout.image.cropped(to: cutout.bounds)

        // The figure's alpha as a grayscale mask (alpha copied into RGB),
        // seated on a clear board first: clamping would replicate the
        // bounds-edge alpha outward and paint the whole canvas.
        let board = CIImage(color: .clear).cropped(to: canvas)
        let mask = figure.composited(over: board)
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
            ])

        var layers: [CIImage] = []
        if options.shadow {
            let shadowMask = mask
                .applyingFilter("CIMorphologyMaximum", parameters: ["inputRadius": border * 0.6])
                .applyingFilter("CIGaussianBlur", parameters: ["inputRadius": shadowBlur])
            let ink = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0.35)).cropped(to: canvas)
            let clear = CIImage(color: .clear).cropped(to: canvas)
            let shadow = ink.applyingFilter("CIBlendWithMask", parameters: [
                "inputBackgroundImage": clear, "inputMaskImage": shadowMask,
            ]).transformed(by: CGAffineTransform(translationX: 0, y: -shadowDrop))
            layers.append(shadow.cropped(to: canvas))
        }
        if border > 0 {
            let rim = mask.applyingFilter("CIMorphologyMaximum", parameters: ["inputRadius": border])
            let paint = CIImage(color: options.borderColor.ciColor).cropped(to: canvas)
            let clear = CIImage(color: .clear).cropped(to: canvas)
            let outline = paint.applyingFilter("CIBlendWithMask", parameters: [
                "inputBackgroundImage": clear, "inputMaskImage": rim,
            ])
            layers.append(outline.cropped(to: canvas))
        }
        var sticker = figure
        for layer in layers.reversed() { sticker = sticker.composited(over: layer) }
        sticker = sticker.cropped(to: canvas)
            .transformed(by: CGAffineTransform(translationX: -canvas.minX, y: -canvas.minY))

        guard let size = options.size, size > 0 else { return sticker }
        // Fit into the square with a whisker of padding, centred.
        let side = CGFloat(size)
        let pad = side * 0.03
        let scale = min((side - 2 * pad) / sticker.extent.width, (side - 2 * pad) / sticker.extent.height)
        let scaled = sticker.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let dx = (side - scaled.extent.width) / 2 - scaled.extent.minX
        let dy = (side - scaled.extent.height) / 2 - scaled.extent.minY
        let placed = scaled.transformed(by: CGAffineTransform(translationX: dx, y: dy))
        let square = CIImage(color: .clear).cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
        return placed.composited(over: square).cropped(to: square.extent)
    }
}
