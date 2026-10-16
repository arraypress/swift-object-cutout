//
//  EdgeRefiner.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//
//  The edge of a cutout, finished on plain pixel buffers (so it is the same
//  on every device and testable without a model).
//
//  Near the mask's edge every pixel is a mix: `I = a·F + (1 − a)·B`, some of
//  the subject's colour F and some of the background's B. A halo is what
//  happens when that pixel is kept with its colour as it is: the
//  background's share shows as a rim. So, in a band round the mask's edge:
//
//  - F and B are estimated for each pixel from what lies near it: the mean
//    colour of the confidently-inside pixels, and of the confidently-outside
//    ones (widening the search until some are found).
//  - `crisp`: where F and B differ enough to tell apart, the pixel's alpha
//    is solved from its colour — its position between B and F — rather than
//    taken from the mask, so a strand of hair keeps its own width and the
//    gaps between strands open. Where the two-colour model does not explain
//    the pixel (a third colour: detail inside the subject), the mask is
//    trusted instead, steepened to about a pixel of anti-aliasing.
//  - Every mode: the pixel's colour is the subject's — `(I − (1 − a)·B) / a`,
//    blended toward the interior estimate where alpha is too small for that
//    to be stable — so what remains of the edge carries no background.
//
//  Buffers are RGBA8 (straight colour; the source is opaque), rows from the
//  top; the mask is one byte a pixel. The result is premultiplied RGBA8.
//

import Foundation

enum EdgeRefiner {

    /// Mask values at or above this are certainly the subject.
    static let inside: UInt8 = 250
    /// At or below this, certainly not.
    static let outside: UInt8 = 5
    /// Colours closer than this (sRGB, 0…1 per channel) cannot be told
    /// apart: the colour solve is skipped and the mask is trusted.
    static let separation = 0.1
    /// The share of the subject–background distance a pixel may sit off the
    /// line between them before the two-colour model stops being trusted.
    static let residualLimit = 0.5
    /// Inside the mask, a pixel this close to the background beside it (sRGB
    /// distance) is the background showing through a gap.
    static let gapDistance = 0.08
    /// A background sample this far from its neighbourhood's mean is not background.
    static let outlier = 0.15

    /// The finished cutout: premultiplied RGBA8, the same size.
    ///
    /// - Parameters:
    ///   - rgba: The picture, straight RGBA8, top row first.
    ///   - mask: The subject's mask, one byte a pixel (255 = subject).
    ///   - width: Pixels across.
    ///   - height: Pixels down.
    ///   - edge: The finish. The mask is already blurred for `feather`.
    ///   - choke: Pixels to pull the finished edge in.
    static func refine(rgba: [UInt8], mask: [UInt8], width: Int, height: Int, edge: CutoutEdge,
                       choke: Double) -> [UInt8] {
        let n = width * height
        precondition(rgba.count == n * 4 && mask.count == n)
        let ramp = rampWidth(mask, width: width, height: height)
        let crisp = edge == .crisp

        // The band: within a few pixels of the mask's half-way line, and
        // everywhere the mask is undecided.
        let reach = max(3, Int(ramp.rounded(.up)))
        var band = dilate(contour(mask, width: width, height: height), radius: reach, width: width, height: height)
        for i in 0..<n where mask[i] > outside && mask[i] < inside { band[i] = true }

        let first = Estimates(rgba: rgba, mask: mask, band: band, width: width, height: height, radius: max(8, 3 * reach))
        var estimates = Estimates(rgba: rgba, mask: mask, band: band, width: width, height: height,
                                  radius: max(8, 3 * reach), previous: first)
        if crisp {
            // Gaps: inside the mask, near its outline, the background's own
            // colour showing through (between strands of hair). They join the
            // band, and the subject's colour is re-estimated without them.
            let depth = max(8, min(width, height) / 50)
            let near = dilate(contour(mask, width: width, height: height), radius: depth, width: width, height: height)
            var gaps = [Bool](repeating: false, count: n)
            var found = false
            for i in 0..<n where near[i] && !band[i] && mask[i] >= 128 {
                guard let bg = estimates.mean(x: i % width, y: i / width, inside: false) else { continue }
                let dr = Float(rgba[i * 4]) / 255 - bg.0, dg = Float(rgba[i * 4 + 1]) / 255 - bg.1, db = Float(rgba[i * 4 + 2]) / 255 - bg.2
                if dr * dr + dg * dg + db * db < Float(gapDistance * gapDistance) { gaps[i] = true; found = true }
            }
            if found {
                let grown = dilate(gaps, radius: 2, width: width, height: height)
                var skip = band
                for i in 0..<n where gaps[i] { skip[i] = true }
                estimates = Estimates(rgba: rgba, mask: mask, band: skip, width: width, height: height,
                                      radius: max(8, 3 * reach), previous: first)
                for i in 0..<n where grown[i] && near[i] { band[i] = true }
            }
        }
        // Steepness of the tightened mask: its own 10–90 % width squeezed to about 1.2 px.
        let steep = Float(max(1, ramp / 1.2))

        var alpha = [Float](repeating: 0, count: n)
        var colour = [Float](repeating: 0, count: n * 3)
        for i in 0..<n {
            let m = Float(mask[i]) / 255
            let r = Float(rgba[i * 4]) / 255, g = Float(rgba[i * 4 + 1]) / 255, b = Float(rgba[i * 4 + 2]) / 255
            guard band[i] else {
                alpha[i] = crisp ? (m >= 0.5 ? 1 : 0) : m
                colour[i * 3] = r; colour[i * 3 + 1] = g; colour[i * 3 + 2] = b
                continue
            }
            let x = i % width, y = i / width
            let fg = estimates.mean(x: x, y: y, inside: true) ?? (r, g, b)
            let bg = estimates.mean(x: x, y: y, inside: false)
            var a = crisp ? min(1, max(0, (m - 0.5) * steep + 0.5)) : m
            var solved: (Float, Float, Float)?
            if let bg {
                let d = (fg.0 - bg.0, fg.1 - bg.1, fg.2 - bg.2)
                let dd = d.0 * d.0 + d.1 * d.1 + d.2 * d.2
                if dd >= Float(separation * separation) {
                    let v = (r - bg.0, g - bg.1, b - bg.2)
                    let ac = min(1, max(0, (v.0 * d.0 + v.1 * d.1 + v.2 * d.2) / dd))
                    let off = (v.0 - ac * d.0, v.1 - ac * d.1, v.2 - ac * d.2)
                    let residual = (off.0 * off.0 + off.1 * off.1 + off.2 * off.2).squareRoot() / dd.squareRoot()
                    let trust = max(0, min(1, 1 - residual / Float(residualLimit)))
                    if crisp { a = trust * ac + (1 - trust) * a }
                }
                if a > 0.001 {
                    // The subject's colour: the background's share taken out.
                    let k = 1 - a
                    solved = (min(1, max(0, (r - k * bg.0) / a)), min(1, max(0, (g - k * bg.1) / a)),
                              min(1, max(0, (b - k * bg.2) / a)))
                }
            }
            // Where alpha is small the solve divides by little: lean on the interior's colour.
            let lean = smoothstep(0.2, 0.7, a)
            let pick = solved ?? (r, g, b)
            let base = solved == nil ? (m >= 0.5 ? (r, g, b) : fg) : fg
            colour[i * 3] = base.0 + (pick.0 - base.0) * lean
            colour[i * 3 + 1] = base.1 + (pick.1 - base.1) * lean
            colour[i * 3 + 2] = base.2 + (pick.2 - base.2) * lean
            alpha[i] = a
        }

        if choke >= 0.5 { alpha = erode(alpha, radius: Int(choke.rounded()), width: width, height: height) }

        var out = [UInt8](repeating: 0, count: n * 4)
        for i in 0..<n {
            let a = alpha[i]
            guard a > 0 else { continue }
            out[i * 4] = byte(colour[i * 3] * a)
            out[i * 4 + 1] = byte(colour[i * 3 + 1] * a)
            out[i * 4 + 2] = byte(colour[i * 3 + 2] * a)
            out[i * 4 + 3] = byte(a)
        }
        return out
    }

    static func byte(_ value: Float) -> UInt8 { UInt8(min(255, max(0, (value * 255).rounded()))) }

    static func smoothstep(_ low: Float, _ high: Float, _ value: Float) -> Float {
        let t = min(1, max(0, (value - low) / (high - low)))
        return t * t * (3 - 2 * t)
    }

    // MARK: Measures

    /// The mask's mean 10–90 % transition, in pixels: undecided pixels per
    /// pixel of outline. `1` for a hard mask.
    static func rampWidth(_ mask: [UInt8], width: Int, height: Int) -> Double {
        var partial = 0, outline = 0
        guard width > 2, height > 2 else { return 1 }
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let i = y * width + x, m = mask[i]
                if m > 25 && m < 230 { partial += 1 }
                if m >= 128 && (mask[i - 1] < 128 || mask[i + 1] < 128 || mask[i - width] < 128 || mask[i + width] < 128) {
                    outline += 1
                }
            }
        }
        return outline == 0 ? 1 : max(1, Double(partial) / Double(outline))
    }

    /// The pixels on either side of the mask's half-way line.
    static func contour(_ mask: [UInt8], width: Int, height: Int) -> [Bool] {
        var line = [Bool](repeating: false, count: width * height)
        for y in 0..<height {
            for x in 0..<width {
                let i = y * width + x, on = mask[i] >= 128
                if x + 1 < width, (mask[i + 1] >= 128) != on { line[i] = true; line[i + 1] = true }
                if y + 1 < height, (mask[i + width] >= 128) != on { line[i] = true; line[i + width] = true }
            }
        }
        return line
    }

    /// `marks` grown by `radius` pixels (a square), in two passes.
    static func dilate(_ marks: [Bool], radius: Int, width: Int, height: Int) -> [Bool] {
        func pass(_ input: [Bool], count: Int, step: Int, lines: Int, lineStride: Int) -> [Bool] {
            var output = input
            for line in 0..<lines {
                let base = line * lineStride
                var last = -Int.max / 2
                for k in 0..<count {
                    if input[base + k * step] { last = k }
                    if k - last <= radius { output[base + k * step] = true }
                }
                last = Int.max / 2
                for k in stride(from: count - 1, through: 0, by: -1) {
                    if input[base + k * step] { last = k }
                    if last - k <= radius { output[base + k * step] = true }
                }
            }
            return output
        }
        let across = pass(marks, count: width, step: 1, lines: height, lineStride: width)
        return pass(across, count: height, step: width, lines: width, lineStride: 1)
    }

    /// `alpha` eroded by `radius` pixels (a square): each value the least in its window.
    static func erode(_ alpha: [Float], radius: Int, width: Int, height: Int) -> [Float] {
        func pass(_ input: [Float], count: Int, step: Int, lines: Int, lineStride: Int) -> [Float] {
            var output = input
            for line in 0..<lines {
                let base = line * lineStride
                for k in 0..<count {
                    var least: Float = 1
                    for j in max(0, k - radius)...min(count - 1, k + radius) { least = min(least, input[base + j * step]) }
                    output[base + k * step] = least
                }
            }
            return output
        }
        let across = pass(alpha, count: width, step: 1, lines: height, lineStride: width)
        return pass(across, count: height, step: width, lines: width, lineStride: 1)
    }

    // MARK: Local colours

    /// The mean colour of the certain-subject and certain-background pixels
    /// round any point, from per-cell sums and a summed-area table over the
    /// cells: a query is four reads, at any radius.
    struct Estimates {
        let cell: Int
        let columns: Int
        let rows: Int
        /// Per side: red, green, blue, count — summed over cells.
        private let tables: [[Double]]
        private let reach: Int

        /// With `previous`, background pixels far from its estimate are left
        /// out — a strand tip the mask missed is not background.
        init(rgba: [UInt8], mask: [UInt8], band: [Bool], width: Int, height: Int, radius: Int, previous: Estimates? = nil) {
            let cell = max(2, radius / 4)
            let columns = (width + cell - 1) / cell, rows = (height + cell - 1) / cell
            self.cell = cell
            self.columns = columns
            self.rows = rows
            reach = max(1, (radius + cell - 1) / cell)
            var sums = [[Double]](repeating: [Double](repeating: 0, count: columns * rows), count: 8)
            for y in 0..<height {
                for x in 0..<width {
                    let i = y * width + x
                    guard !band[i] else { continue }
                    let side: Int
                    if mask[i] >= EdgeRefiner.inside { side = 0 } else if mask[i] <= EdgeRefiner.outside { side = 4 } else { continue }
                    if side == 4, let previous, let bg = previous.mean(x: x, y: y, inside: false) {
                        let dr = Float(rgba[i * 4]) / 255 - bg.0, dg = Float(rgba[i * 4 + 1]) / 255 - bg.1
                        let db = Float(rgba[i * 4 + 2]) / 255 - bg.2
                        if dr * dr + dg * dg + db * db > Float(EdgeRefiner.outlier * EdgeRefiner.outlier) { continue }
                    }
                    let c = (y / cell) * columns + x / cell
                    sums[side][c] += Double(rgba[i * 4])
                    sums[side + 1][c] += Double(rgba[i * 4 + 1])
                    sums[side + 2][c] += Double(rgba[i * 4 + 2])
                    sums[side + 3][c] += 1
                }
            }
            let stride = columns + 1
            tables = sums.map { grid in
                var table = [Double](repeating: 0, count: stride * (rows + 1))
                for r in 0..<rows {
                    var run = 0.0
                    for c in 0..<columns {
                        run += grid[r * columns + c]
                        table[(r + 1) * stride + c + 1] = table[r * stride + c + 1] + run
                    }
                }
                return table
            }
        }

        private func sum(_ k: Int, _ c0: Int, _ r0: Int, _ c1: Int, _ r1: Int) -> Double {
            let s = columns + 1, t = tables[k]
            return t[r1 * s + c1] - t[r0 * s + c1] - t[r1 * s + c0] + t[r0 * s + c0]
        }

        /// The mean colour (0…1) of one side near pixel (x, y), blended
        /// between the four nearest cells so it has no steps; `nil` when none
        /// lies within eight times the radius.
        func mean(x: Int, y: Int, inside: Bool) -> (Float, Float, Float)? {
            let fx = (Double(x) + 0.5) / Double(cell) - 0.5, fy = (Double(y) + 0.5) / Double(cell) - 0.5
            let x0 = Int(fx.rounded(.down)), y0 = Int(fy.rounded(.down))
            let tx = Float(fx - Double(x0)), ty = Float(fy - Double(y0))
            var total: (Float, Float, Float) = (0, 0, 0), weight: Float = 0
            for (cx, cy, w) in [(x0, y0, (1 - tx) * (1 - ty)), (x0 + 1, y0, tx * (1 - ty)),
                                (x0, y0 + 1, (1 - tx) * ty), (x0 + 1, y0 + 1, tx * ty)] where w > 0 {
                guard let m = mean(cell: min(columns - 1, max(0, cx)), min(rows - 1, max(0, cy)), side: inside ? 0 : 4) else { continue }
                total = (total.0 + m.0 * w, total.1 + m.1 * w, total.2 + m.2 * w)
                weight += w
            }
            guard weight > 0 else { return nil }
            return (total.0 / weight, total.1 / weight, total.2 / weight)
        }

        /// The mean round one cell, widening until some pixels are found.
        private func mean(cell cx: Int, _ cy: Int, side: Int) -> (Float, Float, Float)? {
            var k = reach
            for _ in 0..<4 {
                let c0 = max(0, cx - k), r0 = max(0, cy - k), c1 = min(columns, cx + k + 1), r1 = min(rows, cy + k + 1)
                let count = sum(side + 3, c0, r0, c1, r1)
                if count >= 3 {
                    return (Float(sum(side, c0, r0, c1, r1) / count / 255), Float(sum(side + 1, c0, r0, c1, r1) / count / 255),
                            Float(sum(side + 2, c0, r0, c1, r1) / count / 255))
                }
                k *= 2
            }
            return nil
        }
    }
}
