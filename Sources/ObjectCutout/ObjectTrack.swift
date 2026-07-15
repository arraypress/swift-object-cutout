//
//  ObjectTrack.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//

import AVFoundation
import CoreImage
import Foundation

/// One sampled position of a tracked object on the source timeline
/// (normalized, top-left origin).
public struct ObjectTrackFrame: Codable, Equatable, Sendable {
    public var time: TimeInterval
    /// The object's mask centroid (`0…1`), for a tracked sticker's position.
    public var center: CGPoint
    /// The object's tight bounds (`0…1`, top-left), for size.
    public var bounds: CGRect

    public init(time: TimeInterval, center: CGPoint, bounds: CGRect) {
        self.time = time
        self.center = center
        self.bounds = bounds
    }
}

/// Follows a clicked object through a video by re-segmenting each sampled frame
/// with the point riding the previous frame's mask centroid.
///
/// Apple's Core ML SAM 2 is the image path (no memory module), so this is
/// point-propagation tracking — great for a cutout sticker that follows an
/// object; it can lose fast-moving or occluded objects.
public enum SAM2VideoTracker {

    /// Tracks the object at `startPoint` (normalized) from `startTime`.
    public static func track(
        videoURL: URL,
        startPoint: CGPoint,
        startTime: TimeInterval = 0,
        using segmenter: SAM2Segmenter,
        fps: Double = 4,
        progress: (@Sendable (Double) -> Void)? = nil
    ) async throws -> [ObjectTrackFrame] {
        let asset = AVURLAsset(url: videoURL)
        let duration = try await asset.load(.duration).seconds
        guard duration > 0 else { return [] }

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.maximumSize = CGSize(width: 1024, height: 1024)
        generator.requestedTimeToleranceBefore = CMTime(seconds: 0.1, preferredTimescale: 600)
        generator.requestedTimeToleranceAfter = CMTime(seconds: 0.1, preferredTimescale: 600)

        let step = 1.0 / max(1, fps)
        let times = stride(from: startTime, to: duration, by: step).map { $0 }

        var frames: [ObjectTrackFrame] = []
        var point = startPoint
        for (i, t) in times.enumerated() {
            guard let cg = try? generator.copyCGImage(at: CMTime(seconds: t, preferredTimescale: 600), actualTime: nil)
            else { continue }
            let image = CIImage(cgImage: cg)
            guard let mask = segmenter.mask(for: image, at: point),
                  let (center, bounds) = analyze(mask: mask, extent: image.extent)
            else { continue }
            frames.append(ObjectTrackFrame(time: t, center: center, bounds: bounds))
            point = center                       // ride the centroid into the next frame
            progress?(Double(i + 1) / Double(times.count))
        }
        return smoothed(frames)
    }

    /// Centroid + tight bounds (normalized, top-left) of a mask, via a low-res scan.
    static func analyze(mask: CIImage, extent: CGRect) -> (center: CGPoint, bounds: CGRect)? {
        let side = 128
        var buf = [UInt8](repeating: 0, count: side * side * 4)
        let scaled = mask
            .transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
            .transformed(by: CGAffineTransform(scaleX: CGFloat(side) / extent.width, y: CGFloat(side) / extent.height))
        CIContext().render(scaled, toBitmap: &buf, rowBytes: side * 4,
                           bounds: CGRect(x: 0, y: 0, width: side, height: side),
                           format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
        var minX = side, minY = side, maxX = -1, maxY = -1, sumX = 0, sumY = 0, n = 0
        for y in 0..<side {
            for x in 0..<side where buf[(y * side + x) * 4] > 128 {
                minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
                sumX += x; sumY += y; n += 1
            }
        }
        guard n > 8 else { return nil }          // lost the object
        let f = CGFloat(side)
        let center = CGPoint(x: CGFloat(sumX) / CGFloat(n) / f, y: CGFloat(sumY) / CGFloat(n) / f)
        let bounds = CGRect(x: CGFloat(minX) / f, y: CGFloat(minY) / f,
                            width: CGFloat(maxX - minX + 1) / f, height: CGFloat(maxY - minY + 1) / f)
        return (center, bounds)
    }

    private static func smoothed(_ s: [ObjectTrackFrame]) -> [ObjectTrackFrame] {
        guard s.count > 2 else { return s }
        return s.indices.map { i in
            let w = s[max(0, i - 1)...min(s.count - 1, i + 1)]
            let n = CGFloat(w.count)
            return ObjectTrackFrame(
                time: s[i].time,
                center: CGPoint(x: w.map(\.center.x).reduce(0, +) / n, y: w.map(\.center.y).reduce(0, +) / n),
                bounds: CGRect(
                    x: w.map(\.bounds.minX).reduce(0, +) / n, y: w.map(\.bounds.minY).reduce(0, +) / n,
                    width: w.map(\.bounds.width).reduce(0, +) / n, height: w.map(\.bounds.height).reduce(0, +) / n)
            )
        }
    }
}
