//
//  SubjectMatter.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//
//  The subject matte from the downloaded model: BiRefNet general (MIT) on
//  Core AI. The picture is resized to the network's 1024² square, the
//  network returns a soft alpha there (its normalisation and sigmoid are
//  inside the asset), and the alpha is stretched back over the picture.
//  Load it once and reuse it: the first load in a program compiles the
//  model for this Mac (cached by the OS after that).
//
//  Compiled only where the SDK has Core AI: the iOS Simulator does not, and
//  there `Matte` lifts the subject with Vision.
//

#if canImport(CoreAI)
import CoreAI
import CoreGraphics
import CoreImage
import Foundation

/// The salient subject's soft matte, from the downloaded model.
///
/// ```swift
/// let matter = try await SubjectMatter(contentsOf: SubjectModel.installedURL)
/// let mask = try await matter.mask(for: image)   // grey, image extent
/// ```
@available(macOS 27, iOS 27, *)
public final class SubjectMatter: Sendable {

    /// The asset this was loaded from.
    public let url: URL

    private let function: InferenceFunction
    private let context = CIContext()

    /// Loads the asset, on the GPU (Core AI's Neural Engine preference
    /// measured up to 90× slower on another fleet model and still ended on
    /// the GPU).
    public init(contentsOf url: URL) async throws(MatteError) {
        guard FileManager.default.fileExists(atPath: url.appendingPathComponent("main.hash").path) else {
            throw .modelUnavailable("no Core AI asset at \(url.path)")
        }
        self.url = url
        do {
            let model = try await AIModel(contentsOf: url, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
            guard let main = try model.loadFunction(named: "main") else {
                throw MatteError.modelUnavailable("\(url.lastPathComponent) has no 'main' entry point; it has \(model.functionNames)")
            }
            function = main
        } catch let error as MatteError {
            throw error
        } catch {
            throw .modelUnavailable("\(url.lastPathComponent) could not be loaded: \(error)")
        }
    }

    /// The network on a prepared input: planar RGB in 0…1 at
    /// ``SubjectModel/inputSize``² in, the alpha at that size out (rows
    /// from the top).
    public func alpha(planarRGB input: [Float]) async throws(MatteError) -> [Float] {
        let side = SubjectModel.inputSize
        guard input.count == 3 * side * side else {
            throw .inference("the input holds \(input.count) values; \(3 * side * side) were expected")
        }
        var image = NDArray(shape: [1, 3, side, side], scalarType: .float32)
        image.mutableView(as: Float.self).withUnsafeMutablePointer { pointer, _, _ in
            input.withUnsafeBufferPointer { pointer.update(from: $0.baseAddress!, count: input.count) }
        }
        var out = NDArray(shape: [1, 1, side, side], scalarType: .float32)
        var views = InferenceFunction.MutableViews()
        views.insert(out.mutableRawView(), for: "alpha")
        do {
            _ = try await function.run(inputs: ["image": image], states: InferenceFunction.MutableViews(),
                                       outputViews: consume views)
        } catch {
            throw .inference("\(error)")
        }
        var alpha = [Float](repeating: 0, count: side * side)
        out.view(as: Float.self).withUnsafePointer { pointer, _, _ in
            alpha.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: pointer, count: side * side) }
        }
        return alpha
    }

    /// The subject's soft matte over `image`'s extent: grey, white = subject.
    public func mask(for image: CGImage) async throws(MatteError) -> CIImage {
        let side = SubjectModel.inputSize
        guard let input = MattePixels.planarRGB(of: image, side: side) else {
            throw .imageUnreadable("\(image.width)×\(image.height) could not be drawn")
        }
        let alpha = try await alpha(planarRGB: input)
        return MattePixels.mask(alpha: alpha, side: side,
                                extent: CGRect(x: 0, y: 0, width: image.width, height: image.height))
    }

    /// The subject's soft matte over `image`'s extent.
    public func mask(for image: CIImage) async throws(MatteError) -> CIImage {
        let extent = image.extent.integral
        guard extent.width >= 1, extent.height >= 1, !extent.isInfinite,
              let picture = context.createCGImage(image, from: extent, format: .RGBA8,
                                                  colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!) else {
            throw .imageUnreadable("the image could not be rendered")
        }
        return try await mask(for: picture).transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY))
    }
}

// MARK: - One loaded model per program

@available(macOS 27, iOS 27, *)
extension SubjectMatter {

    /// The model at `url`, loaded once per program and kept: every later
    /// call — every image of a folder, every layer of a design — reuses it.
    public static func shared(at url: URL = SubjectModel.installedURL) async throws(MatteError) -> SubjectMatter {
        try await SharedMatters.store.matter(at: url)
    }
}

/// The loaded models, by path.
@available(macOS 27, iOS 27, *)
actor SharedMatters {

    static let store = SharedMatters()

    private var loaded: [URL: SubjectMatter] = [:]
    private var loading: [URL: Task<SubjectMatter, any Error>] = [:]

    func matter(at url: URL) async throws(MatteError) -> SubjectMatter {
        let key = url.standardizedFileURL
        if let matter = loaded[key] { return matter }
        // One load per path even when several callers ask at once.
        let task = loading[key] ?? Task { try await SubjectMatter(contentsOf: key) }
        loading[key] = task
        do {
            let matter = try await task.value
            loaded[key] = matter
            loading[key] = nil
            return matter
        } catch {
            loading[key] = nil
            throw (error as? MatteError) ?? .modelUnavailable("\(error)")
        }
    }
}

#endif
