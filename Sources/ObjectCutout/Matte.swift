//
//  Matte.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//
//  Whole-figure mattes from the models Vision ships with the OS — no
//  downloaded weights and no click. `person` mattes every human in the
//  frame (the remove-the-background-from-a-person case); `subject` lifts
//  whatever the system judges salient. The SAM click remains the way to
//  choose ONE PARTICULAR object; these are the ways to take the obvious
//  one whole — a click on a torso returns the garment, a person matte
//  returns the person.
//

import CoreImage
import CoreVideo
import Vision

/// Built-in matting: people and salient subjects, no models to install.
public enum Matte {

    /// Which built-in matte to take.
    public enum Mode: String, CaseIterable, Sendable {
        /// Every person in the frame, whole figures.
        case person
        /// The salient subject, whatever kind of thing it is.
        case subject
    }

    /// A soft mask of every person in the image, scaled to the image's
    /// extent, or `nil` when Vision finds nobody.
    public static func personMask(for image: CIImage) throws -> CIImage? {
        let request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = .accurate
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        let handler = VNImageRequestHandler(ciImage: image)
        try handler.perform([request])
        guard let buffer = request.results?.first?.pixelBuffer else { return nil }
        let mask = scaled(CIImage(cvPixelBuffer: buffer), to: image.extent)
        // An all-dark mask is Vision saying "no person"; reject it here so
        // callers get nil rather than a cutout of nothing.
        guard MaskComposite.contentBounds(of: mask, extent: image.extent, context: context) != nil else { return nil }
        return mask
    }

    /// A full-resolution mask of the salient subject(s), or `nil` when the
    /// system lifts nothing.
    public static func subjectMask(for image: CIImage) throws -> CIImage? {
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(ciImage: image)
        try handler.perform([request])
        guard let result = request.results?.first, !result.allInstances.isEmpty else { return nil }
        let buffer = try result.generateScaledMaskForImage(forInstances: result.allInstances, from: handler)
        return scaled(CIImage(cvPixelBuffer: buffer), to: image.extent)
    }

    /// The matte as a cutout — the same edge treatment as the click path,
    /// so a person matte and a SAM sticker read as one tool's work.
    public static func cutout(from image: CIImage, mode: Mode, edge: EdgeOptions = .crisp) throws -> ObjectCutout.Result? {
        let mask: CIImage?
        switch mode {
        case .person: mask = try personMask(for: image)
        case .subject: mask = try subjectMask(for: image)
        }
        guard let mask else { return nil }
        return MaskComposite.cutout(from: image, mask: mask, edge: edge, context: context)
    }

    /// A cutout from an already-computed matte mask — so a caller that also
    /// wants the mask itself segments once, not twice.
    public static func cutout(from image: CIImage, mask: CIImage, edge: EdgeOptions = .crisp) -> ObjectCutout.Result? {
        MaskComposite.cutout(from: image, mask: mask, edge: edge, context: context)
    }

    /// The 1.2 form: `feather` `nil` now means the crisp edge, a number a
    /// feathered one — both decontaminated.
    @available(*, deprecated, message: "use cutout(from:mode:edge:)")
    public static func cutout(from image: CIImage, mode: Mode, feather: CGFloat?) throws -> ObjectCutout.Result? {
        try cutout(from: image, mode: mode, edge: EdgeOptions(feather: feather))
    }

    @available(*, deprecated, message: "use cutout(from:mask:edge:)")
    public static func cutout(from image: CIImage, mask: CIImage, feather: CGFloat?) -> ObjectCutout.Result? {
        cutout(from: image, mask: mask, edge: EdgeOptions(feather: feather))
    }

    /// Vision's masks come back at analysis resolution (person) or image
    /// resolution (subject); either way, pin them to the image's extent.
    private static func scaled(_ mask: CIImage, to extent: CGRect) -> CIImage {
        let sx = extent.width / mask.extent.width
        let sy = extent.height / mask.extent.height
        return mask
            .transformed(by: CGAffineTransform(scaleX: sx, y: sy))
            .transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY))
            .cropped(to: extent)
    }

    /// CIContext is documented thread-safe; one shared context for mattes.
    nonisolated(unsafe) private static let context = CIContext()
}
