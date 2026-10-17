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
//  1.4: the subject comes from the downloaded model (BiRefNet on Core AI)
//  wherever it is installed and the OS can run it, and from Vision
//  otherwise — `MatteEngine` chooses explicitly. Where the SDK has no
//  Core AI (the iOS Simulator) the model path is compiled out and Vision
//  lifts the subject.
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
    /// extent, or `nil` when nobody is found.
    ///
    /// - Parameter engine: ``MatteEngine/automatic`` (the default): where
    ///   the downloaded model is installed, its edge kept where Vision's
    ///   person segmentation finds people (``gated(_:byPeople:extent:)``);
    ///   elsewhere Vision's person mask alone.
    public static func personMask(for image: CIImage, engine: MatteEngine = .automatic) throws -> CIImage? {
        guard let people = try visionPersonMask(for: image) else { return nil }
        if let modelled = try modelMask(for: image, engine: engine) {
            return nonEmpty(gated(modelled, byPeople: people, extent: image.extent), extent: image.extent)
        }
        return people
    }

    /// The subject model's mask kept only where `people` (Vision's person
    /// mask) finds people, grown by 5 % of the short side and softened: the
    /// model's edge, Vision's choice of what counts. SAD 7.5 on AIM-500
    /// portraits, against 24.9 for Vision's person mask with the crisp edge.
    public static func gated(_ modelMask: CIImage, byPeople people: CIImage, extent: CGRect) -> CIImage {
        PersonGate.apply(model: modelMask, person: people, extent: extent, context: context)
    }

    /// Vision's person segmentation (accurate), scaled to the image's
    /// extent, or `nil` when it finds nobody.
    public static func visionPersonMask(for image: CIImage) throws -> CIImage? {
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

    /// A full-resolution soft mask of the salient subject(s), or `nil` when
    /// nothing is lifted.
    ///
    /// - Parameter engine: ``MatteEngine/automatic`` (the default) uses the
    ///   downloaded model where it is installed and Vision elsewhere.
    /// - Throws: ``MatteError`` when the model is asked for (or installed)
    ///   and cannot run; Vision's errors as they come.
    public static func subjectMask(for image: CIImage, engine: MatteEngine = .automatic) throws -> CIImage? {
        if let mask = try modelMask(for: image, engine: engine) {
            return nonEmpty(mask, extent: image.extent)
        }
        return try visionSubjectMask(for: image)
    }

    /// The downloaded model's mask over `image`'s extent, or `nil` when
    /// `engine` does not put the subject to it here — always `nil` where
    /// the SDK has no Core AI (the iOS Simulator), so Vision answers.
    static func modelMask(for image: CIImage, engine: MatteEngine) throws -> CIImage? {
        guard try usesModel(engine) else { return nil }
        #if canImport(CoreAI)
        if #available(macOS 27, iOS 27, *) {
            let url = SubjectModel.installedURL
            return try Blocking.run { try await SubjectMatter.shared(at: url).mask(for: image) }
        }
        #endif
        return nil
    }

    /// Whether `engine` puts the subject to the downloaded model here.
    ///
    /// - Throws: ``MatteError/modelUnavailable(_:)`` when the model is asked
    ///   for by name and is not installed, or the OS cannot run it.
    public static func usesModel(_ engine: MatteEngine) throws(MatteError) -> Bool {
        let runs = SubjectModel.runsHere
        switch engine {
        case .vision: return false
        case .automatic: return runs && SubjectModel.isInstalled
        case .model:
            guard runs else { throw .modelUnavailable("the subject model runs on Core AI: macOS 27 or iOS 27, not the Simulator") }
            guard SubjectModel.isInstalled else { throw .modelUnavailable("not installed at \(SubjectModel.installedURL.path)") }
            return true
        }
    }

    /// `mask`, or `nil` when nothing in it reaches half-on.
    public static func nonEmpty(_ mask: CIImage, extent: CGRect) -> CIImage? {
        MaskComposite.contentBounds(of: mask, extent: extent, context: context) == nil ? nil : mask
    }

    /// Vision's mask of the salient subject(s), full resolution, or `nil`
    /// when it lifts nothing.
    public static func visionSubjectMask(for image: CIImage) throws -> CIImage? {
        let request = VNGenerateForegroundInstanceMaskRequest()
        let handler = VNImageRequestHandler(ciImage: image)
        try handler.perform([request])
        guard let result = request.results?.first, !result.allInstances.isEmpty else { return nil }
        let buffer = try result.generateScaledMaskForImage(forInstances: result.allInstances, from: handler)
        return scaled(CIImage(cvPixelBuffer: buffer), to: image.extent)
    }

    /// The matte as a cutout — the same edge treatment as the click path,
    /// so a person matte and a SAM sticker read as one tool's work.
    ///
    /// When the downloaded model lifts the subject, `.crisp` is taken as
    /// ``EdgeOptions/forModel(_:)`` says — its own alpha, decontaminated —
    /// because re-solving that alpha measured worse (see the README).
    public static func cutout(from image: CIImage, mode: Mode, edge: EdgeOptions = .crisp,
                              engine: MatteEngine = .automatic) throws -> ObjectCutout.Result? {
        let mask: CIImage?
        var finish = edge
        switch mode {
        case .person:
            mask = try personMask(for: image, engine: engine)
            if try usesModel(engine) { finish = .forModel(edge) }
        case .subject:
            mask = try subjectMask(for: image, engine: engine)
            if try usesModel(engine) { finish = .forModel(edge) }
        }
        guard let mask else { return nil }
        return MaskComposite.cutout(from: image, mask: mask, edge: finish, context: context)
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
    private static let context = CIContext()
}
