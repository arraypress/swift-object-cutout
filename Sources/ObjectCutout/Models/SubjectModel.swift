//
//  SubjectModel.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//

import Foundation

/// The downloadable subject matte: BiRefNet general (Zheng et al., MIT —
/// code and weights) exported to Core AI at 1024², hosted for download.
///
/// Measured against Vision's subject mask with the crisp edge on true-alpha
/// sets BiRefNet was not trained on (AIM-500): about a third of the error on
/// objects, plants and glass, and on animals' fur. The README has the table.
public enum SubjectModel {

    /// The Hugging Face repository holding the asset.
    public static let repository = "arraypress/cutout-birefnet"

    /// The asset in ``repository``: float16 weights and arithmetic, float32
    /// input and output. Within noise of the float32 export, at half the
    /// size and memory.
    public static let assetName = "cutout-birefnet-float16.aimodel"

    /// The commit of ``repository`` the measurements were made on — pin
    /// downloads to it, so a later push cannot change what a run fetches.
    public static let revision = "5a64a48ca4cedd1b5e689eab5c4450a735d26e0b"

    /// The Application Support folder the asset is kept in. One name for
    /// every tool, so one download serves `cutout` and `design` alike.
    public static let sharedFolder = "cutout-birefnet"

    /// The authors' licence, which travels with the weights.
    public static let licence = "MIT"

    /// The side of the square the network sees; the host resizes to it and back.
    public static let inputSize = 1024

    /// Where the asset is installed: `~/Library/Application Support/<sharedFolder>/<assetName>`.
    public static var installedURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(sharedFolder, isDirectory: true)
            .appendingPathComponent(assetName, isDirectory: true)
    }

    /// Whether this build and OS can run the model at all: Core AI in the
    /// SDK (the iOS Simulator has none) and macOS 27 / iOS 27 at run time.
    public static var runsHere: Bool {
        #if canImport(CoreAI)
        if #available(macOS 27, iOS 27, *) { return true }
        #endif
        return false
    }

    /// Whether the asset is installed where ``installedURL`` says.
    public static var isInstalled: Bool {
        FileManager.default.fileExists(atPath: installedURL.appendingPathComponent("main.hash").path)
    }
}
