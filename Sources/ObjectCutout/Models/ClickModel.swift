//
//  ClickModel.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//

import Foundation

/// The downloadable click segmenter: SAM 2.1 hiera tiny (Meta, Apache-2.0 —
/// code and weights) exported to Core AI, hosted for download.
///
/// One asset, two entry points: `encode` (the picture, once) and `decode`
/// (a click on it, ~5 ms), so several clicks share one encoding. Measured on
/// 60 clicks over 30 generated pictures against the PyTorch original in
/// float32: the chosen mask's IoU 0.996 at worst, the same mask chosen every
/// time, scores within 0.006. Apple's Core ML conversion, which 1.4 ran,
/// managed 0.50 at worst and chose differently on 4 of the 60.
public enum ClickModel {

    /// The Hugging Face repository holding the asset.
    public static let repository = "arraypress/cutout-sam2-tiny"

    /// The asset in ``repository``: float16 weights and arithmetic, float32
    /// inputs and outputs.
    public static let assetName = "cutout-sam2-tiny-float16.aimodel"

    /// The commit of ``repository`` the measurements were made on — pin
    /// downloads to it, so a later push cannot change what a run fetches.
    public static let revision = "a0b4f4d230089c3a389d8d7f087b82a6d2fc113e"

    /// The Application Support folder the asset is kept in, one for every tool.
    public static let sharedFolder = "cutout-sam2-tiny"

    /// The authors' licence, which travels with the weights.
    public static let licence = "Apache-2.0"

    /// The side of the square the network sees; the host stretches to it.
    public static let inputSize = 1024

    /// The side of the masks it returns (logits); the host stretches them back.
    public static let maskSize = 256

    /// Where the asset is installed: `~/Library/Application Support/<sharedFolder>/<assetName>`.
    public static var installedURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(sharedFolder, isDirectory: true)
            .appendingPathComponent(assetName, isDirectory: true)
    }

    /// Whether the asset is installed where ``installedURL`` says.
    public static var isInstalled: Bool {
        FileManager.default.fileExists(atPath: installedURL.appendingPathComponent("main.hash").path)
    }
}
