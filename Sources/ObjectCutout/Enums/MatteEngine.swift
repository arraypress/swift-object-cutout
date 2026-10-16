//
//  MatteEngine.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//

import Foundation

/// What finds the subject.
public enum MatteEngine: String, CaseIterable, Sendable {

    /// The downloaded model when it is installed and the OS runs Core AI
    /// (macOS 27 / iOS 27), Vision otherwise — so nothing has to be
    /// downloaded for a cutout to work, and a downloaded model is used
    /// wherever it is.
    case automatic

    /// The model the OS ships (Vision): no download, softer and less exact.
    case vision

    /// The downloaded model (BiRefNet general on Core AI) — an error when
    /// it is not installed or the OS cannot run it.
    case model
}
