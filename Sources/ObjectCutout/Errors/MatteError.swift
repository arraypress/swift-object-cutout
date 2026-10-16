//
//  MatteError.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//

import Foundation

/// Why the downloaded subject matte could not be had or run.
public enum MatteError: Error, Sendable, Equatable, LocalizedError {

    /// The asset is missing, unreadable, or not the one this library expects.
    case modelUnavailable(String)

    /// The model loaded but the run failed.
    case inference(String)

    /// The picture could not be read into pixels.
    case imageUnreadable(String)

    public var errorDescription: String? {
        switch self {
        case .modelUnavailable(let why): "The subject model is unavailable: \(why)"
        case .inference(let why): "The subject model failed: \(why)"
        case .imageUnreadable(let why): "The picture could not be read: \(why)"
        }
    }
}
