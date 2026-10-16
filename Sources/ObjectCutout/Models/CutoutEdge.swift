//
//  CutoutEdge.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//

import Foundation

/// How a cutout's edge is finished.
///
/// A mask from a model is soft at the edge, and the photo's pixels there
/// are part subject, part background. Left alone, that mix is a halo: a
/// rim of the old background's colour round the subject (a sky-blue glow
/// round hair, a green fringe round a product shot). Every edge here is
/// **decontaminated** — the background's share is taken out of each edge
/// pixel's colour, estimated from the subject's own interior and the
/// background beside it — and the modes differ only in the alpha.
public enum CutoutEdge: Sendable, Equatable, CustomStringConvertible {

    /// The default: the alpha drawn tight round the subject (about a pixel
    /// of anti-aliasing), each edge pixel's alpha re-solved from its colour
    /// where subject and background differ — so the gaps between strands of
    /// hair open up instead of filling with the background's colour.
    case crisp

    /// The model's own soft mask, decontaminated.
    case soft

    /// The mask blurred by this many pixels, decontaminated — a deliberately soft edge.
    case feather(Double)

    /// `crisp`, `soft`, or `feather:N` (pixels); `nil` for anything else.
    public init?(_ text: String) {
        let word = text.trimmingCharacters(in: .whitespaces).lowercased()
        switch word {
        case "crisp": self = .crisp
        case "soft": self = .soft
        default:
            let parts = word.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2, parts[0] == "feather", let radius = Double(parts[1]), radius.isFinite,
                  (0...200).contains(radius) else { return nil }
            self = .feather(radius)
        }
    }

    /// As written: `crisp`, `soft`, `feather:4`.
    public var description: String {
        switch self {
        case .crisp: "crisp"
        case .soft: "soft"
        case .feather(let radius): "feather:\(radius == radius.rounded() ? String(Int(radius)) : String(radius))"
        }
    }

    /// The words the edge is written with, for help and refusals.
    public static let forms = "crisp, soft, or feather:N (pixels)"
}

/// A cutout's edge: its finish, and how far it is pulled in.
public struct EdgeOptions: Sendable, Equatable {

    /// The finish.
    public var edge: CutoutEdge

    /// Pixels the edge is pulled in (eroded) after it is finished — to drop
    /// a last line of background a mask reached past. `0` leaves it.
    public var choke: Double

    public init(edge: CutoutEdge = .crisp, choke: Double = 0) {
        self.edge = edge
        self.choke = choke
    }

    /// The default: crisp, not choked.
    public static let crisp = EdgeOptions()

    /// The finish for a mask from the downloaded subject model: `.crisp`
    /// becomes `.soft` — the model's own alpha, decontaminated — and any
    /// other finish and the choke are kept. Measured on AIM-500 (true
    /// alpha): re-solving BiRefNet's alpha from colour raised SAD on
    /// objects from 23.4 to 30.7; Vision's softer mask is what crisp helps.
    public static func forModel(_ options: EdgeOptions) -> EdgeOptions {
        guard options.edge == .crisp else { return options }
        return EdgeOptions(edge: .soft, choke: options.choke)
    }
}
