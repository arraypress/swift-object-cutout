//
//  ModelShelf.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//
//  The loaded models of one kind, by path: loaded once per program and kept,
//  so every image of a folder, every layer of a design and every frame of a
//  track reuses one copy — and one load per path even when several callers
//  ask at once.
//

import Foundation

actor ModelShelf<Model: Sendable> {

    private var loaded: [URL: Model] = [:]
    private var loading: [URL: Task<Model, any Error>] = [:]

    func model(at url: URL, load: @escaping @Sendable (URL) async throws -> Model) async throws(MatteError) -> Model {
        let key = url.standardizedFileURL
        if let model = loaded[key] { return model }
        let task = loading[key] ?? Task { try await load(key) }
        loading[key] = task
        do {
            let model = try await task.value
            loaded[key] = model
            loading[key] = nil
            return model
        } catch {
            loading[key] = nil
            throw (error as? MatteError) ?? .modelUnavailable("\(error)")
        }
    }
}
