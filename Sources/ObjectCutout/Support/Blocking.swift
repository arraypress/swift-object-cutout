//
//  Blocking.swift
//  ObjectCutout
//
//  Created by David Sherlock on 2026.
//
//  The synchronous `Matte` calls predate the downloaded model, whose runtime
//  is async only. They wait for it here: the async work runs on a queue of
//  its own (a task executor), so the waiting thread — even one of Swift
//  concurrency's own — is never the one the work needs.
//

import Dispatch
import Synchronization

enum Blocking {

    /// Runs `work` to completion and returns its result, from synchronous code.
    static func run<T: Sendable>(_ work: @escaping @Sendable () async throws -> T) throws -> T {
        let result = Mutex<Result<T, any Error>?>(nil)
        let done = DispatchSemaphore(value: 0)
        Task(executorPreference: executor) {
            let outcome: Result<T, any Error>
            do { outcome = .success(try await work()) } catch { outcome = .failure(error) }
            result.withLock { $0 = outcome }
            done.signal()
        }
        done.wait()
        return try result.withLock { $0! }.get()
    }

    static let executor = QueueExecutor()
}

/// A task executor on a private serial queue.
final class QueueExecutor: TaskExecutor {

    private let queue = DispatchQueue(label: "ObjectCutout.matte", qos: .userInitiated)

    func enqueue(_ job: consuming ExecutorJob) {
        let job = UnownedJob(job)
        queue.async { job.runSynchronously(on: self.asUnownedTaskExecutor()) }
    }
}
