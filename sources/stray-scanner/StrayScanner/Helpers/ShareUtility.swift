//
//  ShareUtility.swift
//  StrayScanner
//
//  Created by Claude on 6/24/25.
//

import Foundation

/// Utility class for creating shareable archives from recording datasets
class ShareUtility {
    
    /// Creates a shareable ZIP archive from a recording's dataset
    /// - Parameter directory: A completed RGB-D bundle directory, read on the UI queue.
    /// - Returns: URL of the created ZIP file
    static func createShareableArchive(directory sourceDirectory: URL) async throws -> URL {
        
        let tempDirectory = FileManager.default.temporaryDirectory
        let archiveURL = tempDirectory.appendingPathComponent(UUID().uuidString + "-" + sourceDirectory.lastPathComponent + ".zip")
        
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let data = try Data(contentsOf: sourceDirectory.appendingPathComponent("scan_bundle.json"))
                    guard let manifest = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                          manifest["format"] as? String == "sku-rgbd",
                          manifest["version"] as? Int == 1,
                          manifest["status"] as? String == "ready",
                          let frames = manifest["frames"] as? [[String: Any]], !frames.isEmpty,
                          let errors = manifest["errors"] as? [String], errors.isEmpty else {
                        throw CaptureError.invalid("数据包尚未完成或存在写入错误，无法导出。")
                    }
                    try createZipArchive(sourceDirectory: sourceDirectory, destinationURL: archiveURL)
                    continuation.resume(returning: archiveURL)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    private static func createZipArchive(sourceDirectory: URL, destinationURL: URL) throws {
        let coordinator = NSFileCoordinator()
        var coordinatorError: NSError?
        var copyError: Error?

        coordinator.coordinate(readingItemAt: sourceDirectory, options: [.forUploading], error: &coordinatorError) { zipURL in
            do {
                try FileManager.default.copyItem(at: zipURL, to: destinationURL)
            } catch {
                copyError = error
            }
        }

        if let error = coordinatorError ?? copyError {
            throw error
        }
    }
}
