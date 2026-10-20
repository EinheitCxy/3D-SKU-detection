// Copyright © 2021 Stray Robots. MIT license; see LICENSE.md.
// SKU workflow: one synchronized RGB-D sample per elapsed second.

import ARKit
import CoreImage
import Foundation
import ImageIO

enum CaptureError: LocalizedError {
    case invalid(String)

    var errorDescription: String? {
        switch self {
        case .invalid(let message): return message
        }
    }
}

struct CaptureSample {
    let frameID: Int
    let timestamp: Double
    let relativeTimestamp: Double
    let targetSeconds: Int
}

/// Called for every ARFrame. Commit only after a complete sample is queued.
/// Deadlines stay anchored to the first timestamp, so jitter never accumulates.
struct TimestampSampler {
    private(set) var sourceFrameCount = 0
    private var origin: Double?
    private var previousTimestamp: Double?
    private var selectedSecond = -1

    mutating func candidate(timestamp: Double) throws -> CaptureSample? {
        guard timestamp.isFinite, timestamp >= 0,
              previousTimestamp.map({ timestamp > $0 }) ?? true else {
            throw CaptureError.invalid("ARFrame 时间戳没有严格递增，请重新采集。")
        }
        let frameID = sourceFrameCount
        sourceFrameCount += 1
        previousTimestamp = timestamp
        if origin == nil { origin = timestamp }
        let relative = timestamp - origin!
        let second = Int(floor(relative))
        guard second > selectedSecond else { return nil }
        return CaptureSample(frameID: frameID, timestamp: timestamp,
                             relativeTimestamp: relative, targetSeconds: second)
    }

    mutating func commit(_ sample: CaptureSample) {
        selectedSecond = sample.targetSeconds
    }
}

struct RGBDFrame: Encodable {
    let frameID: Int
    let timestampSeconds: Double
    let relativeTimestampSeconds: Double
    let targetSeconds: Int
    let rgbSize: [Int] // width, height
    let depthSize: [Int]
    let intrinsics: [[Float]]
    let cameraToWorld: [[Float]]
    let trackingState: String
}

struct RGBDBundle: Encodable {
    let format = "sku-rgbd"
    let version = 1
    let sampleIntervalSeconds = 1
    let samplingPolicy = "first_complete_normal_tracking_frame_per_elapsed_second"
    let timestampClock = "ARFrame.timestamp_seconds_since_boot"
    let rgbEncoding = "png_srgb8"
    let rgbOrientation = "captured_image_unrotated_unmirrored"
    let depthEncoding = "uint16_png_millimeters_zero_invalid"
    let depthAlignment = "same_normalized_extent_as_rgb"
    let confidenceEncoding = "uint8_png_levels_0_1_2"
    let cameraConvention = "opencv_x_right_y_down_z_forward"
    let worldConvention = "arkit_right_handed_y_up_meters"
    let poseConvention = "camera_to_world"
    let matrixLayout = "row_major"
    let id: UUID
    let status: String
    let sourceFrameCount: Int
    let rejectedCandidateFrames: [String: Int]
    let errors: [String]
    let frames: [RGBDFrame]
}

final class DatasetEncoder {
    let id: UUID
    let directory: URL
    let previewRelativePath: String
    let framesRelativePath: String
    var onFrameSaved: ((Int) -> Void)?
    var onFailure: ((String) -> Void)?

    // Admission and the sampler belong to the main ARSession delegate queue.
    private var sampler = TimestampSampler()
    private var accepting = true
    private var rejected: [String: Int] = [:]
    private let slots = DispatchSemaphore(value: 2)
    private let queue = DispatchQueue(label: "sku.rgbd.writer", qos: .userInitiated)
    private let context = CIContext(options: [.cacheIntermediates: false])
    // The serial writer queue owns these two fields.
    private var frames: [RGBDFrame] = []
    private var failure: String?

    init() throws {
        id = UUID()
        let name = "sku-\(id.uuidString.lowercased()).rgbd"
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        directory = documents.appendingPathComponent(name, isDirectory: true)
        previewRelativePath = "\(name)/preview.png"
        framesRelativePath = "\(name)/frames"
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("frames"),
                                                withIntermediateDirectories: true)
        try writeManifest(status: "recording", sourceFrameCount: 0, rejected: [:])
    }

    func add(frame: ARFrame) {
        dispatchPrecondition(condition: .onQueue(.main))
        guard accepting else { return }
        do {
            guard let sample = try sampler.candidate(timestamp: frame.timestamp) else { return }
            guard case .normal = frame.camera.trackingState else {
                rejected["tracking_not_normal", default: 0] += 1
                return
            }
            guard let depth = frame.sceneDepth, let confidence = depth.confidenceMap else {
                rejected["missing_depth_or_confidence", default: 0] += 1
                return
            }
            guard slots.wait(timeout: .now()) == .success else {
                rejected["writer_busy", default: 0] += 1
                return
            }
            sampler.commit(sample)
            queue.async {
                defer { self.slots.signal() }
                guard self.failure == nil else { return }
                do {
                    try autoreleasepool {
                        let record = try self.write(frame: frame, depth: depth.depthMap,
                                                    confidence: confidence, sample: sample)
                        self.frames.append(record)
                    }
                    let count = self.frames.count
                    DispatchQueue.main.async { self.onFrameSaved?(count) }
                } catch {
                    let message = error.localizedDescription
                    self.failure = message
                    DispatchQueue.main.async {
                        self.accepting = false
                        self.onFailure?(message)
                    }
                }
            }
        } catch {
            accepting = false
            let message = error.localizedDescription
            queue.async { self.failure = message }
            onFailure?(message)
        }
    }

    /// Drain the bounded writer before publishing ready. Never block the UI queue.
    func finish(completion: @escaping (Result<Void, Error>) -> Void) {
        dispatchPrecondition(condition: .onQueue(.main))
        accepting = false
        let sourceCount = sampler.sourceFrameCount
        let rejectedFrames = rejected
        queue.async {
            if self.frames.isEmpty && self.failure == nil {
                self.failure = "没有采集到完整且跟踪稳定的 RGB-D 帧，请重新采集。"
            }
            do {
                try self.writeManifest(status: self.failure == nil ? "ready" : "error",
                                       sourceFrameCount: sourceCount, rejected: rejectedFrames)
                let result: Result<Void, Error>
                if let message = self.failure {
                    result = .failure(CaptureError.invalid(message))
                } else {
                    result = .success(())
                }
                DispatchQueue.main.async { completion(result) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
        }
    }

    static func cameraToWorldCV(_ arCameraToWorld: simd_float4x4) -> simd_float4x4 {
        var cvToAR = matrix_identity_float4x4
        cvToAR[1][1] = -1
        cvToAR[2][2] = -1
        // Change camera axes exactly once; retain ARKit world axes and meters.
        return arCameraToWorld * cvToAR
    }

    static func rows(_ matrix: simd_float3x3) -> [[Float]] {
        (0..<3).map { row in (0..<3).map { column in matrix[column][row] } }
    }

    static func rows(_ matrix: simd_float4x4) -> [[Float]] {
        (0..<4).map { row in (0..<4).map { column in matrix[column][row] } }
    }

    private func write(frame: ARFrame, depth: CVPixelBuffer, confidence: CVPixelBuffer,
                       sample: CaptureSample) throws -> RGBDFrame {
        let width = CVPixelBufferGetWidth(frame.capturedImage)
        let height = CVPixelBufferGetHeight(frame.capturedImage)
        let depthWidth = CVPixelBufferGetWidth(depth)
        let depthHeight = CVPixelBufferGetHeight(depth)
        guard Int(frame.camera.imageResolution.width) == width,
              Int(frame.camera.imageResolution.height) == height,
              CVPixelBufferGetWidth(confidence) == depthWidth,
              CVPixelBufferGetHeight(confidence) == depthHeight,
              width * depthHeight == height * depthWidth else {
            throw CaptureError.invalid("RGB、内参或 LiDAR/confidence 网格尺寸不一致。")
        }
        let name = String(format: "%06d", sample.frameID)
        let root = directory.appendingPathComponent("frames")
        let pending = root.appendingPathComponent(".pending-\(name)", isDirectory: true)
        let committed = root.appendingPathComponent(name, isDirectory: true)
        let record = RGBDFrame(
            frameID: sample.frameID, timestampSeconds: sample.timestamp,
            relativeTimestampSeconds: sample.relativeTimestamp, targetSeconds: sample.targetSeconds,
            rgbSize: [width, height], depthSize: [depthWidth, depthHeight],
            intrinsics: Self.rows(frame.camera.intrinsics),
            cameraToWorld: Self.rows(Self.cameraToWorldCV(frame.camera.transform)), trackingState: "normal")
        try FileManager.default.createDirectory(at: pending, withIntermediateDirectories: false)
        do {
            try writeRGB(frame.capturedImage, to: pending.appendingPathComponent("rgb.png"))
            try writeSensor(depth, isDepth: true, to: pending.appendingPathComponent("depth.png"))
            try writeSensor(confidence, isDepth: false, to: pending.appendingPathComponent("confidence.png"))
            try json(record).write(to: pending.appendingPathComponent("frame.json"), options: .atomic)
            // The directory rename commits all four members together.
            try FileManager.default.moveItem(at: pending, to: committed)
        } catch {
            try? FileManager.default.removeItem(at: pending)
            throw error
        }
        if frames.isEmpty {
            try FileManager.default.copyItem(at: committed.appendingPathComponent("rgb.png"),
                                             to: directory.appendingPathComponent("preview.png"))
        }
        return record
    }

    private func writeRGB(_ buffer: CVPixelBuffer, to url: URL) throws {
        let image = CIImage(cvPixelBuffer: buffer)
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let cgImage = context.createCGImage(image, from: image.extent, format: .RGBA8,
                                                 colorSpace: colorSpace),
              let output = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
            throw CaptureError.invalid("无法编码 RGB PNG。")
        }
        // No displayTransform, EXIF rotation, mirroring or resizing.
        CGImageDestinationAddImage(output, cgImage, [kCGImagePropertyOrientation as String: 1] as CFDictionary)
        guard CGImageDestinationFinalize(output) else {
            throw CaptureError.invalid("无法写入 RGB PNG，请检查剩余空间。")
        }
    }

    private func writeSensor(_ buffer: CVPixelBuffer, isDepth: Bool, to url: URL) throws {
        let format = isDepth ? kCVPixelFormatType_DepthFloat32 : kCVPixelFormatType_OneComponent8
        guard CVPixelBufferGetPixelFormatType(buffer) == format,
              CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else {
            throw CaptureError.invalid("不支持的 LiDAR/confidence 像素格式。")
        }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw CaptureError.invalid("无法读取 LiDAR/confidence 像素。")
        }
        let width = Int32(CVPixelBufferGetWidth(buffer))
        let height = Int32(CVPixelBufferGetHeight(buffer))
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let encoder: PngEncoder?
        if isDepth {
            encoder = PngEncoder(depth: base.assumingMemoryBound(to: Float.self), width: width,
                                 height: height, bytesPerRow: stride)
        } else {
            encoder = PngEncoder(confidence: base.assumingMemoryBound(to: UInt8.self), width: width,
                                 height: height, bytesPerRow: stride)
        }
        guard let data = encoder?.fileContents() else {
            throw CaptureError.invalid("LiDAR/confidence PNG 编码失败。")
        }
        try data.write(to: url, options: .atomic)
    }

    private func writeManifest(status: String, sourceFrameCount: Int, rejected: [String: Int]) throws {
        let manifest = RGBDBundle(id: id, status: status, sourceFrameCount: sourceFrameCount,
                                  rejectedCandidateFrames: rejected,
                                  errors: failure.map { [$0] } ?? [], frames: frames)
        try json(manifest).write(to: directory.appendingPathComponent("scan_bundle.json"), options: .atomic)
    }

    private func json<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(value)
    }
}
