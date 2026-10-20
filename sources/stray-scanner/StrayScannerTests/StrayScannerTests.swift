import ARKit
import UIKit
import XCTest
@testable import StrayScanner

final class StrayScannerTests: XCTestCase {
    func testRecordButtonLayoutKeepsOnlyOneTapAction() {
        let button = RecordButton(frame: CGRect(x: 0, y: 0, width: 80, height: 80))
        button.layoutSubviews()
        button.layoutSubviews()
        XCTAssertEqual(button.gestureRecognizers?.count, 1)
        XCTAssertEqual(button.layer.sublayers?.count, 2)
    }

    func testTimestampSamplingDoesNotDriftOrFillMissingSeconds() throws {
        var sampler = TimestampSampler()
        var selected: [CaptureSample] = []
        for timestamp in [100.0, 100.49, 100.99, 101.01, 101.98, 102.001, 104.02] {
            if let sample = try sampler.candidate(timestamp: timestamp) {
                selected.append(sample)
                sampler.commit(sample)
            }
        }
        XCTAssertEqual(selected.map { $0.frameID }, [0, 3, 5, 6])
        XCTAssertEqual(selected.map { $0.targetSeconds }, [0, 1, 2, 4])
        XCTAssertEqual(selected.last!.timestamp, 104.02)
        XCTAssertEqual(sampler.sourceFrameCount, 7)
    }

    func testIncompleteCandidateCanBeRetriedWithinTheSameSecond() throws {
        var sampler = TimestampSampler()
        _ = try sampler.candidate(timestamp: 10) // Missing LiDAR: not committed.
        let second = try XCTUnwrap(sampler.candidate(timestamp: 10.04))
        XCTAssertEqual(second.frameID, 1)
        XCTAssertEqual(second.targetSeconds, 0)
        sampler.commit(second)
        XCTAssertNil(try sampler.candidate(timestamp: 10.5))
        XCTAssertEqual(try sampler.candidate(timestamp: 11.02)?.targetSeconds, 1)
    }

    func testNonmonotonicOrNonfiniteTimeIsAnError() throws {
        for invalid in [10.0, 9.0, Double.nan, Double.infinity] {
            var sampler = TimestampSampler()
            _ = try sampler.candidate(timestamp: 10)
            XCTAssertThrowsError(try sampler.candidate(timestamp: invalid))
        }
    }

    func testCameraAxesChangeExactlyOnceAndTranslationStaysInMeters() {
        var ar = matrix_identity_float4x4
        ar.columns.0 = SIMD4<Float>(0, 1, 0, 0)
        ar.columns.1 = SIMD4<Float>(-1, 0, 0, 0)
        ar.columns.3 = SIMD4<Float>(2, 3, 4, 1)
        let cv = DatasetEncoder.cameraToWorldCV(ar)
        XCTAssertEqual(DatasetEncoder.rows(cv),
                       [[0, 1, 0, 2], [1, 0, 0, 3], [0, 0, -1, 4], [0, 0, 0, 1]])
        XCTAssertEqual(cv * SIMD4<Float>(0, 0, 2, 1), SIMD4<Float>(2, 3, 2, 1))
    }

    func testJSONUsesRowMajorIntrinsicsAndTheServerWireKeys() throws {
        let k = simd_float3x3(columns: (SIMD3<Float>(8, 0, 0),
                                      SIMD3<Float>(0, 9, 0), SIMD3<Float>(3, 2, 1)))
        let frame = RGBDFrame(frameID: 61, timestampSeconds: 101.02,
                              relativeTimestampSeconds: 1.02, targetSeconds: 1,
                              rgbSize: [8, 6], depthSize: [4, 3],
                              intrinsics: DatasetEncoder.rows(k),
                              cameraToWorld: DatasetEncoder.rows(matrix_identity_float4x4),
                              trackingState: "normal")
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let data = try encoder.encode(frame)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["frame_id"] as? Int, 61)
        XCTAssertEqual(json["rgb_size"] as? [Int], [8, 6])
        XCTAssertEqual(json["intrinsics"] as? [[Float]], [[8, 0, 3], [0, 9, 2], [0, 0, 1]])
        XCTAssertNotNil(json["camera_to_world"])
        XCTAssertEqual(json["relative_timestamp_seconds"] as? Double, 1.02)
    }
}
