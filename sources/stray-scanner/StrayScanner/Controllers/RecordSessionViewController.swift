// Copyright © 2020 Stray Robots. MIT license; see LICENSE.md.

import ARKit
import CoreData
import Metal
import UIKit

class MetalView: UIView {
    override class var layerClass: AnyClass { CAMetalLayer.self }
    override var layer: CAMetalLayer { super.layer as! CAMetalLayer }
}

class RecordSessionViewController: UIViewController, ARSessionDelegate {
    private let session = ARSession()
    private var renderer: CameraRenderer?
    private var dataContext: NSManagedObjectContext!
    private var datasetEncoder: DatasetEncoder?
    private var startedRecording: Date?
    private var updateLabelTimer: Timer?
    private var isFinishing = false
    private var isVisible = false
    private var savedFrames = 0
    private var backgroundTask: UIBackgroundTaskIdentifier = .invalid
    private var dismissFunction: (() -> Void)?

    @IBOutlet private var rgbView: MetalView!
    @IBOutlet private var depthView: MetalView!
    @IBOutlet private var recordButton: RecordButton!
    @IBOutlet private var timeLabel: UILabel!
    @IBOutlet private var fpsButton: UIButton!

    func setDismissFunction(_ callback: (() -> Void)?) {
        dismissFunction = callback
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        guard let appDelegate = UIApplication.shared.delegate as? AppDelegate else { return }
        dataContext = appDelegate.persistentContainer.viewContext
        renderer = CameraRenderer(rgbLayer: rgbView.layer, depthLayer: depthView.layer)
        depthView.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(viewTapped)))
        rgbView.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(viewTapped)))
        view.backgroundColor = UIColor(named: "BackgroundColor")
        session.delegate = self
        session.delegateQueue = .main
        recordButton.setCallback { [weak self] recording in
            if recording { self?.startRecording() } else { self?.stopRecording() }
        }
        fpsButton.layer.masksToBounds = true
        fpsButton.layer.cornerRadius = 12
        fpsButton.isUserInteractionEnabled = false
        updateSampleCount()
        NotificationCenter.default.addObserver(self, selector: #selector(willResignActive),
                                               name: UIApplication.willResignActiveNotification, object: nil)
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        isVisible = true
        guard ARWorldTrackingConfiguration.isSupported,
              ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) else {
            recordButton.isUserInteractionEnabled = false
            showError("采集需要配备 LiDAR 的 iPhone Pro 或 iPad Pro。")
            return
        }
        let config = ARWorldTrackingConfiguration()
        config.frameSemantics.insert(.sceneDepth)
        // Keep continuous AR tracking and the device's native capturedImage size.
        session.run(config)
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        isVisible = false
        stopRecording(dismiss: false)
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        session.pause()
    }

    deinit {
        NotificationCenter.default.removeObserver(self)
        updateLabelTimer?.invalidate()
    }

    private func startRecording() {
        guard datasetEncoder == nil, !isFinishing else { return }
        do {
            let encoder = try DatasetEncoder()
            encoder.onFrameSaved = { [weak self] count in
                self?.savedFrames = count
                self?.updateSampleCount()
            }
            encoder.onFailure = { [weak self] _ in self?.stopRecording() }
            datasetEncoder = encoder
            savedFrames = 0
            startedRecording = Date()
            UIApplication.shared.isIdleTimerDisabled = true
            updateTime()
            updateSampleCount()
            updateLabelTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
                self?.updateTime()
            }
        } catch {
            recordButton.reset()
            showError(error.localizedDescription)
        }
    }

    private func stopRecording(dismiss: Bool = true) {
        guard let encoder = datasetEncoder, let started = startedRecording, !isFinishing else { return }
        let duration = Date().timeIntervalSince(started)
        startedRecording = nil
        isFinishing = true
        recordButton.reset()
        recordButton.isUserInteractionEnabled = false
        updateLabelTimer?.invalidate()
        updateLabelTimer = nil
        UIApplication.shared.isIdleTimerDisabled = false
        fpsButton.setTitle("保存中…", for: .normal)
        backgroundTask = UIApplication.shared.beginBackgroundTask(withName: "Finish RGB-D export") { [weak self] in
            self?.endBackgroundTask()
        }
        encoder.finish { [self] result in
            defer {
                datasetEncoder = nil
                isFinishing = false
                recordButton.isUserInteractionEnabled = true
                endBackgroundTask()
            }
            do {
                try result.get()
                try saveRecording(started: started, duration: duration, encoder: encoder)
                if dismiss && isVisible { dismissFunction?() }
            } catch {
                if isVisible {
                    showError(error.localizedDescription)
                } else {
                    print("RGB-D 保存失败：\(error.localizedDescription)")
                }
            }
        }
    }

    private func endBackgroundTask() {
        if backgroundTask != .invalid {
            UIApplication.shared.endBackgroundTask(backgroundTask)
            backgroundTask = .invalid
        }
    }

    private func saveRecording(started: Date, duration: TimeInterval, encoder: DatasetEncoder) throws {
        let recording = Recording(context: dataContext)
        recording.id = encoder.id
        recording.duration = duration
        recording.createdAt = started
        recording.name = "SKU RGB-D"
        // Existing storage fields now point to the first-frame preview and frame directory.
        recording.rgbFilePath = encoder.previewRelativePath
        recording.depthFilePath = encoder.framesRelativePath
        do {
            try dataContext.save()
        } catch {
            dataContext.delete(recording)
            throw error
        }
        NotificationCenter.default.post(name: NSNotification.Name("sessionsChanged"), object: nil)
    }

    private func showError(_ message: String) {
        let alert = UIAlertController(title: "无法完成采集", message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "关闭", style: .default) { [weak self] _ in
            self?.dismissFunction?()
        })
        present(alert, animated: true)
    }

    private func updateTime() {
        guard let started = startedRecording else { return }
        let seconds = Int(Date().timeIntervalSince(started))
        timeLabel.text = String(format: "%02d:%02d:%02d", seconds / 3600, seconds / 60 % 60, seconds % 60)
    }

    private func updateSampleCount() {
        if !isFinishing { fpsButton.setTitle("1 Hz · \(savedFrames)", for: .normal) }
    }

    @objc private func willResignActive() {
        stopRecording()
    }

    @objc func viewTapped() {
        guard let renderer = renderer else { return }
        let showRGB = renderer.renderMode == .depth
        renderer.renderMode = showRGB ? .rgb : .depth
        rgbView.isHidden = !showRGB
        depthView.isHidden = showRGB
    }

    // Retain the existing storyboard action; sampling is fixed at one elapsed second.
    @IBAction func fpsButtonTapped() {
        updateSampleCount()
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        renderer?.render(frame: frame)
        datasetEncoder?.add(frame: frame)
    }

    func sessionWasInterrupted(_ session: ARSession) {
        stopRecording()
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        if datasetEncoder != nil {
            stopRecording()
        } else if isVisible {
            showError(error.localizedDescription)
        }
    }
}
