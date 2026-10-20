//
//  SessionDetailView.swift
//  StrayScanner
//
//  Created by Kenneth Blomqvist on 12/30/20.
//  Copyright © 2020 Stray Robots. All rights reserved.
//

import SwiftUI
import UIKit
import CoreData

class SessionDetailViewModel: ObservableObject {
    private var dataContext: NSManagedObjectContext?

    init() {
        let appDelegate = UIApplication.shared.delegate as? AppDelegate
        self.dataContext = appDelegate?.persistentContainer.viewContext
    }
    
    func title(recording: Recording) -> String {
        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .long
        dateFormatter.timeStyle = .short
        
        if let created = recording.createdAt {
            return dateFormatter.string(from: created)
        } else {
            return recording.name ?? "Recording"
        }
    }

    func delete(recording: Recording) {
        recording.deleteFiles()
        self.dataContext?.delete(recording)
        do {
            try self.dataContext?.save()
        } catch let error as NSError {
            print("Could not save recording. \(error), \(error.userInfo)")
        }
    }
}

struct SessionDetailView: View {
    @ObservedObject var viewModel = SessionDetailViewModel()
    var recording: Recording
    @Environment(\.presentationMode) var presentationMode: Binding<PresentationMode>
    @State private var showingShareSheet = false
    @State private var tempPackageURL: URL?
    @State private var isCreatingPackage = false
    @State private var shareError: String?

    var body: some View {
        let width = UIScreen.main.bounds.size.width
        let height = width * 0.75
        ZStack {
        Color("BackgroundColor")
            .edgesIgnoringSafeArea(.all)
        VStack {
            if let path = recording.absoluteRgbPath(),
               let preview = UIImage(contentsOfFile: path.path) {
                Image(uiImage: preview)
                    .resizable()
                    .scaledToFit()
                    .frame(width: width, height: height)
            }
            Text("同步 RGB + LiDAR · 每秒一帧")
                .foregroundColor(Color("TextColor"))
                .font(.caption)
            
            HStack(spacing: 20) {
                Button(action: shareItem) {
                    HStack {
                        if isCreatingPackage {
                            ProgressView()
                                .scaleEffect(0.8)
                        } else {
                            Image(systemName: "square.and.arrow.up")
                        }
                        Text(isCreatingPackage ? "打包中…" : "导出数据包")
                            .fixedSize()
                    }
                    .foregroundColor(isCreatingPackage ? .gray : .blue)
                    .frame(minWidth: 100)
                }
                .disabled(isCreatingPackage)
                
                Button(action: deleteItem) {
                    Text("删除").foregroundColor(Color("DangerColor"))
                }
                .disabled(isCreatingPackage)
            }
            .padding(.top, 20)
        }
        .navigationBarTitle(viewModel.title(recording: recording))
        .background(Color("BackgroundColor"))
        .sheet(isPresented: $showingShareSheet) {
            if let packageURL = tempPackageURL {
                ShareSheet(activityItems: [packageURL]) { activityType, completed, returnedItems, activityError in
                    DispatchQueue.main.async {
                        // Clean up temporary package after sharing
                        try? FileManager.default.removeItem(at: packageURL)
                        tempPackageURL = nil
                        showingShareSheet = false
                    }
                }
            }
        }
        .alert("Export Failed", isPresented: Binding(
            get: { shareError != nil },
            set: { if !$0 { shareError = nil } }
        )) {
            Button("OK", role: .cancel) { shareError = nil }
        } message: {
            Text(shareError ?? "")
        }
        }
    }

    func deleteItem() {
        viewModel.delete(recording: recording)
        self.presentationMode.wrappedValue.dismiss()
    }
    
    func shareItem() {
        guard let directory = recording.directoryPath() else {
            shareError = "找不到采集目录。"
            return
        }
        isCreatingPackage = true
        
        Task {
            do {
                let packageURL = try await ShareUtility.createShareableArchive(directory: directory)
                await MainActor.run {
                    tempPackageURL = packageURL
                    isCreatingPackage = false
                    showingShareSheet = true
                }
            } catch {
                await MainActor.run {
                    isCreatingPackage = false
                    shareError = error.localizedDescription
                }
            }
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]
    let applicationActivities: [UIActivity]? = nil
    let completionWithItemsHandler: UIActivityViewController.CompletionWithItemsHandler?
    
    init(activityItems: [Any], completionHandler: UIActivityViewController.CompletionWithItemsHandler? = nil) {
        self.activityItems = activityItems
        self.completionWithItemsHandler = completionHandler
    }
    
    func makeUIViewController(context: UIViewControllerRepresentableContext<ShareSheet>) -> UIActivityViewController {
        let controller = UIActivityViewController(
            activityItems: activityItems,
            applicationActivities: applicationActivities
        )
        controller.completionWithItemsHandler = completionWithItemsHandler
        return controller
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: UIViewControllerRepresentableContext<ShareSheet>) {
    }
}

struct SessionDetailView_Previews: PreviewProvider {
    static var recording: Recording = { () -> Recording in
        let rec = Recording()
        rec.id = UUID()
        rec.name = "Placeholder name"
        rec.createdAt = Date()
        rec.duration = 30.0
        return rec
    }()

    static var previews: some View {
        SessionDetailView(recording: recording)
    }
}
