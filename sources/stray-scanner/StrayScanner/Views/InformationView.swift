import SwiftUI

struct InformationView: View {
    var body: some View {
        ZStack {
            Color("BackgroundColor").ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    Text("SKU Scanner").font(.title).bold()
                    Text("保持镜头缓慢移动，并确认保存帧数持续增长。相机跟踪稳定且深度数据齐全时，每秒保存一组同步图像和深度。")
                    Text("导出到电脑").font(.headline)
                    Text("结束采集后，进入扫描详情，点击“导出数据包”。可使用 AirDrop 或“存储到文件”。也可连接电脑，从 Finder 的设备“文件”页复制 SKU Scanner 中的扫描文件夹。")
                    Text("导出会保留原始图像尺寸。保存失败或未结束的扫描不能导出为完成包。")
                    Text("开源许可").font(.headline)
                    Link("基于 Stray Scanner（MIT）",
                         destination: URL(string: "https://github.com/strayrobots/scanner")!)
                    Text(mitLicense).font(.caption)
                }
                .foregroundColor(Color("TextColor"))
                .padding(20)
            }
        }
    }
}

private let mitLicense = """
Copyright 2023 Stray Robots, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
