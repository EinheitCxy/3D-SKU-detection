# Execution journal

## T01 — 2026-10-18 23:23 CST，开始

使用commit-helper、autoresearch、get-available-resources和work-journal。三个medium workers按STATUS分工；root保存初始main/HEAD/index/所有tracked diff与untracked名称。初始暂存为空，未执行add/reset/stash/clean。

首个sandbox网络读取无法解析github.com；按明确push授权提升权限进行remote/visibility只读核验，尚无自动审批拒绝。资源读取成功：GPU2约19GiB余量，原两个resident在场；GPU0/1不用于本次实验。不会重新跑上一轮已通过的完整测试或GPU模型pipeline。

实验第一候选：CPU source_world projection不变，Torch显式gather/float64模拟NumPy bilinear提升，整合mask和固定方向延展；不直接把grid_sample视为等价替换。先小规模正确性再实际GPU计时；未通过等价门槛的实现不改默认。

## T02 — 2026-10-18 23:29 CST，提交与真实GPU样本

上一轮改动拆成21ea29c（DA3）、092047d（ranges）、bb5e276（ground opt-in）、741a52d（GLB）、37019bd（profiling文档）。初次push因公开报告含本机元数据被自动审批拒绝；Rick明确回复“允许公开报告元数据并推送全部提交”后，按同一origin/main目标重试。没有绕过拒绝或强推。

提交核查发现CSV默认CRLF与生成SVG行尾空格触发diff-check，仅将本人报告规范为LF/去行尾空格；数值不变。修正历史snapshot搬移后的相对链接，以及算法文档当前默认encoder=2。其他tracked patches与初态逐项相同，原有未跟踪文件保留；混合core/surfel文档的其他hunks未提交。静态37脚本AST通过，159本地链接中只有README原有fd-videos-sdr链接不存在，未扩展到无关修复；12runtime链接明确为本机证据。

新候选CPU30项通过，静态shape方向延展后仅受影响10项补测通过；12tiny smoke全部RGBA/geometry一致。修复probe错误状态和空比较通过问题；固定tile8/batch≤1024，每frame RGB/depth各上传一次。阶段整合不是single fused kernel。

GPU2 sample-01 complete/equal：29frames×1024input=29696，baked29316，29组corners/raw RGBA全相等。same-frame已解码输入上的native wall总1.526704s、candidate总1.341692s（含首次context/H2D/D2H）；首framecandidate.388182s，后28frame合计native1.476881s/candidate.953510s。decode/input准备1.112241s单列，imports3.413846s单列。Torch allocated peak38892544bytes、reserved56623104bytes；不等于driver显存或完整pipeline。

通过sample门槛，进入同generation完整GLB对照：新native CPU进程（encoder2/source_world），随后新GPU2 Torch进程。两臂OMP/BLAS/MKL=4、无并发heavy实验；保留首次context、上传下载及最终编码/写入。完整交付收益和等价仍待结果。

## T03 — 2026-10-18 23:36 CST，完整GLB通过／push成功

Native complete66.589322s，Torch complete44.671011s，减少21.918312s／32.916%；脚本含import至就绪69.928963→48.084271s（不含外部uv／解释器启动）。GPU计时覆盖首context、H2D／D2H、CPU投影、mask／RGB／方向延展及LOD／编码／写入。候选whole59.170519s包含就绪后约11.086s验证，未混入交付时间。

两臂完整GLB document、accessor bufferView字节及31页128295600decodedRGBA texels全相等，缺页／shape差异0。Viewer输入2402589，LOD1297686，1280batch，baked1282759／三角2565518，文件121197580bytes。RGB／depth各上传29次，batch≤1024。没有检查完整全程raw tiles或整个GLB文件字节，保留sample raw与全交付decoded的声明边界。

Torch allocated峰38922240bytes（37.119MiB）、reserved56623104bytes（54MiB）是本进程allocator生命周期峰，不含完整context／driver；candidate RSS还含验证，不能与native export RSS比较。GPU2收尾仅原6221/8550resident，19038MiB空闲；不触碰GPU0/1新出现的其他任务。

两位medium最终独立结果复核PASS，canonical新增§13。生产默认仍CPU，不把本组件21.918s从上一轮186.894s直接扣成新pipeline结果。

Git发布：公开元数据授权后，首次直连push长时间无HTTP响应；curl证实TLS已建立但15s无HTTP，既有本机代理返回301，代理只读确认origin/main仍8554aaa。核实本人进程组后仅终止该挂起push，保留exit143；用单次git http.proxy配置重试，成功8554aaa→37019bd。无全局Git配置修改／TLS降级／强推。当前实验报告、脚本和测试另行提交并push；其他原工作保留。
