# 实验记录

## E01 — 2026-09-24（Asia/Shanghai）— 实验准备

用户明确授权调用 subagents 尝试三个方向，并要求主线程指导、监督、校验。已有实验版与生产版不同；原始和融合版既有灰面面积35.18%与42.91%不能直接解释为真实覆盖。固定比较条件：方案1固定融合几何；方案2/3相对未融合uniform stride4，各保持217049点。记录未贴图面积、原相机深度支持覆盖、视觉缺陷、成本；保留负结果。范围见CONTRACT。执行 planned，证据结论未评估。

## E02 — 2026-09-24（Asia/Shanghai）— 实现和首轮运行

三名worker分别交付texture/adaptive/coverage模块，focused测试回执为8/11/3项通过；主线程射线camera-Z与灰色前景遮挡测试1项通过。独立审查未发现阻断几何/射线/严格选源的错误，但要求区分相对primary-only恢复面积与相对旧patch基线的实际变化，并明确整货架mask为深度置信度有效域、非商品分割。已增加逐面source变化统计；仅澄清候选源码中的mask文案，不改变门槛或结果。资源检查显示可用内存约386GB、磁盘33.9GB；本轮仅CPU，GPU不可见不作主机无GPU结论。

uniform与fused重新生成，灰面面积35.1765%和42.9132%，与既有基线一致。纹理候选固定206465个surfel，灰面35.5701%，此时尚未视觉验收；绝不能据此宣布外观改善。独立对照采样候选和7视角raycast继续。运行日志与产物：runtime/surfel-coverage-20260924/。

## E03 — 2026-09-24（Asia/Shanghai）— 首轮判定和一次纠偏

七视角独立raycast完成。方案1相对fused贴图支持覆盖增加7.19个百分点，几何完全一致；逐面统计也发现0.552%总面积丢失旧贴图，不能称为只增不减。方案2相对uniform深度支持覆盖增加5.11个百分点、贴图支持覆盖增加5.12个百分点，七视角均有深度支持收益，但无命中和后方不一致仍小幅增加。方案3初版虽然灰面接近零，无命中却达到21.67%，侧视/正视出现大量表面消失，否决。

主线程及独立reviewer检查18组固定视角与来源色截图，前两项只保留为研究候选，未发布。方案3根因是空间单元数大于点预算，第一轮实际变成全局质量筛选。按既定每候选最多一次修复的限制，记录coverage_bounded：每16×16源图块至少保留15/16基线点，只释放受支持的内部锚点，按空间分布交换新观测单元，点数和scale4不变。原worker未启动修复，主线程撤回其写权限并接手；实现测试1项通过，再交新的只读reviewer检查。初版负结果永久保留，不把修正版当预注册独立假设。

首轮Vite构建意外复制无关public资产，临时目录曾约5.1GB，超过自定3GiB限制；删除仅本轮生成副本并设置publicDir:false，未动原资产。随后本轮目录降至约438MB。最终目录大小在E04记录。

## E04 — 2026-09-24 15:04（Asia/Shanghai）— 修复负结果与交付

coverage_bounded保留205542点、交换11507点（5.30%），官方运行52.72秒，独立七视角raycast约1.92秒。相对uniform深度支持覆盖下降0.367个百分点，贴图支持覆盖下降0.304个百分点，七视角深度支持全部略降；无命中降至0.00095%，避免了初版大面积漏空，却没有达到覆盖收益。移除/新增圆盘世界三角面积和12.99/54.91，说明固定点数与scale仍不足以控制世界footprint。独立reviewer未发现使固定输入实验失效的blocker，强调source锚点保护与物理表面novelty均不能直接作为可见覆盖证明。

修正版六组截图检查完成：原有灰色长尖盘、层叠和接缝仍在，整景增加少量漂浮灰片；无可确认文字连续性收益。初次浏览器因前轮服务已停止而连接失败，保留失败回执后重启相同127.0.0.1:8769服务，重试通过。四组候选共24组、48张单面板截图，均无最终pageerror，相机拖动同步通过。追加一次默认页检查确认uniform/adaptive和side60，并保存默认整景对照。

最终24项针对性测试通过（不重跑无关项），独立页面Vite构建成功（保留624kB单chunk体积提示，不作无关拆分），本轮目录约543MiB。git diff为空：本轮仅新增隔离实验、页面和文档；既有大量untracked用户资产未改。所有实验与截图命令已结束，仅静态服务保留。决策：优先方案2、保留方案1继续研究、方案3两版否决；没有组合、部署或生产准确率声明。完整证据见runtime/surfel-coverage-20260924/REPORT.md。

## E05 — 2026-09-24 15:08（Asia/Shanghai）— 授权局域网展示

Rick明确要求使用192.168.2.6局域网接口。主机读取确认eno0为192.168.2.6/23且UP；原服务仅监听127.0.0.1:8769。新增同端口、仅绑定192.168.2.6的静态服务，根目录仍是本轮产物，原本机入口保留。浏览器经LAN IP实际加载页面、manifest、uniform和adaptive两个GLB，均HTTP 200、pageerror为空，默认方法正确。回执lan-browser-report.json；这是服务器浏览器对自身LAN IP的验证，没有声称独立测试另一台局域网客户端。未改防火墙、系统服务、重建模型或重复算法测试。

## E06 — 2026-09-24 16:09（Asia/Shanghai）— 新增生产部署授权与接入

Rick调用commit-helper要求部署方案2并push，随后指定中英文Viewer都使用便宜subagents更新。main=1450f97、Docker=746ad3a、英文=8d9a537、中文=6d4a6d9，暂存区均为空，原有实验/数据/配置等untracked保留。中文分支为并行更新建立runtime/worktrees/viewer-zn-surfel；两名Luna负责中英文协议适配，主线程负责生产采样、打包、部署和最终验收。

正式采样保留200万商品/50万背景预算、global ID均分/回流、背景过滤/体素代表。深度/有效域/global ID边界提高采样权重，密集几何与候选池分离，背景尺度计入体素抽稀。native U/V保持原义，新增Surfel v3逐点两维Float16尺度；Viewer只在圆盘展开使用，深度预测与容忍不放宽，旧v2单位尺度继续受支持。独立代码审查没有blocker，修正边界绝不跨越的过强注释。

针对验证通过：Python18项，根Viewer13项、英文14项、中文10项及TypeScript检查；打包2项首次因带连字符worktree目录的pytest包导入失败，改用--import-mode=importlib后通过，没有改变产品代码规避问题。主机有约34GB可用磁盘、361GB可用RAM。使用现有离线build_code_update.sh从运行镜像sam31-20260918构建surfel-adaptive-20260924；不是完整模型/依赖重建。真实导出、浏览器、部署和push待完成。

## E07 — 2026-09-24 16:35（Asia/Shanghai）— 真实导出及浏览器验证

新后端镜像复用29帧504缓存，导出2,363,298点（1,863,298商品+500,000背景）、393个global ID、711项观测；v3尺度文件9,453,192字节，范围[0.5,8]，中位数[1,1]。CPU导出78.374秒、打包0.410秒，ZIP 109,896,191字节。首次容器用UID1000导致只读输入权限失败，改为当前用户UID1006成功，未修改输入权限。启动检查openapi HTTP200；没有GPU模型重推理或COS上传。

完整236万点场景在SwiftShader下加载到canvas，但截图先后30秒/180秒超时，不能声称整景WebGL或性能通过。保留失败日志后从实际包原slot提取global ID3的18,771点切片，保留native U/V、scale、纹理和深度。中英文候选镜像实际HTTP读取该ZIP，显示、旋转、放大、sidebar选择、canvas拾取、隐藏后不可拾取均通过，pageerror/consoleError为空；主线程检查两语言截图，接缝和漂浮边缘仍存在。

额外由Luna在隔离runtime目录构建单点WebGL probe：同native U/V，scale(1,1)→(3,1)可见宽度54→162px，两个中心pick均slot0、隐藏后null；破坏原生+x邻格深度后宽度162→96px，证实原生连续性裁切仍生效。blend tolerance只作shader静态检查，没有独立遮挡重叠实验。57项测试、三套TS/Vite构建通过；独立审查无blocker。实验覆盖和+4.1%耗时结论不能转用于此生产预算。

## E08 — 2026-09-24 16:43（Asia/Shanghai）— 部署、提交及push完成

先替换中文global-id-viewer（5175），新增英文global-id-viewer-en（192.168.2.6:5178），再替换global-id-mapping-local（8011）。三个镜像标签均surfel-adaptive-20260924；后端保留GPU2、只读.env绑定和原运行参数。实际LAN的两个HTML/JS与openapi全部HTTP200。原中文和后端容器分别保留为global-id-viewer-before-adaptive-20260924、global-id-mapping-local-before-adaptive-20260924；本轮三个candidate容器已清理。

四个精确allowlist提交已推送：origin/main df8e001（12文件）；origin/docker ac35bcd（3文件）；gitlab/visualization_en d23c166（6文件）；gitlab/visualization_zn a7a07ee（6文件）。未发布运行数据、env、模型、构建dist或原有untracked；没有强推、改写历史或向Harbor推镜像。

首次GitHub/GitLab push均被自动审批要求补足归属证据。只读确认GitHub现有公开仓库的viewerPermission=ADMIN；GitLab SSH身份为chenxingyu，该身份可读取既有两个upstream分支，匿名API不公开项目。补充证据后原命令获准并均成功；没有绕过审批或改换目的地。旧ZIP不会自动更新，需要重新生成任务包才能获得新优化。完整场景硬件GPU性能和实际GPU模型/COS业务链未重跑，这些不是本轮已完成验证的范围。
