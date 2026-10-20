# PromptDA selective skip-init 隔离原型

原型作者仅修改 `load_probe.py` 和本文档；没有修改生产 loader、PromptDA 或 DINO 源码。以下是原型阶段说明；coordinator 后续完整模型、性能与 GPU 验证见 [README.md](README.md)。

## API

- `skip_promptda_initialization()`：contextmanager，返回 `InitializationAudit`。只匹配本线程构造的 `nn.Linear` / `_ConvNd` 的真实 `weight`，阈值为 65,536 elements。跳 Kaiming 第一轮及 DINO Linear trunc-normal 第二轮；不跳 bias、tokens、norm、buffers。原生 DINO 对 `trunc_normal_` 早绑定，因此临时拦截 `nn.init._no_grad_trunc_normal_`；退出和异常恢复所有补丁。仅用于隔离 loader，不支持嵌套。
- `load_promptda_model(repo, checkpoint, skip=True)`：推荐 coordinator 使用的完整 API，返回 `(CPU eval model, metadata)`。沿用官方 `PromptDA.__init__` / `load_checkpoint`，官方 `torch.load(map_location='cpu')` 和 `state_dict` 的 `key[9:]` 映射未改。候选子类仅在 `load_state_dict` 前审计实际解码 state，随后 `super(... strict=True, assign=False)`。不存在文件立即失败，不沿用官方缺文件仅 warn 行为。
- 审计要求完整 state 名称、shape、dtype 一致，并要求所有 skipped weight 都仍存在于模型且被实际 checkpoint state 覆盖。metadata 记录每个 weight 的名称、aliases、shape、dtype、大小、跳过操作和 checkpoint shape/dtype；不写 tensor 或 hash。审计通过后才允许模型返回。
- `compare_models(native, candidate)`：直接 `torch.equal` 比较完整 state 和全部 named buffers（含 nonpersistent）；同时比较名称/shape/dtype。失败直接异常。

## CPU CLI（由 coordinator 串行执行）

```bash
UV_CACHE_DIR=/tmp/promptda-probe-uv-cache uv run --no-project \
  --python runtime/viewer-latency-20261017/test-env/bin/python python \
  reports/promptda-acceleration-validation-20261020/load_probe.py \
  --repo runtime/promptda-counterpart-20261019/PromptDA \
  --checkpoint /absolute/path/model.ckpt \
  --output reports/promptda-acceleration-validation-20261020/load_cpu.json
```

Docker 验证应使用相同源码/checkout/checkpoint 和目标环境 Python，并保留该环境 `torch_version` 结果。

CLI 分别记录首次 torch import、PromptDA import、hubconf dependency import。最终版本显式预热 hubconf 和工厂函数延迟导入的 vision_transformer，不构造模型；再按 native→skip、skip→native 的两对顺序计 CPU 构造+读取+strict copy，candidate 额外含 coverage audit（单独报告 audit 秒数）。直接比较不计入 loader 时间。两轮不是冷进程端到端收益，首次 import 不消失，也不外推 GPU/完整交付时间。只预热 hubconf 的第一版结果留作 preliminary，不用于最终对照。

同时保留一对完整 CPU 模型用于直接比较，峰值至少约两套模型加一套解码 checkpoint；报告 `ru_maxrss` 是整个进程峰值 KiB，包含比较，不能当成独立每 loader 峰值。每对结束释放模型、运行 GC。未测实际内存；若资源不允许应由 coordinator 停止并报告，而不是改用 hash 或磁盘 tensor dump。

## 已做验证与限制

2026-10-20，通过 `uv` 对文件 AST parse 和单个 256×256 Linear toy 验证（合计约2.4秒）：两轮初始化操作均记录；strict copy 后与原模型 state 完全相等；空 checkpoint 被拒绝；异常恢复底层初始化函数。没有完整模型、GPU、安装或计时实验。

toy 不证明完整模型 buffers 或输出等价；这正是 CPU CLI 和后续真实推理必须验证的部分。跳过 RNG 消耗可能改变后续保留的随机张量，CLI 的最终直接比较会捕获未被覆盖的差异，不用 RNG 补填去掩盖差异。

底层 `_no_grad_trunc_normal_` 是当前 PyTorch 的内部函数；目标 Docker torch 2.7.1 仍需实测。若不匹配则直接失败，不加兼容路径。全局 monkeypatch 只隔离本调用线程的目标，其他线程仍执行原生 initializer；不要与另一个初始化 monkeypatch 同时运行。预热 imports 后单进程 AB/BA 仍有 page cache、分配器和 CPU 频率等顺序影响，两个样本仅用于判定是否值得真实冷启动实验。
