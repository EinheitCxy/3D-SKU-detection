# 独立工程测试环境恢复

2026-10-17：恢复成功，仅完成环境前置检查。正式 CPU 性能实验预算仍 pending；未执行 pytest、benchmark、GPU 工作、模型推理或导出。

## 路径与边界

- 独立环境：`runtime/viewer-latency-20261017/test-env`。
- 根 `.venv` 保持原 dangling symlink：`/data/www/comfyui/3d-recognition-build/runtime/root-venv`。
- 未修改 `pyproject.toml`、`uv.lock` 或源码；未下载、换依赖、降级或改用 Python 3.12。
- uv cache 与 runtime 的 device 均为 `64768`，复用 cache 的 hardlinks。
- 唯一一次 sync 经过自动 escalation review 放行，为离线命令提供 uv cache 写锁权限；没有重试或替代恢复方案。

## 恢复命令与结果

```bash
ulimit -t 100
TIMEFORMAT='wall_s=%R user_cpu_s=%U system_cpu_s=%S'
{ time timeout --signal=TERM --kill-after=5s 120s env UV_PROJECT_ENVIRONMENT=/home/xingyu/3D_Recognization/runtime/viewer-latency-20261017/test-env UV_LINK_MODE=hardlink UV_CACHE_DIR=/home/xingyu/.cache/uv uv sync --offline --frozen --extra dev --no-install-project; } > runtime/viewer-latency-20261017/environment-sync.log 2>&1
```

退出码 0；CPython 3.11.11；181 packages，uv reported install 245 ms。Shell time：wall 0.340 s，user CPU 0.148 s，system CPU 0.930 s。低于本次恢复的 120 s wall / 100 s CPU 限额。

恢复后独立环境中 link count 为 1 的普通文件分配量为 **5,660,672 bytes**；这是独占文件数据块统计，未把共享 hardlinks 的体积当新增磁盘消耗，不含目录元数据和两个小日志。未达到 2 GiB 新增文件限额。

完整包清单和时间：[environment-sync.log](../../runtime/viewer-latency-20261017/environment-sync.log)。

## 实际导入检查

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-quiet-outputs-cache uv run --no-project --python /home/xingyu/3D_Recognization/runtime/viewer-latency-20261017/test-env/bin/python python -c 'import importlib, json, sys; versions = {name: importlib.import_module(name).__version__ for name in ("numpy", "PIL", "open3d", "torch", "pytest", "yaml")}; print(json.dumps({"python": sys.version, "executable": sys.executable, "versions": versions}, indent=2))' > runtime/viewer-latency-20261017/environment-imports.log 2>&1
```

退出码 0，`sys.executable` 指向独立 test-env；六个模块均实际导入成功：

| Module | Version |
| --- | --- |
| numpy | 1.26.4 |
| PIL | 12.3.0 |
| open3d | 0.19.0 |
| torch | 2.7.1+cu126 |
| pytest | 8.4.2 |
| yaml | 6.0.2 |

导入日志：[environment-imports.log](../../runtime/viewer-latency-20261017/environment-imports.log)。检查命令通过 `CUDA_VISIBLE_DEVICES=''` 屏蔽设备，未显式调用 CUDA API 或初始化模型。导入成功仅证明该环境可加载这六个模块；focused project tests 由主线程安排。
