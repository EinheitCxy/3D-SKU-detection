import argparse
import logging
import sys
import threading
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from re import T
from time import perf_counter
from typing import Any, Dict, Optional, TypedDict

import colorlog
import numpy as np

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SAVE_ROOT = PROJECT_ROOT / "Output"


def _resolve_save_root(value: str | None) -> Path:
    """Resolve omitted and relative output roots against the repository root."""
    if value is None or not value.strip():
        return DEFAULT_SAVE_ROOT
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _is_reusable_da3_cache(cache_path: Path) -> bool:
    """Return whether a DA3 cache has the minimum schema-v3 metric contract."""
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            schema = cache["cache_schema_version"]
            is_metric = cache["is_metric"]
            return (
                schema.shape == ()
                and schema.dtype.kind in "iu"
                and int(schema.item()) == 3
                and is_metric.shape == ()
                and is_metric.dtype.kind in "iu"
                and int(is_metric.item()) == 1
            )
    except (KeyError, OSError, ValueError):
        return False


# 确保可以从仓库根或任意 CWD 导入本目录模块
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


class _StartEndColorFilter(logging.Filter):
    """Colorize messages that contain START/END without changing file logs.

    - Lines containing 'start' → cyan message
    - Lines containing 'end' → green message (or red when contains 'fail')
    """

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            msg = record.getMessage()
            lower = msg.lower()
            from colorlog import escape_codes

            color = ""
            if "start" in lower:
                color = escape_codes.get("cyan", "")
            elif "end" in lower:
                color = (
                    escape_codes.get("red", "")
                    if "fail" in lower
                    else escape_codes.get("green", "")
                )

            # Prefix only the message part; keep level color as-is
            record.msg_color = color
            # ColoredFormatter appends %(reset)s at the end, no need for extra reset
        except (AttributeError, KeyError, ImportError):
            # Gracefully handle missing colorlog or attribute errors in log formatting
            record.msg_color = ""
        return True


def _configure_logging_to_save_root(save_root: Path) -> logging.Logger:
    """配置全局日志，使每次运行仅在 save_root 中生成一个日志文件。

    - 文件: <save_root>/run_YYYYMMDD_HHMMSS.log
    - 同时输出到控制台
    - 清理已存在的 root handlers，避免重复日志
    """
    save_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = save_root / f"run_{ts}.log"

    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    # Root logger at DEBUG so file captures debug-only details.
    root_logger.setLevel(logging.DEBUG)
    # 文件日志保留完整格式，控制台在 TTY 下使用彩色格式，在重定向到文件时使用纯文本，避免ANSI转义序列写入日志文件
    file_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = RotatingFileHandler(
        str(log_file), maxBytes=10_000_000, backupCount=1, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    root_logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)

    if sys.stdout.isatty():
        # 仅在交互式终端中启用彩色输出
        console_fmt = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s - %(msg_color)s%(message)s%(reset)s",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        sh.setFormatter(console_fmt)
        # Only colorize console output based on message content
        sh.addFilter(_StartEndColorFilter())
    else:
        # 非TTY（例如重定向到文件）时使用纯文本，避免ANSI转义序列写入外部日志
        sh.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    root_logger.addHandler(sh)

    logger = logging.getLogger(__name__)
    logger.info(f"日志已写入: {log_file}")
    return logger


logger = logging.getLogger(__name__)
_MATCHING_INFERENCE_LOCK = threading.RLock()


def _serialize_matching_inference(call):
    """Serialize matching because its sampling stack uses process-global RNG."""
    @wraps(call)
    def wrapped(self, *args, **kwargs):
        reference_idx = kwargs.get("reference_idx")
        if reference_idx is None:
            reference_idx = args[2]
        with _MATCHING_INFERENCE_LOCK:
            logger.info(
                "Correctness serialization: reference %d enters the global "
                "Python/NumPy/torch RNG matching boundary",
                reference_idx,
            )
            return call(self, *args, **kwargs)

    return wrapped


class StepResult(TypedDict, total=False):
    success: bool
    error: Optional[str]
    duration_s: float
    details: Dict[str, Any]


class SKUDetectionMain:
    """3D SKU Detection系统主控制器"""

    def __init__(self) -> None:
        # 以仓库根为基准，避免依赖当前工作目录
        self.default_dataset = str(PROJECT_ROOT / "imdata" / "floor_display2")
        self.save_root: Optional[Path] = None  # 可选的输出保存根目录
        self.config_path: Optional[Path] = None
        # DA3 is the repository default for 3D matching.
        self.match_backend: str = "da3"
        logger.info("初始化3D SKU Detection主程序")

    def show_banner(self) -> None:
        """显示程序横幅（自适应对齐，宽字符友好）。"""
        import sys
        from datetime import datetime

        # 运行时信息
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        # 安全获取torch信息
        try:
            import torch

            torch_version = torch.__version__
            cuda_available = "Yes" if torch.cuda.is_available() else "No"
        except ImportError:
            # torch未安装或不可用
            torch_version = "N/A"
            cuda_available = "N/A"

        # 宽字符显示宽度（CJK/Emoji）
        def _disp_width(s: str) -> int:
            try:
                from wcwidth import wcswidth  # type: ignore

                w = wcswidth(s)
                return max(w, 0)
            except ImportError:
                # wcwidth未安装，降级为ASCII长度
                return len(s)

        # 文本内容（不含边框）
        title1 = "3D SKU Detection System"
        title2 = "RetailEye 商品计数分析平台 v2.0"
        lines = [
            " 核心功能:",
            " 1. SKU匹配推理（点追踪 + 3D投影）  2. 检出框可视化",
            " 3. SKU聚类分析                       4. 匹配准确性评估",
            " 5. 改进的SKU计数（去重优化）          6. 3D场景重建",
            " 运行环境:",
            f" 时间: {current_time}    Python: {python_version}",
            f" PyTorch: {torch_version}    CUDA: {cuda_available}",
            " 快速开始:",
            " • 完整流水线: --mode pipeline --dataset <dataset_dir>",
            " • 仅匹配推理: --mode concise --algorithm point_tracking",
            " • 交互模式:   --mode interactive",
            " • 帮助文档:   --help",
        ]

        # 计算内部最大宽度（包含行首一个空格）
        max_inner = max(_disp_width(title1), _disp_width(title2))
        for s in lines:
            max_inner = max(max_inner, _disp_width(s))

        # 预留左右边距各1空格
        inner_width = max_inner
        total_width = inner_width + 2  # 左右各一个空格

        # 构造边框
        top = "╔" + ("═" * total_width) + "╗"
        sep = "╠" + ("═" * total_width) + "╣"
        bottom = "╚" + ("═" * total_width) + "╝"

        def pad_line(text: str, center: bool = False) -> str:
            w = _disp_width(text)
            if w > inner_width:
                text = text[: max(0, len(text) - (w - inner_width))]
                w = _disp_width(text)
            if center:
                # 居中：左右尽量均衡，右侧补齐
                left_spaces = (inner_width - w) // 2
                right_spaces = inner_width - w - left_spaces
                return f"║{' ' * (left_spaces + 1)}{text}{' ' * (right_spaces + 1)}║"
            else:
                # 左对齐：右侧补齐
                pad = inner_width - w
                return f"║ {text}{' ' * pad} ║"

        out = [
            top,
            pad_line(title1, center=True),
            pad_line(title2, center=True),
            sep,
        ]
        # 分段插入
        out.append(pad_line(lines[0]))  # 核心功能
        out.extend(pad_line(x) for x in lines[1:4])
        out.append(sep)
        out.append(pad_line(lines[4]))  # 运行环境
        out.extend(pad_line(x) for x in lines[5:7])
        out.append(sep)
        out.append(pad_line(lines[7]))  # 快速开始
        out.extend(pad_line(x) for x in lines[8:])
        out.append(bottom)

        print("\n".join(out))

    def validate_dataset(self, dataset_path: str) -> bool:
        """验证数据集目录结构"""
        dataset = Path(dataset_path)
        required_dirs = ["images", "detections_results"]

        if not dataset.exists():
            logger.error(f"数据集目录不存在: {dataset_path}")
            return False

        for req_dir in required_dirs:
            if not (dataset / req_dir).exists():
                logger.error(f"缺少必需目录: {dataset_path}/{req_dir}")
                return False

        return True

    def run_sku_matching(
        self,
        dataset_path: str,
        algorithm: str = "point_tracking",
        *,
        reference_idx: int = 0,
        max_images: int = 50,
        device: str = "cuda",
        save_json: bool = False,
        batch_all_refs: bool = True,
        backend: str = "vggt",
        parallel_refs: int = 1,
        match_overrides: dict = None,
        enable_profiling: bool = False,
    ) -> StepResult:
        """运行SKU匹配推理，支持批量将每张图片作为参考图像运行。

        - 当 batch_all_refs=True 时：遍历 images/ 中数字命名且在 detections_results/ 有有效 objects 的每个图片，依次作为参考图运行。
        - 否则：仅以 reference_idx 指定的单张图片作为参考图运行。
        - backend: 3D重建模型后端 (vggt/pi3/da3)，用于3D算法时选择数据源
        - parallel_refs: 并行处理的参考图片数（>1 时启用线程池；推荐 pi3/da3 后端使用，vggt 不支持）
        - enable_profiling: 启用 per-stage 计时 instrumentation（默认 False，零开销）
        """
        from utils.profiling import StageTimer as _StageTimer
        from utils.profiling import dump_stages as _dump_prof
        from utils.profiling import log_stages_sorted as _log_prof
        from utils.profiling import set_enabled as _set_prof_enabled

        _set_prof_enabled(enable_profiling)
        start = perf_counter()
        try:
            logger.info("开始SKU匹配推理")
            if batch_all_refs:
                # 批量处理所有有效图片作为参考图片
                # 使用 utils.data_utils.load_detections 作为唯一标准源
                from utils.data_utils import load_detections

                dataset = Path(dataset_path)
                detection_dir = dataset / "detections_results"

                # 使用标准load_detections获取有效检测文件索引
                try:
                    detections_with_index = load_detections(
                        str(detection_dir), return_index_map=True
                    )
                    # 提取文件编号（即图片索引）
                    valid_indices = sorted(
                        [file_num for file_num, _ in detections_with_index]
                    )
                    logger.info(f"找到 {len(valid_indices)} 个有效参考图片")
                except (FileNotFoundError, ValueError) as e:
                    logger.error(f"无法加载检测结果: {e}")
                    return {
                        "success": False,
                        "error": str(e),
                        "duration_s": perf_counter() - start,
                    }

                # 构建参数列表（list of (system_ref_idx, filename_idx)）
                tasks = [(i, fn_idx) for i, fn_idx in enumerate(valid_indices)]

                if parallel_refs > 1:
                    # 并行处理：适合 pi3/da3 缓存后端（3D 数据只读，线程安全）
                    from concurrent.futures import ThreadPoolExecutor, as_completed

                    workers = min(parallel_refs, len(tasks))
                    logger.info(
                        f"并行处理 {len(tasks)} 个参考图片（worker 数: {workers}）"
                    )

                    def _task(args):
                        sys_idx, fn_idx = args
                        logger.debug(
                            f"[并行] 处理参考图片 {fn_idx} -> 系统索引: {sys_idx}"
                        )
                        return self._run_single_matching(
                            dataset_path,
                            algorithm,
                            sys_idx,
                            max_images,
                            device,
                            save_json,
                            backend,
                            match_overrides,
                            enable_profiling=enable_profiling,
                        )

                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = {executor.submit(_task, t): t for t in tasks}
                        for fut in as_completed(futures):
                            t = futures[fut]
                            try:
                                fut.result()
                            except Exception as exc:
                                logger.warning(f"参考图片 {t[1]} 处理失败: {exc}")
                else:
                    # 串行处理（默认）
                    for i, filename_idx in tasks:
                        logger.debug(
                            f"处理参考图片 {filename_idx} ({i+1}/{len(valid_indices)}) -> 系统索引: {i}"
                        )
                        self._run_single_matching(
                            dataset_path,
                            algorithm,
                            i,
                            max_images,
                            device,
                            save_json,
                            backend,
                            match_overrides,
                            enable_profiling=enable_profiling,
                        )

                duration = perf_counter() - start
                _StageTimer.record("batch_all_refs_total", duration)
                logger.info(
                    f"匹配完成 - 耗时 {duration:.2f}s，处理 {len(valid_indices)} 个参考图片"
                )
                return {"success": True, "duration_s": duration}
            else:
                # 单个参考图片处理
                return self._run_single_matching(
                    dataset_path,
                    algorithm,
                    reference_idx,
                    max_images,
                    device,
                    save_json,
                    backend,
                    match_overrides,
                    enable_profiling=enable_profiling,
                )

        except (
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
        ) as e:
            duration = perf_counter() - start
            _StageTimer.record("batch_all_refs_total", duration)
            logger.error(
                f"END matching duration={duration:.2f}s result=fail error={e}",
                exc_info=True,
            )
            return {"success": False, "error": str(e), "duration_s": duration}
        finally:
            if enable_profiling:
                _prof_dir = (
                    (self.save_root / Path(dataset_path).name)
                    if self.save_root
                    else Path(dataset_path)
                )
                _prof_path = (
                    _prof_dir
                    / f"profiling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                _dump_prof(str(_prof_path))
                _log_prof(logger)
                logger.info(f"[PROF] dumped to {_prof_path}")

    @_serialize_matching_inference
    def _run_single_matching(
        self,
        dataset_path: str,
        algorithm: str,
        reference_idx: int,
        max_images: int,
        device: str,
        save_json: bool,
        backend: str = "vggt",
        match_overrides: dict = None,
        enable_profiling: bool = False,
    ) -> StepResult:
        """运行单个参考图片的SKU匹配推理（内部使用）。

        Args:
            backend: 3D重建模型后端 (vggt/pi3)
        """
        start = perf_counter()
        try:
            logger.debug(
                f"单次匹配 - 算法: {algorithm}, 后端: {backend}, 参考索引: {reference_idx}"
            )

            from src.inference import main as inference_main

            dataset = Path(dataset_path)
            image_folder = dataset / "images"
            detection_dir = dataset / "detections_results"
            # 输出根目录：优先使用 save_root，其次使用数据集目录
            output_dir = (self.save_root / dataset.name) if self.save_root else dataset

            argv = [
                "--image_folder",
                str(image_folder),
                "--detection_dir",
                str(detection_dir),
                "--output_dir",
                str(output_dir),
                "--sam3_mask_cache_root",
                str(output_dir / "sam3_mask_cache" / "v2"),
                "--algorithm",
                algorithm,
                "--reference_idx",
                str(reference_idx),
                "--max_images",
                str(max_images),
                "--device",
                device,
                "--backend",
                backend,
            ]
            if save_json:
                argv.append("--save_json")
            if enable_profiling:
                argv.append("--enable_profiling")
            if self.config_path is not None:
                argv.extend(["--config", str(self.config_path)])
            # 透传 3D 阈值覆盖（网格扫描用）
            if match_overrides:
                for _k, _v in match_overrides.items():
                    argv.extend([f"--{_k}", str(_v)])

            inference_main(argv)

            duration = perf_counter() - start
            logger.debug(f"单次匹配完成 - 耗时 {duration:.2f}s")
            return {"success": True, "duration_s": duration}
        except (
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
        ) as e:
            duration = perf_counter() - start
            logger.error(
                f"END matching_single duration={duration:.2f}s result=fail error={e}",
                exc_info=True,
            )
            return {"success": False, "error": str(e), "duration_s": duration}
    def run_detection_visualization(
        self,
        dataset_path: str,
        detection_dir: str = None,
        output_suffix: str = "imgs_w_bboxes",
    ) -> StepResult:
        """运行检出框可视化

        Args:
            dataset_path: 数据集路径
            detection_dir: 检测结果目录（默认使用 detections_results）
            output_suffix: 输出目录后缀（默认 imgs_w_bboxes）
        """
        start = perf_counter()
        original_argv = sys.argv.copy()
        try:
            logger.info("开始检出框可视化")

            from src.draw_detection_boxes import main as viz_main

            dataset = Path(dataset_path)
            image_dir = dataset / "images"

            # 如果未指定detection_dir，使用默认的detections_results
            if detection_dir is None:
                detection_dir = dataset / "detections_results"
            else:
                detection_dir = Path(detection_dir)

            # 输出目录：若指定 save_root，则写到 save_root/<dataset_name>/<output_suffix>
            output_viz_dir = (
                (self.save_root / dataset.name / output_suffix)
                if self.save_root
                else (DEFAULT_SAVE_ROOT / dataset.name / output_suffix)
            ).resolve()
            output_viz_dir.mkdir(parents=True, exist_ok=True)

            sys.argv = [
                "draw_detection_boxes.py",
                "--image_dir",
                str(image_dir),
                "--detection_dir",
                str(detection_dir),
                "--output_dir",
                str(output_viz_dir),
                "--no_confidence",
                "--no_class",
            ]

            viz_main()

            duration = perf_counter() - start
            logger.info(f"可视化完成 - 耗时 {duration:.2f}s")
            logger.debug(f"输出目录: {output_viz_dir}")
            return {
                "success": True,
                "duration_s": duration,
                "details": {"output_dir": str(output_viz_dir)},
            }
        except (
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
        ) as e:
            duration = perf_counter() - start
            logger.error(
                f"END visualization duration={duration:.2f}s result=fail error={e}",
                exc_info=True,
            )
            return {"success": False, "error": str(e), "duration_s": duration}
        finally:
            sys.argv = original_argv

    def run_improved_sku_analysis(
        self,
        dataset_path: str,
        algorithm: str = "point_tracking",
        backend: str | None = None,
    ) -> StepResult:
        """运行改进的SKU计数分析 (去重优化)，报告写入 <dataset_name>/output_reports/report_*.txt（或 --save_root）"""
        start = perf_counter()
        try:
            logger.info("开始SKU计数分析")

            from src.improved_sku_analyzer import ImprovedSKUCountAnalyzer

            dataset = Path(dataset_path)
            detection_dir = dataset / "detections_results"

            # 根据算法类型动态选择匹配结果目录
            base_dir = self.save_root / dataset.name if self.save_root else dataset
            if algorithm == "point_tracking":
                summary_dir = base_dir / "output_pt"
            elif algorithm in ("3d", "3d_mapping"):
                if backend:
                    summary_dir = base_dir / f"output_3dmapping_{backend}"
                else:
                    summary_dir = base_dir / "output_3dmapping"
            else:
                summary_dir = base_dir / "output_pt"

            if not summary_dir.exists():
                msg = f"匹配结果目录不存在: {summary_dir}，请先运行SKU匹配推理"
                logger.warning(msg)
                duration = perf_counter() - start
                return {"success": False, "error": msg, "duration_s": duration}

            # 检测使用的算法
            algorithm_name = (
                "Point Tracking" if "output_pt" in str(summary_dir) else "3D Mapping"
            )

            analyzer = ImprovedSKUCountAnalyzer(str(detection_dir), str(summary_dir))
            result = analyzer.analyze_with_filtering()

            # 计算统计信息
            pairs = result["pairs"]
            hit_ratios = [p["hit_ratio"] for p in pairs]
            avg_hit_ratio = sum(hit_ratios) / len(hit_ratios) if hit_ratios else 0
            ref_images = len(set(p["ref_idx"] for p in pairs))
            target_images = len(set(p["target_idx"] for p in pairs))

            # 报告目录：若指定 save_root，则保存到 save_root/output_reports/<dataset_name>
            reports_dir = (
                self.save_root / dataset.name / "output_reports"
                if self.save_root
                else DEFAULT_SAVE_ROOT / dataset.name / "output_reports"
            )
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_file = (
                reports_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )

            with report_file.open("w", encoding="utf-8") as f:
                f.write("=" * 70 + "\n")
                f.write("SKU 计数分析报告\n")
                f.write("=" * 70 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据集: {dataset_path}\n")
                f.write(f"匹配算法: {algorithm_name}\n")
                f.write("-" * 70 + "\n\n")

                f.write("【匹配统计】\n")
                f.write(f"  原始匹配数: {result['original_matches']}\n")
                f.write(f"  过滤后匹配数: {result['filtered_matches']}\n")
                f.write(
                    f"  去重减少: {result['original_matches'] - result['filtered_matches']} 个冗余匹配 "
                    f"({(result['original_matches'] - result['filtered_matches']) / result['original_matches'] * 100:.1f}%)\n"
                )
                f.write(f"  平均 Hit Ratio: {avg_hit_ratio:.3f}\n")
                f.write(
                    f"  涉及图片: {ref_images} 个参考图片, {target_images} 个目标图片\n\n"
                )

                f.write("【详细匹配结果】\n")
                for i, pair in enumerate(pairs, 1):
                    f.write(
                        f"{i:3d}. Ref({pair['ref_idx']},{pair['ref_id']}) → "
                        f"Target({pair['target_idx']},{pair['target_id']}) "
                        f"hit_ratio={pair['hit_ratio']:.3f}\n"
                    )
                f.write("\n" + "=" * 70 + "\n")

            duration = perf_counter() - start
            logger.info(
                f"SKU分析完成 - 最终匹配数: {result['filtered_matches']}, 耗时 {duration:.2f}s"
            )
            logger.debug(f"报告文件: {report_file}")
            return {
                "success": True,
                "duration_s": duration,
                "details": {"report_file": str(report_file)},
            }
        except (
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
        ) as e:
            duration = perf_counter() - start
            logger.error(
                f"END improved_analysis duration={duration:.2f}s result=fail error={e}",
                exc_info=True,
            )
            return {"success": False, "error": str(e), "duration_s": duration}

    def run_accuracy_evaluation(
        self, dataset_path: str, *, backend: str = "pt"
    ) -> StepResult:
        """Evaluate this run's backend-specific matching output."""
        start = perf_counter()
        try:
            logger.info("开始准确性评估")

            benchmark_csv = PROJECT_ROOT / "imdata" / "picture_mapping_benchmark.csv"
            if not benchmark_csv.exists():
                msg = f"基准数据文件不存在: {benchmark_csv}"
                logger.error(msg)
                duration = perf_counter() - start
                return {"success": False, "error": msg, "duration_s": duration}

            dataset = Path(dataset_path)
            output_root = self.save_root if self.save_root is not None else DEFAULT_SAVE_ROOT
            normalized_backend = "pt" if backend == "point_tracking" else backend
            output_subdir = (
                "output_pt"
                if normalized_backend == "pt"
                else f"output_3dmapping_{normalized_backend}"
            )
            output_dir = output_root / dataset.name / output_subdir
            if not output_dir.exists():
                msg = "匹配结果目录不存在，请先运行SKU匹配推理"
                logger.warning(msg)
                duration = perf_counter() - start
                return {"success": False, "error": msg, "duration_s": duration}

            script_path = (
                PROJECT_ROOT / "scripts" / "3d" / "evaluation" / "accuracy_evaluation.sh"
            )
            if script_path.exists():
                import subprocess

                result = subprocess.run(
                    [
                        "bash",
                        str(script_path),
                        dataset.name,
                        "--backend",
                        normalized_backend,
                        "--save-root",
                        str(output_root),
                    ],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    duration = perf_counter() - start
                    logger.info(f"评估完成 - 耗时 {duration:.2f}s")
                    return {"success": True, "duration_s": duration}
                else:
                    duration = perf_counter() - start
                    logger.error(f"评估失败: {result.stderr}")
                    return {
                        "success": False,
                        "error": result.stderr,
                        "duration_s": duration,
                    }

            msg = f"准确性评估脚本不存在: {script_path}"
            logger.error(msg)
            return {"success": False, "error": msg, "duration_s": perf_counter() - start}
        except (OSError, subprocess.SubprocessError, RuntimeError) as e:
            duration = perf_counter() - start
            logger.error(
                f"END evaluation duration={duration:.2f}s result=fail error={e}",
                exc_info=True,
            )
            return {"success": False, "error": str(e), "duration_s": duration}

    def run_reconstruction(
        self,
        dataset_path: str,
        *,
        device: str | None = None,
        output_filename: str = "reconstruction.glb",
        backend: str = "vggt",
        conf_thres: float = 50.0,
        model_path: str | None = None,
        show_cam: bool = True,
        mask_black_bg: bool = True,
        mask_white_bg: bool = True,
        mask_sky: bool = True,
    ) -> StepResult:
        """生成3D点云/GLB（支持后端：已注册的 ReconstructorBase 子类，当前 da3/pi3；vggt 可选）。

        - 输入图片目录：<dataset>/images
        - 输出GLB：<save_root>/<dataset_name>/reconstruction_{backend}.glb（或 <dataset>/reconstruction_{backend}.glb）
        - 后端实例化走 RECONSTRUCTOR_REGISTRY（src/__init__.py 注册），新增后端无需改本方法。
        """
        start = perf_counter()
        try:
            from src import RECONSTRUCTOR_REGISTRY, get_reconstructor

            use_backend = (backend or "vggt").lower()
            if use_backend not in RECONSTRUCTOR_REGISTRY:
                available = ", ".join(sorted(RECONSTRUCTOR_REGISTRY)) or "(无)"
                hint = (
                    "（如需启用 vggt，请恢复 src/__init__.py 中 VGGT3DReconstructor 的注册）"
                    if use_backend == "vggt"
                    else ""
                )
                raise ValueError(
                    f"未知/未启用的重建后端: {backend}. 已注册: {available}{hint}"
                )

            dataset = Path(dataset_path)
            if not dataset.exists():
                raise ValueError(f"数据集路径不存在: {dataset_path}")
            image_dir = dataset / "images"
            if not image_dir.exists():
                msg = f"图片目录不存在: {image_dir}"
                logger.error(msg)
                return {"success": False, "error": msg, "duration_s": 0.0}

            # 选择输出位置：GLB文件放到对应的cache目录中
            output_dir = (self.save_root / dataset.name) if self.save_root else dataset
            cache_dir = output_dir / f"{use_backend}_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            # 自动在文件名中添加模型名称（如果文件名中还没有）
            output_path = Path(output_filename)
            if (
                use_backend not in output_path.stem
            ):  # 检查文件名（不含扩展名）中是否已包含模型名
                # 在扩展名前插入模型名称：reconstruction.glb -> reconstruction_vggt.glb
                new_filename = f"{output_path.stem}_{use_backend}{output_path.suffix}"
                output_file = cache_dir / new_filename
            else:
                output_file = cache_dir / output_filename

            logger.info(f"开始3D重建[{use_backend}]: {image_dir} → {output_file}")

            # 通过注册表获取后端类（新增后端只需 @register_reconstructor + 在 src/__init__.py 导入）
            recon_cls = get_reconstructor(use_backend)
            recon = recon_cls(device=device, model_path=model_path)
            # vggt 的 export_glb 需要 mask_* 参数（经 reconstruct_from_directory 的 **kwargs 透传）；
            # da3/pi3 无此参数，忽略即可。
            extra_kwargs: dict = {}
            if use_backend == "vggt":
                extra_kwargs = {
                    "mask_black_bg": mask_black_bg,
                    "mask_white_bg": mask_white_bg,
                    "mask_sky": mask_sky,
                }
            result_path = recon.reconstruct_from_directory(
                input_dir=str(image_dir),
                output_path=str(output_file),
                conf_thres=conf_thres,
                show_cam=show_cam,
                save_predictions=True,
                **extra_kwargs,
            )

            duration = perf_counter() - start
            logger.info(f"3D重建完成 - 耗时 {duration:.2f}s")
            return {
                "success": True,
                "duration_s": duration,
                "details": {"output_file": str(result_path)},
            }
        except (
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
        ) as e:
            duration = perf_counter() - start
            logger.error(
                f"END reconstruct duration={duration:.2f}s result=fail error={e}",
                exc_info=True,
            )
            return {"success": False, "error": str(e), "duration_s": duration}

    def run_dedup_sequence(
        self,
        dataset_path: str,
        algorithm: str = "point_tracking",
        backend: str | None = None,
    ) -> StepResult:
        """顺序去重：对 1..N（或指定上界）生成去重后的检测 JSON。

        Args:
            dataset_path: 数据集路径
            algorithm: 算法类型 'point_tracking'/'3d_mapping'
            backend: 3D算法后端 'vggt'/'pi3'，仅在algorithm='3d_mapping'时生效
        """
        start = perf_counter()
        try:
            logger.info(f"开始顺序去重 (algorithm: {algorithm}, backend: {backend})")
            from src.deduplicate_detections import (
                deduplicate_sequence,
                resolve_dataset_paths,
            )

            dataset_dir = Path(dataset_path)
            if not dataset_dir.exists():
                raise ValueError(f"数据集路径不存在: {dataset_path}")

            paths = resolve_dataset_paths(dataset_dir)
            dataset_name = dataset_dir.name

            # 输出目录：Output/<dataset_name>/dedup_detections/
            output_base = (
                self.save_root if self.save_root is not None else DEFAULT_SAVE_ROOT
            )

            result = deduplicate_sequence(
                paths,
                output_root=output_base,  # 模块内部会追加 dataset_name
                max_image=None,  # 处理所有图片
                same_names=True,  # 默认同名输出 (1.json, 2.json, ...)
                dedup_mode="any",  # 默认使用所有匹配进行去重
                min_hit_ratio=0.0,  # 默认不过滤命中率
                output_subdir="dedup_detections",  # 指定子目录名
                algorithm=algorithm,  # 传递算法类型
                backend=backend,  # 传递后端类型
            )

            # 实际输出路径是 output_base/dataset_name/dedup_detections/
            actual_output_dir = output_base / dataset_name / "dedup_detections"
            duration = perf_counter() - start
            logger.info(f"去重完成 - 处理 {len(result)} 个文件, 耗时 {duration:.2f}s")
            logger.debug(f"输出目录: {actual_output_dir}")
            return {
                "success": True,
                "duration_s": duration,
                "details": {"count": len(result), "output_dir": str(actual_output_dir)},
            }
        except (
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
        ) as e:
            duration = perf_counter() - start
            logger.error(
                f"END dedup_sequence duration={duration:.2f}s result=fail error={e}",
                exc_info=True,
            )
            return {"success": False, "error": str(e), "duration_s": duration}

    def run_complete_pipeline(
        self,
        dataset_path: str,
        algorithm: str = "point_tracking",
        model_path: str | None = None,
    ) -> Dict[str, bool]:
        """运行完整的SKU计数流水线（包含3D重建），返回每步是否成功的摘要。"""
        logger.info("开始完整的SKU计数流水线（包含3D重建）")
        summary: Dict[str, bool] = {}

        if not self.validate_dataset(dataset_path):
            return {"validation": False}
        summary["validation"] = True

        # 1. 3D重建（如果使用3D算法）
        if "3d" in algorithm:
            match_backend = (
                self.match_backend if hasattr(self, "match_backend") else "vggt"
            )
            # DA3 的 canonical 产物是 schema-v3 metric predictions.npz；其他后端用 GLB。
            dataset = Path(dataset_path)
            output_dir = (
                (self.save_root / dataset.name)
                if self.save_root is not None
                else dataset
            )
            cache_dir = output_dir / f"{match_backend}_cache"
            if match_backend == "da3":
                expected_result = cache_dir / "predictions.npz"
                reusable = _is_reusable_da3_cache(expected_result)
            else:
                base_output = Path("reconstruction.glb")
                if match_backend not in base_output.stem:
                    filename = f"{base_output.stem}_{match_backend}{base_output.suffix}"
                    expected_result = cache_dir / filename
                else:
                    expected_result = cache_dir / base_output
                reusable = expected_result.exists()

            if reusable:
                logger.info(f"步骤1: 检测到可复用3D重建结果 {expected_result}，跳过3D重建")
                summary["reconstruction"] = True
            else:
                logger.info(f"步骤1: 3D重建 (backend: {match_backend})")
                recon = self.run_reconstruction(
                    dataset_path, backend=match_backend, model_path=model_path
                )
                summary["reconstruction"] = bool(recon.get("success", False))

                if not summary["reconstruction"]:
                    logger.error("3D重建失败，无法继续3D匹配流程")
                    return summary
        else:
            logger.info("步骤1: 跳过3D重建（使用 Point Tracking 算法）")
            summary["reconstruction"] = True  # 标记为成功（不需要）

        # 2. 原始检测框可视化
        logger.info("步骤2: 原始检测框可视化")
        viz = self.run_detection_visualization(dataset_path)
        summary["visualization"] = bool(viz.get("success", False))

        # 3. SKU匹配推理
        logger.info(f"步骤3: SKU匹配推理 (algorithm: {algorithm})")
        match_backend = self.match_backend if "3d" in algorithm else "vggt"
        match = self.run_sku_matching(
            dataset_path, algorithm, batch_all_refs=True, backend=match_backend
        )
        summary["matching"] = bool(match.get("success", False))

        # 3. SKU计数分析
        analysis = self.run_improved_sku_analysis(
            dataset_path, algorithm=algorithm, backend=match_backend
        )
        summary["improved_analysis"] = bool(analysis.get("success", False))

        # 4. 顺序去重（默认包含以便一键产出去重JSON）
        dedup = self.run_dedup_sequence(
            dataset_path, algorithm=algorithm, backend=match_backend
        )
        summary["dedup"] = bool(dedup.get("success", False))

        # 5. 去重后的检测框可视化
        if summary["dedup"]:
            dataset = Path(dataset_path)
            dataset_name = dataset.name
            output_base = (
                self.save_root if self.save_root is not None else DEFAULT_SAVE_ROOT
            )
            # deduplicate_sequence 输出到 output_base/dataset_name/dedup_detections/
            dedup_detection_dir = output_base / dataset_name / "dedup_detections"

            if dedup_detection_dir.exists() and any(dedup_detection_dir.glob("*.json")):
                logger.info("开始可视化去重后的检测框...")
                dedup_viz = self.run_detection_visualization(
                    dataset_path,
                    detection_dir=str(dedup_detection_dir),
                    output_suffix="dedup_imgs_w_bboxes",
                )
                summary["dedup_visualization"] = bool(dedup_viz.get("success", False))
            else:
                logger.warning(f"去重检测目录为空或不存在: {dedup_detection_dir}")
                summary["dedup_visualization"] = False
        else:
            summary["dedup_visualization"] = False

        # 6. 准确性评估 (可选)
        acc = self.run_accuracy_evaluation(
            dataset_path, backend=match_backend if "3d" in algorithm else "pt"
        )
        summary["accuracy_evaluation"] = bool(acc.get("success", False))

        logger.info("=== 流水线执行结果 ===")
        for step, ok in summary.items():
            status = "成功" if ok else "失败"
            logger.info(f"{step:20s}: {status}")

        return summary

    def run_concise_pipeline(
        self, dataset_path: str, algorithm: str = "point_tracking"
    ) -> Dict[str, bool]:
        """运行精简流水线 - 仅SKU匹配和准确性评估"""
        logger.info("开始精简流水线 - SKU Matching + Accuracy evaluation")
        summary: Dict[str, bool] = {}

        if not self.validate_dataset(dataset_path):
            return {"validation": False}
        summary["validation"] = True

        match_backend = self.match_backend if "3d" in algorithm else "vggt"
        match = self.run_sku_matching(dataset_path, algorithm, backend=match_backend)
        summary["matching"] = bool(match.get("success", False))

        acc = self.run_accuracy_evaluation(
            dataset_path, backend=backend if "3d" in algorithm else "pt"
        )
        summary["accuracy_evaluation"] = bool(acc.get("success", False))

        logger.info("=== 精简流水线执行结果 ===")
        for step, ok in summary.items():
            status = "成功" if ok else "失败"
            logger.info(f"{step:20s}: {status}")

        return summary

    def interactive_mode(self) -> None:
        """交互模式"""
        self.show_banner()

        while True:
            print("\n请选择操作:")
            print("1. 运行完整流水线")
            print("2. 运行精简流水线 (SKU Matching + Accuracy evaluation)")
            print("3. 更改数据集路径")
            print("4. 3D重建 (VGGT/PI3/DA3)")
            print("0. 退出")

            # 显示数据集路径（如果是绝对路径，显示相对于 PROJECT_ROOT 的路径）
            try:
                dataset_display = Path(self.default_dataset).relative_to(PROJECT_ROOT)
            except ValueError:
                dataset_display = self.default_dataset

            choice = input(
                f"\n当前数据集: {dataset_display}\n请输入选择 (0-4): "
            ).strip()

            if choice == "0":
                logger.info("退出程序")
                break
            elif choice == "1":
                algorithm = (
                    input(
                        "选择算法 (point_tracking/3d) [默认: point_tracking]: "
                    ).strip()
                    or "point_tracking"
                )
                if "3d" in algorithm:
                    while True:
                        backend = (
                            input("选择3D匹配后端 (vggt/pi3/da3): ").strip().lower()
                        )
                        if backend in ("vggt", "pi3", "da3"):
                            break
                        logger.warning(f"无效的后端 '{backend}'，请重新输入")
                    self.match_backend = backend
                self.run_complete_pipeline(self.default_dataset, algorithm)
            elif choice == "2":
                algorithm = (
                    input("选择算法 (point_tracking/3d/both) [默认: both]: ").strip()
                    or "both"
                )
                if "3d" in algorithm:
                    while True:
                        backend = (
                            input("选择3D匹配后端 (vggt/pi3/da3): ").strip().lower()
                        )
                        if backend in ("vggt", "pi3", "da3"):
                            break
                        logger.warning(f"无效的后端 '{backend}'，请重新输入")
                    self.match_backend = backend
                self.run_concise_pipeline(self.default_dataset, algorithm)
            elif choice == "3":
                dataset_name = input(
                    "输入数据集名称 (如 floor_display2，或仅输入数字如 15): "
                ).strip()
                if dataset_name:
                    # 支持仅输入数字：自动补全为 floor_display{num}
                    if dataset_name.isdigit():
                        dataset_name = f"floor_display{dataset_name}"

                    # 自动拼接完整路径: PROJECT_ROOT / "imdata" / dataset_name
                    new_path = str(PROJECT_ROOT / "imdata" / dataset_name)
                    if self.validate_dataset(new_path):
                        self.default_dataset = new_path
                        # 显示完整路径和输出目录信息
                        output_base = (
                            self.save_root
                            if self.save_root is not None
                            else DEFAULT_SAVE_ROOT
                        )
                        output_dir = output_base / dataset_name
                        logger.info(f"数据集已更改为: {new_path}")
                        logger.info(f"输出目录将使用: {output_dir}")
                    else:
                        logger.warning(
                            f"数据集 '{dataset_name}' 验证失败，保持当前数据集"
                        )
            elif choice == "4":
                while True:
                    backend = input("选择重建后端 (vggt/pi3/da3): ").strip().lower()
                    if backend in ("vggt", "pi3", "da3"):
                        break
                    logger.warning(f"无效的后端 '{backend}'，请重新输入")
                res = self.run_reconstruction(self.default_dataset, backend=backend)
                print(
                    f"3D重建: {'成功' if res.get('success') else '失败'}，耗时 {res.get('duration_s', 0):.2f}s"
                )
            else:
                print("无效选择，请重试")


def main() -> None:
    # 预解析 --config
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径（默认根 config.yaml，如存在）",
    )
    known, _ = pre.parse_known_args()

    # 从 YAML 读取默认值
    from utils import (
        extract_main_settings,
        extract_reconstruction_settings,
        load_yaml_config,
    )

    config_path = None
    default_cfg = PROJECT_ROOT / "config.yaml"
    if known.config:
        config_path = Path(known.config)
    elif default_cfg.exists():
        config_path = default_cfg

    if config_path is None or not config_path.exists():
        logger.error(f"配置文件不存在: {config_path or default_cfg}")
        sys.exit(1)

    try:
        data = load_yaml_config(config_path)
        yaml_main = extract_main_settings(data)
        yaml_recon = extract_reconstruction_settings(data)
    except Exception as e:
        logger.error(f"YAML 配置文件读取失败: {config_path}, 错误: {e}")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="3D SKU Detection系统主程序", parents=[pre]
    )

    # 处理数据集路径：如果是相对路径，转换为绝对路径
    dataset_from_yaml = yaml_main.get(
        "dataset", str(PROJECT_ROOT / "imdata" / "floor_display2")
    )
    dataset_path = Path(dataset_from_yaml)
    if not dataset_path.is_absolute():
        # 相对路径：相对于 PROJECT_ROOT
        dataset_path = PROJECT_ROOT / dataset_path
    dataset_default = str(dataset_path)

    parser.add_argument(
        "--dataset",
        type=str,
        default=dataset_default,
        help="数据集目录路径（绝对或相对PROJECT_ROOT）",
    )
    parser.add_argument(
        "--floor",
        type=int,
        default=None,
        help="楼层展示数据集编号，例如 15 表示使用 imdata/floor_display15",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=yaml_main.get("mode", "interactive"),
        choices=[
            "interactive",
            "pipeline",
            "concise",
            "analyzer",
            "dedup",
            "ground-stack-area",
            "reconstruct",
            "viewer-web",
        ],
        help="运行模式: interactive(交互), pipeline(完整), concise(匹配), analyzer(仅分析), dedup(去重), ground-stack-area(DA3地堆footprint并集面积), reconstruct(3D重建), viewer-web(静态bundle导出)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default=yaml_main.get("algorithm", "3d"),
        choices=["point_tracking", "3d", "both"],
        help="匹配算法选择",
    )
    # 透传给 inference.py 的关键参数
    parser.add_argument(
        "--reference_idx",
        type=int,
        default=int(yaml_main.get("reference_idx", 0)),
        help="参考图像索引",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=int(yaml_main.get("max_images", 50)),
        help="最大处理图像数量",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=yaml_main.get("device", "cuda"),
        help="计算设备 (cuda/cpu)",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        default=bool(yaml_main.get("save_json", False)),
        help="保存匹配结果为 JSON",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=yaml_main.get("save_root", str(DEFAULT_SAVE_ROOT)),
        help="输出保存根目录。例如：/path/to/outputs",
    )
    # 3D重建/匹配专用参数
    parser.add_argument(
        "--recon_conf_thres",
        type=float,
        default=float(yaml_recon.get("conf_thres", 50.0)),
        help="3D导出置信度阈值(0-100)",
    )
    parser.add_argument(
        "--recon_output",
        type=str,
        default=yaml_recon.get("output", "reconstruction.glb"),
        help="3D重建输出文件名",
    )
    parser.add_argument(
        "--recon_backend",
        type=str,
        default=yaml_recon.get("backend", "vggt"),
        choices=["vggt", "pi3", "da3"],
        help="3D重建后端 (vggt|pi3|da3)",
    )
    parser.add_argument(
        "--recon_model_path",
        type=str,
        default=yaml_recon.get("model_path", None),
        help="3D重建模型权重路径",
    )
    parser.add_argument(
        "--match_backend",
        type=str,
        default=yaml_recon.get("backend", "pi3"),
        choices=["vggt", "pi3", "da3"],
        help="SKU匹配 3D 后端 (vggt|pi3|da3)，仅当算法包含3d时生效",
    )
    parser.add_argument(
        "--parallel_refs",
        type=int,
        default=1,
        help="batch_all_refs 并行线程数（>1 时启用线程池，推荐 pi3/da3 后端）",
    )
    # 可选 3D 阈值覆盖（网格扫描用，default=None 表示用 config 默认）
    parser.add_argument(
        "--plane_normal_alignment_threshold",
        type=float,
        default=None,
        help="平面法向对齐阈值(覆盖config)",
    )
    parser.add_argument(
        "--max_3d_distance", type=float, default=None, help="3D空间距离阈值(覆盖config)"
    )
    parser.add_argument(
        "--max_depth", type=float, default=None, help="最大深度(覆盖config)"
    )
    parser.add_argument(
        "--depth_confidence_threshold",
        type=float,
        default=None,
        help="深度置信度阈值(覆盖config)",
    )
    parser.add_argument(
        "--min_3d_sample_points",
        type=int,
        default=None,
        help="3D采样最少有效点数(覆盖config)",
    )
    parser.add_argument(
        "--pairing_3d",
        type=str,
        default=None,
        choices=["all", "next"],
        help="3D配对策略 all/next(覆盖config)",
    )
    parser.add_argument(
        "--enable_profiling",
        action="store_true",
        default=False,
        help="启用 per-stage 计时 instrumentation（默认关闭，零开销 no-op；输出 profiling_<ts>.json）",
    )

    parser.add_argument(
        "--viewer-web-output",
        type=str,
        default=None,
        help="viewer-web: bundle输出目录（默认：modules/viewer_web/public/data）",
    )
    parser.add_argument(
        "--viewer-web-voxel-size",
        type=float,
        default=0.005,
        help="viewer-web: 点云voxel大小（默认0.005）",
    )
    parser.add_argument(
        "--viewer-web-max-points",
        type=int,
        default=1500000,
        help="viewer-web: 点云最大点数（默认1500000）",
    )

    args = parser.parse_args()

    # 若指定了 --floor，则覆盖 dataset 为 imdata/floor_display{floor}
    if args.floor is not None:
        floor_name = f"floor_display{args.floor}"
        dataset_path = PROJECT_ROOT / "imdata" / floor_name
        args.dataset = str(dataset_path)

    # 统一日志
    save_root_path = _resolve_save_root(args.save_root)
    _configure_logging_to_save_root(save_root_path)

    app = SKUDetectionMain()
    app.default_dataset = args.dataset
    app.save_root = save_root_path
    # 将命令行或配置中的匹配后端设置到应用实例（仅3D算法生效）
    app.match_backend = args.match_backend
    app.config_path = (
        Path(args.config).resolve()
        if args.config
        else (config_path.resolve() if config_path else None)
    )

    if args.mode == "interactive":
        app.interactive_mode()
    elif args.mode == "pipeline":
        app.run_complete_pipeline(
            args.dataset, args.algorithm, model_path=args.recon_model_path
        )
    elif args.mode == "concise":
        # 在精简流水线中，先匹配后评估；匹配透传关键参数
        _ov = {}
        for _k in (
            "plane_normal_alignment_threshold",
            "max_3d_distance",
            "max_depth",
            "depth_confidence_threshold",
            "min_3d_sample_points",
            "pairing_3d",
        ):
            _v = getattr(args, _k, None)
            if _v is not None:
                _ov[_k] = _v
        app.run_sku_matching(
            args.dataset,
            args.algorithm,
            reference_idx=args.reference_idx,
            max_images=args.max_images,
            device=args.device,
            save_json=args.save_json,
            backend=(args.match_backend if "3d" in args.algorithm else "vggt"),
            parallel_refs=args.parallel_refs,
            match_overrides=_ov,
            enable_profiling=args.enable_profiling,
        )
        app.run_accuracy_evaluation(
            args.dataset, backend=args.match_backend if "3d" in args.algorithm else "pt"
        )
    elif args.mode == "analyzer":
        # 仅执行改进的SKU计数分析
        analyzer_backend = args.match_backend if "3d" in args.algorithm else None
        app.run_improved_sku_analysis(
            args.dataset, algorithm=args.algorithm, backend=analyzer_backend
        )
    elif args.mode == "dedup":
        # dedup 模式需要明确的单一算法类型
        if args.algorithm == "both":
            logger.error(
                "dedup 模式不支持 algorithm='both'，请指定 'point_tracking' 或 '3d'"
            )
            sys.exit(1)
        # 根据算法类型选择后端
        dedup_backend = args.match_backend if "3d" in args.algorithm else None
        app.run_dedup_sequence(
            args.dataset, algorithm=args.algorithm, backend=dedup_backend
        )
    elif args.mode == "ground-stack-area":
        from src.da3_footprint_stage import run_da3_footprint

        result = run_da3_footprint(args.dataset, app.save_root)
        if not result["success"]:
            logger.error(
                "ground-stack-area rejected: %s (report: %s)",
                result["status"],
                result["report_path"],
            )
            sys.exit(2)
        logger.info(
            "ground-stack-area %s: %s",
            result["status"],
            result["report_path"],
        )
    elif args.mode == "reconstruct":
        app.run_reconstruction(
            args.dataset,
            device=args.device,
            output_filename=args.recon_output,
            backend=args.recon_backend,
            conf_thres=args.recon_conf_thres,
            model_path=args.recon_model_path,
        )
    elif args.mode == "viewer-web":
        dataset = Path(args.dataset)
        dataset_output = app.save_root / dataset.name
        viewer_web_output = (
            Path(args.viewer_web_output).expanduser().resolve()
            if args.viewer_web_output
            else PROJECT_ROOT / "modules" / "viewer_web" / "public" / "data"
        )

        from src.web_viewer_export import export_web_viewer_bundle

        result = export_web_viewer_bundle(
            da3_cache_path=dataset_output / "da3_cache" / "predictions.npz",
            global_mapping_path=dataset_output
            / "dedup_detections"
            / "global_mapping.json",
            footprint_root=dataset_output / "ground_stack_footprint",
            output_dir=viewer_web_output,
            source_images_dir=dataset / "images",
            sam3_mask_cache_root=dataset_output / "sam3_mask_cache" / "v1",
            voxel_size_m=float(args.viewer_web_voxel_size),
            max_points=int(args.viewer_web_max_points),
        )
        logger.info(
            "viewer-web export: output_dir=%s manifest_path=%s point_count=%s "
            "footprint_status=%s thumbnails=%s",
            result["output_dir"],
            result["manifest_path"],
            result["point_count"],
            result["footprint_status"],
            result["thumbnail_count"],
        )
        default_viewer_web_output = PROJECT_ROOT / "modules" / "viewer_web" / "public" / "data"
        if viewer_web_output == default_viewer_web_output:
            print(f'Next step: npm --prefix {PROJECT_ROOT / "modules" / "viewer_web"} run dev')
        else:
            print(
                f"Custom viewer-web output: {viewer_web_output}; "
                "it must be served or mounted at browser URL /data/ before starting the frontend."
            )


if __name__ == "__main__":
    main()
