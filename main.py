import argparse
import json
import logging
import re
import subprocess
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from re import T
from time import perf_counter
from typing import Any, Dict, Optional, TypedDict

from da3_defaults import DEFAULT_PROCESS_RES, PREPROCESS_METHOD, validate_batch_grid

import colorlog
import numpy as np

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SAVE_ROOT = PROJECT_ROOT / "Output"
CLASSIFIER_ROOT = PROJECT_ROOT / "modules" / "personalcare_classifier"
CLASSIFIER_SCRIPT = CLASSIFIER_ROOT / "source" / "classify_dataset.py"
_CLASSIFIER_PAYLOAD_FIELDS = frozenset(
    {
        "success",
        "run_id",
        "detection_dir",
        "result_path",
        "frame_count",
        "object_count",
        "unavailable_count",
    }
)
_CLASSIFIER_RUN_ID_RE = re.compile(r"[1-9][0-9]*-[1-9][0-9]*")


def _is_valid_classifier_payload(payload: object) -> bool:
    if not isinstance(payload, dict) or set(payload) != _CLASSIFIER_PAYLOAD_FIELDS:
        return False
    if (
        payload["success"] is not True
        or not isinstance(payload["run_id"], str)
        or _CLASSIFIER_RUN_ID_RE.fullmatch(payload["run_id"]) is None
        or not isinstance(payload["detection_dir"], str)
        or not isinstance(payload["result_path"], str)
    ):
        return False
    return all(
        not isinstance(payload[field], bool)
        and isinstance(payload[field], int)
        and payload[field] >= 0
        for field in ("frame_count", "object_count", "unavailable_count")
    )


def _same_typed_classifier_payload(
    left: dict[str, object], right: dict[str, object]
) -> bool:
    return all(
        type(left[field]) is type(right[field]) and left[field] == right[field]
        for field in _CLASSIFIER_PAYLOAD_FIELDS
    )


def _is_complete_classifier_current(current: object, run_id: str) -> bool:
    return (
        isinstance(current, dict)
        and set(current) == {"run_id", "complete"}
        and isinstance(current["run_id"], str)
        and _CLASSIFIER_RUN_ID_RE.fullmatch(current["run_id"]) is not None
        and current["run_id"] == run_id
        and current["complete"] is True
    )


def validate_external_classification_directory(dataset: Path) -> Path:
    """Validate externally enriched detections before using them for deduplication."""
    from utils.classification_aggregation import validate_classification
    from utils.detection_objects import flatten_detection_objects

    detection_dir = dataset / "detections_results"
    if not detection_dir.is_dir():
        raise ValueError(
            f"external classification directory does not exist: {detection_dir}"
        )

    detection_files: list[Path] = []
    for detection_file in detection_dir.glob("*.json"):
        if not detection_file.stem.isdecimal():
            raise ValueError(
                f"external classification filename must be numeric: {detection_file.name}"
            )
        detection_files.append(detection_file)
    if not detection_files:
        raise ValueError(f"external classification directory is empty: {detection_dir}")

    for detection_file in sorted(detection_files, key=lambda path: int(path.stem)):
        with detection_file.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for object_index, detection in enumerate(flatten_detection_objects(payload)):
            if not isinstance(detection, dict):
                raise ValueError(
                    f"external detection object is invalid: "
                    f"{detection_file.name}[{object_index}]"
                )
            validate_classification(detection.get("classification"))
    return detection_dir


def _resolve_save_root(value: str | None) -> Path:
    """Resolve omitted and relative output roots against the repository root."""
    if value is None or not value.strip():
        return DEFAULT_SAVE_ROOT
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _check_geometry_refinement_report(output_dir: Path, reusable: bool) -> None:
    """A rejected refinement must not silently trigger an ordinary reconstruction."""
    report_path = output_dir / "geometry_refinement.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text())
    if report["accepted"] is not True or report["cache_published"] is not True or not reusable:
        raise ValueError(
            f"几何优化未通过验收或优化缓存不可复用，拒绝重建覆盖：{report_path}"
        )


def _is_reusable_da3_cache(cache_path: Path) -> bool:
    """Return whether a DA3 cache has the schema-v3 metric contract and current preprocessing settings."""
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            schema = cache["cache_schema_version"]
            is_metric = cache["is_metric"]
            resolution = cache["preprocess_resolution"]
            method = cache["preprocess_method"]
            if "use_ray_pose" in cache.files:
                use_ray_pose = cache["use_ray_pose"]
                if (
                    use_ray_pose.shape != ()
                    or use_ray_pose.dtype != np.dtype(np.bool_)
                    or bool(use_ray_pose.item())
                ):
                    return False
            sizes = cache["source_image_sizes"]
            if sizes.ndim != 2 or sizes.shape[1] != 2:
                return False
            validate_batch_grid(sizes, DEFAULT_PROCESS_RES)
            return (
                schema.shape == ()
                and schema.dtype.kind in "iu"
                and int(schema.item()) == 3
                and is_metric.shape == ()
                and is_metric.dtype.kind in "iu"
                and int(is_metric.item()) == 1
                and resolution.shape == ()
                and resolution.dtype.kind in "iu"
                and int(resolution.item()) == DEFAULT_PROCESS_RES
                and method.shape == ()
                and str(method.item()) == PREPROCESS_METHOD
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
        self.classifier_device: str = "cuda:0"
        self.classifier_enabled: bool = True
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
        quiet_outputs: bool = False,
        sam3_mask_cache_root: Optional[str] = None,
    ) -> StepResult:
        """匹配全部或单个参考帧；任一参考帧失败直接传播，保留 profiling 清理。"""
        from concurrent.futures import as_completed
        from utils.data_utils import load_detections
        from utils.profiling import StageTimer, dump_stages, log_stages_sorted, set_enabled

        set_enabled(enable_profiling)
        start = perf_counter()

        def run_reference(index):
            return self._run_single_matching(
                dataset_path,
                algorithm,
                index,
                max_images,
                device,
                save_json,
                backend,
                match_overrides,
                enable_profiling=enable_profiling,
                quiet_outputs=quiet_outputs,
                sam3_mask_cache_root=sam3_mask_cache_root,
            )

        try:
            if not batch_all_refs:
                return run_reference(reference_idx)
            detections = load_detections(
                str(Path(dataset_path) / "detections_results"),
                return_index_map=True,
            )
            reference_count = len(detections)
            logger.info("开始匹配 %d 个参考图片", reference_count)
            if parallel_refs > 1:
                with ThreadPoolExecutor(
                    max_workers=min(parallel_refs, reference_count)
                ) as executor:
                    futures = [
                        executor.submit(run_reference, index)
                        for index in range(reference_count)
                    ]
                    for future in as_completed(futures):
                        future.result()
            else:
                for index in range(reference_count):
                    run_reference(index)
            duration = perf_counter() - start
            logger.info(
                "匹配完成 - 耗时 %.2fs，处理 %d 个参考图片", duration, reference_count
            )
            return {"success": True, "duration_s": duration}
        finally:
            StageTimer.record("batch_all_refs_total", perf_counter() - start)
            if enable_profiling:
                profile_dir = (
                    self.save_root / Path(dataset_path).name
                    if self.save_root
                    else Path(dataset_path)
                )
                profile_path = (
                    profile_dir
                    / f"profiling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                dump_stages(str(profile_path))
                log_stages_sorted(logger)
                logger.info("[PROF] dumped to %s", profile_path)

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
        quiet_outputs: bool = False,
        sam3_mask_cache_root: Optional[str] = None,
    ) -> StepResult:
        """运行单个参考图片的SKU匹配推理（内部使用）。

        Args:
            backend: 3D重建模型后端 (vggt/pi3)
        """
        start = perf_counter()
        logger.debug(
            f"单次匹配 - 算法: {algorithm}, 后端: {backend}, 参考索引: {reference_idx}"
        )

        from src.inference import main as inference_main
        from utils.config import default_sam3_mask_cache_root

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
            str(sam3_mask_cache_root or default_sam3_mask_cache_root(output_dir)),
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
        if quiet_outputs:
            argv.append("--quiet_outputs")
        if save_json and not quiet_outputs:
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
            raise RuntimeError(msg)

        # 检测使用的算法
        algorithm_name = (
            "Point Tracking" if "output_pt" in str(summary_dir) else "3D Mapping"
        )

        analyzer = ImprovedSKUCountAnalyzer(str(detection_dir), str(summary_dir))
        result = analyzer.analyze_with_filtering()

        # 计算统计信息
        removed = result["original_matches"] - result["filtered_matches"]
        reduction_suffix = (
            f"({removed / result['original_matches'] * 100:.1f}%)"
            if result["original_matches"]
            else ""
        )
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
        report_file = reports_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

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
                f"{reduction_suffix}\n"
            )
            f.write(f"  平均 Hit Ratio: {avg_hit_ratio:.3f}\n")
            f.write(f"  涉及图片: {ref_images} 个参考图片, {target_images} 个目标图片\n\n")

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

    def run_accuracy_evaluation(
        self, dataset_path: str, *, backend: str = "pt"
    ) -> StepResult:
        """评估当前后端的匹配结果；输入缺失或评估失败直接报错。"""
        start = perf_counter()
        benchmark_csv = PROJECT_ROOT / "imdata" / "picture_mapping_benchmark.csv"
        if not benchmark_csv.is_file():
            raise FileNotFoundError(f"基准数据文件不存在: {benchmark_csv}")
        output_root = self.save_root if self.save_root is not None else DEFAULT_SAVE_ROOT
        normalized_backend = "pt" if backend == "point_tracking" else backend
        output_subdir = (
            "output_pt"
            if normalized_backend == "pt"
            else f"output_3dmapping_{normalized_backend}"
        )
        output_dir = output_root / Path(dataset_path).name / output_subdir
        if not output_dir.is_dir():
            raise FileNotFoundError(f"匹配结果目录不存在: {output_dir}")
        script_path = PROJECT_ROOT / "scripts/3d/evaluation/accuracy_evaluation.sh"
        if not script_path.is_file():
            raise FileNotFoundError(f"准确性评估脚本不存在: {script_path}")
        result = subprocess.run(
            [
                "bash",
                str(script_path),
                Path(dataset_path).name,
                "--backend",
                normalized_backend,
                "--save-root",
                str(output_root),
            ],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"准确性评估失败 ({result.returncode}): {result.stderr}\n{result.stdout}"
            )
        return {"success": True, "duration_s": perf_counter() - start}

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
        from src import RECONSTRUCTOR_REGISTRY, get_reconstructor

        use_backend = (backend or "vggt").lower()
        if use_backend not in RECONSTRUCTOR_REGISTRY:
            available = ", ".join(sorted(RECONSTRUCTOR_REGISTRY)) or "(无)"
            hint = (
                "（如需启用 vggt，请恢复 src/__init__.py 中 VGGT3DReconstructor 的注册）"
                if use_backend == "vggt"
                else ""
            )
            raise ValueError(f"未知/未启用的重建后端: {backend}. 已注册: {available}{hint}")

        dataset = Path(dataset_path)
        if not dataset.exists():
            raise ValueError(f"数据集路径不存在: {dataset_path}")
        image_dir = dataset / "images"
        if not image_dir.exists():
            msg = f"图片目录不存在: {image_dir}"
            logger.error(msg)
            raise RuntimeError(msg)

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
        # vggt 的 export_glb 需要 mask_* 参数（经 reconstruct_from_directory 的 **kwargs 透传）；
        # da3/pi3 无此参数，忽略即可。
        extra_kwargs: dict = {}
        if use_backend == "vggt":
            extra_kwargs = {
                "mask_black_bg": mask_black_bg,
                "mask_white_bg": mask_white_bg,
                "mask_sky": mask_sky,
            }
        with recon_cls(device=device, model_path=model_path) as recon:
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

    def run_dedup_sequence(
        self,
        dataset_path: str,
        algorithm: str = "point_tracking",
        backend: str | None = None,
        detection_dir: str | Path | None = None,
    ) -> StepResult:
        """顺序去重：对 1..N（或指定上界）生成去重后的检测 JSON。

        Args:
            dataset_path: 数据集路径
            algorithm: 算法类型 'point_tracking'/'3d_mapping'
            backend: 3D算法后端 'vggt'/'pi3'，仅在algorithm='3d_mapping'时生效
            detection_dir: 已完成 personalcare 分类的检测目录；省略时使用原始输入
        """
        start = perf_counter()
        logger.info(f"开始顺序去重 (algorithm: {algorithm}, backend: {backend})")
        from src.deduplicate_detections import (
            deduplicate_sequence,
            resolve_dataset_paths,
        )

        dataset_dir = Path(dataset_path)
        if not dataset_dir.exists():
            raise ValueError(f"数据集路径不存在: {dataset_path}")

        classified_detection_dir = (
            Path(detection_dir) if detection_dir is not None else None
        )
        if classified_detection_dir is not None and not classified_detection_dir.is_dir():
            raise ValueError(f"分类检测目录不存在: {classified_detection_dir}")
        paths = resolve_dataset_paths(dataset_dir, classified_detection_dir)
        dataset_name = dataset_dir.name

        # 输出目录：Output/<dataset_name>/dedup_detections/
        output_base = self.save_root if self.save_root is not None else DEFAULT_SAVE_ROOT

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
            detections_dir=paths.detections_dir,
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

    def run_personalcare_classification(self, dataset_path: str) -> StepResult:
        """Run the isolated personalcare classifier and accept only a published result."""
        start = perf_counter()
        dataset = Path(dataset_path)
        output_root = self.save_root if self.save_root is not None else DEFAULT_SAVE_ROOT
        command = [
            "uv",
            "run",
            "--project",
            str(CLASSIFIER_ROOT),
            "python",
            str(CLASSIFIER_SCRIPT),
            "--dataset",
            str(dataset),
            "--output-root",
            str(output_root),
            "--device",
            self.classifier_device,
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        duration = perf_counter() - start
        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            raise RuntimeError(
                f"personalcare classifier exited {completed.returncode}: {stderr}"
            )
        payload = json.loads(completed.stdout)
        if not _is_valid_classifier_payload(payload):
            raise RuntimeError(
                f"personalcare classifier emitted an invalid payload; stderr: {stderr}"
            )
        run_id = payload["run_id"]
        assert isinstance(run_id, str)
        run_dir = (
            output_root / dataset.name / "personalcare_classification" / "runs" / run_id
        ).resolve()
        detection_dir = Path(payload["detection_dir"])
        result_path = Path(payload["result_path"])
        if (
            detection_dir.resolve() != run_dir / "detections"
            or not detection_dir.is_dir()
            or result_path.resolve() != run_dir / "result.json"
            or not result_path.is_file()
        ):
            raise RuntimeError(
                f"personalcare classifier published unexpected paths; stderr: {stderr}"
            )
        published_payload = json.loads(result_path.read_text(encoding="utf-8"))
        current_payload = json.loads(
            (run_dir.parents[1] / "CURRENT").read_text(encoding="utf-8")
        )
        if (
            not _is_valid_classifier_payload(published_payload)
            or not _same_typed_classifier_payload(published_payload, payload)
            or not _is_complete_classifier_current(current_payload, run_id)
        ):
            raise RuntimeError(
                f"personalcare classifier publication does not match stdout; stderr: {stderr}"
            )
        return {
            "success": True,
            "duration_s": duration,
            "details": payload,
            "detection_dir": str(detection_dir),
        }

    def run_complete_pipeline(
        self,
        dataset_path: str,
        algorithm: str = "point_tracking",
        model_path: str | None = None,
        evaluate_accuracy: bool = True,
        quiet_outputs: bool = False,
    ) -> Dict[str, bool]:
        """运行 SKU 计数流水线；无人工标注的服务请求可显式关闭准确率评估。"""
        logger.info("开始完整的SKU计数流水线（包含3D重建）")
        summary: Dict[str, bool] = {}

        if not self.validate_dataset(dataset_path):
            raise ValueError(f"数据集验证失败: {dataset_path}")
        summary["validation"] = True

        classifier_future: Future[StepResult] | None = None
        classification_result: StepResult | None = None

        if not self.classifier_enabled:
            external_detection_dir = validate_external_classification_directory(
                Path(dataset_path)
            )
            classification_result = {
                "success": True,
                "detection_dir": str(external_detection_dir),
            }

        def join_classification() -> StepResult:
            nonlocal classification_result
            if classification_result is None:
                if classifier_future is None:
                    raise RuntimeError("personalcare classification was not submitted")
                classification_result = classifier_future.result()
            if not isinstance(classification_result, dict) or not isinstance(
                classification_result.get("detection_dir"), str
            ):
                raise ValueError("personalcare classification returned an invalid result")
            return classification_result

        classifier_executor: ThreadPoolExecutor | None = None
        try:
            if self.classifier_enabled:
                classifier_executor = ThreadPoolExecutor(max_workers=1)
                classifier_future = classifier_executor.submit(
                    self.run_personalcare_classification, dataset_path
                )
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
                    _check_geometry_refinement_report(output_dir, reusable)
                else:
                    base_output = Path("reconstruction.glb")
                    if match_backend not in base_output.stem:
                        filename = f"{base_output.stem}_{match_backend}{base_output.suffix}"
                        expected_result = cache_dir / filename
                    else:
                        expected_result = cache_dir / base_output
                    reusable = expected_result.exists()

                if reusable:
                    logger.info(
                        f"步骤1: 检测到可复用3D重建结果 {expected_result}，跳过3D重建"
                    )
                    summary["reconstruction"] = True
                else:
                    logger.info(f"步骤1: 3D重建 (backend: {match_backend})")
                    self.run_reconstruction(
                        dataset_path, backend=match_backend, model_path=model_path
                    )
                    summary["reconstruction"] = True
            else:
                logger.info("步骤1: 跳过3D重建（使用 Point Tracking 算法）")
                summary["reconstruction"] = True  # 标记为成功（不需要）

            if not quiet_outputs:
                self.run_detection_visualization(dataset_path)
                summary["visualization"] = True

            match_backend = self.match_backend if "3d" in algorithm else "vggt"
            self.run_sku_matching(
                dataset_path, algorithm, batch_all_refs=True, backend=match_backend,
                quiet_outputs=quiet_outputs,
            )
            summary["matching"] = True
            if not quiet_outputs:
                self.run_improved_sku_analysis(
                    dataset_path, algorithm=algorithm, backend=match_backend
                )
                summary["improved_analysis"] = True

            classification = join_classification()
            summary["classification"] = True
            self.run_dedup_sequence(
                dataset_path,
                algorithm=algorithm,
                backend=match_backend,
                detection_dir=classification["detection_dir"],
            )
            summary["dedup"] = True
            if not quiet_outputs:
                output_base = (
                    self.save_root if self.save_root is not None else DEFAULT_SAVE_ROOT
                )
                dedup_detection_dir = output_base / Path(dataset_path).name / "dedup_detections"
                self.run_detection_visualization(
                    dataset_path,
                    detection_dir=str(dedup_detection_dir),
                    output_suffix="dedup_imgs_w_bboxes",
                )
                summary["dedup_visualization"] = True
            if evaluate_accuracy and not quiet_outputs:
                self.run_accuracy_evaluation(
                    dataset_path, backend=match_backend if "3d" in algorithm else "pt"
                )
                summary["accuracy_evaluation"] = True
            logger.info("流水线全部阶段完成")
            return summary
        finally:
            if classifier_executor is not None:
                classifier_executor.shutdown(wait=True)

    def run_concise_pipeline(
        self,
        dataset_path: str,
        algorithm: str = "point_tracking",
        **matching_options,
    ) -> Dict[str, bool]:
        """CLI 与交互菜单共享的匹配、评估流程；匹配失败不执行评估。"""
        if not self.validate_dataset(dataset_path):
            raise ValueError(f"数据集验证失败: {dataset_path}")
        match_backend = self.match_backend if "3d" in algorithm else "vggt"
        self.run_sku_matching(
            dataset_path, algorithm, backend=match_backend, **matching_options
        )
        self.run_accuracy_evaluation(
            dataset_path, backend=match_backend if "3d" in algorithm else "pt"
        )
        return {"validation": True, "matching": True, "accuracy_evaluation": True}

    def interactive_mode(self) -> None:
        """交互模式"""
        self.show_banner()

        while True:
            print("\n请选择操作:")
            print("1. 运行完整流水线")
            print("2. 运行精简流水线 (SKU Matching + Accuracy evaluation)")
            print("3. 更改数据集路径")
            print("4. 3D重建 (VGGT/PI3/PI3X/DA3/MapAnything)")
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
                            input(
                                f"选择3D匹配后端 ({'/'.join(BACKEND_CHOICES)}): "
                            )
                            .strip()
                            .lower()
                        )
                        if backend in BACKEND_CHOICES:
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
                            input(
                                f"选择3D匹配后端 ({'/'.join(BACKEND_CHOICES)}): "
                            )
                            .strip()
                            .lower()
                        )
                        if backend in BACKEND_CHOICES:
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
                    backend = (
                        input(f"选择重建后端 ({'/'.join(BACKEND_CHOICES)}): ")
                        .strip()
                        .lower()
                    )
                    if backend in BACKEND_CHOICES:
                        break
                    logger.warning(f"无效的后端 '{backend}'，请重新输入")
                res = self.run_reconstruction(self.default_dataset, backend=backend)
                print(
                    f"3D重建: {'成功' if res.get('success') else '失败'}，耗时 {res.get('duration_s', 0):.2f}s"
                )
            else:
                print("无效选择，请重试")


"""3D 重建/匹配后端集合；interactive 菜单与 argparse choices 共用。

注意与 src 注册表（RECONSTRUCTOR_REGISTRY）解耦：此处是 CLI 面板的
用户可选后端（含 pi3x/mapanything 等只读/对比后端），新增后端时两边同步。
"""
BACKEND_CHOICES = ["vggt", "pi3", "pi3x", "da3", "mapanything"]


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
        "--classifier-device",
        type=str,
        default="cuda:0",
        help="personalcare 分类器设备（默认 cuda:0）",
    )
    parser.add_argument(
        "--classifier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="运行本地分类；--no-classifier 使用外部 enriched detections",
    )
    parser.add_argument("--quiet_outputs", action="store_true", help="完整pipeline关闭额外调试图片和辅助报告")
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
        choices=BACKEND_CHOICES,
        help=f"3D重建后端 ({'|'.join(BACKEND_CHOICES)})",
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
        choices=BACKEND_CHOICES,
        help=f"SKU匹配 3D 后端 ({'|'.join(BACKEND_CHOICES)})，仅当算法包含3d时生效",
    )
    parser.add_argument(
        "--sam3_mask_cache_root",
        type=str,
        default=None,
        help="SAM3 mask 缓存根目录（默认 <output_dir>/sam3_mask_cache_sam31/v2）。"
        "节点缓存 v2 key 无 checkpoint 身份，换 checkpoint 必须换根以免跨版本复用。",
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
        "--viewer-web-sku-masterdata-csv",
        type=str,
        default=str(PROJECT_ROOT / "runtime" / "sku_masterdata.csv"),
        help="viewer-web: 窄SKU主数据CSV（默认：runtime/sku_masterdata.csv）",
    )
    parser.add_argument(
        "--viewer-web-voxel-size",
        type=float,
        default=0.005,
        help="viewer-web: 点云voxel大小（默认0.005）",
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
    app.classifier_device = args.classifier_device
    app.classifier_enabled = args.classifier
    app.config_path = (
        Path(args.config).resolve()
        if args.config
        else (config_path.resolve() if config_path else None)
    )

    if args.mode == "interactive":
        app.interactive_mode()
    elif args.mode == "pipeline":
        app.run_complete_pipeline(
            args.dataset, args.algorithm, model_path=args.recon_model_path,
            quiet_outputs=args.quiet_outputs,
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
        app.run_concise_pipeline(
            args.dataset,
            args.algorithm,
            reference_idx=args.reference_idx,
            max_images=args.max_images,
            device=args.device,
            save_json=args.save_json,
            parallel_refs=args.parallel_refs,
            match_overrides=_ov,
            enable_profiling=args.enable_profiling,
            sam3_mask_cache_root=args.sam3_mask_cache_root,
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
        from utils.config import default_sam3_mask_cache_root

        dataset = Path(args.dataset)
        dataset_output = app.save_root / dataset.name
        viewer_web_output = (
            Path(args.viewer_web_output).expanduser().resolve()
            if args.viewer_web_output
            else PROJECT_ROOT / "modules" / "viewer_web" / "public" / "data"
        )
        sku_masterdata_csv = Path(args.viewer_web_sku_masterdata_csv).expanduser()
        if not sku_masterdata_csv.is_absolute():
            sku_masterdata_csv = PROJECT_ROOT / sku_masterdata_csv

        from src.web_viewer_export import export_web_viewer_bundle

        result = export_web_viewer_bundle(
            dataset_name=dataset.name,
            da3_cache_path=dataset_output / "da3_cache" / "predictions.npz",
            global_mapping_path=dataset_output
            / "dedup_detections"
            / "global_mapping.json",
            output_dir=viewer_web_output,
            source_images_dir=dataset / "images",
            sam3_mask_cache_root=default_sam3_mask_cache_root(dataset_output),
            sku_masterdata_csv=sku_masterdata_csv,
            voxel_size_m=float(args.viewer_web_voxel_size),
        )
        logger.info(
            "viewer-web export: output_dir=%s manifest_path=%s point_count=%s thumbnails=%s",
            result["output_dir"],
            result["manifest_path"],
            result["point_count"],
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
