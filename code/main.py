from re import T
import sys
import argparse
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
from time import perf_counter
from typing import Optional, Dict, Any, TypedDict
import colorlog

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = Path(__file__).resolve().parent

# 确保可以从仓库根或任意 CWD 导入本目录模块
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))


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

            color = ''
            if 'start' in lower:
                color = escape_codes.get('cyan', '')
            elif 'end' in lower:
                color = escape_codes.get('red', '') if 'fail' in lower else escape_codes.get('green', '')

            # Prefix only the message part; keep level color as-is
            record.msg_color = color
            # ColoredFormatter appends %(reset)s at the end, no need for extra reset
        except (AttributeError, KeyError, ImportError):
            # Gracefully handle missing colorlog or attribute errors in log formatting
            record.msg_color = ''
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
    # 文件日志保留完整格式，控制台使用彩色格式
    file_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建彩色控制台格式器
    console_fmt = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s - %(msg_color)s%(message)s%(reset)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        },
    )

    fh = RotatingFileHandler(str(log_file), maxBytes=10_000_000, backupCount=1, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(console_fmt)

    root_logger.addHandler(fh)
    # Only colorize console output based on message content
    sh.addFilter(_StartEndColorFilter())
    root_logger.addHandler(sh)

    logger = logging.getLogger(__name__)
    logger.info(f"日志已写入: {log_file}")
    return logger


logger = logging.getLogger(__name__)


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
        logger.info("初始化3D SKU Detection主程序")

    def show_banner(self) -> None:
        """显示程序横幅（自适应对齐，宽字符友好）。"""
        from datetime import datetime
        import sys

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
        required_dirs = ['images', 'detections_results']

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
    ) -> StepResult:
        """运行SKU匹配推理，支持批量将每张图片作为参考图像运行。

        - 当 batch_all_refs=True 时：遍历 images/ 中数字命名且在 detections_results/ 有有效 objects 的每个图片，依次作为参考图运行。
        - 否则：仅以 reference_idx 指定的单张图片作为参考图运行。
        - backend: 3D重建模型后端 (vggt/pi3)，用于3D算法时选择数据源
        """
        start = perf_counter()
        original_argv = sys.argv.copy()
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
                    detections_with_index = load_detections(str(detection_dir), return_index_map=True)
                    # 提取文件编号（即图片索引）
                    valid_indices = sorted([file_num for file_num, _ in detections_with_index])
                    logger.info(f"找到 {len(valid_indices)} 个有效参考图片")
                except (FileNotFoundError, ValueError) as e:
                    logger.error(f"无法加载检测结果: {e}")
                    return {"success": False, "error": str(e), "duration_s": perf_counter() - start}

                # 依次以每个有效图片为参考图片运行推理
                for i, filename_idx in enumerate(valid_indices):
                    # 文件名从1开始，但系统内部索引从0开始，所以需要减1
                    system_ref_idx = filename_idx - 1
                    logger.debug(f"处理参考图片 {filename_idx} ({i+1}/{len(valid_indices)}) -> 系统索引: {system_ref_idx}")
                    self._run_single_matching(dataset_path, algorithm, system_ref_idx, max_images, device, save_json, backend)

                duration = perf_counter() - start
                logger.info(f"✓ 匹配完成 - 耗时 {duration:.2f}s，处理 {len(valid_indices)} 个参考图片")
                return {"success": True, "duration_s": duration}
            else:
                # 单个参考图片处理
                return self._run_single_matching(dataset_path, algorithm, reference_idx, max_images, device, save_json, backend)

        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as e:
            duration = perf_counter() - start
            logger.error(f"END matching duration={duration:.2f}s result=fail error={e}", exc_info=True)
            return {"success": False, "error": str(e), "duration_s": duration}
        finally:
            sys.argv = original_argv

    def _run_single_matching(self, dataset_path: str, algorithm: str, reference_idx: int,
                           max_images: int, device: str, save_json: bool, backend: str = "vggt") -> StepResult:
        """运行单个参考图片的SKU匹配推理（内部使用）。

        Args:
            backend: 3D重建模型后端 (vggt/pi3)
        """
        start = perf_counter()
        original_argv = sys.argv.copy()
        try:
            logger.debug(f"单次匹配 - 算法: {algorithm}, 后端: {backend}, 参考索引: {reference_idx}")

            from modules.inference import main as inference_main

            dataset = Path(dataset_path)
            image_folder = dataset / "images"
            detection_dir = dataset / "detections_results"
            # 输出根目录：优先使用 save_root，其次使用数据集目录
            output_dir = (self.save_root / dataset.name) if self.save_root else dataset

            argv = [
                'inference.py',
                '--image_folder', str(image_folder),
                '--detection_dir', str(detection_dir),
                '--output_dir', str(output_dir),
                '--algorithm', algorithm,
                '--reference_idx', str(reference_idx),
                '--max_images', str(max_images),
                '--device', device,
                '--backend', backend,
            ]
            if save_json:
                argv.append('--save_json')
            if self.config_path is not None:
                argv.extend(['--config', str(self.config_path)])

            sys.argv = argv
            inference_main()

            duration = perf_counter() - start
            logger.debug(f"✓ 单次匹配完成 - 耗时 {duration:.2f}s")
            return {"success": True, "duration_s": duration}
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as e:
            duration = perf_counter() - start
            logger.error(f"END matching_single duration={duration:.2f}s result=fail error={e}", exc_info=True)
            return {"success": False, "error": str(e), "duration_s": duration}
        finally:
            sys.argv = original_argv

    

    def run_detection_visualization(self, dataset_path: str, detection_dir: str = None, output_suffix: str = "imgs_w_bboxes") -> StepResult:
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

            from modules.draw_detection_boxes import main as viz_main

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
                if self.save_root else (CODE_DIR / dataset.name / output_suffix)
            ).resolve()
            output_viz_dir.mkdir(parents=True, exist_ok=True)

            sys.argv = [
                'draw_detection_boxes.py',
                '--image_dir', str(image_dir),
                '--detection_dir', str(detection_dir),
                '--output_dir', str(output_viz_dir),
            ]

            viz_main()

            duration = perf_counter() - start
            logger.info(f"✓ 可视化完成 - 耗时 {duration:.2f}s")
            logger.debug(f"输出目录: {output_viz_dir}")
            return {"success": True, "duration_s": duration, "details": {"output_dir": str(output_viz_dir)}}
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as e:
            duration = perf_counter() - start
            logger.error(f"END visualization duration={duration:.2f}s result=fail error={e}", exc_info=True)
            return {"success": False, "error": str(e), "duration_s": duration}
        finally:
            sys.argv = original_argv

    def run_improved_sku_analysis(self, dataset_path: str) -> StepResult:
        """运行改进的SKU计数分析 (去重优化)，报告写入 <dataset_name>/output_reports/report_*.txt（或 --save_root）"""
        start = perf_counter()
        try:
            logger.info("开始SKU计数分析")

            from modules.improved_sku_analyzer import ImprovedSKUCountAnalyzer

            dataset = Path(dataset_path)
            detection_dir = dataset / "detections_results"
            # 匹配结果目录：若指定 save_root，则读取 save_root/<dataset_name>/output_pt
            summary_dir = (
                (self.save_root / dataset.name / "output_pt") if self.save_root else (dataset / "output_pt")
            )

            if not summary_dir.exists():
                msg = f"匹配结果目录不存在: {summary_dir}，请先运行SKU匹配推理"
                logger.warning(msg)
                duration = perf_counter() - start
                return {"success": False, "error": msg, "duration_s": duration}

            analyzer = ImprovedSKUCountAnalyzer(str(detection_dir), str(summary_dir))
            result = analyzer.analyze_with_filtering()

            # 报告目录：若指定 save_root，则保存到 save_root/output_reports/<dataset_name>
            reports_dir = (
                self.save_root / dataset.name / "output_reports"
                if self.save_root else CODE_DIR / dataset.name / "output_reports"
            )
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_file = reports_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with report_file.open('w', encoding='utf-8') as f:
                f.write("改进的SKU计数分析报告\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据集: {dataset_path}\n\n")
                f.write(f"原始匹配数: {result['original_matches']}\n")
                f.write(f"过滤后匹配数: {result['filtered_matches']}\n")
                f.write(f"去重减少: {result['original_matches'] - result['filtered_matches']} 个冗余匹配\n\n")
                f.write("最终匹配结果:\n")
                for i, pair in enumerate(result['pairs'], 1):
                    f.write(
                        f"{i:3d}. Ref({pair['ref_idx']},{pair['ref_id']}) → "
                        f"Target({pair['target_idx']},{pair['target_id']}) hit_ratio={pair['hit_ratio']:.3f}\n"
                    )

            duration = perf_counter() - start
            logger.info(f"✓ SKU分析完成 - 最终匹配数: {result['filtered_matches']}, 耗时 {duration:.2f}s")
            logger.debug(f"报告文件: {report_file}")
            return {"success": True, "duration_s": duration, "details": {"report_file": str(report_file)}}
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError, KeyError) as e:
            duration = perf_counter() - start
            logger.error(f"END improved_analysis duration={duration:.2f}s result=fail error={e}", exc_info=True)
            return {"success": False, "error": str(e), "duration_s": duration}

    def run_accuracy_evaluation(self, dataset_path: str) -> StepResult:
        """运行准确性评估；缺少匹配结果时提示先运行匹配。"""
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
            output_pt_dir = (self.save_root / dataset.name / "output_pt") if self.save_root else (dataset / "output_pt")
            if not output_pt_dir.exists():
                msg = "匹配结果目录不存在，请先运行SKU匹配推理"
                logger.warning(msg)
                duration = perf_counter() - start
                return {"success": False, "error": msg, "duration_s": duration}

            script_path = CODE_DIR / "batch_accuracy_evaluation.sh"
            if script_path.exists():
                import subprocess
                result = subprocess.run(
                    ['bash', str(script_path)],
                    cwd=str(script_path.parent),
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    duration = perf_counter() - start
                    logger.info(f"✓ 评估完成 - 耗时 {duration:.2f}s")
                    return {"success": True, "duration_s": duration}
                else:
                    duration = perf_counter() - start
                    logger.error(f"评估失败: {result.stderr}")
                    return {"success": False, "error": result.stderr, "duration_s": duration}

            duration = perf_counter() - start
            logger.info(f"✓ 评估完成 - 耗时 {duration:.2f}s")
            return {"success": True, "duration_s": duration}
        except (OSError, subprocess.SubprocessError, RuntimeError) as e:
            duration = perf_counter() - start
            logger.error(f"END evaluation duration={duration:.2f}s result=fail error={e}", exc_info=True)
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
        """生成3D点云/GLB（支持后端：VGGT 或 PI3）。

        - 输入图片目录：<dataset>/images
        - 输出GLB：<save_root>/<dataset_name>/reconstruction_{backend}.glb（或 <dataset>/reconstruction_{backend}.glb）
        """
        start = perf_counter()
        try:
            use_backend = (backend or "vggt").lower()
            if use_backend not in ("vggt", "pi3"):
                raise ValueError(f"未知重建后端: {backend}. 仅支持 vggt|pi3")

            # 允许相对路径；当从 code/ 目录执行时，尝试回退到仓库根
            dataset = Path(dataset_path)
            if not dataset.exists():
                candidate = CODE_DIR.parent / dataset_path
                if candidate.exists():
                    dataset = candidate
                    logger.info(f"已将数据集路径解析为: {dataset}")
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
            if use_backend not in output_path.stem:  # 检查文件名（不含扩展名）中是否已包含模型名
                # 在扩展名前插入模型名称：reconstruction.glb -> reconstruction_vggt.glb
                new_filename = f"{output_path.stem}_{use_backend}{output_path.suffix}"
                output_file = cache_dir / new_filename
            else:
                output_file = cache_dir / output_filename

            logger.info(f"开始3D重建[{use_backend}]: {image_dir} → {output_file}")

            if use_backend == "vggt":
                from modules.vggt_3d_reconstructor import VGGT3DReconstructor
                recon = VGGT3DReconstructor(device=device, model_path=model_path)
                result_path = recon.reconstruct_from_directory(
                    input_dir=str(image_dir),
                    output_path=str(output_file),
                    conf_thres=conf_thres,
                    show_cam=show_cam,
                    mask_black_bg=mask_black_bg,
                    mask_white_bg=mask_white_bg,
                    mask_sky=mask_sky,
                )
            else:
                from modules.pi3_3d_reconstructor import PI33DReconstructor
                recon = PI33DReconstructor(device=device, model_path=model_path)
                result_path = recon.reconstruct_from_directory(
                    input_dir=str(image_dir),
                    output_path=str(output_file),
                    conf_thres=conf_thres,
                    show_cam=show_cam,
                    save_predictions=True,
                )

            duration = perf_counter() - start
            logger.info(f"✓ 3D重建完成 - 耗时 {duration:.2f}s")
            return {"success": True, "duration_s": duration, "details": {"output_file": str(result_path)}}
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as e:
            duration = perf_counter() - start
            logger.error(f"END reconstruct duration={duration:.2f}s result=fail error={e}", exc_info=True)
            return {"success": False, "error": str(e), "duration_s": duration}

    def run_dedup_sequence(self, dataset_path: str) -> StepResult:
        """顺序去重：对 1..N（或指定上界）生成去重后的检测 JSON。"""
        start = perf_counter()
        try:
            logger.info("开始顺序去重")
            from modules.deduplicate_detections import resolve_dataset_paths, deduplicate_sequence

            dataset_dir = Path(dataset_path)
            if not dataset_dir.exists():
                candidate = CODE_DIR.parent / dataset_path
                if candidate.exists():
                    dataset_dir = candidate

            paths = resolve_dataset_paths(dataset_dir)
            dataset_name = dataset_dir.name

            # 输出目录：code/Output/<dataset_name>/dedup_detections/
            output_base = self.save_root if self.save_root is not None else (CODE_DIR / "Output")

            result = deduplicate_sequence(
                paths,
                output_root=output_base,       # 模块内部会追加 dataset_name
                max_image=None,                # 处理所有图片
                same_names=True,               # 默认同名输出 (1.json, 2.json, ...)
                dedup_mode='any',              # 默认使用所有匹配进行去重
                min_hit_ratio=0.0,             # 默认不过滤命中率
                output_subdir='dedup_detections'  # 指定子目录名
            )

            # 实际输出路径是 output_base/dataset_name/dedup_detections/
            actual_output_dir = output_base / dataset_name / "dedup_detections"
            duration = perf_counter() - start
            logger.info(f"✓ 去重完成 - 处理 {len(result)} 个文件, 耗时 {duration:.2f}s")
            logger.debug(f"输出目录: {actual_output_dir}")
            return {"success": True, "duration_s": duration, "details": {"count": len(result), "output_dir": str(actual_output_dir)}}
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError, KeyError) as e:
            duration = perf_counter() - start
            logger.error(f"END dedup_sequence duration={duration:.2f}s result=fail error={e}", exc_info=True)
            return {"success": False, "error": str(e), "duration_s": duration}

    def run_complete_pipeline(self, dataset_path: str, algorithm: str = 'point_tracking') -> Dict[str, bool]:
        """运行完整的SKU计数流水线，返回每步是否成功的摘要。"""
        logger.info("开始完整的SKU计数流水线")
        summary: Dict[str, bool] = {}

        if not self.validate_dataset(dataset_path):
            return {"validation": False}
        summary['validation'] = True

        # 1. 原始检测框可视化
        viz = self.run_detection_visualization(dataset_path)
        summary['visualization'] = bool(viz.get('success', False))

        # 2. SKU匹配推理
        match = self.run_sku_matching(dataset_path, algorithm, batch_all_refs=True)
        summary['matching'] = bool(match.get('success', False))

        # 3. SKU计数分析
        analysis = self.run_improved_sku_analysis(dataset_path)
        summary['improved_analysis'] = bool(analysis.get('success', False))

        # 4. 顺序去重（默认包含以便一键产出去重JSON）
        dedup = self.run_dedup_sequence(dataset_path)
        summary['dedup'] = bool(dedup.get('success', False))

        # 5. 去重后的检测框可视化
        if summary['dedup']:
            dataset = Path(dataset_path)
            dataset_name = dataset.name
            output_base = self.save_root if self.save_root is not None else (CODE_DIR / "Output")
            # deduplicate_sequence 输出到 output_base/dataset_name/dedup_detections/
            dedup_detection_dir = output_base / dataset_name / "dedup_detections"

            if dedup_detection_dir.exists() and any(dedup_detection_dir.glob("*.json")):
                logger.info("开始可视化去重后的检测框...")
                dedup_viz = self.run_detection_visualization(
                    dataset_path,
                    detection_dir=str(dedup_detection_dir),
                    output_suffix="dedup_imgs_w_bboxes"
                )
                summary['dedup_visualization'] = bool(dedup_viz.get('success', False))
            else:
                logger.warning(f"去重检测目录为空或不存在: {dedup_detection_dir}")
                summary['dedup_visualization'] = False
        else:
            summary['dedup_visualization'] = False

        # 6. 准确性评估 (可选)
        acc = self.run_accuracy_evaluation(dataset_path)
        summary['accuracy_evaluation'] = bool(acc.get('success', False))

        logger.info("=== 流水线执行结果 ===")
        for step, ok in summary.items():
            status = "✓ 成功" if ok else "✗ 失败"
            logger.info(f"{step:20s}: {status}")

        return summary

    def run_concise_pipeline(self, dataset_path: str, algorithm: str = "point_tracking") -> Dict[str, bool]:
        """运行精简流水线 - 仅SKU匹配和准确性评估"""
        logger.info("开始精简流水线 - SKU Matching + Accuracy evaluation")
        summary: Dict[str, bool] = {}

        if not self.validate_dataset(dataset_path):
            return {"validation": False}
        summary['validation'] = True

        match = self.run_sku_matching(dataset_path, algorithm)
        summary['matching'] = bool(match.get('success', False))

        acc = self.run_accuracy_evaluation(dataset_path)
        summary['accuracy_evaluation'] = bool(acc.get('success', False))

        logger.info("=== 精简流水线执行结果 ===")
        for step, ok in summary.items():
            status = "✓ 成功" if ok else "✗ 失败"
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
            print("4. 3D重建 (VGGT/PI3)")
            print("5. 3D可视化 (Viewer)")
            print("0. 退出")

            choice = input(f"\n当前数据集: {self.default_dataset}\n请输入选择 (0-5): ").strip()

            if choice == '0':
                logger.info("退出程序")
                break
            elif choice == '1':
                algorithm = input("选择算法 (point_tracking/3d) [默认: point_tracking]: ").strip() or 'point_tracking'
                self.run_complete_pipeline(self.default_dataset, algorithm)
            elif choice == '2':
                algorithm = input("选择算法 (point_tracking/3d/both) [默认: both]: ").strip() or 'both'
                self.run_concise_pipeline(self.default_dataset, algorithm)
            elif choice == '3':
                dataset_name = input("输入数据集名称 (如 floor_display2): ").strip()
                if dataset_name:
                    # 自动拼接完整路径: PROJECT_ROOT / "imdata" / dataset_name
                    new_path = str(PROJECT_ROOT / "imdata" / dataset_name)
                    if self.validate_dataset(new_path):
                        self.default_dataset = new_path
                        # 显示完整路径和输出目录信息
                        output_base = self.save_root if self.save_root is not None else (CODE_DIR / "Output")
                        output_dir = output_base / dataset_name
                        logger.info(f"数据集已更改为: {new_path}")
                        logger.info(f"输出目录将使用: {output_dir}")
                    else:
                        logger.warning(f"数据集 '{dataset_name}' 验证失败，保持当前数据集")
            elif choice == '4':
                backend = (input("选择重建后端 (vggt/pi3) [默认 vggt]: ").strip() or 'vggt').lower()
                if backend not in ('vggt','pi3'):
                    logger.warning(f"无效的后端 '{backend}'，使用默认 vggt")
                    backend = 'vggt'
                res = self.run_reconstruction(self.default_dataset, backend=backend)
                print(f"3D重建: {'成功' if res.get('success') else '失败'}，耗时 {res.get('duration_s', 0):.2f}s")
            elif choice == '5':
                # 3D 可视化（viewer），基于智能路径推导
                dataset = Path(self.default_dataset)
                dataset_name = dataset.name
                save_base = self.save_root if self.save_root is not None else (CODE_DIR / 'Output')
                output_dir = save_base / dataset_name

                # 基于统一的 output_dir 自动推导所有路径
                gm_default = output_dir / 'dedup_detections' / 'global_mapping.json'

                # 从cache目录查找reconstruction文件
                recon_default = None
                cache_default = None
                for backend in ['vggt', 'pi3']:
                    cache_dir = output_dir / f'{backend}_cache'
                    candidate = cache_dir / f'reconstruction_{backend}.glb'
                    if candidate.exists():
                        recon_default = candidate
                        cache_default = cache_dir
                        break

                det_default = output_dir / 'dedup_detections'

                img_default = dataset / 'images'

                print("\n即将启动 3D 可视化 (Viewer)")
                print(f"output_dir:     {output_dir}")
                print(f"global_mapping: {gm_default}")
                print(f"reconstruction: {recon_default}")
                print(f"image_dir:      {img_default}")
                print(f"detection_dir:  {det_default}")
                print(f"cache_dir:      {cache_default}")
                pts_src = (input("点云来源 (glb/predictions) [默认 glb]: ").strip() or 'glb')
                try:
                    port = int(input("端口 [默认 8080]: ").strip() or '8080')
                except ValueError:
                    logger.warning("Invalid port, using default 8080")
                    port = 8080
                force = (input("强制重建缓存? (y/N): ").strip().lower() == 'y')
                open_browser = not ((input("启动后自动打开浏览器? (Y/n): ").strip().lower() or 'y') == 'n')

                from modules.viewer_runner import run_viewer as viewer_run
                viewer_run(
                    global_mapping=str(gm_default),
                    reconstruction=str(recon_default),
                    image_dir=str(img_default),
                    detection_dir=str(det_default),
                    cache_dir=str(cache_default),
                    downsample_ratio=1.0,
                    points_source=pts_src,
                    port=port,
                    force_rebuild=force,
                    open_browser=open_browser,
                )
            else:
                print("无效选择，请重试")


def main() -> None:
    # 预解析 --config
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None, help='YAML 配置文件路径（默认 code/config.yaml，如存在）')
    known, _ = pre.parse_known_args()

    # 从 YAML 读取默认值
    from utils import load_yaml_config, extract_main_settings, extract_reconstruction_settings
    yaml_main = {}
    yaml_recon = {}
    config_path = None
    default_cfg = CODE_DIR / 'config.yaml'
    try:
        if known.config:
            config_path = Path(known.config)
        elif default_cfg.exists():
            config_path = default_cfg
        if config_path and config_path.exists():
            data = load_yaml_config(config_path)
            yaml_main = extract_main_settings(data)
            yaml_recon = extract_reconstruction_settings(data)
    except (FileNotFoundError, KeyError) as e:
        logger.warning(f"Failed to load config: {e}, using defaults")
        yaml_main = {}
        yaml_recon = {}

    parser = argparse.ArgumentParser(description="3D SKU Detection系统主程序", parents=[pre])
    parser.add_argument('--dataset', type=str, default=yaml_main.get('dataset', str(PROJECT_ROOT / "imdata" / "floor_display2")),
                       help="数据集目录路径")
    parser.add_argument('--mode', type=str, default=yaml_main.get('mode', "interactive"),
                       choices=['interactive', 'pipeline', 'concise', 'analyzer', 'dedup', 'reconstruct', 'viewer'],
                       help="运行模式: interactive(交互), pipeline(完整), concise(匹配), analyzer(仅分析), dedup(去重), reconstruct(3D重建), viewer(3D可视化)")
    parser.add_argument('--algorithm', type=str, default=yaml_main.get('algorithm', "both"),
                       choices=['point_tracking', '3d', 'both'],
                       help="匹配算法选择")
    # 透传给 inference.py 的关键参数
    parser.add_argument('--reference_idx', type=int, default=int(yaml_main.get('reference_idx', 0)), help="参考图像索引")
    parser.add_argument('--max_images', type=int, default=int(yaml_main.get('max_images', 50)), help="最大处理图像数量")
    parser.add_argument('--device', type=str, default=yaml_main.get('device', 'cuda'), help="计算设备 (cuda/cpu)")
    parser.add_argument('--save_json', action='store_true', default=bool(yaml_main.get('save_json', False)), help="保存匹配结果为 JSON")
    parser.add_argument('--save_root', type=str, default=yaml_main.get('save_root', 'Output'),
                        help="输出保存根目录。例如：/path/to/outputs")
    # 3D重建专用参数
    parser.add_argument('--recon_conf_thres', type=float, default=float(yaml_recon.get('conf_thres', 50.0)), help="3D导出置信度阈值(0-100)")
    parser.add_argument('--recon_output', type=str, default=yaml_recon.get('output', 'reconstruction.glb'), help="3D重建输出文件名")
    parser.add_argument('--recon_backend', type=str, default=yaml_recon.get('backend', 'vggt'), choices=['vggt','pi3'], help="3D重建后端 (vggt|pi3)")
    parser.add_argument('--recon_model_path', type=str, default=yaml_recon.get('model_path', None), help="3D重建模型权重路径")

    # Viewer 参数：复用 --save_root 和 --dataset，无需额外路径参数
    #   - output_dir: <save_root>/<dataset_name>
    #   - image_dir: <dataset>/images
    #   - global_mapping: <output_dir>/dedup_detections/global_mapping.json
    #   - reconstruction: <output_dir>/reconstruction.glb (fallback: <dataset>/reconstruction.glb)
    #   - detection_dir: <output_dir>/dedup_detections
    #   - cache_dir: <output_dir>/viewer_cache
    parser.add_argument('--viewer-global-mapping', type=str, default=None,
                       help='viewer: global_mapping.json 路径（默认：<save_root>/<dataset_name>/dedup_detections/global_mapping.json）')
    parser.add_argument('--viewer-reconstruction', type=str, default=None,
                       help='viewer: reconstruction.glb 路径（默认：<save_root>/<dataset_name>/reconstruction.glb）')
    parser.add_argument('--viewer-image-dir', type=str, default=None,
                       help='viewer: images 目录（默认：<dataset>/images）')
    parser.add_argument('--viewer-detection-dir', type=str, default=None,
                       help='viewer: 检测结果目录（默认：<save_root>/<dataset_name>/dedup_detections）')
    parser.add_argument('--viewer-cache-dir', type=str, default=None,
                       help='viewer: 缓存目录（默认：<save_root>/<dataset_name>/viewer_cache）')
    parser.add_argument('--viewer-points-source', type=str, default='glb', choices=['glb', 'predictions'],
                       help='viewer: 点云来源（默认glb）')
    parser.add_argument('--viewer-downsample', type=float, default=1.0,
                       help='viewer: 下采样比例 0-1（默认1.0）')
    parser.add_argument('--viewer-port', type=int, default=8080,
                       help='viewer: 端口（默认8080）')
    parser.add_argument('--no-viewer-open', action='store_true',
                       help='viewer: 启动后不自动打开浏览器（默认开启自动打开）')
    parser.add_argument('--viewer-force-rebuild', action='store_true',
                       help='viewer: 强制重建缓存')

    args = parser.parse_args()

    # 统一日志
    save_root_path = Path(args.save_root).expanduser().resolve() if args.save_root else (PROJECT_ROOT / 'Output').resolve()
    _configure_logging_to_save_root(save_root_path)

    app = SKUDetectionMain()
    app.default_dataset = args.dataset
    app.save_root = save_root_path
    app.config_path = Path(args.config).resolve() if args.config else (config_path.resolve() if config_path else None)

    if args.mode == 'interactive':
        app.interactive_mode()
    elif args.mode == 'pipeline':
        app.run_complete_pipeline(args.dataset, args.algorithm)
    elif args.mode == 'concise':
        # 在精简流水线中，先匹配后评估；匹配透传关键参数
        app.run_sku_matching(
            args.dataset,
            args.algorithm,
            reference_idx=args.reference_idx,
            max_images=args.max_images,
            device=args.device,
            save_json=args.save_json,
        )
        app.run_accuracy_evaluation(args.dataset)
    elif args.mode == 'analyzer':
        # 仅执行改进的SKU计数分析
        app.run_improved_sku_analysis(args.dataset)
    elif args.mode == 'dedup':
        app.run_dedup_sequence(args.dataset)
    elif args.mode == 'reconstruct':
        app.run_reconstruction(
            args.dataset,
            device=args.device,
            output_filename=args.recon_output,
            backend=args.recon_backend,
            conf_thres=args.recon_conf_thres,
            model_path=args.recon_model_path,
        )
    elif args.mode == 'viewer':
        # 完全复用 --save_root 和 --dataset，无需额外参数
        dataset = Path(args.dataset)
        dataset_name = dataset.name
        output_dir = app.save_root / dataset_name  # save_root已在前面初始化

        # 基于约定自动推导所有路径（可被显式参数覆盖）
        gm_default = output_dir / 'dedup_detections' / 'global_mapping.json'

        # 从cache目录查找reconstruction文件
        recon_default = None
        cache_default = None
        for backend in ['vggt', 'pi3']:
            cache_dir = output_dir / f'{backend}_cache'
            candidate = cache_dir / f'reconstruction_{backend}.glb'
            if candidate.exists():
                recon_default = candidate
                cache_default = cache_dir
                break

        det_default = output_dir / 'detections_results'
        img_default = dataset / 'images'  # images 始终在数据集目录

        # 通过 modules.viewer_runner 调用
        from modules.viewer_runner import run_viewer as viewer_run
        viewer_run(
            global_mapping=str(Path(args.viewer_global_mapping) if args.viewer_global_mapping else gm_default),
            reconstruction=str(Path(args.viewer_reconstruction) if args.viewer_reconstruction else recon_default),
            image_dir=str(Path(args.viewer_image_dir) if args.viewer_image_dir else img_default),
            detection_dir=str(Path(args.viewer_detection_dir) if args.viewer_detection_dir else det_default),
            cache_dir=str(Path(args.viewer_cache_dir)) if args.viewer_cache_dir else str(cache_default),
            downsample_ratio=float(args.viewer_downsample),
            points_source=str(args.viewer_points_source) if args.viewer_points_source else 'glb',
            port=int(args.viewer_port),
            force_rebuild=bool(args.viewer_force_rebuild),
            open_browser=not bool(getattr(args, 'no_viewer_open', False)),
        )


if __name__ == "__main__":
    main()
