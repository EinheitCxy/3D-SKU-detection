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
PROJECT_ROOT = Path(__file__).parent.parent
CODE_DIR = Path(__file__).parent

# 确保可以从仓库根或任意 CWD 导入本目录模块
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))


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
        '%(log_color)s%(levelname)s - %(message)s%(reset)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'WARNING': 'yellow', 
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    fh = RotatingFileHandler(str(log_file), maxBytes=10_000_000, backupCount=1, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(console_fmt)

    root_logger.addHandler(fh)
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
        """显示程序横幅"""
        from datetime import datetime
        import sys
        
        # 获取运行时信息
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # 安全获取torch信息
        try:
            import torch
            torch_version = torch.__version__
            cuda_available = "Yes" if torch.cuda.is_available() else "No"
        except ImportError:
            torch_version = "N/A"
            cuda_available = "N/A"
        
        banner = f"""
╔════════════════════════════════════════════════════════════════════════╗
║                        3D SKU Detection System                         ║
║                     RetailEye 商品计数分析平台 v2.0                       ║
╠════════════════════════════════════════════════════════════════════════╣
║  核心功能:                                                              ║
║  1. SKU匹配推理 (点追踪 + 3D投影)     2. 检出框可视化                       ║
║  3. SKU聚类分析                      4. 匹配准确性评估                     ║
║  5. 改进的SKU计数 (去重优化)          6. 3D场景重建                        ║
╠════════════════════════════════════════════════════════════════════════╣
║  运行环境:                                                               ║
║  时间: {current_time:<20} Python: {python_version}                      ║
║  PyTorch: {torch_version:<12} CUDA: {cuda_available}                   ║
╠════════════════════════════════════════════════════════════════════════╣
║  快速开始:                                                              ║
║  • 完整流水线: --mode pipeline --dataset imdata/floor_display2          ║
║  • 仅匹配推理: --mode concise --algorithm point_tracking                         ║
║  • 交互模式:   --mode interactive (当前模式)                             ║
║  • 帮助文档:   --help                                                   ║
╚════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)

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

        logger.info(f"数据集验证通过: {dataset_path}")
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
    ) -> StepResult:
        """运行SKU匹配推理，支持批量将每张图片作为参考图像运行。

        - 当 batch_all_refs=True 时：遍历 images/ 中数字命名且在 detections_results/ 有有效 objects 的每个图片，依次作为参考图运行。
        - 否则：仅以 reference_idx 指定的单张图片作为参考图运行。
        """
        start = perf_counter()
        original_argv = sys.argv.copy()
        try:
            logger.info("START matching")
            if batch_all_refs:
                # 批量处理所有有效图片作为参考图片
                import json
                dataset = Path(dataset_path)
                image_folder = dataset / "images"
                detection_dir = dataset / "detections_results"

                # 检测有效图片索引（文件名为纯数字且 JSON 含非空 objects）
                valid_indices: list[int] = []
                for img_file in image_folder.glob("*"):
                    if img_file.is_file() and img_file.stem.isdigit():
                        detection_file = detection_dir / f"{img_file.stem}.json"
                        if detection_file.exists():
                            try:
                                with detection_file.open('r', encoding='utf-8') as f:
                                    data = json.load(f)
                                if ((isinstance(data, list) and data and isinstance(data[0], dict) and data[0].get('objects')) or
                                    (isinstance(data, dict) and ((isinstance(data.get('skus'), list) and data['skus'] and data['skus'][0].get('objects')) or data.get('objects')))):
                                    valid_indices.append(int(img_file.stem))
                            except Exception:
                                continue

                valid_indices.sort()
                logger.info(f"找到 {len(valid_indices)} 个有效图片作为参考图片")

                # 依次以每个有效图片为参考图片运行推理
                for i, filename_idx in enumerate(valid_indices):
                    # 文件名从1开始，但系统内部索引从0开始，所以需要减1
                    system_ref_idx = filename_idx - 1
                    logger.info(f"处理参考图片 {filename_idx} ({i+1}/{len(valid_indices)}) -> 系统索引: {system_ref_idx}")
                    self._run_single_matching(dataset_path, algorithm, system_ref_idx, max_images, device, save_json)

                duration = perf_counter() - start
                logger.info(f"END matching duration={duration:.2f}s result=ok processed_refs={len(valid_indices)}")
                return {"success": True, "duration_s": duration}
            else:
                # 单个参考图片处理
                return self._run_single_matching(dataset_path, algorithm, reference_idx, max_images, device, save_json)

        except Exception as e:
            duration = perf_counter() - start
            logger.error(f"END matching duration={duration:.2f}s result=fail error={e}", exc_info=True)
            return {"success": False, "error": str(e), "duration_s": duration}
        finally:
            sys.argv = original_argv

    def _run_single_matching(self, dataset_path: str, algorithm: str, reference_idx: int,
                           max_images: int, device: str, save_json: bool) -> StepResult:
        """运行单个参考图片的SKU匹配推理（内部使用）。"""
        start = perf_counter()
        original_argv = sys.argv.copy()
        try:
            logger.info(f"START matching_single algo={algorithm} ref={reference_idx}")

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
            ]
            if save_json:
                argv.append('--save_json')
            if self.config_path is not None:
                argv.extend(['--config', str(self.config_path)])

            sys.argv = argv
            inference_main()

            duration = perf_counter() - start
            logger.info(f"END matching_single duration={duration:.2f}s result=ok")
            return {"success": True, "duration_s": duration}
        except Exception as e:
            duration = perf_counter() - start
            logger.error(f"END matching_single duration={duration:.2f}s result=fail error={e}", exc_info=True)
            return {"success": False, "error": str(e), "duration_s": duration}
        finally:
            sys.argv = original_argv

    

    def run_detection_visualization(self, dataset_path: str) -> StepResult:
        """运行检出框可视化，输出到 output_viz/<dataset_name>/（或 --save_root）"""
        start = perf_counter()
        original_argv = sys.argv.copy()
        try:
            logger.info("START visualization")

            from modules.draw_detection_boxes import main as viz_main

            dataset = Path(dataset_path)
            image_dir = dataset / "images"
            detection_dir = dataset / "detections_results"

            # 输出目录：若指定 save_root，则写到 save_root/output_viz/<dataset_name>
            output_viz_dir = (
                (self.save_root / dataset.name / "imgs_w_bboxes")
                if self.save_root else (CODE_DIR / dataset.name / "imgs_w_bboxes")
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
            logger.info(f"END visualization duration={duration:.2f}s result=ok")
            return {"success": True, "duration_s": duration, "details": {"output_dir": str(output_viz_dir)}}
        except Exception as e:
            duration = perf_counter() - start
            logger.error(f"END visualization duration={duration:.2f}s result=fail error={e}", exc_info=True)
            return {"success": False, "error": str(e), "duration_s": duration}
        finally:
            sys.argv = original_argv

    def run_improved_sku_analysis(self, dataset_path: str) -> StepResult:
        """运行改进的SKU计数分析 (去重优化)，报告写入 <dataset_name>/output_reports/report_*.txt（或 --save_root）"""
        start = perf_counter()
        try:
            logger.info("START improved_analysis")

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
            logger.info(f"END improved_analysis duration={duration:.2f}s result=ok output={report_file}")
            logger.info(f"最终SKU匹配数: {result['filtered_matches']}")
            return {"success": True, "duration_s": duration, "details": {"report_file": str(report_file)}}
        except Exception as e:
            duration = perf_counter() - start
            logger.error(f"END improved_analysis duration={duration:.2f}s result=fail error={e}", exc_info=True)
            return {"success": False, "error": str(e), "duration_s": duration}

    def run_accuracy_evaluation(self, dataset_path: str) -> StepResult:
        """运行准确性评估；缺少匹配结果时提示先运行匹配。"""
        start = perf_counter()
        try:
            logger.info("START evaluation")

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
                    logger.info(f"END evaluation duration={duration:.2f}s result=ok (batch)")
                    return {"success": True, "duration_s": duration}
                else:
                    duration = perf_counter() - start
                    logger.error(f"END evaluation duration={duration:.2f}s result=fail error={result.stderr}")
                    return {"success": False, "error": result.stderr, "duration_s": duration}

            duration = perf_counter() - start
            logger.info(f"END evaluation duration={duration:.2f}s result=ok")
            return {"success": True, "duration_s": duration}
        except Exception as e:
            duration = perf_counter() - start
            logger.error(f"END evaluation duration={duration:.2f}s result=fail error={e}", exc_info=True)
            return {"success": False, "error": str(e), "duration_s": duration}

    def run_reconstruction(
        self,
        dataset_path: str,
        *,
        device: str | None = None,
        output_filename: str = "reconstruction.glb",
        conf_thres: float = 50.0,
        model_path: str | None = None,
        show_cam: bool = True,
        mask_black_bg: bool = False,
        mask_white_bg: bool = False,
        mask_sky: bool = False,
    ) -> StepResult:
        """使用 VGGT 生成3D点云/GLB。

        - 输入图片目录：<dataset>/images
        - 输出GLB：<save_root>/<dataset_name>/reconstruction.glb（或 <dataset>/reconstruction.glb）
        """
        start = perf_counter()
        try:
            from modules.vggt_3d_reconstructor import VGGT3DReconstructor

            dataset = Path(dataset_path)
            image_dir = dataset / "images"
            if not image_dir.exists():
                msg = f"图片目录不存在: {image_dir}"
                logger.error(msg)
                return {"success": False, "error": msg, "duration_s": 0.0}

            # 选择输出位置
            output_dir = (self.save_root / dataset.name) if self.save_root else dataset
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / output_filename

            logger.info("START reconstruct")
            logger.info(f"输入图片目录: {image_dir}")
            logger.info(f"输出GLB文件: {output_file}")

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

            duration = perf_counter() - start
            logger.info(f"END reconstruct duration={duration:.2f}s result=ok output={result_path}")
            return {"success": True, "duration_s": duration, "details": {"output_file": str(result_path)}}
        except Exception as e:
            duration = perf_counter() - start
            logger.error(f"END reconstruct duration={duration:.2f}s result=fail error={e}", exc_info=True)
            return {"success": False, "error": str(e), "duration_s": duration}

    def run_dedup_sequence(self, dataset_path: str) -> StepResult:
        """顺序去重：对 1..N（或指定上界）生成去重后的检测 JSON。"""
        start = perf_counter()
        try:
            logger.info("START dedup_sequence")
            from modules.deduplicate_detections import resolve_dataset_paths, deduplicate_sequence

            dataset_dir = Path(dataset_path)
            if not dataset_dir.exists():
                candidate = CODE_DIR.parent / dataset_path
                if candidate.exists():
                    dataset_dir = candidate

            paths = resolve_dataset_paths(dataset_dir)

            output_root = self.save_root if self.save_root is not None else (CODE_DIR / "output_dedup")

            result = deduplicate_sequence(
                paths,
                output_root=output_root,
                max_image=None,           # 处理所有图片
                same_names=True,          # 默认同名输出 (1.json, 2.json, ...)
                dedup_mode='any',         # 默认使用所有匹配进行去重
                min_hit_ratio=0.0,        # 默认不过滤命中率
            )

            duration = perf_counter() - start
            dataset_name = Path(dataset_path).name
            logger.info(f"END dedup_sequence duration={duration:.2f}s result=ok output_dir={output_root / dataset_name} count={len(result)}")
            return {"success": True, "duration_s": duration, "details": {"count": len(result), "output_root": str(output_root)}}
        except Exception as e:
            duration = perf_counter() - start
            logger.error(f"END dedup_sequence duration={duration:.2f}s result=fail error={e}", exc_info=True)
            return {"success": False, "error": str(e), "duration_s": duration}

    def run_complete_pipeline(self, dataset_path: str) -> Dict[str, bool]:
        """运行完整的SKU计数流水线，返回每步是否成功的摘要。"""
        logger.info("开始完整的SKU计数流水线")
        summary: Dict[str, bool] = {}

        if not self.validate_dataset(dataset_path):
            return {"validation": False}
        summary['validation'] = True

        viz = self.run_detection_visualization(dataset_path)
        summary['visualization'] = bool(viz.get('success', False))

        match = self.run_sku_matching(dataset_path, 'point_tracking', batch_all_refs=True)  # TODO: algorithm is hardcoded now
        summary['matching'] = bool(match.get('success', False))

        analysis = self.run_improved_sku_analysis(dataset_path)
        summary['improved_analysis'] = bool(analysis.get('success', False))

        # 6. 顺序去重（默认包含以便一键产出去重JSON）
        dedup = self.run_dedup_sequence(dataset_path)
        summary['dedup'] = bool(dedup.get('success', False))

        # 7. 准确性评估 (可选)
        acc = self.run_accuracy_evaluation(dataset_path)
        summary['accuracy_evaluation'] = bool(acc.get('success', False))

        logger.info("=== 流水线执行结果 ===")
        for step, ok in summary.items():
            status = "✓ 成功" if ok else "✗ 失败"
            logger.info(f"{step:20s}: {status}")

        return summary

    def run_concise_pipeline(self, dataset_path: str, algorithm: str = "point_tracking") -> Dict[str, bool]:
        """运行精简流水线 - 仅SKU匹配和准确性评估"""
        logger.info("开始精简流水线 - SKU匹配 + 准确性评估")
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
            print("2. 运行精简流水线 (SKU匹配 + 准确性评估)")
            print("3. SKU匹配推理")
            print("4. 检出框可视化")
            print("5. 改进的SKU计数分析")
            print("6. 准确性评估")
            print("7. 更改数据集路径")
            print("8. 3D重建 (VGGT)")
            print("0. 退出")

            choice = input(f"\n当前数据集: {self.default_dataset}\n请输入选择 (0-8): ").strip()

            if choice == '0':
                logger.info("退出程序")
                break
            elif choice == '1':
                self.run_complete_pipeline(self.default_dataset)
            elif choice == '2':
                algorithm = input("选择算法 (point_tracking/3d/both) [默认: both]: ").strip() or 'both'
                self.run_concise_pipeline(self.default_dataset, algorithm)
            elif choice == '3':
                algorithm = input("选择算法 (point_tracking/3d/both) [默认: both]: ").strip() or 'both'
                res = self.run_sku_matching(self.default_dataset, algorithm)
                print(f"匹配: {'成功' if res.get('success') else '失败'}，耗时 {res.get('duration_s', 0):.2f}s")
            elif choice == '4':
                res = self.run_detection_visualization(self.default_dataset)
                print(f"可视化: {'成功' if res.get('success') else '失败'}，耗时 {res.get('duration_s', 0):.2f}s")
            elif choice == '5':
                res = self.run_improved_sku_analysis(self.default_dataset)
                print(f"分析: {'成功' if res.get('success') else '失败'}，耗时 {res.get('duration_s', 0):.2f}s")
            elif choice == '6':
                res = self.run_accuracy_evaluation(self.default_dataset)
                print(f"评估: {'成功' if res.get('success') else '失败'}，耗时 {res.get('duration_s', 0):.2f}s")
            elif choice == '7':
                new_path = input("输入新的数据集路径: ").strip()
                if new_path and self.validate_dataset(new_path):
                    self.default_dataset = new_path
                    logger.info(f"数据集已更改为: {new_path}")
            elif choice == '8':
                res = self.run_reconstruction(self.default_dataset)
                print(f"3D重建: {'成功' if res.get('success') else '失败'}，耗时 {res.get('duration_s', 0):.2f}s")
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
    except Exception:
        yaml_main = {}
        yaml_recon = {}

    parser = argparse.ArgumentParser(description="3D SKU Detection系统主程序", parents=[pre])
    parser.add_argument('--dataset', type=str, default=yaml_main.get('dataset', str(PROJECT_ROOT / "imdata" / "floor_display2")),
                       help="数据集目录路径")
    parser.add_argument('--mode', type=str, default=yaml_main.get('mode', "interactive"),
                       choices=['interactive', 'pipeline', 'concise', 'analyzer', 'dedup', 'reconstruct'],
                       help="运行模式: interactive(交互), pipeline(完整), concise(匹配), analyzer(仅分析), dedup(去重), reconstruct(3D重建)")
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
    parser.add_argument('--recon_model_path', type=str, default=yaml_recon.get('model_path', None), help="3D重建模型权重路径")

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
        app.run_complete_pipeline(args.dataset)
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
            conf_thres=args.recon_conf_thres,
            model_path=args.recon_model_path,
        )


if __name__ == "__main__":
    main()
