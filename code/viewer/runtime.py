from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from utils.kdtree_utils import build_kdtree

from .paths import _resolve_da3_cache_dir
from .types import ViewerArtifacts, ViewerRuntimeConfig

logger = logging.getLogger(__name__)


def _closed_form_inverse_se3(se3: np.ndarray) -> np.ndarray:
    """w2c 的 SE3 求逆（c2w）：R^T, -R^T·t。

    支持输入 (3,4)/(4,4)/(N,3,4)/(N,4,4)，返回同形状逆矩阵。
    输入不足 4x4 时先补 [0,0,0,1] 行，求逆后裁回原形状。
    （替代 vggt.utils.geometry.closed_form_inverse_se3，da3 缓存 extrinsic 为 (N,3,4) w2c。）
    """
    arr = np.asarray(se3, dtype=np.float64)
    orig_shape = arr.shape
    single = arr.ndim == 2
    if single:
        arr = arr[None, ...]  # (1, H, W)
    # 补齐到 4x4（若输入是 3x4 / (N,3,4)）
    if arr.shape[-2] == 3:
        pad_row = np.zeros(arr.shape[:-2] + (1, arr.shape[-1]), dtype=arr.dtype)
        pad_row[..., -1] = 1.0
        arr = np.concatenate([arr, pad_row], axis=-2)
    # arr: (N, 4, 4) 形如 [R|t; 0 1]
    R = arr[..., :3, :3]
    t = arr[..., :3, 3:4]
    R_inv = np.swapaxes(R, -1, -2)  # R^T
    t_inv = -np.matmul(R_inv, t)  # -R^T·t
    inv = np.empty_like(arr)
    inv[..., :3, :3] = R_inv
    inv[..., :3, 3:4] = t_inv
    inv[..., 3, :3] = 0.0
    inv[..., 3, 3] = 1.0
    if single:
        inv = inv[0]
    # 裁回原形状（输入为 3x4 时返回 3x4）
    if inv.shape != orig_shape:
        inv = inv[..., : orig_shape[-2], : orig_shape[-1]]
    return inv


class ViserViewer:
    def __init__(
        self,
        artifacts: ViewerArtifacts,
        runtime: ViewerRuntimeConfig,
        image_dir: Path,
        global_mapping: Path,
    ):
        self.artifacts = artifacts
        self.runtime = runtime
        self.image_dir = image_dir
        self.global_mapping_path = global_mapping
        self.server = None
        self.kdtree = None

    @staticmethod
    def _is_right_or_middle_button(evt) -> bool:
        """检查鼠标事件是否为右键或中键"""
        try:
            btn = (
                str(evt.button).lower()
                if hasattr(evt, "button") and evt.button is not None
                else "left"
            )
            return "right" in btn or "middle" in btn
        except (AttributeError, ValueError):
            return False

    @staticmethod
    def _is_shift_pressed(evt) -> bool:
        """检查Shift键是否按下（兼容多种viser API版本）"""
        try:
            if hasattr(evt, "shift") and bool(evt.shift):
                return True
            if hasattr(evt, "shift_key") and bool(evt.shift_key):
                return True
            if hasattr(evt, "modifiers") and evt.modifiers is not None:
                mods = str(evt.modifiers).lower()
                return ("shift" in mods) or ("mod.shift" in mods)
        except (AttributeError, ValueError):
            return False
        return False

    def _register_pointer_event(
        self, event_type: str, callback: Callable, skip_right_middle: bool = True
    ) -> bool:
        """安全注册指针事件，封装try/except和按钮过滤逻辑。

        兼容说明：某些 viser 版本不支持 'down'/'move'/'up' 类型，
        在这种情况下，on_pointer_event 会在装饰器创建时触发断言。
        这里捕获 AssertionError 并跳过注册，以保证服务器不崩溃。
        """
        try:

            @self.server.scene.on_pointer_event(event_type=event_type)
            def _handler(evt) -> None:
                # 统一的按钮过滤
                if skip_right_middle and self._is_right_or_middle_button(evt):
                    return
                # 调用实际业务逻辑
                callback(evt)

            return True
        except (AttributeError, RuntimeError, AssertionError, ValueError) as e:
            logger.debug(f"Failed to register {event_type} pointer event: {e}")
            return False

    def _load_cache(self) -> None:
        data = np.load(self.artifacts.pcd_cache_path)
        self.points = data["points"]
        self.colors = data["colors"]
        self.global_ids = data["global_ids"]
        self.confidences = data["confidences"]
        self.frame_indices = data["frame_indices"] if "frame_indices" in data else None
        self.scene_center = np.mean(self.points, axis=0)
        self.points_centered = self.points - self.scene_center
        with self.artifacts.index_cache_path.open("r", encoding="utf-8") as f:
            self.index = json.load(f)
        if self.colors.dtype != np.uint8:
            self.colors = np.clip(
                self.colors * (255.0 if self.colors.max() <= 1.0 else 1.0), 0, 255
            ).astype(np.uint8)

    def _load_camera_data(
        self,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]]:
        try:
            dataset_root = self.artifacts.pcd_cache_path.parent.parent
            pred = _resolve_da3_cache_dir(dataset_root) / "predictions.npz"
            if not pred.exists():
                return None
            data = np.load(pred)
            extr = data["extrinsic"]
            intr = data["intrinsic"]
            images = data["images"]
            if images.ndim == 4 and images.shape[1] == 3:
                images = images.transpose(0, 2, 3, 1)
            if images.dtype != np.uint8:
                images = np.clip(
                    images * (255.0 if images.max() <= 1.0 else 1.0), 0, 255
                ).astype(np.uint8)
            if "image_ids" in data:
                image_ids = data["image_ids"].tolist()
            else:
                image_ids = list(range(extr.shape[0]))

            # 本地实现 SE3 求逆（替代 vggt.utils.geometry.closed_form_inverse_se3）
            cam2world = _closed_form_inverse_se3(extr)[:, :3, :]
            cam2world[..., -1] -= self.scene_center
            return cam2world, intr, images, image_ids
        except (FileNotFoundError, KeyError) as e:
            logger.debug(f"Failed to load camera data: {e}")
            return None

    def start(self) -> None:
        global viser
        if "viser" not in globals() or globals().get("viser") is None:
            import viser as _viser

            viser = _viser

        self._load_cache()

        logger.info(f"Starting Viser server on port {self.runtime.port}")
        self.server = viser.ViserServer(host="0.0.0.0", port=self.runtime.port)
        self.server.gui.configure_theme(
            titlebar_content=None, control_layout="collapsible"
        )

        points_centered = self.points_centered
        span = (np.max(points_centered, axis=0) - np.min(points_centered, axis=0)).max()
        dist = span * 2.0 if span > 0 else 1.0
        initial_camera_position = (
            float(dist * 0.7),
            float(dist * 0.5),
            float(dist * 0.7),
        )

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            try:
                client.camera.position = initial_camera_position
            except (AttributeError, RuntimeError):
                # Client API可能不支持position属性
                logger.debug("Failed to set initial camera position")

        # Build KDTree on rotated points (initial no-rotation)
        self.R_cur = np.eye(3, dtype=float)
        self.rotated_points = points_centered
        self.kdtree = build_kdtree(self.rotated_points)

        # Controls
        gui_conf = self.server.gui.add_slider(
            "Confidence %",
            min=0,
            max=100,
            step=0.1,
            initial_value=float(self.runtime.default_conf_percentile),
        )
        gui_point_size = self.server.gui.add_slider(
            "Point Size",
            min=0.0001,
            max=0.005,
            step=0.0001,
            initial_value=float(self.runtime.default_point_size),
        )
        unique_gids = np.unique(self.global_ids[self.global_ids >= 0])
        gid_options = [str(x) for x in sorted(unique_gids)]
        gui_gid = self.server.gui.add_dropdown(
            "Show Global ID", options=["All"] + gid_options, initial_value="All"
        )
        # Navigation buttons: All / Prev / Next
        btn_all = self.server.gui.add_button("All")
        btn_prev = self.server.gui.add_button("Prev")
        btn_next = self.server.gui.add_button("Next")

        @btn_all.on_click
        def _(_):
            gui_gid.value = "All"

        @btn_prev.on_click
        def _(_):
            if not gid_options:
                return
            cur = gui_gid.value
            if cur == "All":
                gui_gid.value = gid_options[0]
                return
            try:
                idx = gid_options.index(str(cur))
            except ValueError:
                gui_gid.value = gid_options[0]
                return
            gui_gid.value = gid_options[(idx - 1) % len(gid_options)]

        @btn_next.on_click
        def _(_):
            if not gid_options:
                return
            cur = gui_gid.value
            if cur == "All":
                gui_gid.value = gid_options[0]
                return
            try:
                idx = gid_options.index(str(cur))
            except ValueError:
                gui_gid.value = gid_options[0]
                return
            gui_gid.value = gid_options[(idx + 1) % len(gid_options)]

        with self.server.gui.add_folder("📋 Selected ID Info", expand_by_default=True):
            gui_info_gid = self.server.gui.add_text(
                "Global ID", initial_value="(None)", disabled=True
            )
            gui_info_images = self.server.gui.add_text(
                "Images", initial_value="-", disabled=True
            )
            gui_info_objects = self.server.gui.add_text(
                "Objects ID", initial_value="-", disabled=True
            )
            gui_info_status = self.server.gui.add_text(
                "Status (Active/Filtered)", initial_value="-", disabled=True
            )
            self.server.gui.add_markdown(
                "*Tip: Enable Pick Mode and click on point cloud, or select from dropdown above.*"
            )

        with self.server.gui.add_folder("🔍 Point Picking"):
            gui_pick_mode = self.server.gui.add_checkbox(
                "Enable Pick Mode", initial_value=False
            )
            # radius proportional to span
            rmin, rmax, rinit = span * 0.0002, span * 0.02, span * 0.0005
            gui_pick_radius = self.server.gui.add_slider(
                "Pick Radius",
                min=float(max(rmin, 1e-9)),
                max=float(max(rmax, 1e-6)),
                step=float(max((rmax - rmin) / 100.0, 1e-6)),
                initial_value=float(max(rinit, 1e-9)),
            )
            gui_show_pick = self.server.gui.add_checkbox(
                "Show Pick Sphere",
                initial_value=bool(self.runtime.show_pick_sphere_default),
            )

        with self.server.gui.add_folder("🔷 Mesh Generation", expand_by_default=False):
            gui_mesh_method = self.server.gui.add_dropdown(
                "Method", options=["poisson", "ball_pivoting"], initial_value="poisson"
            )
            gui_mesh_depth = self.server.gui.add_slider(
                "Poisson Depth", min=6, max=12, step=1, initial_value=9
            )
            btn_generate_mesh = self.server.gui.add_button("Generate Mesh")
            gui_show_mesh = self.server.gui.add_checkbox(
                "Show Mesh", initial_value=False
            )
            gui_show_points = self.server.gui.add_checkbox(
                "Show Points", initial_value=True
            )
            gui_mesh_status = self.server.gui.add_text(
                "Status", initial_value="No mesh generated", disabled=True
            )

        with self.server.gui.add_folder("⚙️ Advanced Options", expand_by_default=False):
            gui_sampling = self.server.gui.add_slider(
                "Display Sample %", min=10, max=100, step=10, initial_value=100
            )
            gui_hide_unknown = self.server.gui.add_checkbox(
                "Hide Unknown IDs (-1)",
                initial_value=bool(self.runtime.hide_unknown_default),
            )
            # 说明：默认相机交互始终可用（左键旋转 / 右键平移 / 滚轮缩放）
            self.server.gui.add_markdown(
                "Default camera: left=rotate, right=pan, wheel=zoom"
            )

        # Camera viz
        camera_data = self._load_camera_data()
        frames = []
        frustums = []
        if camera_data is not None:
            extr, intr, images, image_ids = camera_data
            import viser.transforms as viser_tf

            for idx in range(extr.shape[0]):
                T = viser_tf.SE3.from_matrix(extr[idx])
                frame_axis = self.server.scene.add_frame(
                    f"camera_{idx}",
                    wxyz=T.rotation().wxyz,
                    position=T.translation(),
                    axes_length=0.1,
                    axes_radius=0.005,
                    origin_radius=0.005,
                )
                frames.append(frame_axis)
                img = images[idx]
                h, w = img.shape[:2]
                fy = intr[idx][1, 1]
                fov = 2 * np.arctan2(h / 2, fy)
                fr = self.server.scene.add_camera_frustum(
                    f"camera_{idx}/frustum",
                    fov=float(fov),
                    aspect=w / h,
                    scale=0.1,
                    image=img,
                    line_width=1.0,
                )
                frustums.append(fr)

                @fr.on_click
                def _(_, frame=frame_axis):
                    for client in self.server.get_clients().values():
                        client.camera.wxyz = frame.wxyz
                        client.camera.position = frame.position

            gui_show_cameras = self.server.gui.add_checkbox(
                "Show Cameras", initial_value=bool(self.runtime.show_cameras_default)
            )

            @gui_show_cameras.on_update
            def _(_):
                for f in frames:
                    f.visible = gui_show_cameras.value
                for fr in frustums:
                    fr.visible = gui_show_cameras.value

        # Frame selector
        gui_frame_selector = None
        if self.frame_indices is not None:
            fids = sorted(np.unique(self.frame_indices).tolist())
            gui_frame_selector = self.server.gui.add_dropdown(
                "Show Points from Frame",
                options=["All"] + [str(x) for x in fids],
                initial_value="All",
            )

        # Status bar (always visible)
        gui_status = self.server.gui.add_text(
            "Status", initial_value="-", disabled=True
        )

        # Point cloud and pick sphere
        point_cloud = self.server.scene.add_point_cloud(
            name="sku_pcd",
            points=self.rotated_points[:1],
            colors=self.colors[:1],
            point_size=gui_point_size.value,
            point_shape="circle",
        )
        pick_sphere = self.server.scene.add_icosphere(
            name="pick_sphere",
            radius=gui_pick_radius.value,
            color=(255, 255, 0),
            position=(0.0, 0.0, 0.0),
            visible=False,
        )

        # Mesh state
        mesh_handle = None
        mesh_data = {"vertices": None, "faces": None, "colors": None}

        rng = np.random.default_rng(42)
        display_rand = rng.random(len(self.points))

        def clear_info():
            gui_info_gid.value = "(None)"
            gui_info_images.value = "-"
            gui_info_objects.value = "-"
            gui_info_status.value = "-"

        def update_info(gid_str: str):
            info = self.index.get(gid_str)
            if not info:
                gui_info_gid.value = f"ID {gid_str} (No data)"
                gui_info_images.value = "-"
                gui_info_objects.value = "-"
                gui_info_status.value = "-"
                return
            gui_info_gid.value = gid_str
            gui_info_images.value = ", ".join(map(str, info["images"]))
            gui_info_objects.value = ", ".join(map(str, info["objects"][:10])) + (
                "..." if len(info["objects"]) > 10 else ""
            )
            gui_info_status.value = f"{info['active_count']} / {info['removed_count']}"

        def update_point_cloud():
            thr = np.percentile(self.confidences, gui_conf.value)
            conf_mask = self.confidences >= thr
            if gui_gid.value == "All":
                id_mask = np.ones(len(self.global_ids), dtype=bool)
            else:
                id_mask = self.global_ids == int(gui_gid.value)
            if gui_frame_selector is not None and gui_frame_selector.value != "All":
                frame_mask = self.frame_indices == int(gui_frame_selector.value)
            else:
                frame_mask = np.ones(len(self.global_ids), dtype=bool)
            known_mask = (
                (self.global_ids >= 0)
                if gui_hide_unknown.value
                else np.ones(len(self.global_ids), dtype=bool)
            )
            sample_mask = display_rand <= (gui_sampling.value / 100.0)
            mask = conf_mask & id_mask & frame_mask & known_mask & sample_mask

            n_vis = int(mask.sum())

            # Update status
            try:
                total = int(len(self.points))
                frame_lbl = (
                    gui_frame_selector.value
                    if gui_frame_selector is not None
                    else "All"
                )
                id_lbl = gui_gid.value
                gui_status.value = f"Visible: {n_vis}/{total} | ID: {id_lbl} | Frame: {frame_lbl} | Conf≥{gui_conf.value:.1f}%"
            except (AttributeError, ValueError) as e:
                logger.debug(f"Failed to update status: {e}")
            point_cloud.points = self.rotated_points[mask]
            point_cloud.colors = self.colors[mask]

        def handle_pick(click_pos: np.ndarray):
            """处理点云拾取：找到最近的点并更新显示"""
            pick_sphere.position = tuple(float(x) for x in click_pos)
            pick_sphere.radius = gui_pick_radius.value
            pick_sphere.visible = bool(gui_show_pick.value)
            # nearest
            d, idx = self.kdtree.query(click_pos, k=1)
            if d <= gui_pick_radius.value:
                gid = int(self.global_ids[int(idx)])
                if gid >= 0:
                    gid_str = str(gid)
                    update_info(gid_str)
                    gui_gid.value = gid_str
                    update_point_cloud()
                else:
                    clear_info()
            else:
                clear_info()

        # ============================================================
        # 交互控制策略（参考demo_viser.py）:
        # 1. 默认不注册pointer事件，让viser完全控制相机
        # 2. 只在Pick Mode启用时注册click事件
        # 3. Click事件尽可能轻量，避免阻塞相机交互
        # ============================================================

        click_handler_registered = {"val": False}

        @gui_pick_mode.on_update
        def _(_):
            """动态注册/注销点击事件，避免干扰相机控制"""
            if gui_pick_mode.value and not click_handler_registered["val"]:
                # 启用Pick Mode - 注册click事件
                try:

                    @self.server.scene.on_pointer_event(event_type="click")
                    def _on_click(evt) -> None:
                        if not gui_pick_mode.value:
                            return
                        if (
                            not hasattr(evt, "ray_direction")
                            or evt.ray_direction is None
                        ):
                            return
                        ray = np.asarray(evt.ray_direction)
                        o = np.asarray(evt.ray_origin)
                        ray = ray / (np.linalg.norm(ray) + 1e-9)
                        depth_est = np.linalg.norm(o)
                        best_idx = None
                        best_d = float("inf")
                        # 增加采样点提高命中率
                        for t in np.linspace(0, depth_est * 2, 50):
                            p = o + ray * t
                            d, idx = self.kdtree.query(p, k=1)
                            search_radius = gui_pick_radius.value * 2.0
                            if d < best_d and d < search_radius:
                                best_d = d
                                best_idx = int(idx)
                        if best_idx is not None:
                            handle_pick(self.rotated_points[best_idx])
                        else:
                            clear_info()

                    click_handler_registered["val"] = True
                    logger.info("Pick Mode enabled - click on point cloud to select")
                except Exception as e:
                    logger.warning(f"Failed to register click event: {e}")
            elif not gui_pick_mode.value:
                # 禁用Pick Mode - 无需注销（viser会自动处理）
                click_handler_registered["val"] = False
                pick_sphere.visible = False
                logger.info("Pick Mode disabled - camera interaction fully restored")

        # GUI callbacks - 仅处理GUI控件更新
        @gui_conf.on_update
        def _(_):
            update_point_cloud()

        @gui_gid.on_update
        def _(_):
            if gui_gid.value != "All":
                update_info(gui_gid.value)
            else:
                clear_info()
            update_point_cloud()

        @gui_point_size.on_update
        def _(_):
            point_cloud.point_size = gui_point_size.value

        @gui_sampling.on_update
        def _(_):
            update_point_cloud()

        if gui_frame_selector is not None:

            @gui_frame_selector.on_update
            def _(_):
                update_point_cloud()

        @gui_pick_radius.on_update
        def _(_):
            pick_sphere.radius = gui_pick_radius.value

        # Mesh generation callbacks
        @btn_generate_mesh.on_click
        def _(_):
            nonlocal mesh_handle, mesh_data
            try:
                gui_mesh_status.value = "Generating mesh..."
                logger.info("Starting mesh generation...")

                # Import mesh utils
                from utils.mesh_utils import pointcloud_to_mesh

                # Get current visible points
                thr = np.percentile(self.confidences, gui_conf.value)
                conf_mask = self.confidences >= thr
                if gui_gid.value == "All":
                    id_mask = np.ones(len(self.global_ids), dtype=bool)
                else:
                    id_mask = self.global_ids == int(gui_gid.value)
                known_mask = (
                    (self.global_ids >= 0)
                    if gui_hide_unknown.value
                    else np.ones(len(self.global_ids), dtype=bool)
                )
                mask = conf_mask & id_mask & known_mask

                points_for_mesh = self.rotated_points[mask]
                colors_for_mesh = self.colors[mask]

                if len(points_for_mesh) < 100:
                    gui_mesh_status.value = "Error: Too few points"
                    logger.warning("Too few points for meshing")
                    return

                # Generate mesh
                vertices, faces, vertex_colors = pointcloud_to_mesh(
                    points_for_mesh,
                    colors_for_mesh,
                    method=gui_mesh_method.value,
                    depth=int(gui_mesh_depth.value),
                )

                # Store mesh data
                mesh_data["vertices"] = vertices
                mesh_data["faces"] = faces
                mesh_data["colors"] = (
                    vertex_colors
                    if vertex_colors is not None
                    else np.tile([200, 200, 200], (len(vertices), 1)).astype(np.uint8)
                )

                # Remove old mesh if exists
                if mesh_handle is not None:
                    mesh_handle.remove()

                # Add mesh to scene
                mesh_handle = self.server.scene.add_mesh_simple(
                    name="sku_mesh",
                    vertices=vertices,
                    faces=faces,
                    color=(200, 200, 200),
                    wireframe=False,
                    visible=True,
                )

                gui_show_mesh.value = True
                gui_mesh_status.value = (
                    f"Mesh: {len(vertices)} vertices, {len(faces)} faces"
                )
                logger.info(
                    f"Mesh generated: {len(vertices)} vertices, {len(faces)} faces"
                )

            except ImportError as e:
                gui_mesh_status.value = "Error: Open3D not installed"
                logger.error(f"Import error: {e}")
            except Exception as e:
                gui_mesh_status.value = f"Error: {str(e)[:50]}"
                logger.error(f"Mesh generation failed: {e}")

        @gui_show_mesh.on_update
        def _(_):
            nonlocal mesh_handle
            if mesh_handle is not None:
                mesh_handle.visible = gui_show_mesh.value

        @gui_show_points.on_update
        def _(_):
            point_cloud.visible = gui_show_points.value

        # initial draw
        update_point_cloud()

        logger.info(f"Open http://localhost:{self.runtime.port} in your browser")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
