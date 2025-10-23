from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .types import ViewerArtifacts, ViewerRuntimeConfig
from utils.kdtree_utils import build_kdtree

logger = logging.getLogger(__name__)


class ViserViewer:
    def __init__(self, artifacts: ViewerArtifacts, runtime: ViewerRuntimeConfig, image_dir: Path, global_mapping: Path):
        self.artifacts = artifacts
        self.runtime = runtime
        self.image_dir = image_dir
        self.global_mapping_path = global_mapping
        self.server = None
        self.kdtree = None

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
            self.colors = np.clip(self.colors * (255.0 if self.colors.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)

    def _load_camera_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]]:
        try:
            dataset_root = self.artifacts.pcd_cache_path.parent.parent
            pred = dataset_root / "vggt_cache" / "predictions.npz"
            if not pred.exists():
                return None
            data = np.load(pred)
            extr = data["extrinsic"]
            intr = data["intrinsic"]
            images = data["images"]
            if images.ndim == 4 and images.shape[1] == 3:
                images = images.transpose(0, 2, 3, 1)
            if images.dtype != np.uint8:
                images = np.clip(images * (255.0 if images.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            if "image_ids" in data:
                image_ids = data["image_ids"].tolist()
            else:
                image_ids = list(range(extr.shape[0]))
            from vggt.utils.geometry import closed_form_inverse_se3

            cam2world = closed_form_inverse_se3(extr)[:, :3, :]
            cam2world[..., -1] -= self.scene_center
            return cam2world, intr, images, image_ids
        except Exception:
            return None

    def start(self) -> None:
        global viser
        if "viser" not in globals() or globals().get("viser") is None:
            import viser as _viser

            viser = _viser

        self._load_cache()

        logger.info(f"Starting Viser server on port {self.runtime.port}")
        self.server = viser.ViserServer(host="0.0.0.0", port=self.runtime.port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        points_centered = self.points_centered
        span = (np.max(points_centered, axis=0) - np.min(points_centered, axis=0)).max()
        dist = span * 2.0 if span > 0 else 1.0
        initial_camera_position = (float(dist * 0.7), float(dist * 0.5), float(dist * 0.7))

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            try:
                client.camera.position = initial_camera_position
            except Exception:
                pass

        # Build KDTree on rotated points (initial no-rotation)
        self.R_cur = np.eye(3, dtype=float)
        self.rotated_points = points_centered
        self.kdtree = build_kdtree(self.rotated_points)

        # Controls
        gui_conf = self.server.gui.add_slider(
            "Confidence %", min=0, max=100, step=0.1, initial_value=float(self.runtime.default_conf_percentile)
        )
        gui_point_size = self.server.gui.add_slider(
            "Point Size", min=0.0001, max=0.005, step=0.0001, initial_value=float(self.runtime.default_point_size)
        )
        unique_gids = np.unique(self.global_ids[self.global_ids >= 0])
        gui_gid = self.server.gui.add_dropdown(
            "Show Global ID", options=["All"] + [str(x) for x in sorted(unique_gids)], initial_value="All"
        )

        with self.server.gui.add_folder("📋 Selected ID Info", expand_by_default=True):
            gui_info_gid = self.server.gui.add_text("Global ID", initial_value="(None)", disabled=True)
            gui_info_images = self.server.gui.add_text("Images", initial_value="-", disabled=True)
            gui_info_objects = self.server.gui.add_text("Objects ID", initial_value="-", disabled=True)
            gui_info_status = self.server.gui.add_text("Status (Active/Filtered)", initial_value="-", disabled=True)
            self.server.gui.add_markdown("*Tip: Enable Pick Mode and click on point cloud, or select from dropdown above.*")

        with self.server.gui.add_folder("🔍 Point Picking"):
            gui_pick_mode = self.server.gui.add_checkbox("Enable Pick Mode", initial_value=False)
            # radius proportional to span
            rmin, rmax, rinit = span * 0.0002, span * 0.02, span * 0.0005
            gui_pick_radius = self.server.gui.add_slider(
                "Pick Radius", min=float(max(rmin, 1e-9)), max=float(max(rmax, 1e-6)), step=float(max((rmax - rmin) / 100.0, 1e-6)), initial_value=float(max(rinit, 1e-9))
            )
            gui_show_pick = self.server.gui.add_checkbox("Show Pick Sphere", initial_value=bool(self.runtime.show_pick_sphere_default))

        with self.server.gui.add_folder("⚙️ Advanced Options", expand_by_default=False):
            gui_sampling = self.server.gui.add_slider("Display Sample %", min=10, max=100, step=10, initial_value=100)
            # gui_rotate disabled - Viser version doesn't support drag events
            # gui_rotate = self.server.gui.add_checkbox("Rotate Model (Shift+Drag)", initial_value=bool(self.runtime.rotate_model_default))
            gui_hide_unknown = self.server.gui.add_checkbox("Hide Unknown IDs (-1)", initial_value=bool(self.runtime.hide_unknown_default))

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
                    f"camera_{idx}", wxyz=T.rotation().wxyz, position=T.translation(), axes_length=0.1, axes_radius=0.005, origin_radius=0.005
                )
                frames.append(frame_axis)
                img = images[idx]
                h, w = img.shape[:2]
                fy = intr[idx][1, 1]
                fov = 2 * np.arctan2(h / 2, fy)
                fr = self.server.scene.add_camera_frustum(
                    f"camera_{idx}/frustum", fov=float(fov), aspect=w / h, scale=0.1, image=img, line_width=1.0
                )
                frustums.append(fr)
                @fr.on_click
                def _(_, frame=frame_axis):
                    for client in self.server.get_clients().values():
                        client.camera.wxyz = frame.wxyz
                        client.camera.position = frame.position

            gui_show_cameras = self.server.gui.add_checkbox("Show Cameras", initial_value=bool(self.runtime.show_cameras_default))
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
            gui_frame_selector = self.server.gui.add_dropdown("Show Points from Frame", options=["All"] + [str(x) for x in fids], initial_value="All")

        # Point cloud and pick sphere
        point_cloud = self.server.scene.add_point_cloud(
            name="sku_pcd", points=self.rotated_points[:1], colors=self.colors[:1], point_size=gui_point_size.value, point_shape="circle"
        )
        pick_sphere = self.server.scene.add_icosphere(
            name="pick_sphere", radius=gui_pick_radius.value, color=(255, 255, 0), position=(0.0, 0.0, 0.0), visible=False
        )

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
            gui_info_objects.value = ", ".join(map(str, info["objects"][:10])) + ("..." if len(info["objects"]) > 10 else "")
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
            known_mask = (self.global_ids >= 0) if gui_hide_unknown.value else np.ones(len(self.global_ids), dtype=bool)
            sample_mask = display_rand <= (gui_sampling.value / 100.0)
            mask = conf_mask & id_mask & frame_mask & known_mask & sample_mask
            point_cloud.points = self.rotated_points[mask]
            point_cloud.colors = self.colors[mask]

        def handle_pick(click_pos: np.ndarray):
            if not gui_pick_mode.value:
                return
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

        @self.server.scene.on_pointer_event(event_type="click")
        def _on_click(evt: "viser.ScenePointerEvent") -> None:
            try:
                btn = str(evt.button).lower() if hasattr(evt, "button") and evt.button is not None else "left"
                if "right" in btn or "middle" in btn:
                    return
            except Exception:
                pass
            if not hasattr(evt, "ray_direction") or evt.ray_direction is None:
                return
            ray = np.asarray(evt.ray_direction)
            o = np.asarray(evt.ray_origin)
            ray = ray / (np.linalg.norm(ray) + 1e-9)
            depth_est = np.linalg.norm(o)
            best_idx = None
            best_d = float("inf")
            for t in np.linspace(0, depth_est * 2, 20):
                p = o + ray * t
                d, idx = self.kdtree.query(p, k=1)
                if d < best_d and d < gui_pick_radius.value:
                    best_d = d
                    best_idx = int(idx)
            if best_idx is not None:
                handle_pick(self.rotated_points[best_idx])
            else:
                clear_info()

        # rotation (Shift+Drag)
        arcball_active = {"v": False}
        prev = {"p": None}

        def _shift(evt) -> bool:
            try:
                if hasattr(evt, "shift") and bool(evt.shift):
                    return True
                if hasattr(evt, "shift_key") and bool(evt.shift_key):
                    return True
                if hasattr(evt, "modifiers") and evt.modifiers is not None:
                    mods = str(evt.modifiers).lower()
                    return ("shift" in mods) or ("mod.shift" in mods)
            except Exception:
                return False
            return False

        def _ray_sphere_intersect(o: np.ndarray, d: np.ndarray, R: float):
            b = 2.0 * float(np.dot(o, d))
            c = float(np.dot(o, o) - R * R)
            disc = b * b - 4.0 * c
            if disc < 0:
                return None
            t1 = (-b - float(np.sqrt(disc))) / 2.0
            t2 = (-b + float(np.sqrt(disc))) / 2.0
            t = t1 if t1 > 0 else (t2 if t2 > 0 else None)
            if t is None:
                return None
            return o + t * d

        def _rot_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            a_n = a / (np.linalg.norm(a) + 1e-9)
            b_n = b / (np.linalg.norm(b) + 1e-9)
            v = np.cross(a_n, b_n)
            s = np.linalg.norm(v)
            c = float(np.dot(a_n, b_n))
            if s < 1e-9:
                return np.eye(3)
            vx = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]], dtype=float)
            return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))

        # NOTE: Arcball rotation disabled - Viser version only supports 'click' and 'rect-select' events
        # The following event types are not supported: 'down', 'move', 'up'
        # Use Viser's built-in camera controls (drag to rotate) instead

        # @self.server.scene.on_pointer_event(event_type="down")
        # def _on_down(evt: "viser.ScenePointerEvent") -> None:
        #     # respect right/middle buttons for pan
        #     try:
        #         btn = str(evt.button).lower()
        #         if "right" in btn or "middle" in btn:
        #             return
        #     except Exception:
        #         pass
        #     if not (gui_rotate.value and _shift(evt)):
        #         return
        #     o = np.asarray(evt.ray_origin) - self.scene_center
        #     d = np.asarray(evt.ray_direction)
        #     d = d / (np.linalg.norm(d) + 1e-9)
        #     R_sphere = float(np.linalg.norm(o)) * 0.6
        #     p = _ray_sphere_intersect(o, d, R_sphere)
        #     if p is None:
        #         return
        #     arcball_active["v"] = True
        #     prev["p"] = p

        # @self.server.scene.on_pointer_event(event_type="move")
        # def _on_move(evt: "viser.ScenePointerEvent") -> None:
        #     try:
        #         btn = str(evt.button).lower()
        #         if "right" in btn or "middle" in btn:
        #             return
        #     except Exception:
        #         pass
        #     if not arcball_active["v"] or not (gui_rotate.value and _shift(evt)):
        #         return
        #     o = np.asarray(evt.ray_origin) - self.scene_center
        #     d = np.asarray(evt.ray_direction)
        #     d = d / (np.linalg.norm(d) + 1e-9)
        #     R_sphere = float(np.linalg.norm(o)) * 0.6
        #     p = _ray_sphere_intersect(o, d, R_sphere)
        #     if p is None or prev["p"] is None:
        #         return
        #     R_delta = _rot_from_to(prev["p"], p)
        #     self.R_cur = R_delta @ self.R_cur
        #     self.rotated_points = (self.R_cur @ self.points_centered.T).T
        #     self.kdtree = build_kdtree(self.rotated_points)
        #     update_point_cloud()
        #     prev["p"] = p

        # @self.server.scene.on_pointer_event(event_type="up")
        # def _on_up(_: "viser.ScenePointerEvent") -> None:
        #     arcball_active["v"] = False
        #     prev["p"] = None

        # GUI callbacks
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

        # initial draw
        update_point_cloud()

        logger.info(f"Open http://localhost:{self.runtime.port} in your browser")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Server stopped by user")

