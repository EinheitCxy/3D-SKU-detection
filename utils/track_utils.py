"""RAFT optical-flow point-tracking utilities for SKU matching.

Warps 2D query points (sampled from SAM3 masks in the ref image) to target
frames via RAFT optical flow, **bypassing da3 depth/3D projection**. This is the
correspondence source for `--algorithm point_track`, attacking the "projection
miss" recall wall.

Why RAFT (not CoTracker3): the floor_display images are wide-baseline photos
with large inter-frame parallax motion (RAFT-measured ~100px / 26% of width on
fd2). CoTracker3 (a video point tracker with a small-motion assumption) returns
near-static tracks with falsely-high visibility on this motion scale. RAFT
optical flow correctly captures the large per-pixel motion. RAFT is pairwise
(ref->target) which fits `pairing_3d=next` (adjacent pairs only) perfectly.

Occlusion handling: forward-backward consistency. Warp ref point forward to
target (via fwd flow), then back (via bwd flow); if the round-trip lands within
`fb_tol` px of the original, the point is reliable (visible, well-tracked).
Unreliable points (occluded / flow divergence) are dropped via the visibility
mask, preventing the C4c-style FP flood.

Coordinates stay in the `images` tensor (final/model) space. RAFT requires
H,W divisible by 8; the video width is right-padded to a multiple of 8 (content
coords unaffected - flow indexed only at valid content x).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

_RAFT_MODEL = None
# Flow cache keyed by (video data_ptr, src, dst) -> (flow_fwd (2,H,W), flow_bwd (2,H,W))
# video is cached module-level in sku_matching_system (_DA3_IMAGE_CACHE) so data_ptr
# is stable across refs within one batch_all_refs run -> flows computed once per pair.
_FLOW_CACHE: Dict[Tuple[int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def _pad_to_8(video: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Right- and bottom-pad H,W to multiples of 8 (RAFT requirement).

    Returns (padded, pad_h, pad_w). Content coords [0,H) x [0,W) are unaffected
    by the padding (flow indexed only at valid content coords).
    """
    H, W = video.shape[-2], video.shape[-1]
    pad_h = (8 - H % 8) % 8
    pad_w = (8 - W % 8) % 8
    if pad_h == 0 and pad_w == 0:
        return video, 0, 0
    # F.pad last two dims (W, H) order: (left, right, top, bottom)
    return torch.nn.functional.pad(video, (0, pad_w, 0, pad_h)), pad_h, pad_w


def get_raft(device: str):
    """Load (and cache) the RAFT-large optical flow model."""
    global _RAFT_MODEL
    if _RAFT_MODEL is None:
        from torchvision.models.optical_flow import raft_large

        _RAFT_MODEL = raft_large(weights="DEFAULT").to(device).eval()
        logger.info("Loaded RAFT-large optical flow model")
    return _RAFT_MODEL


@torch.no_grad()
def _compute_flow_pair(
    video: torch.Tensor, src: int, dst: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute forward (src->dst) and backward (dst->src) RAFT flow.

    Returns (flow_fwd, flow_bwd), each (2, H, W) with [0]=dx, [1]=dy, on device,
    in the UNPADDED content space (width trimmed back to original).
    """
    key = (video.data_ptr(), src, dst)
    cached = _FLOW_CACHE.get(key)
    if cached is not None:
        return cached

    raft = get_raft(device)
    H, W = video.shape[-2], video.shape[-1]
    v0 = video[src]
    v1 = video[dst]
    # pad H,W to multiples of 8 (RAFT requirement)
    v0p, pad_h, pad_w = _pad_to_8(v0)
    v1p, _, _ = _pad_to_8(v1)
    flow_fwd = raft(v0p.unsqueeze(0), v1p.unsqueeze(0))[0].squeeze(0)  # (2, H_pad, W_pad)
    flow_bwd = raft(v1p.unsqueeze(0), v0p.unsqueeze(0))[0].squeeze(0)
    # trim padding back to content dims (flow indexed at original coords thereafter)
    if pad_h > 0 or pad_w > 0:
        flow_fwd = flow_fwd[..., :H, :W]
        flow_bwd = flow_bwd[..., :H, :W]
    _FLOW_CACHE[key] = (flow_fwd, flow_bwd)
    return flow_fwd, flow_bwd


def track_query_points(
    video: torch.Tensor,
    queries_xy: np.ndarray,
    query_frame: int,
    checkpoint_path: str,  # unused (RAFT); kept for config API compatibility
    device: str,
    batch_size: int = 256,  # unused; kept for config API compatibility
    fb_tol: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Warp 2D query points from `query_frame` to ALL frames via RAFT flow.

    Args:
        video: images tensor (S, C, H, W) float [0,1], final/model space.
        queries_xy: (N, 2) [x, y] in final space.
        query_frame: the ref frame index.
        device: cuda device.
        fb_tol: forward-backward round-trip tolerance (px) for reliability.

    Returns:
        tracks: (S, N, 2) warped [x, y] in final space.
        vis: (S, N) float reliability in {0, 1} (fb-consistent & in-bounds).
    """
    S = video.shape[0]
    H, W = video.shape[-2], video.shape[-1]
    N = queries_xy.shape[0]
    device = torch.device(device)

    tracks = torch.zeros((S, N, 2), dtype=torch.float32, device=device)
    vis = torch.zeros((S, N), dtype=torch.float32, device=device)

    qx = torch.from_numpy(queries_xy[:, 0].astype(np.float32)).to(device)
    qy = torch.from_numpy(queries_xy[:, 1].astype(np.float32)).to(device)

    for t in range(S):
        if t == query_frame:
            tracks[t, :, 0] = qx
            tracks[t, :, 1] = qy
            vis[t, :] = 1.0
            continue

        flow_fwd, flow_bwd = _compute_flow_pair(video, query_frame, t, str(device))
        # index flow at query points (clamp to content bounds)
        xi = qx.round().long().clamp(0, W - 1)
        yi = qy.round().long().clamp(0, H - 1)
        fx = flow_fwd[0, yi, xi]  # dx
        fy = flow_fwd[1, yi, xi]  # dy
        tx = qx + fx
        ty = qy + fy

        # forward-backward consistency with RELATIVE tolerance: RAFT flow has
        # ~10-20px error on 100px+ motion, so absolute tol=2 rejects everything.
        # Scale tol with motion magnitude: fb_tol = max(fb_tol, 0.15*|flow|).
        flow_mag = torch.sqrt(fx * fx + fy * fy)
        tol = torch.clamp(0.15 * flow_mag, min=fb_tol)
        # warp target pos back via bwd flow
        txi = tx.round().long().clamp(0, W - 1)
        tyi = ty.round().long().clamp(0, H - 1)
        bx = tx + flow_bwd[0, tyi, txi]
        by = ty + flow_bwd[1, tyi, txi]

        reliable = (
            ((bx - qx).abs() < tol)
            & ((by - qy).abs() < tol)
            & (tx >= 0) & (tx < W)
            & (ty >= 0) & (ty < H)
        )

        tracks[t, :, 0] = tx
        tracks[t, :, 1] = ty
        vis[t, :] = reliable.float()

    return tracks, vis


def clear_flow_cache() -> None:
    """Clear the flow cache (call when video changes / memory tight)."""
    _FLOW_CACHE.clear()
