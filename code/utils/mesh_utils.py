"""Mesh generation utilities for point cloud surface reconstruction."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def pointcloud_to_mesh(points, colors=None, method="poisson", depth=9):
    """
    Convert point cloud to triangle mesh.

    Args:
        points: (N, 3) point coordinates
        colors: (N, 3) point colors, uint8
        method: "poisson" or "ball_pivoting"
        depth: Poisson octree depth (6-12)

    Returns:
        vertices: (M, 3) mesh vertices
        faces: (F, 3) mesh faces
        vertex_colors: (M, 3) vertex colors or None
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D required: pip install open3d")

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if colors is not None:
        colors_norm = colors.astype(np.float64) / 255.0 if colors.dtype == np.uint8 else colors
        pcd.colors = o3d.utility.Vector3dVector(colors_norm)

    # Estimate normals
    bbox = pcd.get_axis_aligned_bounding_box()
    radius = np.linalg.norm(bbox.get_extent()) * 0.01
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # Surface reconstruction
    if method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        # Remove low-density vertices (3-5% threshold for robustness)
        densities = np.asarray(densities)
        threshold = np.quantile(densities, 0.05)  # 调整为5%，更稳健
        mesh.remove_vertices_by_mask(densities < threshold)
    elif method == "ball_pivoting":
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist * r for r in [1.0, 2.0, 4.0]]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract data
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)

    # Transfer colors from point cloud to mesh vertices using KDTree
    vertex_colors = None
    if colors is not None:
        from scipy.spatial import KDTree

        # Build KDTree on original point cloud
        tree = KDTree(points)

        # Find nearest point for each mesh vertex
        distances, indices = tree.query(vertices, k=1)

        # Transfer colors (use original colors, not normalized)
        vertex_colors = colors[indices]

        logger.info(f"Mesh: {len(vertices)} vertices, {len(faces)} faces (with colors)")
    else:
        logger.info(f"Mesh: {len(vertices)} vertices, {len(faces)} faces (no colors)")

    return vertices, faces, vertex_colors


def save_mesh(vertices, faces, output_path, vertex_colors=None):
    """Save mesh to file (PLY/OBJ/STL)."""
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Open3D required: pip install open3d")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))

    if vertex_colors is not None:
        colors_norm = vertex_colors.astype(np.float64) / 255.0 if vertex_colors.dtype == np.uint8 else vertex_colors
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors_norm)

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    logger.info(f"Mesh saved: {output_path}")
