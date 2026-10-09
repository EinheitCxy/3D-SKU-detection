"""The coverage metric must use camera Z and count a nearer gray occluder."""
import numpy as np
import open3d as o3d

from perf.surfel_coverage.evaluate import make_rays, pixel_metrics


def test_rotated_translated_camera_z_and_gray_first_hit():
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = [1., -2., 3.]
    intrinsic = np.array([[2., 0., 0.], [0., 2., 0.], [0., 0., 1.]])
    # A front gray square at camera Z=2 and a textured one at Z=3.
    square = np.array([[-2., -2., 0.], [2., -2., 0.], [2., 2., 0.], [-2., 2., 0.]])
    camera_vertices = np.concatenate((square + [0., 0., 2.], square + [0., 0., 3.]))
    world = (camera_vertices - extrinsic[:3, 3]) @ rotation
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.uint32)
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(world.astype(np.float32)), o3d.core.Tensor(faces))
    result = scene.cast_rays(o3d.core.Tensor(make_rays(intrinsic, extrinsic, (1, 2))), nthreads=1)
    z = result['t_hit'].numpy()
    np.testing.assert_allclose(z, 2., atol=1e-6)
    sources = np.array([-1, -1, 0, 0])
    textured = sources[result['primitive_ids'].numpy()] >= 0
    counts = pixel_metrics(z, np.isfinite(z), textured,
                          np.array([[2., 3.]]), np.ones((1, 2), dtype=bool), .01)
    assert counts['matched'] == 1
    assert counts['textured_matched'] == 0
    assert counts['inconsistent_nearer'] == 1
