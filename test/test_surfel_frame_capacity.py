import json
import numpy as np
import pytest
from PIL import Image
from src.surfel_export import grid_tangents, prepare_surfels

@pytest.mark.parametrize('count', [33, 256])
def test_export_preserves_every_source_frame(tmp_path, count):
    y,x=np.mgrid[:3,:3]
    points=np.tile(np.stack([x*.01,y*.01,np.ones_like(x)],-1).astype(np.float32),(count,1,1,1))
    e=np.tile(np.c_[np.eye(3),np.zeros(3)],(count,1,1))
    k=np.tile(np.diag([100.,100.,1.]),(count,1,1))
    cache={'points':points,'extrinsic':e,'image_ids':np.arange(count),'source_image_sizes':np.tile([3,3],(count,1)),'affine':np.tile(np.eye(3)[:2],(count,1,1))}
    images=tmp_path/'images'; images.mkdir()
    for i in range(count): Image.new('RGB',(3,3),(i%256,0,0)).save(images/f'{i}.jpg')
    path=tmp_path/'predictions.npz';np.savez(path,intrinsic=k)
    indices=np.arange(count)*9+4
    u, v, depth = grid_tangents(points, e)
    scales = np.column_stack((np.full(count, .5), np.full(count, 3.0)))
    files=prepare_surfels(cache,indices,path,images,256,scales=scales,
        geometry=(u.reshape(-1,3)[indices],v.reshape(-1,3)[indices],depth))
    assert np.frombuffer(files['surfel-frame.u8.bin'],dtype=np.uint8).tolist()==list(range(count))
    assert len(json.loads(files['surfel.json'])['frames'])==count
    assert len(files['surfel-depth.f16.bin'])==count*3*3*2
    assert json.loads(files['surfel.json'])['version'] == 3
    assert np.array_equal(np.frombuffer(files['surfel-scale.f16.bin'], dtype='<f2').reshape(count, 2), scales)
    assert np.allclose(np.frombuffer(files['surfel-u.f16.bin'], dtype='<f2').reshape(count, 3), u.reshape(-1,3)[indices], rtol=.001)


def test_over_byte_capacity_fails_before_encoding(tmp_path):
    count=257
    path=tmp_path/'predictions.npz';np.savez(path,intrinsic=np.tile(np.eye(3),(count,1,1)))
    with pytest.raises(ValueError,match='256'):
        prepare_surfels({'points':np.ones((count,2,2,3))},np.array([0]),path,tmp_path,256,scales=None,geometry=None)
