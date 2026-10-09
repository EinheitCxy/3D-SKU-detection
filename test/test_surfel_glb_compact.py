"""Quantization preserves triangle geometry and embedded portable assets."""
import json
import struct

import numpy as np
import pytest

from src.surfel_glb_compact import _chunks, _read, compact_surfel_glb
from src.surfel_glb import export_surfel_glb
from test.test_surfel_glb import bundle, fake_baker


def decode(doc, binary, index):
    acc = doc['accessors'][index]
    view = doc['bufferViews'][acc['bufferView']]
    width = {'VEC3': 3, 'VEC2': 2, 'SCALAR': 1}[acc['type']]
    return np.ndarray((acc['count'], width), '<u2', buffer=binary,
                      offset=view['byteOffset'], strides=(view.get('byteStride', width * 2), 2))


def test_quantized_reconstruction_and_assets(bundle, fake_baker, tmp_path):
    source, output = tmp_path / 'source.glb', tmp_path / 'compact.glb'
    export_surfel_glb(bundle, source)
    before, original, _ = _read(source)
    report = compact_surfel_glb(source, output)
    doc, binary, _ = _read(output)
    assert doc['extensionsRequired'] == ['KHR_mesh_quantization']
    assert doc['nodes'][0]['matrix'] == before['nodes'][0]['matrix']
    assert doc['materials'] == before['materials']
    assert report['geometry_bytes_after'] < report['geometry_bytes_before']
    for node in doc['nodes'][1:]:
        primitive = doc['meshes'][node['mesh']]['primitives'][0]
        pos = decode(doc, binary, primitive['attributes']['POSITION'])
        reconstructed = pos / 65535 * node['scale'] + node['translation']
        np.testing.assert_allclose(reconstructed[:, 2], 1)
        indices = decode(doc, binary, primitive['indices'])
        assert indices.max() < len(pos)
        uv = decode(doc, binary, primitive['attributes']['TEXCOORD_0']) / 65535
        assert np.all((uv >= 0) & (uv <= 1))
    for new, old in zip(doc['images'], before['images']):
        a, b = doc['bufferViews'][new['bufferView']], before['bufferViews'][old['bufferView']]
        assert binary[a['byteOffset']:a['byteOffset'] + a['byteLength']] == original[b['byteOffset']:b['byteOffset'] + b['byteLength']]
    assert doc['buffers'] == [{'byteLength': len(binary)}]
    assert all(v['byteOffset'] % 4 == 0 and v['byteOffset'] + v['byteLength'] <= len(binary)
               for v in doc['bufferViews'])
    with pytest.raises(FileExistsError):
        compact_surfel_glb(source, source)
    with pytest.raises(ValueError, match='PNG'):
        compact_surfel_glb(source, tmp_path / 'bad.glb', image_transform=lambda _: b'bad')
    assert not (tmp_path / 'bad.glb').exists()


def test_chunk_remapping_boundaries():
    # General triangle soup exceeds 65536 vertices; preserve exact triangle ordering.
    triangles = np.arange(120000, dtype=np.uint32).reshape(-1, 3)
    chunks = list(_chunks(triangles))
    assert len(chunks) > 1
    assert all(len(vertices) <= 65535 and indices.max() < min(len(vertices), 65535)
               for vertices, indices in chunks)
    restored = np.concatenate([vertices[indices].reshape(-1, 3) for vertices, indices in chunks])
    np.testing.assert_array_equal(restored, triangles)


def test_nonzero_quantization_error_and_reject_animation(bundle, fake_baker, tmp_path):
    source = tmp_path / 'source.glb'
    export_surfel_glb(bundle, source)
    doc, binary, _ = _read(source)
    data = bytearray(binary)
    positions = np.array([[0, 0, 0], [1, 2, 3], [.1234567, .456789, 1.234567], [1, 0, 0]], '<f4')
    p = doc['accessors'][doc['meshes'][0]['primitives'][0]['attributes']['POSITION']]
    v = doc['bufferViews'][p['bufferView']]
    data[v['byteOffset']:v['byteOffset'] + v['byteLength']] = positions.tobytes()

    def write_document(path):
        encoded = json.dumps(doc).encode()
        encoded += b' ' * (-len(encoded) % 4)
        path.write_bytes(struct.pack('<4sII', b'glTF', 2, 28 + len(encoded) + len(data))
                         + struct.pack('<I4s', len(encoded), b'JSON') + encoded
                         + struct.pack('<I4s', len(data), b'BIN\0') + data)

    changed = tmp_path / 'changed.glb'
    write_document(changed)
    output = tmp_path / 'compact.glb'
    report = compact_surfel_glb(changed, output)
    out, payload, _ = _read(output)
    node = out['nodes'][1]
    acc = out['meshes'][0]['primitives'][0]['attributes']['POSITION']
    restored = decode(out, payload, acc) / 65535 * node['scale'] + node['translation']
    np.testing.assert_allclose(restored, positions, atol=3 / 65535 / 2)
    assert 0 < report['max_position_error'] <= 3 / 65535 / 2
    doc['animations'] = []
    write_document(changed)
    with pytest.raises(ValueError, match='document'):
        compact_surfel_glb(changed, tmp_path / 'invalid.glb')
