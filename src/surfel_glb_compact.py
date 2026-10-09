"""Compact a portable Surfel GLB using standard KHR_mesh_quantization."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import struct

import numpy as np


def _read(path):
    raw = Path(path).read_bytes()
    if len(raw) < 28 or struct.unpack_from('<4sII', raw) != (b'glTF', 2, len(raw)):
        raise ValueError('Expected GLB 2.0')
    size, kind = struct.unpack_from('<I4s', raw, 12)
    if kind != b'JSON' or 28 + size > len(raw):
        raise ValueError('Invalid GLB JSON chunk')
    doc = json.loads(raw[20:20 + size])
    length, kind = struct.unpack_from('<I4s', raw, 20 + size)
    if kind != b'BIN\0' or 28 + size + length != len(raw):
        raise ValueError('Expected one embedded BIN chunk')
    return doc, memoryview(raw)[28 + size:], len(raw)


def _validate(doc, binary):
    allowed = {'asset', 'extensionsUsed', 'scene', 'scenes', 'nodes', 'meshes',
               'bufferViews', 'accessors', 'images', 'textures', 'materials', 'samplers', 'buffers'}
    if set(doc) - allowed or doc.get('extensionsUsed') != ['KHR_materials_unlit']:
        raise ValueError('Expected original portable Surfel exporter document')
    if (len(doc['nodes']) != 1 or len(doc['meshes']) != 1
            or set(doc['nodes'][0]) != {'mesh', 'matrix'} or doc['nodes'][0]['mesh'] != 0
            or doc['scenes'] != [{'nodes': [0]}] or doc['scene'] != 0
            or len(doc['buffers']) != 1 or 'uri' in doc['buffers'][0]
            or doc['buffers'][0]['byteLength'] > len(binary)):
        raise ValueError('Unsupported Surfel scene structure')
    if set(doc['meshes'][0]) != {'primitives'}:
        raise ValueError('Unsupported mesh properties')
    for view in doc['bufferViews']:
        if (set(view) - {'buffer', 'byteOffset', 'byteLength', 'target'}
                or view['buffer'] != 0 or view.get('byteOffset', 0) < 0
                or view['byteLength'] < 0
                or view.get('byteOffset', 0) + view['byteLength'] > len(binary)):
            raise ValueError('Unsupported or invalid buffer view')


def _values(doc, binary, index, component, kind, width):
    acc = doc['accessors'][index]
    if (set(acc) - {'bufferView', 'componentType', 'count', 'type', 'min', 'max'}
            or acc['componentType'] != component or acc['type'] != kind or acc['count'] < 1):
        raise ValueError('Unsupported source accessor')
    view = doc['bufferViews'][acc['bufferView']]
    if view['byteLength'] != acc['count'] * width * 4:
        raise ValueError('Unexpected source accessor byte length')
    result = np.frombuffer(binary, '<f4' if component == 5126 else '<u4',
                           count=acc['count'] * width, offset=view.get('byteOffset', 0))
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite geometry')
    return result.reshape(acc['count'], width)


def _chunks(triangles):
    # Typical Surfel quads fit exactly: 32766 triangles -> 65532 vertices.
    for start in range(0, len(triangles), 32766):
        pending = [triangles[start:start + 32766]]
        while pending:
            part = pending.pop()
            vertices, inverse = np.unique(part, return_inverse=True)
            if len(vertices) > 65535:
                middle = len(part) // 2
                pending.extend([part[middle:], part[:middle]])
            else:
                yield vertices, inverse.astype('<u2').reshape(-1)


def compact_surfel_glb(source, output, *, image_transform=None):
    """Quantize geometry, optionally transform embedded PNGs, and create output exclusively."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    doc, binary, source_bytes = _read(source)
    _validate(doc, binary)
    result = copy.deepcopy(doc)
    result.update(bufferViews=[], accessors=[], meshes=[], images=[])
    result['nodes'][0].pop('mesh')
    result['nodes'][0]['children'] = []
    result['extensionsUsed'].append('KHR_mesh_quantization')
    result['extensionsRequired'] = ['KHR_mesh_quantization']
    data = bytearray()
    report = dict(source=str(source), output=str(output), source_bytes=source_bytes,
                  geometry_bytes_before=0, geometry_bytes_after=0,
                  texture_bytes_before=0, texture_bytes_after=0,
                  max_position_error=0., max_uv_error=0., triangle_count=0)

    def view(raw, target=None, stride=None):
        data.extend(b'\0' * (-len(data) % 4))
        record = dict(buffer=0, byteOffset=len(data), byteLength=len(raw))
        data.extend(raw)
        if target:
            record['target'] = target
        if stride:
            record['byteStride'] = stride
        result['bufferViews'].append(record)
        return len(result['bufferViews']) - 1

    def accessor(values, kind, target, normalized=False, bounds=False):
        record = dict(bufferView=view(values.tobytes(), target, 8 if kind == 'VEC3' else None),
                      componentType=5123, count=len(values), type=kind)
        if normalized:
            record['normalized'] = True
        if bounds:
            record.update(min=values[:, :3].min(0).tolist(), max=values[:, :3].max(0).tolist())
        result['accessors'].append(record)
        report['geometry_bytes_after'] += values.nbytes
        return len(result['accessors']) - 1

    for primitive in doc['meshes'][0]['primitives']:
        if (set(primitive) != {'attributes', 'indices', 'material', 'mode'}
                or primitive['mode'] != 4 or set(primitive['attributes']) != {'POSITION', 'TEXCOORD_0'}):
            raise ValueError('Expected indexed POSITION/TEXCOORD_0 triangles')
        positions = _values(doc, binary, primitive['attributes']['POSITION'], 5126, 'VEC3', 3)
        uv = _values(doc, binary, primitive['attributes']['TEXCOORD_0'], 5126, 'VEC2', 2)
        indices = _values(doc, binary, primitive['indices'], 5125, 'SCALAR', 1).ravel()
        if len(indices) % 3 or indices.max() >= len(positions) or len(uv) != len(positions):
            raise ValueError('Invalid triangle indices or attribute counts')
        if np.any((uv < 0) | (uv > 1)):
            raise ValueError('UV coordinates must be within [0, 1]')
        report['geometry_bytes_before'] += positions.nbytes + uv.nbytes + indices.nbytes
        report['triangle_count'] += len(indices) // 3
        for vertices, local_indices in _chunks(indices.reshape(-1, 3)):
            p = positions[vertices].astype(np.float64)
            low = p.min(0)
            scale = p.max(0) - low
            scale[scale == 0] = 1
            quantized = np.zeros((len(p), 4), dtype='<u2')
            quantized[:, :3] = np.rint((p - low) / scale * 65535).astype('<u2')
            tex = np.rint(uv[vertices].astype(np.float64) * 65535).astype('<u2')
            report['max_position_error'] = max(report['max_position_error'],
                float(np.abs(quantized[:, :3] / 65535 * scale + low - p).max()))
            report['max_uv_error'] = max(report['max_uv_error'],
                float(np.abs(tex / 65535 - uv[vertices]).max()))
            attrs = dict(POSITION=accessor(quantized, 'VEC3', 34962, True, True),
                         TEXCOORD_0=accessor(tex, 'VEC2', 34962, True))
            primitive_out = dict(attributes=attrs, indices=accessor(local_indices, 'SCALAR', 34963),
                                 material=primitive['material'], mode=4)
            result['nodes'][0]['children'].append(len(result['nodes']))
            result['nodes'].append(dict(mesh=len(result['meshes']), translation=low.tolist(), scale=scale.tolist()))
            result['meshes'].append(dict(primitives=[primitive_out]))
    for image in doc['images']:
        if set(image) != {'bufferView', 'mimeType'} or image['mimeType'] != 'image/png':
            raise ValueError('Expected embedded PNG images')
        v = doc['bufferViews'][image['bufferView']]
        raw = bytes(binary[v.get('byteOffset', 0):v.get('byteOffset', 0) + v['byteLength']])
        compact = image_transform(raw) if image_transform else raw
        if not compact.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('Image transform must return PNG bytes')
        report['texture_bytes_before'] += len(raw)
        report['texture_bytes_after'] += len(compact)
        result['images'].append(dict(bufferView=view(compact), mimeType='image/png'))
    data.extend(b'\0' * (-len(data) % 4))
    result['buffers'] = [dict(byteLength=len(data))]
    encoded = json.dumps(result, separators=(',', ':'), allow_nan=False).encode()
    encoded += b' ' * (-len(encoded) % 4)
    length = 28 + len(encoded) + len(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open('xb')
    try:
        with stream:
            stream.write(struct.pack('<4sII', b'glTF', 2, length))
            stream.write(struct.pack('<I4s', len(encoded), b'JSON'))
            stream.write(encoded)
            stream.write(struct.pack('<I4s', len(data), b'BIN\0'))
            stream.write(data)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    report.update(glb_bytes=length, primitive_count=len(result['meshes']))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--keep-textures', action='store_true')
    args = parser.parse_args(argv)
    if args.report is not None and args.report.exists():
        raise FileExistsError(args.report)
    transform = None
    if not args.keep_textures:
        from src.surfel_glb_texture import compact_png
        transform = compact_png
    report = compact_surfel_glb(args.source, args.output, image_transform=transform)
    text = json.dumps(report, indent=2) + '\n'
    if args.report:
        with args.report.open('x') as stream:
            stream.write(text)
    print(text, end='')


if __name__ == '__main__':
    main()
