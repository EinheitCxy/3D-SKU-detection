"""Compact a portable Surfel GLB using standard KHR_mesh_quantization."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import struct
import tempfile

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


def _write_glb(document, binary, output):
    """Write one completed BIN stream; remove only our output after a write failure."""
    binary.write(b'\0' * (-binary.tell() % 4))
    binary_length = binary.tell()
    document['buffers'] = [dict(byteLength=binary_length)]
    encoded = json.dumps(document, separators=(',', ':'), allow_nan=False).encode()
    encoded += b' ' * (-len(encoded) % 4)
    length = 28 + len(encoded) + binary_length
    if length > 0xFFFFFFFF:
        raise ValueError('GLB exceeds the 4 GiB format size limit; export an explicit global-id subset')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = output.open('xb')
    try:
        with stream:
            stream.write(struct.pack('<4sII', b'glTF', 2, length))
            stream.write(struct.pack('<I4s', len(encoded), b'JSON'))
            stream.write(encoded)
            stream.write(struct.pack('<I4s', binary_length, b'BIN\0'))
            binary.seek(0)
            shutil.copyfileobj(binary, stream)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return length


class _CompactWriter:
    """Append quantized primitives and PNG pages to one bounded on-disk stream."""

    def __init__(self, document, binary):
        self.document = document
        self.binary = binary
        document.update(bufferViews=[], accessors=[], meshes=[], images=[])
        document['nodes'][0].pop('mesh')
        document['nodes'][0]['children'] = []
        document['extensionsUsed'].append('KHR_mesh_quantization')
        document['extensionsRequired'] = ['KHR_mesh_quantization']
        self.report = dict(geometry_bytes_before=0, geometry_bytes_after=0,
                           texture_bytes_before=0, texture_bytes_after=0,
                           max_position_error=0., max_uv_error=0., triangle_count=0)

    def view(self, raw, target=None, stride=None):
        self.binary.write(b'\0' * (-self.binary.tell() % 4))
        record = dict(buffer=0, byteOffset=self.binary.tell(), byteLength=len(raw))
        self.binary.write(raw)
        if target:
            record['target'] = target
        if stride:
            record['byteStride'] = stride
        self.document['bufferViews'].append(record)
        return len(self.document['bufferViews']) - 1

    def accessor(self, values, kind, target, normalized=False, bounds=False):
        record = dict(bufferView=self.view(values.tobytes(), target, 8 if kind == 'VEC3' else None),
                      componentType=5123, count=len(values), type=kind)
        if normalized:
            record['normalized'] = True
        if bounds:
            record.update(min=values[:, :3].min(0).tolist(), max=values[:, :3].max(0).tolist())
        self.document['accessors'].append(record)
        self.report['geometry_bytes_after'] += values.nbytes
        return len(self.document['accessors']) - 1

    def geometry(self, positions, uv, indices, material):
        result, report = self.document, self.report
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
            attrs = dict(POSITION=self.accessor(quantized, 'VEC3', 34962, True, True),
                         TEXCOORD_0=self.accessor(tex, 'VEC2', 34962, True))
            primitive_out = dict(attributes=attrs, indices=self.accessor(local_indices, 'SCALAR', 34963),
                                 material=material, mode=4)
            result['nodes'][0]['children'].append(len(result['nodes']))
            result['nodes'].append(dict(mesh=len(result['meshes']), translation=low.tolist(), scale=scale.tolist()))
            result['meshes'].append(dict(primitives=[primitive_out]))

    def image(self, raw, *, source_bytes=None):
        if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('Image transform must return PNG bytes')
        if source_bytes is None:
            self.report['texture_bytes_before'] = None
        elif self.report['texture_bytes_before'] is not None:
            self.report['texture_bytes_before'] += source_bytes
        self.report['texture_bytes_after'] += len(raw)
        self.document['images'].append(dict(bufferView=self.view(raw), mimeType='image/png'))

    def finish(self, output):
        length = _write_glb(self.document, self.binary, output)
        self.report.update(output=str(output), glb_bytes=length,
                           primitive_count=len(self.document['meshes']))
        return self.report


def compact_surfel_glb(source, output, *, image_transform=None):
    """Quantize geometry, optionally transform embedded PNGs, and create output exclusively."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    doc, binary, source_bytes = _read(source)
    _validate(doc, binary)
    with tempfile.TemporaryFile() as stream:
        writer = _CompactWriter(copy.deepcopy(doc), stream)
        for primitive in doc['meshes'][0]['primitives']:
            if (set(primitive) != {'attributes', 'indices', 'material', 'mode'}
                    or primitive['mode'] != 4 or set(primitive['attributes']) != {'POSITION', 'TEXCOORD_0'}):
                raise ValueError('Expected indexed POSITION/TEXCOORD_0 triangles')
            positions = _values(doc, binary, primitive['attributes']['POSITION'], 5126, 'VEC3', 3)
            uv = _values(doc, binary, primitive['attributes']['TEXCOORD_0'], 5126, 'VEC2', 2)
            indices = _values(doc, binary, primitive['indices'], 5125, 'SCALAR', 1).ravel()
            writer.geometry(positions, uv, indices, primitive['material'])
        for image in doc['images']:
            if set(image) != {'bufferView', 'mimeType'} or image['mimeType'] != 'image/png':
                raise ValueError('Expected embedded PNG images')
            v = doc['bufferViews'][image['bufferView']]
            raw = bytes(binary[v.get('byteOffset', 0):v.get('byteOffset', 0) + v['byteLength']])
            writer.image(image_transform(raw) if image_transform else raw, source_bytes=len(raw))
        report = writer.finish(output)
    report.update(source=str(source), source_bytes=source_bytes)
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
