import { describe, expect, it } from "vitest";
import { loadDockerViewerBundle } from "./docker-zip-loader";

type ZipEntry = Readonly<{ name: string; content: Uint8Array; compression?: number }>;

const encoder = new TextEncoder();

function bytes(value: string): Uint8Array {
  return encoder.encode(value);
}

function concat(parts: readonly Uint8Array[]): Uint8Array {
  const output = new Uint8Array(parts.reduce((size, part) => size + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
}

function record(size: number, write: (view: DataView) => void): Uint8Array {
  const output = new Uint8Array(size);
  write(new DataView(output.buffer));
  return output;
}

function blobPart(content: Uint8Array): ArrayBuffer {
  const copy = new Uint8Array(content.byteLength);
  copy.set(content);
  return copy.buffer;
}

function storedZip(entries: readonly ZipEntry[]): Blob {
  let offset = 0;
  const local: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  for (const entry of entries) {
    const name = bytes(entry.name);
    const compression = entry.compression ?? 0;
    local.push(record(30, (view) => {
      view.setUint32(0, 0x04034b50, true);
      view.setUint16(4, 20, true);
      view.setUint16(8, compression, true);
      view.setUint32(18, entry.content.length, true);
      view.setUint32(22, entry.content.length, true);
      view.setUint16(26, name.length, true);
    }), name, entry.content);
    central.push(record(46, (view) => {
      view.setUint32(0, 0x02014b50, true);
      view.setUint16(4, 20, true);
      view.setUint16(6, 20, true);
      view.setUint16(10, compression, true);
      view.setUint32(20, entry.content.length, true);
      view.setUint32(24, entry.content.length, true);
      view.setUint16(28, name.length, true);
      view.setUint32(42, offset, true);
    }), name);
    offset += 30 + name.length + entry.content.length;
  }
  const centralBytes = concat(central);
  const tail = record(22, (view) => {
    view.setUint32(0, 0x06054b50, true);
    view.setUint16(8, entries.length, true);
    view.setUint16(10, entries.length, true);
    view.setUint32(12, centralBytes.length, true);
    view.setUint32(16, offset, true);
  });
  return new Blob([blobPart(concat([...local, centralBytes, tail]))], { type: "application/zip" });
}

function validArchive(overrides: readonly ZipEntry[] = []): Blob {
  return storedZip([
    {
      name: "manifest.json",
      content: bytes(JSON.stringify({
        schema_version: "3.0.0",
        dataset_name: "floor_display6",
        backend: "DA3",
        frame_count: 1,
        display_bounds: [0, 0, 0, 1, 1, 1],
        world_to_view: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      })),
    },
    { name: "positions.f32.bin", content: new Uint8Array(new Float32Array([1, 2, 3]).buffer) },
    { name: "colors.u8.bin", content: new Uint8Array([10, 20, 30]) },
    { name: "normals.i8.bin", content: new Uint8Array([0, 0, 127]) },
    {
      name: "objects.json",
      content: bytes(JSON.stringify({
        "1": {
          ordered_skus: [{ sku_id: "sku-1", sku_name: "产品" }],
          point_ranges: [[0, 1]],
          observations: [{ image_id: 0, object_id: 1, removed: false, thumbnail: "thumbs/1_0.jpg" }],
        },
      })),
    },
    { name: "thumbs/1_0.jpg", content: new Uint8Array([255, 216, 255, 217]) },
    ...overrides,
  ]);
}

describe("loadDockerViewerBundle", () => {
  it("loads Docker's flat stored ZIP and resolves thumbnail bytes locally", async () => {
    const bundle = await loadDockerViewerBundle(validArchive());

    expect(bundle.manifest.dataset_name).toBe("floor_display6");
    expect(Array.from(bundle.positions)).toEqual([1, 2, 3]);
    expect(bundle.pointCount).toBe(1);
    expect(bundle.resolveAssetUrl("thumbs/1_0.jpg")).toMatch(/^blob:/);
  });

  it("rejects non-stored ZIP entries instead of silently accepting an unsupported archive", async () => {
    await expect(loadDockerViewerBundle(validArchive([
      { name: "unexpected.txt", content: bytes("x") },
    ]))).rejects.toThrow(/unexpected member/i);
    await expect(loadDockerViewerBundle(storedZip([
      { name: "manifest.json", content: bytes("{}"), compression: 8 },
    ]))).rejects.toThrow(/stored|compression/i);
  });

  it("rejects nested or duplicate names before decoding the Viewer contract", async () => {
    await expect(loadDockerViewerBundle(validArchive([
      { name: "thumbs/nested/1.jpg", content: bytes("x") },
    ]))).rejects.toThrow(/unexpected member/i);
    await expect(loadDockerViewerBundle(validArchive([
      { name: "manifest.json", content: bytes("{}") },
    ]))).rejects.toThrow(/duplicate/i);
  });
});
