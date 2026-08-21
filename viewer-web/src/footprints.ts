import {
  BufferGeometry,
  Color,
  Float32BufferAttribute,
  Group,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshBasicMaterial,
  Shape,
  ShapeGeometry,
  Vector2,
  Vector3,
} from "three";
import type { FootprintBundle, FootprintGeometry, SupportPlane } from "./contracts";

export const FOOTPRINT_AMBER = new Color("#f5a524");
const FOOTPRINT_UNION_OPACITY = 0.98;

export interface FootprintSelectionVisual {
  readonly fills: readonly MeshBasicMaterial[];
  readonly outlines: readonly LineBasicMaterial[];
}

export interface FootprintObjects {
  readonly group: Group;
  readonly pickMeshes: Mesh[];
  readonly focusTargets: ReadonlyMap<string, Group>;
  readonly selectionVisuals: ReadonlyMap<string, FootprintSelectionVisual>;
}

export function createFootprintObjects(footprints: FootprintBundle): FootprintObjects {
  const group = new Group();
  group.name = "footprints";
  const pickMeshes: Mesh[] = [];
  const focusTargets = new Map<string, Group>();
  const selectionVisuals = new Map<string, FootprintSelectionVisual>();
  if (footprints.status !== "accepted" || footprints.support_plane === null) return { group, pickMeshes, focusTargets, selectionVisuals };

  if (footprints.union !== null) {
    const union = createOutlineGroup("footprint:union:outline", footprints.union, footprints.support_plane, FOOTPRINT_UNION_OPACITY, true);
    group.add(union);
  }
  for (const [globalId, geometry] of Object.entries(footprints.per_global_id)) {
    const visual = createPerIdVisual(globalId, geometry, footprints.support_plane);
    group.add(visual.group);
    pickMeshes.push(...visual.meshes);
    focusTargets.set(globalId, visual.group);
    selectionVisuals.set(globalId, { fills: visual.fillMaterials, outlines: visual.outlineMaterials });
  }
  return { group, pickMeshes, focusTargets, selectionVisuals };
}

function createPerIdVisual(globalId: string, geometry: FootprintGeometry, plane: SupportPlane): {
  group: Group;
  meshes: Mesh[];
  fillMaterials: MeshBasicMaterial[];
  outlineMaterials: LineBasicMaterial[];
} {
  const group = new Group();
  group.name = `footprint:${globalId}`;
  const meshes: Mesh[] = [];
  const fillMaterials: MeshBasicMaterial[] = [];
  const outlineMaterials: LineBasicMaterial[] = [];
  geometry.rings.forEach((polygon, index) => {
    const fillMaterial = new MeshBasicMaterial({
      color: FOOTPRINT_AMBER, transparent: true, opacity: 0.38, side: 2,
      polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
    });
    const mesh = new Mesh(createPolygonGeometry(polygon, plane), fillMaterial);
    mesh.name = `footprint:${globalId}:fill:${index}`;
    mesh.renderOrder = 2;
    mesh.userData.globalId = globalId;
    group.add(mesh);
    meshes.push(mesh);
    fillMaterials.push(fillMaterial);
    const outline = createOutlineGroup(`footprint:${globalId}:outline:${index}`, { rings: [polygon], properties: geometry.properties }, plane, 0.95, false);
    group.add(outline);
    for (const line of outline.children as Line[]) outlineMaterials.push(line.material as LineBasicMaterial);
  });
  if (meshes.length === 0) throw new Error(`Footprint ${globalId} has no polygons`);
  return { group, meshes, fillMaterials, outlineMaterials };
}

function createOutlineGroup(name: string, geometry: FootprintGeometry, plane: SupportPlane, opacity: number, isUnion: boolean): Group {
  const group = new Group();
  group.name = name;
  for (const polygon of geometry.rings) {
    for (const ring of polygon) {
      const lineGeometry = new BufferGeometry();
      lineGeometry.setAttribute("position", new Float32BufferAttribute(ring.flatMap(([u, v]) => toWorld(u, v, plane).toArray()), 3));
      const line = new Line(lineGeometry, new LineBasicMaterial({
        color: FOOTPRINT_AMBER,
        transparent: true,
        opacity,
        depthWrite: !isUnion,
      }));
      line.renderOrder = isUnion ? 5 : 3;
      group.add(line);
    }
  }
  return group;
}

function createPolygonGeometry(polygon: FootprintGeometry["rings"][number], plane: SupportPlane): ShapeGeometry {
  const [outer, ...holes] = polygon;
  const shape = new Shape(outer.map(([u, v]) => new Vector2(u, v)));
  for (const ring of holes) shape.holes.push(new Shape(ring.map(([u, v]) => new Vector2(u, v))));
  const geometry = new ShapeGeometry(shape);
  const positions = geometry.getAttribute("position");
  for (let index = 0; index < positions.count; index += 1) {
    const world = toWorld(positions.getX(index), positions.getY(index), plane);
    positions.setXYZ(index, world.x, world.y, world.z);
  }
  positions.needsUpdate = true;
  geometry.computeVertexNormals();
  return geometry;
}

function toWorld(u: number, v: number, plane: SupportPlane): Vector3 {
  return new Vector3(...plane.point)
    .addScaledVector(new Vector3(...plane.u_axis), u)
    .addScaledVector(new Vector3(...plane.v_axis), v);
}
