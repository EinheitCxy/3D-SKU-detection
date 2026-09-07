import type { ObjectIndex, ObjectIndexEntry, OrderedSku } from "./contracts";

export interface ObservationCounts {
  readonly total: number;
  readonly active: number;
  readonly removed: number;
}

export interface SelectedObjectObservationView {
  readonly imageId: number;
  readonly objectId: number;
  readonly removed: boolean;
  readonly thumbnailUrl: string;
}

export interface SelectedObjectView {
  readonly globalId: string;
  readonly orderedSkus: readonly OrderedSku[];
  readonly observations: readonly SelectedObjectObservationView[];
}

export function formatDatasetSummary(datasetName: string, frameCount: number): string {
  return `${datasetName} · ${frameCount} frames`;
}

export function entryHasGeometry(object: ObjectIndexEntry): boolean {
  return object.point_ranges.some(([start, end]) => end > start);
}

export function canFocusGlobalId(object: ObjectIndexEntry | undefined): boolean {
  return object !== undefined && entryHasGeometry(object);
}

export function listGlobalIds(objects: ObjectIndex): string[] {
  return Object.keys(objects).sort((left, right) => {
    const leftValue = BigInt(left);
    const rightValue = BigInt(right);
    return leftValue === rightValue ? 0 : leftValue < rightValue ? -1 : 1;
  });
}

export function summarizeObjectCounts(
  objects: ObjectIndex,
  visibleGlobalIds: ReadonlySet<string>,
): { readonly total: number; readonly visible: number } {
  let visible = 0;
  for (const globalId of visibleGlobalIds) {
    if (objects[globalId] !== undefined) visible += 1;
  }
  return { total: Object.keys(objects).length, visible };
}

export function summarizeObservationCounts(
  observations: readonly { readonly removed: boolean }[],
): ObservationCounts {
  const removed = observations.filter((observation) => observation.removed).length;
  return { total: observations.length, active: observations.length - removed, removed };
}

export function buildSelectedObjectView(
  objects: ObjectIndex,
  globalId: string,
  assetSource: string | ((relativePath: string) => string),
): SelectedObjectView | null {
  const object = objects[globalId];
  if (object === undefined) return null;
  return {
    globalId,
    orderedSkus: object.ordered_skus,
    observations: object.observations.map((observation) => ({
      imageId: observation.image_id,
      objectId: observation.object_id,
      removed: observation.removed,
      thumbnailUrl: typeof assetSource === "string"
        ? new URL(observation.thumbnail, assetSource).toString()
        : assetSource(observation.thumbnail),
    })),
  };
}
