import type { FootprintBundle, ObjectIndex, ObjectIndexEntry } from "./contracts";
import type { ViewerBundle } from "./bundle-loader";

export interface EvidenceFootprintView {
  readonly available: boolean;
  readonly areaM2: number | null;
  readonly observationsUsed: number | null;
}

export interface EvidenceView {
  readonly globalId: string;
  readonly object: ObjectIndexEntry;
  readonly footprint: EvidenceFootprintView;
}

export function formatFormalMetric(footprints: FootprintBundle): string {
  return footprints.status === "accepted" && footprints.value_m2 !== null
    ? `${footprints.value_m2.toFixed(2)} m²`
    : "—";
}

export function listGlobalIds(objects: ObjectIndex): string[] {
  return Object.keys(objects).sort((left, right) => {
    const leftValue = BigInt(left);
    const rightValue = BigInt(right);
    return leftValue === rightValue ? 0 : leftValue < rightValue ? -1 : 1;
  });
}

export function buildEvidenceView(bundle: ViewerBundle, globalId: string): EvidenceView | null {
  if (globalId === "union") return null;
  const object = bundle.objects[globalId];
  if (object === undefined) return null;
  const geometry = bundle.footprints.status === "accepted" ? bundle.footprints.per_global_id[globalId] : undefined;
  const properties = geometry?.properties;
  const hasFootprint = properties !== undefined && properties.global_id === globalId && "observations_used" in properties;
  return {
    globalId,
    object,
    footprint: hasFootprint
      ? { available: true, areaM2: properties.area_m2, observationsUsed: properties.observations_used }
      : { available: false, areaM2: null, observationsUsed: null },
  };
}
