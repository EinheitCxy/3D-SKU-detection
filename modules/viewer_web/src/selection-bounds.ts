import { Box3 } from "three";

export function cachedSelectionBox(cache: Map<string, Box3 | null>, globalId: string, compute: () => Box3 | null): Box3 | null {
  if (!cache.has(globalId)) cache.set(globalId, compute());
  return cache.get(globalId)?.clone() ?? null;
}
