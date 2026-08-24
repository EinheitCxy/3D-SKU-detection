export type Vec3 = readonly [number, number, number];

/** Places a focused camera on the same side of the new target as before focus. */
export function focusedCameraPosition(camera: Vec3, currentTarget: Vec3, target: Vec3, distance: number): Vec3 {
  const offset: Vec3 = [camera[0] - currentTarget[0], camera[1] - currentTarget[1], camera[2] - currentTarget[2]];
  const length = Math.hypot(...offset);
  if (length === 0) return [target[0] + distance, target[1], target[2]];
  return [target[0] + offset[0] * distance / length, target[1] + offset[1] * distance / length, target[2] + offset[2] * distance / length];
}
