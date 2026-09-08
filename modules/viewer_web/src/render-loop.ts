/** Coalesce invalidations; the callback requests continuation only while animating. */
export function createRenderLoop(draw: (time: number) => boolean) {
  let pending: number | null = null;
  let disposed = false;
  function request() {
    if (disposed || pending !== null) return;
    pending = requestAnimationFrame(time => {
      pending = null;
      if (disposed) return;
      if (draw(time)) request();
    });
  }
  return {
    request,
    dispose() {
      disposed = true;
      if (pending !== null) cancelAnimationFrame(pending);
      pending = null;
    },
  };
}
