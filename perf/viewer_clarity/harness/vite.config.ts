import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

const here = fileURLToPath(new URL(".", import.meta.url));
const viewer = process.env.VIEWER_SRC;
if (!viewer) throw new Error("VIEWER_SRC must point to the selected arm's viewer/src");
const viewerRoot = resolve(viewer);
const runtimeRoot = resolve(here, "../../../runtime/viewer-clarity-20261012");

export default {
  root: here,
  resolve: { alias: [
    { find: "@viewer", replacement: viewerRoot },
    { find: "three/addons", replacement: resolve(viewerRoot, "../node_modules/three/examples/jsm") },
    { find: "three", replacement: resolve(viewerRoot, "../node_modules/three") },
  ] },
  server: { fs: { allow: [here, viewerRoot, resolve(viewerRoot, "../node_modules"), runtimeRoot] } },
};
