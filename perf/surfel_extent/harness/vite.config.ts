import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { createReadStream, statSync } from "node:fs";

const here = fileURLToPath(new URL(".", import.meta.url));
const repository = resolve(here, "../../..");
const viewer = resolve(repository, "docker/viewer/src");
const viewerModules = resolve(repository, "docker/viewer/node_modules");
const visual = resolve(repository, "runtime/viewer-extent-20261012/visual");
const bundles = new Map([
  ["/bundles/video1.zip", resolve(repository, "runtime/viewer-extent-20261012/video12/video1/viewer_bundle.zip")],
  ["/bundles/video2.zip", resolve(repository, "runtime/viewer-extent-20261012/video12/video2/viewer_bundle.zip")],
  ["/bundles/video3.zip", resolve(repository, "runtime/viewer-clarity-20261012/video3/X0/crop/viewer_bundle.zip")],
  ["/bundles/fd6.zip", resolve(repository, "runtime/viewer-clarity-20261012/fd6/X0/crop/viewer_bundle.zip")],
]);

export default {
  root: here,
  resolve: { alias: [
    { find: "@viewer", replacement: viewer },
    { find: "@baseline-renderer", replacement: resolve(repository, "runtime/viewer-extent-20261012/visual/baseline-renderer.ts") },
    { find: "@candidate-renderer", replacement: resolve(viewer, "surfel-renderer.ts") },
    { find: "@diagnostic-renderer", replacement: resolve(repository, "runtime/viewer-extent-20261012/video12/followup/diagnostic-renderer.ts") },
    { find: "@source-selection-renderer", replacement: resolve(repository, "runtime/viewer-extent-20261012/video12/followup/source-selection-renderer.ts") },
    { find: "@uniform-renderer", replacement: resolve(visual, "uniform-renderer.ts") },
    { find: "@major-renderer", replacement: resolve(visual, "major-renderer.ts") },
    { find: "@root-candidate-renderer", replacement: resolve(repository, "modules/viewer_web/src/surfel-renderer.ts") },
    { find: "@root-baseline-renderer", replacement: resolve(visual, "root-baseline-renderer.ts") },
    { find: "three/addons", replacement: resolve(viewerModules, "three/examples/jsm") },
    { find: "three", replacement: resolve(viewerModules, "three") },
  ] },
  plugins: [{
    name: "exact-surfel-extent-bundles",
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        const pathname = new URL(request.url ?? "/", "http://localhost").pathname;
        const file = bundles.get(pathname);
        if (!file) return next();
        response.statusCode = 200;
        response.setHeader("Content-Type", "application/zip");
        response.setHeader("Content-Length", statSync(file).size);
        createReadStream(file).pipe(response);
      });
    },
  }],
  server: {
    host: "127.0.0.1", port: 53110, strictPort: true,
    fs: { allow: [
      here, viewer, viewerModules, resolve(repository, "modules/viewer_web/src"), visual,
      resolve(repository, "runtime/viewer-extent-20261012/video12/followup"),
      resolve(repository, "runtime/viewer-extent-20261012/video12/video1"),
      resolve(repository, "runtime/viewer-extent-20261012/video12/video2"),
      resolve(repository, "runtime/viewer-clarity-20261012/video3/X0/crop"),
      resolve(repository, "runtime/viewer-clarity-20261012/fd6/X0/crop"),
    ] },
  },
};
