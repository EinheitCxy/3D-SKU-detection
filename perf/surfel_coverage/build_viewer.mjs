import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from '../../modules/viewer_web/node_modules/vite/dist/node/index.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
await build({configFile:false, publicDir:false, root:path.join(root,'modules/viewer_web'), base:'./',
  build:{outDir:path.join(root,'runtime/surfel-coverage-20260924/web'), emptyOutDir:false,
    rollupOptions:{input:path.join(root,'modules/viewer_web/surfel-coverage.html')}}});
