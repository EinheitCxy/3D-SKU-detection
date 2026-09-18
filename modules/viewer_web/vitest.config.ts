import { defineConfig } from "vitest/config";

export default defineConfig({
  root: "../..",
  test: { include: ["test/viewer_web/**/*.test.ts"] },
});
