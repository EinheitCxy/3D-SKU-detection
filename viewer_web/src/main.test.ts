import { describe, expect, it } from "vitest";
import { bootstrap, candidateLabel, configureViewControlsState, disabledFacetLabels, selectionModeTransition, selectionStateAfterCanvasPick, viewerBundleUrl, visibleGlobalIdsForFilters } from "./main";

class FakeElement {
  constructor(readonly tagName: string) {}
  className = "";
  textContent = "";
  value = "";
  id = "";
  hidden = false;
  readonly attributes = new Map<string, string>();
  readonly children: FakeElement[] = [];
  onchange: (() => Promise<void>) | null = null;
  append(...children: FakeElement[]): void { this.children.push(...children); }
  replaceChildren(...children: FakeElement[]): void { this.children.splice(0, this.children.length, ...children); }
  setAttribute(name: string, value: string): void { this.attributes.set(name, value); }
  getAttribute(name: string): string | null { return this.attributes.get(name) ?? null; }
}

describe("bootstrap", () => {
  it("downloads the requested task bundle from COS and mounts it", async () => {
    const root = new FakeElement("root");
    const previousDocument = globalThis.document;
    const previousWindow = globalThis.window;
    Object.defineProperty(globalThis, "document", { configurable: true, value: { createElement: (tag: string) => new FakeElement(tag) } });
    Object.defineProperty(globalThis, "window", { configurable: true, value: { location: { search: "?recognition_task_id=QXahlXG7acJ867Myahs8" } } });
    const bundle = {} as import("./bundle-loader").ViewerBundle;
    const fetched: string[] = [];
    let mounted: import("./bundle-loader").ViewerBundle | null = null;
    try {
      await bootstrap(root as unknown as HTMLElement, {
        fetch: async (url) => {
          fetched.push(String(url));
          return new Response(new Blob(["zip"], { type: "application/zip" }));
        },
        loadZip: async (archive) => {
          expect(await archive.text()).toBe("zip");
          return bundle;
        },
        mount: (_root, received) => { mounted = received; },
      });
      expect(fetched).toHaveLength(1);
      expect(fetched[0]).not.toBe("/viewer-config");
      expect(fetched[0]).toMatch(/QXahlXG7acJ867Myahs8\/viewer_bundle\.zip$/);
      expect(mounted).toBe(bundle);
    } finally {
      Object.defineProperty(globalThis, "document", { configurable: true, value: previousDocument });
      Object.defineProperty(globalThis, "window", { configurable: true, value: previousWindow });
    }
  });
});

describe("COS bundle URL", () => {
  it("uses the raw recognition task value as one encoded COS key segment", () => {
    expect(viewerBundleUrl("https://cos.test/global-id-mapping", "task /?x=1")).toBe(
      "https://cos.test/global-id-mapping/task%20%2F%3Fx%3D1/viewer_bundle.zip",
    );
  });
});

describe("SKU viewer labels and filters", () => {
  it("keeps ordinary mode buttons mutually exclusive and clears selection on a switch", () => {
    expect(selectionModeTransition("global", "sku", "11")).toEqual({ mode: "sku", searchQuery: "", clearSelection: true });
    expect(selectionModeTransition("sku", "global", "11")).toEqual({ mode: "global", searchQuery: "11", clearSelection: true });
    expect(selectionModeTransition("sku", "sku", "")).toEqual({ mode: "sku", searchQuery: "", clearSelection: false });
  });

  it("switches a canvas point pick into Global ID mode", () => {
    expect(selectionStateAfterCanvasPick("sku", "A", "11")).toEqual({ mode: "global", selectedSkuId: null, selectedGlobalId: "11" });
  });

  it("formats a minimal ordered SKU label", () => {
    expect(candidateLabel({ sku_id: "430085", sku_name: "产品A" })).toBe("430085 · 产品A");
  });

  it("filters the supplied global ID order by search and primary SKU", () => {
    const objects = {
      "1": { ordered_skus: [{ sku_id: "A" }] },
      "2": { ordered_skus: [{ sku_id: "B" }] },
      "3": { ordered_skus: [] },
    };
    expect(visibleGlobalIdsForFilters(["1", "2", "3"], objects as never, "", "B")).toEqual(["2"]);
    expect(visibleGlobalIdsForFilters(["1", "2", "3"], objects as never, "3", null)).toEqual(["3"]);
  });
});

describe("View Controls state", () => {
  it("writes the linked collapsed and expanded DOM state", () => {
    const toggle = new FakeElement("button");
    const panel = new FakeElement("div");

    configureViewControlsState(toggle as unknown as HTMLButtonElement, panel as unknown as HTMLElement, false);
    expect(panel.id).toBe("scene-controls-panel");
    expect(toggle.getAttribute("aria-controls")).toBe("scene-controls-panel");
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(panel.hidden).toBe(true);

    configureViewControlsState(toggle as unknown as HTMLButtonElement, panel as unknown as HTMLElement, true);
    expect(panel.id).toBe("scene-controls-panel");
    expect(toggle.getAttribute("aria-controls")).toBe("scene-controls-panel");
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    expect(panel.hidden).toBe(false);
  });
});

describe("disabled product facets", () => {
  it("keeps the six product metadata placeholders disabled", () => {
    expect(disabledFacetLabels).toEqual([
      "厂商：主数据待接入",
      "品牌：主数据待接入",
      "品类：主数据待接入",
      "POSM：检测能力待接入",
      "价签：检测能力待接入",
      "空缺位：检测能力待接入",
    ]);
  });
});
