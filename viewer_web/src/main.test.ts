import { describe, expect, it } from "vitest";
import { bootstrap, candidateLabel, configureViewControlsState, disabledFacetLabels, selectionModeTransition, selectionStateAfterCanvasPick, visibleGlobalIdsForFilters } from "./main";

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
  it("only exposes the local ZIP picker and never loads HTTP data roots", async () => {
    const root = new FakeElement("root");
    const previousDocument = globalThis.document;
    const previousWindow = globalThis.window;
    Object.defineProperty(globalThis, "document", { configurable: true, value: { createElement: (tag: string) => new FakeElement(tag) } });
    Object.defineProperty(globalThis, "window", { configurable: true, value: { location: { href: "https://viewer.test/review/?data=https%3A%2F%2Fcdn.test%2Fbundle" } } });
    try {
      await bootstrap(root as unknown as HTMLElement, {
        mount: () => { throw new Error("mount must not run before ZIP selection"); },
      });
      expect(root.children).toHaveLength(1);
      expect(root.children[0].className).toBe("zip-picker");
      expect(root.children[0].children.map((child) => child.tagName)).toEqual(["h2", "p", "pre", "input"]);
    } finally {
      Object.defineProperty(globalThis, "document", { configurable: true, value: previousDocument });
      Object.defineProperty(globalThis, "window", { configurable: true, value: previousWindow });
    }
  });

  it("keeps the ZIP picker available and shows the error when ZIP loading fails", async () => {
    const root = new FakeElement("root");
    const previousDocument = globalThis.document;
    const file = new Blob(["zip"], { type: "application/zip" });
    Object.defineProperty(globalThis, "document", { configurable: true, value: { createElement: (tag: string) => new FakeElement(tag) } });
    try {
      await bootstrap(root as unknown as HTMLElement, {
        loadZip: async () => { throw new Error("invalid ZIP"); },
        mount: () => { throw new Error("mount must not run"); },
      });
      const panel = root.children[0];
      const detail = panel.children[2];
      const picker = panel.children[3] as FakeElement & { files?: Blob[]; disabled?: boolean };
      picker.files = [file];
      picker.value = "viewer_bundle.zip";
      await picker.onchange?.();
      expect(panel.className).toBe("zip-picker");
      expect(picker.disabled).toBe(false);
      expect(picker.value).toBe("");
      expect(detail.textContent).toContain("invalid ZIP");
    } finally {
      Object.defineProperty(globalThis, "document", { configurable: true, value: previousDocument });
    }
  });

  it("loads a user-selected Docker viewer ZIP", async () => {
    const root = new FakeElement("root");
    const previousDocument = globalThis.document;
    const bundle = {} as import("./bundle-loader").ViewerBundle;
    const file = new Blob(["zip"], { type: "application/zip" });
    let mounted: import("./bundle-loader").ViewerBundle | null = null;
    Object.defineProperty(globalThis, "document", { configurable: true, value: { createElement: (tag: string) => new FakeElement(tag) } });
    try {
      await bootstrap(root as unknown as HTMLElement, {
        loadZip: async (selected) => {
          expect(selected).toBe(file);
          return bundle;
        },
        mount: (_root, received) => { mounted = received; },
      });
      const picker = root.children[0].children[3] as FakeElement & { files?: Blob[] };
      picker.files = [file];
      await picker.onchange?.();
      expect(mounted).toBe(bundle);
    } finally {
      Object.defineProperty(globalThis, "document", { configurable: true, value: previousDocument });
    }
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
