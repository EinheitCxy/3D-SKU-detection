import { describe, expect, it } from "vitest";
import { bootstrap, candidateLabel, configureViewControlsState, disabledFacetLabels, selectionModeTransition, selectionStateAfterCanvasPick, visibleGlobalIdsForFilters } from "./main";

class FakeElement {
  constructor(readonly tagName: string) {}
  className = "";
  textContent = "";
  id = "";
  hidden = false;
  readonly attributes = new Map<string, string>();
  readonly children: FakeElement[] = [];
  append(...children: FakeElement[]): void { this.children.push(...children); }
  replaceChildren(...children: FakeElement[]): void { this.children.splice(0, this.children.length, ...children); }
  setAttribute(name: string, value: string): void { this.attributes.set(name, value); }
  getAttribute(name: string): string | null { return this.attributes.get(name) ?? null; }
}

describe("bootstrap", () => {
  it("replaces Loading with the existing load-error panel when href parsing fails", async () => {
    const root = new FakeElement("root");
    let loadCalled = false;
    let mountCalled = false;
    const previousDocument = globalThis.document;
    Object.defineProperty(globalThis, "document", { configurable: true, value: { createElement: (tag: string) => new FakeElement(tag) } });
    try {
      await bootstrap(root as unknown as HTMLElement, {
        href: "https://viewer.test/?data=ftp%3A%2F%2Fcdn.test%2Fbundle",
        load: async () => { loadCalled = true; throw new Error("loader must not run"); },
        mount: () => { mountCalled = true; throw new Error("mount must not run"); },
      });
      expect(root.children).toHaveLength(1);
      expect(root.children[0].className).toBe("load-error");
      expect(root.children[0].children.map((child) => child.tagName)).toEqual(["h2", "p", "pre"]);
      expect(loadCalled).toBe(false);
      expect(mountCalled).toBe(false);
    } finally {
      Object.defineProperty(globalThis, "document", { configurable: true, value: previousDocument });
    }
  });

  it("tries a failing explicit root once and does not fall back to defaults", async () => {
    const root = new FakeElement("root");
    const previousDocument = globalThis.document;
    const calls: string[] = [];
    Object.defineProperty(globalThis, "document", { configurable: true, value: { createElement: (tag: string) => new FakeElement(tag) } });
    try {
      await bootstrap(root as unknown as HTMLElement, {
        href: "https://viewer.test/review/?data=.%2Fcustom%2F",
        load: async (url) => { calls.push(url); throw new Error("missing bundle"); },
        mount: () => { throw new Error("mount must not run"); },
      });
      expect(calls).toEqual(["https://viewer.test/review/custom/"]);
      expect(root.children).toHaveLength(1);
      expect(root.children[0].className).toBe("load-error");
      expect(root.children[0].children.map((child) => child.tagName)).toEqual(["h2", "p", "pre"]);
    } finally {
      Object.defineProperty(globalThis, "document", { configurable: true, value: previousDocument });
    }
  });

  it("tries the second default root and mounts it after the first fails", async () => {
    const root = new FakeElement("root");
    const previousDocument = globalThis.document;
    const calls: string[] = [];
    const bundle = {} as import("./bundle-loader").ViewerBundle;
    let mounted: import("./bundle-loader").ViewerBundle | null = null;
    Object.defineProperty(globalThis, "document", { configurable: true, value: { createElement: (tag: string) => new FakeElement(tag) } });
    try {
      await bootstrap(root as unknown as HTMLElement, {
        href: "https://viewer.test/review/",
        load: async (url) => {
          calls.push(url);
          if (calls.length === 1) throw new Error("first root missing");
          return bundle;
        },
        mount: (_root, received) => { mounted = received; },
      });
      expect(calls).toEqual(["https://viewer.test/review/data/", "https://viewer.test/data/"]);
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
