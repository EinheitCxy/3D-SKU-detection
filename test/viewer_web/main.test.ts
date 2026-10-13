import { beforeEach, describe, expect, it, vi } from "vitest";
import { loadSurfels, type SurfelData } from "../../modules/viewer_web/src/surfel-loader";

vi.mock("../../modules/viewer_web/src/surfel-loader", async (importOriginal) => ({
  ...await importOriginal<typeof import("../../modules/viewer_web/src/surfel-loader")>(),
  loadSurfels: vi.fn(),
}));

beforeEach(() => vi.mocked(loadSurfels).mockReset());

function mockSurfelAssets(): SurfelData {
  const assets = { textures: { dispose: vi.fn() }, depths: { dispose: vi.fn() } } as unknown as SurfelData;
  vi.mocked(loadSurfels).mockResolvedValueOnce(assets);
  return assets;
}

import { bootstrap, configureRenderModeToggle, candidateLabel, configureViewControlsState, displayPosmValue, selectionModeLabels, selectionModeTransition, selectionSummaryLabels, selectionStateAfterCanvasPick, skuFacetResultLabel, visibleGlobalIdsForFilters } from "../../modules/viewer_web/src/main";

class FakeElement {
  constructor(readonly tagName: string) {}
  className = "";
  textContent = "";
  id = "";
  hidden = false;
  disabled = false;
  readonly listeners = new Map<string, (() => void)[]>();
  addEventListener(name: string, listener: () => void): void { this.listeners.set(name, [...this.listeners.get(name) ?? [], listener]); }
  click(): void { if (!this.disabled) this.listeners.get("click")?.forEach(listener => listener()); }
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

  it("loads explicit points assets without requesting surfels", async () => {
    const previousDocument = globalThis.document;
    Object.defineProperty(globalThis, "document", { configurable: true, value: { createElement: (tag: string) => new FakeElement(tag) } });
    const bundle = {} as import("../../modules/viewer_web/src/bundle-loader").ViewerBundle;
    const mount = vi.fn();
    const load = vi.fn(async () => bundle);
    try {
      await bootstrap(new FakeElement("root") as unknown as HTMLElement, { href: "https://viewer.test/?render=points&data=/data/", load, mount });
      expect(load).toHaveBeenCalledExactlyOnceWith("https://viewer.test/data/", expect.any(Function), "points");
      expect(loadSurfels).not.toHaveBeenCalled();
      expect(mount).toHaveBeenCalledExactlyOnceWith(expect.anything(), bundle);
    } finally { Object.defineProperty(globalThis, "document", { configurable: true, value: previousDocument }); }
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
    const bundle = {} as import("../../modules/viewer_web/src/bundle-loader").ViewerBundle;
    const surfels = mockSurfelAssets();
    let mounted: import("../../modules/viewer_web/src/bundle-loader").ViewerBundle | null = null;
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
      expect(loadSurfels).toHaveBeenCalledExactlyOnceWith(bundle, expect.any(Function));
      expect(mounted).toEqual({ ...bundle, surfels });
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
    expect(selectionStateAfterCanvasPick("sku", "A", "11")).toEqual({ mode: "global", selectedFacetId: null, selectedGlobalId: "11" });
  });

  it("formats a minimal ordered SKU label", () => {
    expect(candidateLabel({ sku_id: "430085", sku_name: "产品A" })).toBe("430085 · 产品A");
  });

  it("formats the label used before a matched SKU count", () => {
    expect(skuFacetResultLabel({ skuId: "430085", skuName: "产品A" })).toBe("430085 · 产品A");
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

describe("selection modes", () => {
  it("exposes manufacturer, brand, category, SKU, and Global ID modes", () => {
    expect(selectionModeLabels).toEqual([
      "Manufacturer",
      "Brand",
      "Category",
      "SKU",
      "Global ID",
    ]);
  });
});

describe("POSM display", () => {
  it("always shows N/A regardless of the source marker", () => {
    expect(displayPosmValue(false)).toBe("N/A");
    expect(displayPosmValue(true)).toBe("N/A");
  });
});

describe("selection summary", () => {
  it("shows only total and removed observations", () => {
    expect(selectionSummaryLabels).toEqual(["Total"]);
  });
});

describe("recognition task loading", () => {
  it.each([false, true])("uses the task ZIP without trying local data (HTTP failure: %s)", async (fail) => {
    const previousDocument = globalThis.document;
    Object.defineProperty(globalThis, "document", { configurable: true, value: { createElement: (tag: string) => new FakeElement(tag) } });
    const root = new FakeElement("root");
    const calls: string[] = [];
    let mounted = false;
    const bundle = { disposeAssets: () => {} } as unknown as import("../../modules/viewer_web/src/bundle-loader").ViewerBundle;
    try {
      await bootstrap(root as unknown as HTMLElement, {
        href: "https://viewer.test/?recognition_task_id=task%20id&data=/ignored/&render=points",
        load: async () => { throw new Error("must not load local data"); },
        fetch: (async (url: string | URL | Request) => {
          calls.push(String(url));
          return new Response("archive", { status: fail ? 404 : 200 });
        }) as typeof fetch,
        loadZip: async (_archive: Blob, _data: unknown, mode: string) => {
          expect(mode).toBe("points");
          return bundle;
        },
        mount: (_root: HTMLElement, loaded: typeof bundle) => {
          expect(loaded).toBe(bundle);
          mounted = true;
        },
      });
      expect(calls).toHaveLength(1);
      expect(calls[0]).toMatch(/\/task%20id\/viewer_bundle.zip$/);
      expect(mounted).toBe(!fail);
      expect(loadSurfels).not.toHaveBeenCalled();
      if (fail) expect(root.children[0].children[2].textContent).toContain("404");
    } finally {
      Object.defineProperty(globalThis, "document", { configurable: true, value: previousDocument });
    }
  });
});


describe("render mode resource availability", () => {
  it.each(["points", "surfel"] as const)("hides switching when only %s assets are available", (mode) => {
    const toggle = new FakeElement("button"), setMode = vi.fn(), onSuccess = vi.fn();
    const bundle = { colors: mode === "points" ? new Uint8Array(3) : null, ...(mode === "surfel" ? { surfels: {} as SurfelData } : {}) };
    configureRenderModeToggle(toggle as unknown as HTMLButtonElement, bundle, () => mode, setMode, onSuccess);
    expect(toggle.hidden).toBe(true); expect(toggle.disabled).toBe(true); toggle.click();
    expect(setMode).not.toHaveBeenCalled(); expect(onSuccess).not.toHaveBeenCalled();
  });
  it("updates mode only after the controller successfully switches", () => {
    const toggle = new FakeElement("button"), bundle = { colors: new Uint8Array(3), surfels: {} as SurfelData };
    let active: "points" | "surfel" = "points";
    const order: string[] = [];
    configureRenderModeToggle(toggle as unknown as HTMLButtonElement, bundle, () => active, mode => { order.push(`set:${mode}`); expect(active).toBe("points"); }, mode => { active = mode; order.push(`updated:${mode}`); });
    expect(toggle.hidden).toBe(false); toggle.click(); expect(active).toBe("surfel");
    expect(order).toEqual(["set:surfel", "updated:surfel"]);
    const failed = new FakeElement("button"), updated = vi.fn();
    configureRenderModeToggle(failed as unknown as HTMLButtonElement, bundle, () => "points", () => { throw new Error("failed"); }, updated);
    expect(() => failed.click()).toThrow("failed"); expect(updated).not.toHaveBeenCalled();
  });
});
