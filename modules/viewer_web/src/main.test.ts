import { describe, expect, it } from "vitest";
import { bootstrap } from "./main";

class FakeElement {
  constructor(readonly tagName: string) {}
  className = "";
  textContent = "";
  readonly children: FakeElement[] = [];
  append(...children: FakeElement[]): void { this.children.push(...children); }
  replaceChildren(...children: FakeElement[]): void { this.children.splice(0, this.children.length, ...children); }
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
