import { describe, expect, it } from "vitest";
import { dataCandidates } from "./data-candidates";

describe("dataCandidates", () => {
  it("returns deduplicated absolute default roots when data is absent", () => {
    expect(dataCandidates("https://viewer.test/")).toEqual(["https://viewer.test/data/"]);
    expect(dataCandidates("https://viewer.test/review/")).toEqual([
      "https://viewer.test/review/data/", "https://viewer.test/data/",
    ]);
  });

  it("returns one absolute explicit root and supports relative paths", () => {
    expect(dataCandidates("https://viewer.test/review/?data=./data/")).toEqual([
      "https://viewer.test/review/data/",
    ]);
    expect(dataCandidates("https://viewer.test/?data=https%3A%2F%2Fcdn.test%2Fbundle")).toEqual([
      "https://cdn.test/bundle/",
    ]);
  });

  it("rejects blank, unsupported, and query/hash explicit roots", () => {
    expect(() => dataCandidates("https://viewer.test/?data=%20")).toThrow(/blank/i);
    expect(() => dataCandidates("https://viewer.test/?data=ftp%3A%2F%2Fcdn.test%2Fbundle")).toThrow(/http/i);
    expect(() => dataCandidates("https://viewer.test/?data=https%3A%2F%2Fcdn.test%2Fbundle%3Fv%3D1")).toThrow(/search|hash/i);
  });

  it("rejects bare and non-empty search/hash delimiters with standard URL semantics", () => {
    for (const suffix of ["%3F", "%3Fv%3D1", "%23", "%23part"]) {
      expect(() => dataCandidates(`https://viewer.test/?data=https%3A%2F%2Fcdn.test%2Fbundle${suffix}`)).toThrow(/search|hash/i);
    }
  });
});
