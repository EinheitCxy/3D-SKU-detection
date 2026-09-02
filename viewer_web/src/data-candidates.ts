/** Select either the one requested data root or the conventional local roots. */
export function dataCandidates(locationHref: string): readonly string[] {
  const pageUrl = new URL(locationHref);
  const explicit = pageUrl.searchParams.get("data");
  if (explicit === null) {
    return [...new Set([new URL("./data/", pageUrl).toString(), new URL("/data/", pageUrl).toString()])];
  }
  if (explicit.trim() === "") throw new Error("explicit data root must not be blank");
  const root = new URL(explicit, pageUrl);
  if (root.protocol !== "http:" && root.protocol !== "https:") throw new Error("explicit data root must use http or https");
  const cleanRoot = new URL(root);
  cleanRoot.search = "";
  cleanRoot.hash = "";
  if (root.href !== cleanRoot.href) throw new Error("explicit data root must not contain search or hash");
  const href = root.toString();
  return [href.endsWith("/") ? href : `${href}/`];
}
