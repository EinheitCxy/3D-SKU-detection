export interface NavigationState {
  readonly previousDisabled: boolean;
  readonly nextDisabled: boolean;
}

export function stepVisibleId(visibleIds: readonly string[], selectedId: string | null, direction: -1 | 1): string | null {
  if (selectedId === null) return null;
  const selectedIndex = visibleIds.indexOf(selectedId);
  if (selectedIndex < 0) return null;
  const index = selectedIndex + direction;
  return index >= 0 && index < visibleIds.length ? visibleIds[index] : null;
}

export function navigationState(visibleIds: readonly string[], selectedId: string | null): NavigationState {
  const index = selectedId === null ? -1 : visibleIds.indexOf(selectedId);
  return { previousDisabled: index <= 0, nextDisabled: index < 0 || index >= visibleIds.length - 1 };
}
