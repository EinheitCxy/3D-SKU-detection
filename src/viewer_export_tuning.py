"""Options for the Viewer export experiments."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ViewerExportTuning:
    """Viewer-only export options under A/B evaluation (defaults = current behavior)."""

    quality_filter: bool = False
    product_spill_trim: bool = False
    background_edge_weights: tuple[int, int, int] = (4, 2, 1)
    single_owner: bool = False
    owner_patch_cells: int = 4
    early_background_candidates: int = 0

    def __post_init__(self):
        if type(self.early_background_candidates) is not int or self.early_background_candidates < 0:
            raise ValueError("early_background_candidates must be a nonnegative integer")
        weights = self.background_edge_weights
        if (not isinstance(weights, tuple) or len(weights) != 3
                or any(type(weight) is not int or not 1 <= weight <= 255 for weight in weights)):
            raise ValueError("background_edge_weights must be three integers in [1, 255]")
        if type(self.owner_patch_cells) is not int or not 2 <= self.owner_patch_cells <= 16:
            raise ValueError("owner_patch_cells must be an integer in [2, 16]")
