import numpy as np

from perf.surfel_coverage.coverage_bounded import select
from perf.surfel_coverage.test_coverage import planes


def test_bounded_exchange_keeps_tile_and_edge_anchors_without_changing_budget():
    data = planes([0, .037], 64, 64)
    data['valid'][:, 28:36, 28:36] = False
    baseline = data['baseline_indices']
    baseline = baseline[data['valid'].ravel()[baseline]]
    data['baseline_indices'] = baseline
    result = select(data, len(baseline))
    selected = result['indices']
    assert len(selected) == len(np.unique(selected)) == len(baseline)
    assert data['valid'].ravel()[selected].all()
    np.testing.assert_array_equal(result['scales'], 4)
    removed = np.setdiff1d(baseline, selected)
    added = np.setdiff1d(selected, baseline)
    assert 0 < len(removed) == len(added) <= len(baseline)//16
    for indices in (baseline, removed):
        frame, rem = np.divmod(indices, 64*64)
        y, x = np.divmod(rem, 64)
        tile = (frame*4+y//16)*4+x//16
        counts = np.bincount(tile, minlength=32)
        if indices is baseline:
            base_counts = counts
        else:
            assert np.all(counts <= base_counts//16)
            assert np.all((x >= 4) & (x < 60) & (y >= 4) & (y < 60))
    # Original observations adjacent to the mask hole cannot be released.
    for dy, dx in ((-4, 0), (4, 0), (0, -4), (0, 4)):
        frame, rem = np.divmod(removed, 64*64)
        y, x = np.divmod(rem, 64)
        assert data['valid'][frame, y+dy, x+dx].all()
    assert result['stats']['removed_disc_triangle_area_m2'] > 0
    assert result['stats']['added_disc_triangle_area_m2'] > 0
