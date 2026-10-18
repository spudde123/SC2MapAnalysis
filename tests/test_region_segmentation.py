from types import SimpleNamespace

import numpy as np
import pytest
from sc2.position import Point2

from map_analyzer.MapData import MapData


def make_map(placement, heights, ramps=(), blockers=()):
    map_data = MapData.__new__(MapData)
    map_data.placement_arr = placement
    map_data.path_arr = np.ones_like(placement)
    map_data.terrain_height = heights
    map_data.bot = SimpleNamespace(game_info=SimpleNamespace(map_ramps=ramps))
    map_data.resource_blockers = blockers
    map_data._vision_blockers = set()
    map_data.polygons = []
    map_data.regions = {}
    map_data.base_locations = []
    map_data._calc_grid()
    return map_data


@pytest.mark.parametrize("low_height", [63, 66, 239])
def test_enclosed_lower_plateau_is_not_filled(low_height):
    placement = np.zeros((24, 24), dtype=np.uint8)
    placement[2:22, 2:22] = 1
    heights = np.full_like(placement, low_height + 16)
    heights[8:16, 8:16] = low_height
    # A ramp connects the enclosed low ground to the higher ring.
    ramp_points = {Point2((x, 11)) for x in range(6, 9)}
    ramp = SimpleNamespace(points=ramp_points)
    placement[11, 6:9] = 0
    map_data = make_map(placement, heights, ramps=[ramp])

    assert len(map_data.regions_labels) == 2
    assert map_data.region_grid[10, 10] != map_data.region_grid[4, 4]
    assert map_data.region_grid[10, 10] != 0
    assert map_data.region_grid[4, 4] != 0
    for point in ramp_points:
        assert map_data.region_grid[int(point.y), int(point.x)] == 0


def test_hole_cleanup_preserves_ramps_and_resource_blockers():
    placement = np.zeros((24, 24), dtype=np.uint8)
    placement[2:22, 2:22] = 1
    placement[5, 5] = 0  # Small walkable hole can be filled.
    placement[8:15, 8:15] = 0  # Large hole must remain excluded.
    ramp = SimpleNamespace(points={Point2((17, 17))})
    map_data = make_map(
        placement,
        np.full_like(placement, 191),
        ramps=[ramp],
        blockers=[Point2((19, 5))],
    )

    assert map_data.region_grid[5, 5] != 0
    assert np.all(map_data.region_grid[8:15, 8:15] == 0)
    assert map_data.region_grid[17, 17] == 0
    for x, y in [(19, 5), (18, 5), (20, 5), (19, 4), (19, 6)]:
        assert map_data.region_grid[y, x] == 0


def test_large_region_is_retained_but_background_is_not():
    placement = np.zeros((104, 104), dtype=np.uint8)
    placement[1:-1, 1:-1] = 1
    map_data = make_map(placement, np.full_like(placement, 191))
    map_data._calc_regions()
    map_data._clean_polys()

    assert 0 not in map_data.regions_labels
    assert len(map_data.regions) == 1
    region = next(iter(map_data.regions.values()))
    assert region.area > 8500
    assert region in map_data.polygons
    assert region.array[0, 0] == 0


@pytest.mark.parametrize("offset", [0, 1, 2, 8, 16])
def test_small_surface_variations_are_invariant_under_height_offsets(offset):
    placement = np.ones((20, 20), dtype=np.uint8)
    heights = np.full_like(placement, 198)
    heights[:, 10:] = 199
    baseline = make_map(placement, heights)
    shifted = make_map(placement, heights + offset)

    assert len(baseline.regions_labels) == 1
    np.testing.assert_array_equal(shifted.region_grid, baseline.region_grid)


def test_empty_buildable_mask():
    placement = np.zeros((20, 20), dtype=np.uint8)
    map_data = make_map(placement, np.full_like(placement, 191))

    assert map_data.regions_labels.size == 0
    assert not np.any(map_data.region_grid)
