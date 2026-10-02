import random

import numpy as np
import pytest

from map_analyzer.Pather import _draw_circle, draw_circle


def test_memoized_disk_matches_uncached():
    rng = random.Random(0)
    shape = (176, 184)
    for _ in range(500):
        c = (rng.uniform(-3, 180), rng.uniform(-3, 188))
        r = rng.uniform(0.3, 14)
        expected = _draw_circle(c, r, shape)
        first = draw_circle(c, r, shape)
        again = draw_circle(c, r, shape)
        for e, f in zip(expected, first):
            assert np.array_equal(e, f)
        assert again is first  # repeat calls are served from the cache


def test_memoized_disk_is_read_only():
    rr, cc = draw_circle((50.5, 60.25), 6.0, (176, 184))
    with pytest.raises(ValueError):
        rr += 1
    grid = np.ones((176, 184))
    grid[(rr, cc)] += 5  # indexing a grid with it is unaffected
    assert grid.sum() == 176 * 184 + 5 * len(rr)


def test_no_shape_is_not_cached():
    rr, cc = draw_circle((10.0, 10.0), 3.0)
    rr += 1  # a fresh, writable result
