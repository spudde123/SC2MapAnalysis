from functools import lru_cache
from typing import List, Tuple, TYPE_CHECKING, Union

import numpy as np
from numpy import ndarray
from sc2.position import Point2
from scipy.ndimage import center_of_mass

if TYPE_CHECKING:
    pass


class Polygon:
    """
    Polygon DocString
    """

    def __init__(self, map_data, array):
        # type: ("MapData", ndarray) -> None
        self.map_data = map_data
        self.array = array
        self.indices = np.where(self.array == 1)
        points = map_data.indices_to_points(self.indices)
        self.points = set([Point2(p) for p in points])

    def plot(self):
        import matplotlib.pyplot as plt

        plt.style.use("ggplot")
        plt.imshow(self.array, origin="lower")
        plt.show()

    @property
    @lru_cache()
    def nodes(self):
        return [p for p in self.points]

    @property
    @lru_cache()
    def corner_array(self):
        from skimage.feature import corner_harris, corner_peaks

        array = corner_peaks(
                corner_harris(self.array), min_distance=3, threshold_rel=0.01
        )
        return array


    @property
    @lru_cache()
    def corner_points(self):
        points = [Point2(p) for p in self.corner_array]
        return points

    @property
    @lru_cache()
    def region(self):
        return self.map_data.in_region(self.center)

    @property
    def center(self):
        # type: () -> Tuple[int, int]
        cm = center_of_mass(self.array)
        real_center_idx = self.map_data._closest_node_idx(cm, self.nodes)
        return self.nodes[real_center_idx]

    @lru_cache(100)
    def is_inside_point(self, point: Union[Point2, Tuple]) -> bool:
        if point in self.points:
            return True
        if isinstance(point, Point2):
            point = point.rounded
        return point in self.points

    @lru_cache(100)
    def is_inside_indices(self, point: Union[Point2, Tuple]) -> bool:
        if isinstance(point, Point2):
            point = point.rounded
        return point[0] in self.indices[0] and point[1] in self.indices[1]

    @property
    def perimeter(self) -> np.ndarray:
        isolated_region = self.array
        xx, yy = np.gradient(isolated_region)
        edge_indices = np.argwhere(xx ** 2 + yy ** 2 > 0.1)
        return edge_indices

    @property
    def area(self):
        # type: () -> int
        return len(self.points)

    @property
    def get_holes(self) -> List[Tuple]:
        # fly zones inside the Polygon
        pass
