import numpy as np
import utils.math
from abc import abstractmethod

class Geometry:
    def __init__(self, center: np.ndarray, rpy: np.ndarray=np.zeros(3)) -> None:
        self.center = center
        self.rpy = rpy

    @property
    def rpy(self) -> np.ndarray:
        return self.__rpy

    @rpy.setter
    def rpy(self, value) -> None:
        self.__rpy = value
        self.__rotation = utils.math.extrinsic_xyz_rotation(value)

    def __to_subview(self, pos: np.ndarray) -> np.ndarray:
        return self.center + np.matmul(self.__rotation.T, pos - self.center)

    def contain(self, pos: np.ndarray) -> bool:
        pos_tf = self.__to_subview(pos)
        return self._contain(pos_tf)

    def distance(self, pos: np.ndarray) -> float:
        pos_tf = self.__to_subview(pos)
        return self._distance(pos_tf)

    @abstractmethod
    def _contain(self, pos: np.ndarray) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def _distance(self, pos: np.ndarray) -> float:
        raise NotImplementedError()

class Sphere(Geometry):
    def __init__(self, center: np.ndarray, r: float, **kwarg) -> None:
        assert r > 0
        super().__init__(center, **kwarg)
        self.r = r

    def _contain(self, pos: np.ndarray) -> bool:
        d = utils.math.distance(pos, self.center)
        return d <= self.r

    def _distance(self, pos: np.ndarray) -> float:
        d = utils.math.distance(pos, self.center)
        return max(d - self.r, 0)

class Box(Geometry):
    def __init__(self, center: np.ndarray, size: np.ndarray, **kwarg) -> None:
        assert all(size > 0)
        super().__init__(center, **kwarg)
        self.high = center + size/2
        self.low = center - size/2

    def _contain(self, pos: np.ndarray) -> bool:
        return all(pos <= self.high) and all(pos >= self.low)

    def _distance(self, pos: np.ndarray) -> float:
        direction = np.empty(len(pos))
        for i, dim in enumerate(pos):
            direction[i] = max(self.low[i] - dim, dim - self.high[i], 0)
        return np.linalg.norm(direction)

class GeometryModel(Geometry):
    def __init__(self, objs: list[Geometry]=[], center=np.zeros(3), **kwarg) -> None:
        super().__init__(center, **kwarg)
        self.objs: list[Geometry] = objs

    def _contain(self, pos: np.ndarray) -> bool:
        for obj in self.objs:
            if obj.contain(pos):
                return True
        return False

    def _distance(self, pos: np.ndarray) -> float:
        return min([obj.distance(pos) for obj in self.objs])