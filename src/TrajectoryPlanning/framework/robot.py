import numpy as np
from abc import abstractmethod

class Robot:
    def __init__(self, joint_limits: np.ndarray) -> None:
        assert all(joint_limits[0] < joint_limits[1])
        self.joint_limits = joint_limits

    @staticmethod
    def _from_dh(a, al, d, th):
        rotation = np.array([
            [th[1], -th[0], 0],
            [th[0] * al[1], th[1] * al[1], -al[0]],
            [th[0] * al[0], th[1] * al[0], al[1]]
        ])
        translation = np.array([
            a,
            -d * al[0],
            d * al[1]
        ])
        return rotation, translation

    @abstractmethod
    def _dh(x: np.ndarray) -> list:
        raise NotImplementedError()

    def origins(self, joint_position: np.ndarray) -> list[np.ndarray]:
        s = np.sin(joint_position)[:, np.newaxis]
        c = np.cos(joint_position)[:, np.newaxis]
        rts = self._dh(np.concatenate((s, c), axis=1))

        origin = np.zeros(3)
        rotation = np.identity(3)
        translation = np.zeros(3)
        ans = []

        for r, t in rts:
            translation += np.matmul(rotation, t)
            rotation = np.matmul(rotation, r)
            ans.append(np.matmul(rotation, origin) + translation)
        return ans

    def clip(self, joint_pos: np.ndarray) -> bool:
        return joint_pos.clip(self.joint_limits[0], self.joint_limits[1])

    @abstractmethod
    def collision_points(self) -> list[np.ndarray]:
        raise NotImplementedError()

# User-defined
class Robot1(Robot):
    __joint_limits = np.array([
        [
            -0.000006,
            -1.571984,
            -1.529065,
            -0.785398,
            -3.053450,
        ], [
            3.053448,
            1.571984,
            1.571069,
            0.785398,
            3.053439,
        ]
    ])

    def __init__(self) -> None:
        super().__init__(Robot1.__joint_limits)

    def _dh(self, x: np.ndarray):
        zero = [0, 1]
        return [
            Robot._from_dh(      0, zero, 0.3215, x[0]),
            Robot._from_dh(-0.1405, x[1],      0, zero),
            Robot._from_dh(      0, zero,  0.408, zero),
            Robot._from_dh( 0.1215, x[2],      0, zero),
            Robot._from_dh(      0, zero,  0.376, zero),
            Robot._from_dh(-0.1025, x[3],      0, zero),
            Robot._from_dh(      0, zero, 0.1025, x[4]),
            Robot._from_dh( -0.144, zero,      0, zero),
        ]

    def collision_points(self, joint_position: np.ndarray) -> list[np.ndarray]:
        return self.origins(joint_position)[1:]