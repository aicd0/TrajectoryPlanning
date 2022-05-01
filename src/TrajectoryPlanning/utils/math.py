import numpy as np

# Common
def period_map(val: float, low: float, high: float) -> float:
    span = high - low
    rel = val - low
    count = int(rel / span)
    if rel < 0:
        count -= 1
    return val - span * count

def linear_map(val: float, src_low: float, src_high: float, dst_low: float, dst_high) -> float:
    return (val - src_low) / (src_high - src_low) * (dst_high - dst_low) + dst_low

def distance(pt1, pt2) -> float:
    return np.linalg.norm(pt1 - pt2)

def distance2(pt1, pt2) -> float:
    return np.sum(np.square(pt1 - pt2))

def manhattan_distance(pt1, pt2) -> float:
    return np.sum(np.abs(pt1 - pt2))

# Random
def random_point_on_hypersphere(dim: int):
    pt = np.random.normal(size=dim)
    pt /= np.linalg.norm(pt)
    return pt

def random_point_in_hypersphere(dim: int, low: float=0, high: float=1) -> np.ndarray:
    pt = random_point_on_hypersphere(dim)
    pt *= np.random.uniform(low ** dim, high ** dim) ** (1 / dim)
    return pt

# Transform
def __x_rotation(x: float) -> np.ndarray:
    sinx = np.sin(x)
    cosx = np.cos(x)
    return np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])

def __y_rotation(y: float) -> np.ndarray:
    siny = np.sin(y)
    cosy = np.cos(y)
    return np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])

def __z_rotation(z: float) -> np.ndarray:
    sinz = np.sin(z)
    cosz = np.cos(z)
    return np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

def extrinsic_xyz_rotation(euler_angles: np.ndarray) -> np.ndarray:
    tx = __x_rotation(euler_angles[0])
    ty = __y_rotation(euler_angles[1])
    tz = __z_rotation(euler_angles[2])
    return np.matmul(tz, np.matmul(ty, tx))

# PyTorch
def soft_update(dst, src, tau) -> None:
    for dst_param, src_param in zip(dst.parameters(), src.parameters()):
        dst_param.data.copy_(dst_param.data * (1.0 - tau) + src_param.data * tau)

def hard_update(dst, src) -> None:
    for dst_param, src_param in zip(dst.parameters(), src.parameters()):
        dst_param.data.copy_(src_param.data)