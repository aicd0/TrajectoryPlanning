import numpy as np
import random

def soft_update(dst, src, tau):
    for dst_param, src_param in zip(dst.parameters(), src.parameters()):
        dst_param.data.copy_(dst_param.data * (1.0 - tau) + src_param.data * tau)

def hard_update(dst, src):
    for dst_param, src_param in zip(dst.parameters(), src.parameters()):
        dst_param.data.copy_(src_param.data)

def random_point_in_hypersphere(dim: int, low: float=0, high: float=1) -> np.ndarray:
    pt = np.random.uniform(-1, 1, dim)
    pt /= np.linalg.norm(pt)
    pt *= random.uniform(low**dim, high**dim) ** (1/dim)
    return pt