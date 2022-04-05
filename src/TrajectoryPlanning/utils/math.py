import numpy as np
import random

def soft_update(dst, src, tau):
    for dst_param, src_param in zip(dst.parameters(), src.parameters()):
        dst_param.data.copy_(dst_param.data * (1.0 - tau) + src_param.data * tau)

def hard_update(dst, src):
    for dst_param, src_param in zip(dst.parameters(), src.parameters()):
        dst_param.data.copy_(src_param.data)

def period_map(val: float, low: float, high: float) -> float:
    span = high - low
    rel = val - low
    count = int(rel / span)
    if rel < 0:
        count -= 1
    return val - span * count

def linear_map(val: float, src_low: float, src_high: float, dst_low: float, dst_high) -> float:
    return (val - src_low) / (src_high - src_low) * (dst_high - dst_low) + dst_low

def distance(pt1, pt2):
    return np.linalg.norm(pt1 - pt2)
