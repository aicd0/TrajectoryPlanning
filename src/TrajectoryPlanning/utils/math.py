def soft_update(dst, src, tau):
    for dst_param, src_param in zip(dst.parameters(), src.parameters()):
        dst_param.data.copy_(dst_param.data * (1.0 - tau) + src_param.data * tau)

def hard_update(dst, src):
    for dst_param, src_param in zip(dst.parameters(), src.parameters()):
        dst_param.data.copy_(src_param.data)