import math

def lmtd(delta_t1, delta_t2):
    if delta_t1 <= 0 or delta_t2 <= 0:
        raise ValueError("Temperature differences must be positive.")
    if abs(delta_t1 - delta_t2) < 1e-12:
        return delta_t1
    return (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)
