def required_area(duty, U, lmtd_value):
    if U <= 0 or lmtd_value <= 0:
        raise ValueError("U and LMTD must be positive.")
    return duty / (U * lmtd_value)
