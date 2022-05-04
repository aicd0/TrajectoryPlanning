def AST_EQUAL_FLT(x: float, y: float, error: float=1e-5):
    assert abs(x - y) <= error