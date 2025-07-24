def if_has_sp1(a, b):
    return select({
        "//gnark:zkx_has_sp1": a,
        "//conditions:default": b,
    })
