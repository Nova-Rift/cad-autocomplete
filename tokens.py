"""How a sketch becomes a row of numbers (tokens), shared by every script.

A sketch is a list of shapes in the order the person drew them. Each shape is one "type" token
followed by its coordinates on a 64 x 64 grid:

    line    x1 y1 x2 y2            (start, end)
    circle  cx cy r
    arc     x1 y1 xm ym x2 y2      (start, middle of the arc, end)
    point   x y

Construction shapes (helper geometry such as centre lines) get their own four type tokens.
"""
PAD, BOS, EOS = 0, 1, 2
LINE, CIRCLE, ARC, POINT = 3, 4, 5, 6
CONSTRUCTION = 4                      # add to a type token for its construction version (7..10)
FIRST_TYPE, LAST_TYPE = 3, 10
COORD0 = 11                           # coordinate value v is token COORD0 + v
BINS = 64
VOCAB = COORD0 + BINS                 # 75
BLOCK = 128                           # longest row: 16 shapes x 7 tokens + start + end = 114
N_COORDS = {LINE: 4, CIRCLE: 3, ARC: 6, POINT: 2}
NAMES = {LINE: "line", CIRCLE: "circle", ARC: "arc", POINT: "point"}


def base_type(tok):
    return tok - CONSTRUCTION if tok > POINT else tok


def is_type(tok):
    return FIRST_TYPE <= tok <= LAST_TYPE


def n_coords(tok):
    return N_COORDS[base_type(tok)]


def split_entities(row):
    """Token row -> list of (type_token, [coordinate values])."""
    out, i, row = [], 0, [int(t) for t in row]
    if row and row[0] == BOS:
        i = 1
    while i < len(row) and is_type(row[i]):
        n = n_coords(row[i])
        out.append((row[i], [t - COORD0 for t in row[i + 1:i + 1 + n]]))
        i += 1 + n
    return out


def canonical(ent):
    """Same shape drawn in the other direction counts as the same shape."""
    t, c = ent
    b = base_type(t)
    if b == LINE:
        a, z = (c[0], c[1]), (c[2], c[3])
        if z < a:
            c = [*z, *a]
    elif b == ARC:
        a, z = (c[0], c[1]), (c[4], c[5])
        if z < a:
            c = [*z, c[2], c[3], *a]
    return (t, tuple(c))
