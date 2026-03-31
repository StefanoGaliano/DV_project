'''Utilities for color manipulation and gradient generation.'''

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)

'''Generates a list of colors forming a gradient between two hex colors.'''
def gradient(start_hex: str, end_hex: str, steps: int) -> list[str]:
    s = _hex_to_rgb(start_hex)
    e = _hex_to_rgb(end_hex)
    if steps <= 1:
        return [start_hex]
    out = []
    for i in range(steps):
        t = i / (steps - 1)
        rgb = tuple(round(s[c] + (e[c] - s[c]) * t) for c in range(3))
        out.append(_rgb_to_hex(rgb))
    return out