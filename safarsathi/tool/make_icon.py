#!/usr/bin/env python3
"""Draws the SafarSathi launcher icon and writes every Android density.

    python3 tool/make_icon.py

Rail, road and air on ONE line. The merge is the line itself: it carries
sleepers on the left half, so it is a railway there, and runs clean on the
right, where it is a road. A train sits on the rail half, a car on the road
half, and a plane lifts off the right-hand end. Three ways of travelling, one
journey — which is what the app is about.

Drawn at 4x and downsampled, because PIL has no antialiasing of its own and a
192px icon drawn directly comes out with stepped edges.
"""

import math
import os
from PIL import Image, ImageDraw

# The icon does NOT use the app's green. It is the one thing seen next to
# thirty other icons on a home screen, so it is allowed to shout where the
# app itself stays quiet: a deep violet falling to magenta across the
# diagonal, with the marks in the app's own paper so the two still belong to
# each other.
VIOLET = (76, 29, 149, 255)    # top-left
MAGENTA = (199, 33, 141, 255)  # bottom-right
PAPER = (250, 247, 240, 255)   # from app_tokens.dart, unchanged

# Everything knocked out of the marks samples the ground behind it, so the
# gradient shows through rather than being painted over in a flat colour.
GREEN = None  # replaced by knockout; see _cut()

S = 768                      # working canvas; 4x the largest density
DENSITIES = {
    "mdpi": 48,
    "hdpi": 72,
    "xhdpi": 96,
    "xxhdpi": 144,
    "xxxhdpi": 192,
}


def rotate(points, degrees, origin):
    """Rotates points about origin. Negative degrees tilt up to the right."""
    rad = math.radians(degrees)
    cos, sin = math.cos(rad), math.sin(rad)
    ox, oy = origin
    return [
        (
            ox + (x - ox) * cos - (y - oy) * sin,
            oy + (x - ox) * sin + (y - oy) * cos,
        )
        for x, y in points
    ]


def plane(cx, cy, scale, tilt):
    """A plane seen from above, nose along +x before the tilt is applied."""
    half = [
        (150, 0),      # nose
        (112, 13),
        (34, 17),
        (17, 96),      # wing, leading edge
        (-16, 99),     # wingtip
        (-20, 21),
        (-64, 17),
        (-94, 53),     # tailplane
        (-118, 53),
        (-113, 11),
        (-142, 7),     # tail
    ]
    outline = half + [(x, -y) for x, y in reversed(half)]
    scaled = [(cx + x * scale, cy + y * scale) for x, y in outline]
    return rotate(scaled, tilt, (cx, cy))


def _gradient() -> Image.Image:
    """Violet to magenta, corner to corner."""
    ramp = Image.new("RGBA", (S, S))
    px = ramp.load()
    for y in range(S):
        for x in range(S):
            # Diagonal position, 0 at the top-left corner and 1 at the far one.
            t = (x + y) / (2 * (S - 1))
            px[x, y] = tuple(
                int(a + (b - a) * t) for a, b in zip(VIOLET, MAGENTA)
            )
    return ramp


def draw() -> Image.Image:
    # The marks are drawn into a mask rather than onto the ground, so a
    # knockout (a window, a wheel hub) reveals the gradient underneath
    # instead of a flat colour that would band against it.
    mask = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(mask)


    # --- the line: railway on the left, road on the right ----------------
    line_y = 528
    line_h = 20
    d.rounded_rectangle(
        [128, line_y, 640, line_y + line_h], radius=line_h // 2, fill=255
    )

    # Sleepers, left half only. The asymmetry is what says "rail, then road".
    for x in (158, 238, 318):
        d.rounded_rectangle(
            [x, line_y + line_h + 10, x + 26, line_y + line_h + 48],
            radius=12,
            fill=255,
        )

    # --- train, front three-quarter, on the rail half --------------------
    d.rounded_rectangle([170, 332, 340, line_y - 6], radius=34, fill=255)
    # Windows knocked out of the body.
    d.rounded_rectangle([196, 366, 246, 420], radius=12, fill=0)
    d.rounded_rectangle([264, 366, 314, 420], radius=12, fill=0)
    # The grille band across its face.
    d.rounded_rectangle([196, 450, 314, 476], radius=11, fill=0)

    # --- car, side view, on the road half --------------------------------
    body = [
        (392, 502),
        (392, 456),
        (432, 456),
        (466, 404),
        (556, 404),
        (578, 456),
        (614, 462),
        (620, 502),
    ]
    d.polygon(body, fill=255)
    d.rounded_rectangle([392, 462, 620, 502], radius=14, fill=255)
    # Window, knocked out.
    d.polygon([(446, 452), (472, 418), (516, 418), (516, 452)], fill=0)
    d.polygon([(530, 418), (552, 418), (570, 452), (530, 452)], fill=0)
    # Wheels sit on the line.
    for wx in (444, 578):
        d.ellipse([wx - 32, line_y - 34, wx + 32, line_y + 30], fill=255)
        d.ellipse([wx - 13, line_y - 15, wx + 13, line_y + 11], fill=0)

    # --- plane, lifting away from the right-hand end ---------------------
    d.polygon(plane(cx=458, cy=248, scale=1.0, tilt=-30), fill=255)

    # Compose: gradient ground, paper marks, rounded-square silhouette.
    ground = _gradient()
    img = Image.composite(Image.new("RGBA", (S, S), PAPER), ground, mask)

    corner = Image.new("L", (S, S), 0)
    ImageDraw.Draw(corner).rounded_rectangle(
        [0, 0, S - 1, S - 1], radius=int(S * 0.225), fill=255
    )
    img.putalpha(corner)
    return img


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    res = os.path.join(here, "..", "android", "app", "src", "main", "res")

    master = draw()
    for folder, size in DENSITIES.items():
        out = os.path.join(res, f"mipmap-{folder}", "ic_launcher.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        master.resize((size, size), Image.LANCZOS).save(out)
        print(f"wrote {out} ({size}x{size})")

    preview = os.path.join(here, "icon_preview.png")
    master.resize((384, 384), Image.LANCZOS).save(preview)
    print(f"wrote {preview}")


if __name__ == "__main__":
    main()
