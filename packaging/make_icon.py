"""Draw Aria's app icon: aria/icon.png (512x512).

    python packaging/make_icon.py        # needs Pillow

The icon is drawn from shapes rather than a font, so it comes out the same on
every machine. PyInstaller turns the PNG into the .icns (Mac) and .ico
(Windows) it needs when it builds the apps, and the app's window uses it as is.
"""

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 1024                      # drawn large, then scaled down smoothly
TOP, BOTTOM = (166, 126, 86), (104, 74, 48)     # the page's warm brown
SPARK = (246, 221, 170)


def _tile() -> Image.Image:
    # The rounded square macOS expects: 824 of 1024 pixels, corner radius 185.
    inset, radius = 100, 185
    grad = Image.new("RGB", (SIZE, SIZE))
    d = ImageDraw.Draw(grad)
    for y in range(SIZE):
        t = y / (SIZE - 1)
        d.line([(0, y), (SIZE, y)],
               fill=tuple(round(a + (b - a) * t) for a, b in zip(TOP, BOTTOM)))
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [inset, inset, SIZE - inset, SIZE - inset], radius=radius, fill=255)
    tile = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    tile.paste(grad, (0, 0), mask)
    return tile


def _stroke(d: ImageDraw.ImageDraw, a, b, width: int, fill) -> None:
    d.line([a, b], fill=fill, width=width)
    for x, y in (a, b):                       # round ends
        r = width // 2
        d.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def draw() -> Image.Image:
    img = _tile()
    d = ImageDraw.Draw(img)
    white = (255, 255, 255, 255)
    apex, left, right = (500, 318), (346, 716), (654, 716)
    _stroke(d, apex, left, 92, white)
    _stroke(d, apex, right, 92, white)
    _stroke(d, (420, 566), (580, 566), 70, white)
    # A small four-pointed spark: the learning.
    cx, cy, r, w = 708, 316, 74, 20
    d.polygon([(cx, cy - r), (cx + w, cy - w), (cx + r, cy), (cx + w, cy + w),
               (cx, cy + r), (cx - w, cy + w), (cx - r, cy), (cx - w, cy - w)],
              fill=SPARK + (255,))
    return img.resize((512, 512), Image.LANCZOS)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "aria" / "icon.png"
    draw().save(out, optimize=True)
    print(f"wrote {out}")
