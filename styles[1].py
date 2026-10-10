"""Amharic text styles - procedural effects (numpy + Pillow). No downloads needed."""
import io
import os
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

W = H = 900
HERE = os.path.dirname(os.path.abspath(__file__))
FONT_CANDIDATES = [
    os.path.join(HERE, "font.ttf"),
    "C:/Windows/Fonts/nyala.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
]
Yp, Xp = np.mgrid[0:H, 0:W].astype(np.float32)
Yn, Xn = Yp / H, Xp / W


def _font_path():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Put an Ethiopic font at font.ttf")


# ---------------------------------------------------------------- text mask
def _wrap(text, font, max_w, probe):
    lines, cur = [], ""
    for word in text.split():
        t = (cur + " " + word).strip()
        if probe.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines or [text]


def _text_mask(text):
    path = _font_path()
    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    size = 230
    while True:
        font = ImageFont.truetype(path, size)
        lines = _wrap(text, font, W - 120, probe)
        lh = int(size * 1.18)
        widest = max(probe.textlength(l, font=font) for l in lines)
        if (widest <= W - 100 and lh * len(lines) <= 460) or size <= 64:
            break
        size -= 8
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    y = max(80, (H - lh * len(lines)) // 2 - 70)
    for l in lines:
        w = probe.textlength(l, font=font)
        d.text(((W - w) / 2, y), l, font=font, fill=255, stroke_width=4, stroke_fill=255)
        y += lh
    return mask


# ------------------------------------------------------------------ helpers
def A(img):
    return np.asarray(img, np.float32) / 255.0


def col(c):
    return np.array(c, np.float32) / 255.0


def blur(a, r):
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return A(im.filter(ImageFilter.GaussianBlur(r)))


def grow(m, px):
    if px <= 0:
        return m
    return np.clip((blur(m, px * 0.8) - 0.08) / 0.06, 0, 1)


def shrink(m, px):
    return 1 - grow(1 - m, px)


def over(base, color, alpha):
    a = alpha[..., None]
    return base * (1 - a) + np.asarray(color, np.float32) * a


def add(base, color, alpha):
    return np.clip(base + np.asarray(color, np.float32) * alpha[..., None], 0, 1)


def ramp(t, stops):
    pos = [p for p, _ in stops]
    out = np.empty(t.shape + (3,), np.float32)
    for k in range(3):
        out[..., k] = np.interp(t, pos, [s[k] for _, s in stops])
    return out / 255.0


def roll(a, dy, dx):
    return np.roll(np.roll(a, dy, 0), dx, 1)


def layer_from(img):
    return A(img)


def bevel(o, m, d=4, hi=0.7, lo=0.5):
    h = blur(np.clip(m - roll(m, d, d), 0, 1), 1.5)
    l = blur(np.clip(m - roll(m, -d, -d), 0, 1), 1.5)
    o = over(o, (1, 1, 1), h * hi * m)
    return over(o, (0, 0, 0), l * lo * m)


# ------------------------------------------------------------------ context
class Ctx:
    def __init__(self, text, photo=None, seed=None):
        seed = random.randrange(1 << 30) if seed is None else seed
        self.rng = np.random.default_rng(seed)
        self.rnd = random.Random(seed)
        self.mask = _text_mask(text)
        self.m = A(self.mask)
        ys, xs = np.nonzero(self.m > 0.5)
        self.top, self.bot, self.left, self.right = ys.min(), ys.max(), xs.min(), xs.max()
        self.yb = np.clip((Yp - self.top) / max(1, self.bot - self.top), 0, 1)
        self.xb = np.clip((Xp - self.left) / max(1, self.right - self.left), 0, 1)
        self.photo = None
        if photo:
            p = Image.open(io.BytesIO(photo)).convert("RGB")
            self.photo = A(ImageOps.fit(p, (W, H)))

    def bg(self, builder, dim=0.55):
        """Style's own background, or the user's photo (dimmed) if one was sent."""
        if self.photo is not None:
            return self.photo * dim
        return builder()

    def fbm(self, octaves=5, base=4):
        acc = np.zeros((H, W), np.float32)
        amp, tot = 1.0, 0.0
        for o in range(octaves):
            n = base * 2 ** o
            small = (self.rng.random((n, n)) * 255).astype(np.uint8)
            big = A(Image.fromarray(small).resize((W, H), Image.BICUBIC))
            acc += big * amp
            tot += amp
            amp *= 0.5
        acc /= tot
        lo, hi = acc.min(), acc.max()
        return (acc - lo) / (hi - lo + 1e-6)

    def points_in(self, n):
        ys, xs = np.nonzero(self.m > 0.6)
        idx = self.rng.integers(0, len(ys), n)
        return [(int(xs[i]), int(ys[i])) for i in idx]


# -------------------------------------------------------------- backgrounds
def bg_dark(tint=(28, 10, 12)):
    d = np.clip(np.sqrt((Xn - .5) ** 2 + (Yn - .5) ** 2) / 0.72, 0, 1)
    return ramp(d ** 1.4, [(0, tint), (1, (0, 0, 0))])


def bg_solid(c):
    return np.ones((H, W, 3), np.float32) * col(c)


def bg_grad(stops):
    return ramp(Yn, stops)


def bg_paper(c, base=(244, 236, 220)):
    n = c.fbm(3, 40)
    return np.clip((0.93 + 0.12 * n)[..., None] * col(base), 0, 1)


def stars_layer(c, n=260):
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for _ in range(n):
        x, y = c.rnd.randrange(W), c.rnd.randrange(H)
        r = c.rnd.choice([1, 1, 1, 2, 2, 3])
        v = c.rnd.randint(110, 255)
        d.ellipse([x - r, y - r, x + r, y + r], fill=v)
    s = A(img)
    return np.clip(s + blur(s, 3) * 1.2, 0, 1)


def sparkle_layer(points, size, rnd):
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for x, y in points:
        r = rnd.randint(*size)
        k = r * 0.14
        d.polygon([(x, y - r), (x + k, y - k), (x + r, y), (x + k, y + k),
                   (x, y + r), (x - k, y + k), (x - r, y), (x - k, y - k)], fill=255)
    return A(img)


def drips_mask(c, lmin=40, lmax=200, wmin=9, wmax=17, tri=False):
    has = c.m > 0.5
    valid = has.any(axis=0)
    idx = H - 1 - np.argmax(has[::-1], axis=0)
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    x = int(c.left)
    while x < c.right:
        if valid[x] and c.rnd.random() < 0.9:
            w = c.rnd.randint(wmin, wmax)
            ln = c.rnd.randint(lmin, lmax)
            yb = int(idx[x]) - 5
            if tri:
                d.polygon([(x, yb), (x + w, yb), (x + w / 2, yb + ln)], fill=255)
            else:
                d.rectangle([x, yb, x + w, yb + ln], fill=255)
                d.ellipse([x - 1, yb + ln - w // 2 - 2, x + w + 1, yb + ln + w // 2 + 7], fill=255)
            x += w + c.rnd.randint(14, 50)
        else:
            x += 5
    return A(img.filter(ImageFilter.GaussianBlur(1)))


def bolts_layer(c, n=8):
    edge = (c.m - shrink(c.m, 3)) > 0.5
    pts = np.argwhere(edge)
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for _ in range(n):
        y, x = pts[c.rnd.randrange(len(pts))]
        ang = c.rnd.uniform(0, 2 * math.pi)
        line = [(float(x), float(y))]
        for _k in range(c.rnd.randint(8, 16)):
            ang += c.rnd.uniform(-0.8, 0.8)
            step = c.rnd.randint(14, 26)
            x += math.cos(ang) * step
            y += math.sin(ang) * step
            line.append((float(x), float(y)))
        d.line(line, fill=255, width=3)
    return A(img)


def facets(c, cell, stops):
    ny, nx = H // cell + 3, W // cell + 3
    PX = (np.arange(nx)[None, :] - 1 + c.rng.random((ny, nx))) * cell
    PY = (np.arange(ny)[:, None] - 1 + c.rng.random((ny, nx))) * cell
    V = c.rng.random((ny, nx)).astype(np.float32)
    ci = (Xp // cell).astype(np.int32)
    cj = (Yp // cell).astype(np.int32)
    d1 = np.full((H, W), 1e9, np.float32)
    d2 = np.full((H, W), 1e9, np.float32)
    val = np.zeros((H, W), np.float32)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            jj, ii = cj + dj + 1, ci + di + 1
            d = np.sqrt((Xp - PX[jj, ii]) ** 2 + (Yp - PY[jj, ii]) ** 2)
            closer = d < d1
            d2 = np.where(closer, d1, np.minimum(d2, d))
            val = np.where(closer, V[jj, ii], val)
            d1 = np.where(closer, d, d1)
    fill = ramp(val, stops)
    edge = np.clip(1 - (d2 - d1) / 5, 0, 1)
    return over(fill, (1, 1, 1), edge * 0.6)


def stack(c, o, fill, outline=None, ow=8, glow=None, shadow=None, bev=0, shape=None):
    m = c.m if shape is None else shape
    if glow:
        for color, r, op in glow:
            o = over(o, col(color), blur(m, r) * op)
    if shadow:
        color, dx, dy, b, op = shadow
        o = over(o, col(color), blur(roll(m, dy, dx), b) * op)
    if outline:
        o = over(o, col(outline), grow(m, ow))
    if isinstance(fill, tuple):
        fill = col(fill)
    o = over(o, fill, m)
    if bev:
        o = bevel(o, m, bev)
    return o


# ----------------------------------------------------------------- registry
STYLES = {}
CATEGORIES = [
    ("light", "💡 ብርሃን"),
    ("metal", "🥇 ብረትና ድንጋይ"),
    ("nature", "🔥 እሳትና ተፈጥሮ"),
    ("sky", "🌌 ሰማይ"),
    ("art", "🎨 ጥበብ"),
]


def style(key, label, cat):
    def deco(fn):
        STYLES[key] = (label, cat, fn)
        return fn
    return deco


# =========================================================== 💡 LIGHT
def _neon(c, color, core=(255, 255, 255), bg=(8, 8, 18), shape=None):
    m = c.m if shape is None else shape
    o = c.bg(lambda: bg_dark(bg))
    for r, op in [(60, .5), (28, .7), (12, .9)]:
        o = over(o, col(color), blur(m, r) * op)
    o = over(o, col(color), grow(m, 3))
    return over(o, col(core), shrink(m, 4))


@style("neon_blue", "💙 ሰማያዊ ኒዮን", "light")
def s_neon_blue(c): return _neon(c, (0, 170, 255))


@style("neon_green", "💚 አረንጓዴ ኒዮን", "light")
def s_neon_green(c): return _neon(c, (40, 255, 90))


@style("neon_pink", "💗 ሮዝ ኒዮን", "light")
def s_neon_pink(c): return _neon(c, (255, 40, 170))


@style("neon_purple", "💜 ሐምራዊ ኒዮን", "light")
def s_neon_purple(c): return _neon(c, (170, 60, 255))


@style("glow", "⚪ ነጭ ብርሃን", "light")
def s_glow(c):
    o = c.bg(lambda: bg_dark((14, 14, 14)))
    for r, op in [(50, .5), (22, .7), (8, .9)]:
        o = over(o, (1, 1, 1), blur(c.m, r) * op)
    return over(o, (1, 1, 1), c.m)


@style("toxic", "🧪 መርዛማ አረንጓዴ", "light")
def s_toxic(c):
    shape = np.maximum(c.m, drips_mask(c, 30, 140, 8, 14))
    return _neon(c, (150, 255, 30), (235, 255, 170), (5, 18, 0), shape)


@style("radio", "☢️ ራዲዮአክቲቭ", "light")
def s_radio(c):
    o = c.bg(lambda: bg_dark((15, 25, 0)))
    for r, op in [(70, .6), (30, .8)]:
        o = over(o, col((190, 255, 0)), blur(c.m, r) * op)
    o = over(o, col((200, 255, 0)), grow(c.m, 6))
    return over(o, col((10, 10, 0)), c.m)


@style("cyber", "🌆 ሳይበርፐንክ", "light")
def s_cyber(c):
    def bg():
        o = bg_dark((26, 0, 48))
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        hz = int(H * 0.66)
        for k in range(1, 14):
            y = hz + int((H - hz) * (k / 13) ** 2)
            d.line([(0, y), (W, y)], fill=200, width=2)
        for k in range(-14, 15):
            d.line([(W / 2 + k * 12, hz), (W / 2 + k * 90, H)], fill=200, width=2)
        return add(o, (1, 0.1, 0.8), blur(A(img), 1) * 0.55)
    o = c.bg(bg)
    fill = ramp(c.yb, [(0, (170, 255, 252)), (1, (0, 170, 255))])
    return stack(c, o, fill, outline=(10, 0, 30), ow=4,
                 glow=[((0, 220, 255), 30, .6)], shadow=((255, 0, 170), -9, 6, 0.6, 1.0))


@style("glitch", "📺 የተሰበረ ቲቪ", "light")
def s_glitch(c):
    o = c.bg(lambda: bg_solid((6, 6, 10)))
    r, g, b = roll(c.m, 0, -9), c.m.copy(), roll(c.m, 0, 9)
    for _ in range(9):
        y0 = c.rnd.randint(int(c.top), max(int(c.top) + 1, int(c.bot) - 20))
        h = c.rnd.randint(8, 34)
        dx = c.rnd.randint(-45, 45)
        r[y0:y0 + h] = np.roll(r[y0:y0 + h], dx, 1)
        b[y0:y0 + h] = np.roll(b[y0:y0 + h], -dx, 1)
    o = np.clip(o + np.stack([r, g * 0.9, b], -1), 0, 1)
    o = add(o, (.2, .5, 1), blur(c.m, 22) * 0.35)
    return o * (0.8 + 0.2 * ((Yp % 4) < 2))[..., None]


def _bolt_style(c, color, text_color, bg):
    o = c.bg(lambda: bg_dark(bg))
    bl = bolts_layer(c, 9)
    o = over(o, col(color), blur(c.m, 30) * .7)
    o = add(o, col(color), blur(bl, 6))
    o = add(o, (1, 1, 1), bl)
    o = over(o, col(color), grow(c.m, 4))
    return over(o, col(text_color), c.m)


@style("lightning", "⚡ መብረቅ", "light")
def s_lightning(c): return _bolt_style(c, (90, 150, 255), (238, 246, 255), (6, 8, 30))


@style("plasma", "🔮 ሐምራዊ ፕላዝማ", "light")
def s_plasma(c): return _bolt_style(c, (200, 80, 255), (255, 235, 255), (16, 4, 30))


@style("matrix", "💻 ማትሪክስ", "light")
def s_matrix(c):
    def bg():
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        f = ImageFont.truetype(_font_path(), 20)
        for x in range(0, W, 22):
            y, n = c.rnd.randint(0, H + 200), c.rnd.randint(8, 30)
            for i in range(n):
                v = int(255 * (1 - i / n) ** 1.6 * 0.75)
                d.text((x, y - i * 22), c.rnd.choice("01ABCDEFXYZ"), font=f, fill=v)
        return add(bg_solid((0, 4, 0)), (0, 1, .3), A(img))
    o = c.bg(bg)
    o = over(o, (0, .9, .25), blur(c.m, 24) * .7)
    o = over(o, (0, .1, .03), grow(c.m, 6))
    return over(o, col((190, 255, 200)), c.m)


# =========================================================== 🥇 METAL / STONE
def _metal(c, stops, outline, glow, bg, tex=0.0, sparkle=False):
    o = c.bg(lambda: bg_dark(bg))
    fill = ramp(c.yb, stops)
    if tex:
        fill = np.clip(fill * (1 - tex + 2 * tex * c.fbm(4, 30))[..., None], 0, 1)
    o = stack(c, o, fill, outline=outline, ow=7, glow=glow, bev=4)
    if sparkle:
        sp = sparkle_layer(c.points_in(30), (10, 28), c.rnd)
        o = add(o, (1, 1, .8), blur(sp, 3) * .9)
        o = over(o, (1, 1, 1), sp)
    return o


GOLD = [(0, (255, 246, 175)), (.35, (244, 190, 45)), (.5, (150, 95, 10)), (.65, (255, 214, 90)), (1, (205, 140, 30))]


@style("gold", "🥇 ወርቅ", "metal")
def s_gold(c): return _metal(c, GOLD, (60, 35, 0), [((255, 190, 40), 26, .35)], (20, 14, 4))


@style("gold_glitter", "✨ የሚያንፀባርቅ ወርቅ", "metal")
def s_gold_glitter(c): return _metal(c, GOLD, (60, 35, 0), [((255, 190, 40), 30, .45)], (20, 14, 4), sparkle=True)


@style("silver", "🥈 ብር", "metal")
def s_silver(c):
    return _metal(c, [(0, (255, 255, 255)), (.4, (185, 190, 200)), (.52, (90, 95, 110)), (.7, (225, 228, 235)), (1, (150, 155, 165))],
                  (40, 42, 50), [((200, 210, 255), 24, .3)], (14, 14, 20))


@style("chrome", "🪞 ፈሳሽ ብረት", "metal")
def s_chrome(c):
    return _metal(c, [(0, (255, 255, 255)), (.35, (150, 185, 235)), (.5, (25, 40, 80)), (.52, (235, 240, 250)), (.75, (120, 130, 150)), (1, (240, 245, 255))],
                  (10, 14, 30), [((120, 160, 255), 26, .3)], (8, 10, 22))


@style("copper", "🪙 ነሐስ", "metal")
def s_copper(c):
    return _metal(c, [(0, (255, 205, 160)), (.4, (205, 115, 60)), (.55, (120, 55, 25)), (.75, (220, 135, 80)), (1, (150, 75, 35))],
                  (50, 20, 5), [((255, 130, 60), 24, .3)], (22, 10, 4), tex=0.06)


def _facet_style(c, cell, stops, glowc, outline, bg, sparkle=True, shape=None):
    o = c.bg(lambda: bg_dark(bg))
    fill = facets(c, cell, stops)
    o = stack(c, o, fill, outline=outline, ow=5, glow=[(glowc, 28, .5)], bev=3, shape=shape)
    if sparkle:
        sp = sparkle_layer(c.points_in(14), (14, 34), c.rnd)
        o = add(o, (.9, 1, 1), blur(sp, 3) * .9)
        o = over(o, (1, 1, 1), sp)
    return o


@style("diamond", "💎 አልማዝ", "metal")
def s_diamond(c):
    return _facet_style(c, 34, [(0, (200, 240, 255)), (.4, (255, 255, 255)), (.7, (120, 200, 255)), (1, (170, 150, 255))],
                        (120, 200, 255), (40, 70, 130), (6, 10, 24))


@style("turquoise", "💠 ቱርኮይዝ", "metal")
def s_turquoise(c):
    return _facet_style(c, 30, [(0, (0, 150, 140)), (.5, (40, 230, 200)), (1, (200, 255, 245))],
                        (40, 230, 200), (0, 70, 70), (2, 18, 18))


@style("granite", "🪨 ግራናይት", "metal")
def s_granite(c):
    o = c.bg(lambda: bg_dark((24, 24, 28)))
    g = 0.30 + 0.45 * c.fbm(4, 60) + 0.22 * c.rng.random((H, W)).astype(np.float32)
    g = np.where(c.rng.random((H, W)) > 0.975, g * 0.3, g)
    g = np.where(c.rng.random((H, W)) > 0.975, np.minimum(g + 0.5, 1), g)
    g = np.clip(blur(g, 0.7), 0, 1)
    return stack(c, o, ramp(np.clip((g - 0.1) * 1.3, 0, 1), [(0, (92, 70, 66)), (.5, (150, 140, 138)), (1, (232, 224, 220))]),
                 outline=(25, 25, 28), ow=5, bev=3)


@style("moon", "🌑 የጨረቃ ግርዶሽ", "metal")
def s_moon(c):
    o = c.bg(lambda: add(bg_solid((2, 2, 8)), (1, 1, 1), stars_layer(c, 220) * .8))
    base = ramp(0.35 + 0.4 * c.fbm(4, 8), [(0, (110, 110, 118)), (1, (225, 225, 230))])
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for _ in range(110):
        x, y, r = c.rnd.randrange(W), c.rnd.randrange(H), c.rnd.randint(6, 26)
        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    cr = A(img)
    base = over(base, (.22, .22, .25), blur(cr, 1) * .35)
    base = over(base, (1, 1, 1), blur(np.clip(roll(cr, 4, 4) - cr, 0, 1), 1) * .5)
    return stack(c, o, base, outline=(40, 40, 48), ow=4, glow=[((200, 210, 255), 30, .4)], bev=3)


@style("glass", "🔍 ግልጽ መስታወት", "metal")
def s_glass(c):
    o = c.bg(lambda: bg_grad([(0, (20, 40, 110)), (1, (110, 30, 120))]), dim=0.8)
    m = c.m
    o = over(o, (0, 0, 0), blur(roll(m, 10, 8), 8) * .35)
    o = over(o, (1, 1, 1), m * .24)
    rim = np.clip(m - shrink(m, 3), 0, 1)
    o = over(o, (1, 1, 1), np.clip(blur(rim, 1) * 1.4, 0, 1))
    hl = np.clip(blur(shrink(m, 14), 6) * (1 - c.yb) * 1.2, 0, 1) * m
    return over(o, (1, 1, 1), hl * .45)


# =========================================================== 🔥 NATURE
@style("fire", "🔥 እሳት", "nature")
def s_fire(c):
    o = c.bg(lambda: bg_dark((25, 6, 0)))
    flame = np.clip(blur(roll(c.m, -45, 0), 16) * (0.4 + c.fbm(4, 6)), 0, 1)
    o = over(o, col((255, 100, 0)), flame * .85)
    o = over(o, col((255, 140, 0)), blur(c.m, 18) * .6)
    fill = ramp(c.yb, [(0, (255, 252, 170)), (.45, (255, 180, 25)), (.7, (240, 70, 0)), (1, (150, 10, 0))])
    o = stack(c, o, fill, outline=(70, 8, 0), ow=5)
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for _ in range(80):
        x = c.rnd.randint(int(c.left), int(c.right))
        y = c.rnd.randint(max(0, int(c.top) - 260), int(c.bot))
        r = c.rnd.randint(1, 3)
        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    sp = A(img)
    return add(o, (1, .6, .1), np.clip(sp + blur(sp, 3), 0, 1))


@style("lava", "🌋 የእሳተ ገሞራ ላቫ", "nature")
def s_lava(c):
    o = c.bg(lambda: bg_dark((40, 8, 0)))
    n = c.fbm(5, 4)
    crack = np.clip(1 - np.abs(n - 0.5) * 6.5, 0, 1) ** 1.5
    crust = ramp(0.25 + 0.5 * c.fbm(4, 24), [(0, (25, 15, 12)), (1, (80, 52, 40))])
    fill = over(crust, col((255, 150, 20)), crack)
    o = over(o, col((255, 90, 0)), blur(c.m, 28) * .7)
    return stack(c, o, fill, outline=(20, 8, 4), ow=5, bev=3)


@style("blood", "🩸 ደም የሚንጠባጠብ", "nature")
def s_blood(c):
    shape = np.maximum(c.m, drips_mask(c, 40, 230, 10, 20))
    shape = np.clip((blur(shape, 1.2) - .4) * 5, 0, 1)
    o = c.bg(lambda: bg_dark((30, 10, 12)))
    fill = ramp(c.yb, [(0, (240, 20, 25)), (.5, (190, 0, 10)), (1, (90, 0, 0))])
    o = stack(c, o, fill, outline=(35, 0, 0), ow=5, glow=[((170, 0, 0), 28, .8)], shape=shape)
    hi = blur(np.clip(shape - roll(shape, 8, 5), 0, 1), 2)
    return over(o, (1, .6, .6), hi * .7)


@style("vampire", "🧛 ጥቁርና ቀይ ቫምፓየር", "nature")
def s_vampire(c):
    shape = np.maximum(c.m, drips_mask(c, 25, 90, 7, 12))
    o = c.bg(lambda: bg_dark((24, 0, 6)))
    o = over(o, col((255, 0, 30)), blur(shape, 34) * .6)
    o = over(o, col((210, 10, 30)), grow(shape, 5))
    return over(o, col((12, 5, 8)), shape)


@style("ice", "🧊 የበረዶ ክሪስታል", "nature")
def s_ice(c):
    shape = np.maximum(c.m, drips_mask(c, 30, 130, 10, 18, tri=True))
    return _facet_style(c, 36, [(0, (180, 225, 255)), (.5, (240, 250, 255)), (1, (120, 190, 235))],
                        (120, 210, 255), (60, 120, 190), (4, 14, 30), sparkle=True, shape=shape)


@style("ocean", "🌊 ውቅያኖስ", "nature")
def s_ocean(c):
    o = c.bg(lambda: bg_grad([(0, (0, 20, 50)), (1, (0, 60, 110))]))
    t = np.clip(c.yb + 0.09 * np.sin(c.xb * 22 + c.yb * 6), 0, 1)
    fill = ramp(t, [(0, (190, 245, 255)), (.35, (0, 190, 235)), (.7, (0, 90, 180)), (1, (0, 35, 110))])
    foam = np.clip(1 - np.abs(t - 0.22) * 24, 0, 1)
    fill = over(fill, (1, 1, 1), foam * .8)
    return stack(c, o, fill, outline=(0, 30, 70), ow=5, glow=[((0, 200, 255), 26, .5)], bev=3)


@style("wood", "🪵 እንጨት", "nature")
def s_wood(c):
    o = c.bg(lambda: bg_dark((50, 30, 14)))
    g = 0.5 + 0.5 * np.sin(Yp * 0.12 + 14 * c.fbm(3, 3) + Xp * 0.01)
    fill = ramp(g, [(0, (110, 66, 30)), (1, (200, 140, 78))])
    return stack(c, o, fill, outline=(40, 22, 8), ow=6, bev=4)


@style("camo", "🪖 ወታደራዊ ካሜራ", "nature")
def s_camo(c):
    o = c.bg(lambda: bg_dark((22, 26, 14)))
    n = c.fbm(5, 5)
    cols = np.array([(40, 58, 30), (88, 100, 50), (160, 140, 90), (78, 54, 34)], np.float32) / 255
    fill = cols[np.digitize(n, [.4, .56, .72])]
    return stack(c, o, fill, outline=(16, 20, 8), ow=6, bev=2)


@style("snake", "🐍 የእባብ ቆዳ", "nature")
def s_snake(c):
    o = c.bg(lambda: bg_dark((6, 20, 8)))
    s = 26.0
    r = np.floor(Yp / (s * 0.85))
    off = (r % 2) * s / 2
    cc = np.floor((Xp + off) / s)
    cx = (cc + 0.5) * s - off
    cy = (r + 0.5) * s * 0.85
    d = np.sqrt((Xp - cx) ** 2 + (Yp - cy) ** 2) / (s * 0.62)
    rnd = np.mod(np.sin(r * 12.9898 + cc * 78.233) * 43758.5453, 1.0)
    shade = np.clip(1 - d ** 2, 0, 1) * (0.55 + 0.45 * rnd)
    fill = ramp(shade, [(0, (6, 28, 10)), (.5, (50, 130, 40)), (1, (190, 225, 90))])
    return stack(c, o, fill, outline=(2, 14, 4), ow=5, bev=2)


@style("nature", "🌿 የተፈጥሮ ቅጠል", "nature")
def s_nature(c):
    o = c.bg(lambda: bg_dark((10, 40, 14)))
    fill = ramp(c.yb, [(0, (190, 240, 100)), (1, (30, 120, 45))])
    veins = np.abs(np.sin((Xp * 0.9 - Yp * 1.3) / 9)) < 0.08
    fill = over(fill, (0.05, 0.3, 0.1), veins.astype(np.float32) * .5)
    return stack(c, o, fill, outline=(8, 50, 16), ow=5, glow=[((130, 255, 120), 26, .4)], bev=3)


@style("autumn", "🍂 የመኸር ቅጠል", "nature")
def s_autumn(c):
    o = c.bg(lambda: bg_dark((40, 18, 6)))
    n = c.fbm(5, 5)
    fill = ramp(n, [(0, (120, 40, 10)), (.4, (220, 90, 20)), (.7, (250, 170, 40)), (1, (150, 30, 20))])
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = c.rnd.randrange(W), c.rnd.randrange(H)
        d.ellipse([x - 14, y - 7, x + 14, y + 7], fill=200)
    o = over(o, col((200, 80, 20)), blur(A(img), 1.5) * .55)
    return stack(c, o, fill, outline=(60, 20, 4), ow=5, bev=3)


@style("smoke", "💨 ጭስና ጉም", "nature")
def s_smoke(c):
    o = c.bg(lambda: bg_dark((30, 30, 32)))
    n = c.fbm(5, 5)
    a = np.clip(blur(c.m, 12) * (0.5 + 1.0 * n), 0, 1)
    o = over(o, col((200, 200, 205)), blur(c.m, 40) * .25)
    o = over(o, col((225, 225, 230)), a)
    return over(o, col((255, 255, 255)), shrink(c.m, 6) * .6)


# =========================================================== 🌌 SKY
def _space_bg(c, nebula=True):
    def bg():
        o = bg_solid((3, 3, 12))
        if nebula:
            n = c.fbm(5, 3)
            neb = ramp(n, [(0, (5, 5, 25)), (.45, (50, 18, 100)), (.7, (190, 55, 150)), (1, (60, 160, 255))])
            o = np.clip(o + neb * 0.8 * (0.3 + 0.7 * c.fbm(3, 2))[..., None], 0, 1)
        return add(o, (1, 1, 1), stars_layer(c))
    return c.bg(bg)


@style("galaxy", "✨ ኮከብና ጋላክሲ", "sky")
def s_galaxy(c):
    o = _space_bg(c, nebula=False)
    n = c.fbm(5, 3)
    fill = ramp(n, [(0, (30, 20, 110)), (.4, (130, 50, 190)), (.7, (240, 90, 170)), (1, (110, 200, 255))])
    fill = add(fill, (1, 1, 1), stars_layer(c, 160) * .8)
    return stack(c, o, fill, outline=(255, 255, 255), ow=3, glow=[((140, 100, 255), 28, .5)])


@style("nebula", "🌌 የከዋክብት ጭጋግ", "sky")
def s_nebula(c):
    o = _space_bg(c, nebula=True)
    o = over(o, (.7, .8, 1), blur(c.m, 30) * .5)
    return over(o, (1, 1, 1), c.m)


@style("aurora", "🌠 የሰሜን ብርሃን", "sky")
def s_aurora(c):
    def bg():
        o = add(bg_solid((2, 6, 16)), (1, 1, 1), stars_layer(c, 200) * .8)
        a = 0.5 + 0.5 * np.sin(Xn * 14 + 5 * c.fbm(3, 3))
        w = np.exp(-((Yn - 0.45) / 0.25) ** 2)
        return np.clip(o + ramp(Yn, [(0, (0, 255, 150)), (1, (150, 60, 255))]) * (a * w * 0.8)[..., None], 0, 1)
    o = c.bg(bg)
    fill = ramp(c.xb, [(0, (120, 255, 200)), (.5, (160, 200, 255)), (1, (200, 130, 255))])
    return stack(c, o, fill, outline=(255, 255, 255), ow=3, glow=[((90, 255, 190), 30, .5)])


@style("sunset", "🌅 የጋለ ፀሐይ ጨረር", "sky")
def s_sunset(c):
    def bg():
        o = bg_grad([(0, (18, 8, 40)), (.7, (90, 20, 70)), (1, (150, 40, 40))])
        d = np.sqrt((Xn - .5) ** 2 + (Yn - .55) ** 2)
        return over(o, col((255, 140, 40)), np.clip(1 - d * 2.4, 0, 1) ** 2 * .8)
    o = c.bg(bg)
    fill = ramp(c.yb, [(0, (255, 240, 150)), (.4, (255, 150, 40)), (.75, (232, 45, 100)), (1, (120, 25, 150))])
    return stack(c, o, fill, outline=(60, 10, 40), ow=5, glow=[((255, 120, 40), 30, .55)], bev=3)


@style("fireworks", "🎆 የሚያበራ የሌዘር ጨረር", "sky")
def s_fireworks(c):
    def bg():
        img = Image.new("RGB", (W, H), (0, 0, 0))
        d = ImageDraw.Draw(img)
        pal = [(255, 80, 80), (255, 210, 60), (90, 200, 255), (255, 90, 220), (120, 255, 150)]
        for _ in range(8):
            cx, cy = c.rnd.randint(80, W - 80), c.rnd.randint(60, int(H * .8))
            color = c.rnd.choice(pal)
            for k in range(36):
                ang = k * 2 * math.pi / 36 + c.rnd.uniform(-.05, .05)
                for j in range(14):
                    rr = 10 + j * 8
                    f = 1 - j / 16
                    sz = max(1, int(5 - j * .3))
                    x, y = cx + math.cos(ang) * rr, cy + math.sin(ang) * rr
                    d.ellipse([x - sz, y - sz, x + sz, y + sz], fill=tuple(int(v * f) for v in color))
        a = A(img)
        return np.clip(bg_solid((2, 3, 14)) + a + blur(a.mean(-1), 8)[..., None] * a * 1.5, 0, 1)
    o = c.bg(bg, dim=0.7)
    fill = ramp(c.yb, [(0, (255, 246, 175)), (.5, (244, 190, 45)), (1, (205, 120, 30))])
    return stack(c, o, fill, outline=(40, 20, 0), ow=6, glow=[((255, 190, 40), 26, .5)], bev=3)


@style("candle", "🕯 ሻማ ሰም", "sky")
def s_candle(c):
    def bg():
        d = np.sqrt((Xn - .5) ** 2 + (Yn - .45) ** 2)
        return over(bg_solid((6, 3, 2)), col((255, 150, 50)), np.clip(1 - d * 2.2, 0, 1) ** 2 * .55)
    o = c.bg(bg)
    o = over(o, col((255, 170, 60)), blur(c.m, 40) * .6)
    fill = ramp(c.yb, [(0, (255, 250, 220)), (1, (250, 200, 120))])
    return stack(c, o, fill, outline=(90, 40, 10), ow=4, glow=[((255, 190, 90), 14, .6)], bev=3)


# =========================================================== 🎨 ART
@style("sticker", "🏷 ተለጣፊ", "art")
def s_sticker(c):
    def bg():
        o = bg_solid((60, 200, 190))
        dots = (((Xp % 36) - 18) ** 2 + ((Yp % 36) - 18) ** 2) < 25
        return over(o, col((255, 255, 255)), dots.astype(np.float32) * .25)
    o = c.bg(bg, dim=0.9)
    fill = ramp(c.xb, [(0, (255, 70, 150)), (1, (255, 150, 40))])
    o = over(o, (0, 0, 0), blur(roll(grow(c.m, 18), 9, 7), 5) * .35)
    o = over(o, (1, 1, 1), grow(c.m, 18))
    o = over(o, col((210, 210, 215)), np.clip(grow(c.m, 18) - grow(c.m, 15), 0, 1))
    return over(o, fill, c.m)


@style("graffiti", "🎨 ግራፊቲ", "art")
def s_graffiti(c):
    def bg():
        n = c.fbm(4, 10)
        return np.clip((0.22 + 0.18 * n)[..., None] * np.ones((H, W, 3), np.float32), 0, 1)
    o = c.bg(bg)
    fill = ramp(c.xb, [(0, (255, 235, 40)), (.5, (255, 90, 90)), (1, (210, 60, 255))])
    o = over(o, (0, 0, 0), blur(roll(grow(c.m, 10), 14, 14), 3) * .8)
    o = over(o, (1, 1, 1), grow(c.m, 14))
    o = over(o, (0, 0, 0), grow(c.m, 10))
    o = over(o, fill, c.m)
    spec = (c.rng.random((H, W)) > 0.985).astype(np.float32) * grow(c.m, 40) * (1 - grow(c.m, 16))
    return over(o, col((255, 120, 150)), blur(spec, 0.8) * 1.5)


@style("comic", "💥 ፖፕ አርት ኮሚክ", "art")
def s_comic(c):
    def bg():
        d = np.sqrt(((Xp % 22) - 11) ** 2 + ((Yp % 22) - 11) ** 2)
        dot = np.clip((11 * 0.95 * (1 - Yn * .8) - d) / 1.2, 0, 1)
        return over(bg_solid((255, 222, 0)), col((235, 40, 70)), dot)
    o = c.bg(bg, dim=0.9)
    o = over(o, (0, 0, 0), roll(grow(c.m, 8), 12, 12))
    o = over(o, (0, 0, 0), grow(c.m, 8))
    o = over(o, col((255, 255, 255)), grow(c.m, 3))
    return over(o, col((30, 120, 255)), c.m)


@style("bw", "⬛ ጥቁር በነጭ", "art")
def s_bw(c):
    o = c.bg(lambda: bg_paper(c, (240, 240, 240)), dim=0.9)
    o = over(o, (0, 0, 0), blur(roll(c.m, 10, 8), 8) * .3)
    return over(o, (0.04, 0.04, 0.04), c.m)


@style("outline_bw", "🖤 ጥቁር ከነጭ ዳር", "art")
def s_outline_bw(c):
    o = c.bg(lambda: bg_grad([(0, (60, 60, 66)), (1, (20, 20, 24))]))
    o = over(o, (1, 1, 1), blur(c.m, 30) * .3)
    o = over(o, (1, 1, 1), grow(c.m, 9))
    return over(o, (0.02, 0.02, 0.02), c.m)


@style("pencil", "✏️ የከሰል እርሳስ", "art")
def s_pencil(c):
    o = c.bg(lambda: bg_paper(c), dim=0.9)
    hatch = (((Xp + Yp) / 5) % 2 < 1).astype(np.float32)
    shade = 0.22 + 0.25 * hatch + 0.2 * c.fbm(3, 30)
    o = over(o, (shade[..., None] * np.ones(3, np.float32)), c.m * 0.9)
    edge = np.clip(c.m - shrink(c.m, 2), 0, 1)
    return over(o, col((50, 50, 55)), edge * .9)


@style("ink", "🖋 የካሊግራፊ ቀለም", "art")
def s_ink(c):
    o = c.bg(lambda: bg_paper(c, (248, 242, 228)), dim=0.9)
    rough = np.clip((blur(c.m, 2.2) + (c.fbm(4, 40) - 0.5) * 0.6 - 0.45) * 6, 0, 1)
    o = over(o, (0.05, 0.04, 0.04), blur(rough, 4) * .25)
    o = over(o, col((14, 12, 10)), rough)
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for _ in range(26):
        x = c.rnd.randint(int(c.left) - 30, int(c.right) + 30)
        y = c.rnd.randint(int(c.top) - 30, int(c.bot) + 40)
        r = c.rnd.randint(2, 7)
        d.ellipse([x - r, y - r, x + r, y + r], fill=255)
    return over(o, col((14, 12, 10)), blur(A(img), 0.8))


@style("watercolor", "🎨 የውሃ ቀለም", "art")
def s_watercolor(c):
    o = c.bg(lambda: bg_paper(c, (250, 246, 238)), dim=0.9)
    n = c.fbm(4, 4)
    fill = ramp(n, [(0, (225, 70, 120)), (.35, (245, 160, 50)), (.65, (50, 150, 205)), (1, (120, 70, 200))])
    a = np.clip(blur(c.m, 2.5), 0, 1) * (0.88 + 0.12 * c.fbm(4, 14))
    o = over(o, fill, blur(c.m, 26) * .25)
    o = over(o, fill, a)
    edge = np.clip(blur(c.m, 2) - blur(shrink(c.m, 4), 2), 0, 1)
    return over(o, fill * .65, edge * .5)


@style("papercut", "✂️ የተቆረጠ ወረቀት", "art")
def s_papercut(c):
    o = c.bg(lambda: bg_grad([(0, (40, 130, 150)), (1, (20, 70, 100))]), dim=0.8)
    for dx, dy, b, op in [(4, 6, 3, .35), (10, 14, 8, .3), (18, 24, 16, .25)]:
        o = over(o, (0, 0, 0), blur(roll(c.m, dy, dx), b) * op)
    fill = np.clip((0.96 + 0.05 * c.fbm(3, 60))[..., None] * col((255, 252, 245)), 0, 1)
    return over(o, fill, c.m)


@style("pixel", "🎮 ፒክሴል 8-ቢት", "art")
def s_pixel(c):
    o = c.bg(lambda: bg_solid((24, 14, 48)))
    q = 12
    small = c.mask.resize((W // q, H // q), Image.BOX).point(lambda v: 255 if v > 110 else 0)
    ol = small.filter(ImageFilter.MaxFilter(3))
    m = A(small.resize((W, H), Image.NEAREST))
    mo = A(ol.resize((W, H), Image.NEAREST))
    band = np.floor(c.yb * 4) / 3
    fill = ramp(band, [(0, (255, 240, 90)), (1, (255, 90, 40))])
    o = over(o, (0, 0, 0), mo)
    return over(o, fill, m)


@style("vintage", "📷 የድሮ ፎቶግራፍ", "art")
def s_vintage(c):
    def bg():
        d = np.clip(np.sqrt((Xn - .5) ** 2 + (Yn - .5) ** 2) / .75, 0, 1)
        return ramp(d ** 1.6, [(0, (214, 184, 134)), (1, (96, 68, 38))])
    o = c.bg(bg, dim=0.85)
    o = over(o, col((58, 34, 18)), blur(c.m, 1.2))
    grain = c.rng.normal(0, 0.05, (H, W)).astype(np.float32)
    o = np.clip(o + grain[..., None], 0, 1)
    for _ in range(5):
        x = c.rnd.randrange(W)
        o[:, x:x + 1] = np.clip(o[:, x:x + 1] + 0.12, 0, 1)
    return o


@style("rainbow", "🌈 ቀስተ ደመና", "art")
def s_rainbow(c):
    o = c.bg(lambda: bg_dark((20, 20, 30)))
    fill = ramp(c.xb, [(0, (255, 40, 40)), (.17, (255, 150, 0)), (.34, (255, 235, 0)), (.5, (40, 220, 60)),
                       (.67, (0, 190, 255)), (.84, (70, 70, 255)), (1, (190, 60, 230))])
    return stack(c, o, fill, outline=(255, 255, 255), ow=5, glow=[((255, 255, 255), 24, .3)], bev=4)


@style("circus", "🎪 የሰርከስ ቀለሞች", "art")
def s_circus(c):
    o = c.bg(lambda: bg_dark((60, 10, 10)))
    k = np.floor(((Xp + Yp) / 34) % 3)
    cols = np.array([(230, 30, 40), (255, 255, 255), (255, 205, 30)], np.float32) / 255
    return stack(c, o, cols[k.astype(np.int32)], outline=(40, 8, 8), ow=7, bev=4,
                 glow=[((255, 190, 60), 24, .4)])


@style("mosaic", "🕌 የሞዛይክ ሴራሚክ", "art")
def s_mosaic(c):
    o = c.bg(lambda: bg_dark((10, 30, 50)))
    s = 22
    cid = np.floor(Xp / s) * 31 + np.floor(Yp / s) * 17
    rnd = np.mod(np.sin(cid * 12.9898) * 43758.5453, 1.0)
    fill = ramp(rnd, [(0, (20, 80, 170)), (.35, (0, 170, 190)), (.65, (240, 190, 40)), (1, (250, 245, 230))])
    grout = ((Xp % s) < 2) | ((Yp % s) < 2)
    fill = over(fill, (.1, .1, .1), grout.astype(np.float32) * .8)
    return stack(c, o, fill, outline=(10, 20, 30), ow=5, bev=2)


@style("jelly", "🍓 የፍራፍሬ ጄሊ", "art")
def s_jelly(c):
    o = c.bg(lambda: bg_dark((40, 8, 28)))
    fill = ramp(c.yb, [(0, (255, 110, 130)), (1, (190, 8, 60))])
    o = stack(c, o, fill, outline=(120, 0, 40), ow=4, glow=[((255, 60, 100), 28, .5)], bev=5)
    hl = blur(np.roll(shrink(c.m, 16), -6, 0), 5) * c.m * (1 - c.yb * .6)
    return over(o, (1, 1, 1), np.clip(hl, 0, 1) * .55)


@style("coffee", "☕ የቡና አረፋ", "art")
def s_coffee(c):
    o = c.bg(lambda: bg_dark((40, 22, 10)))
    t = np.clip(c.yb + 0.08 * np.sin(c.xb * 30), 0, 1)
    fill = ramp(t, [(0, (248, 230, 195)), (.3, (236, 205, 160)), (.38, (120, 70, 38)), (1, (55, 28, 14))])
    return stack(c, o, fill, outline=(30, 14, 6), ow=6, glow=[((200, 130, 60), 24, .3)], bev=4)


@style("orange", "🍊 የብርቱካን ልጣጭ", "art")
def s_orange(c):
    o = c.bg(lambda: bg_dark((50, 24, 4)))
    fill = ramp(c.yb, [(0, (255, 185, 50)), (1, (232, 100, 8))])
    pores = c.rng.random((H, W)).astype(np.float32)
    pores = blur((pores > 0.93).astype(np.float32), 1.2)
    fill = over(fill, (0.6, 0.25, 0.0), pores * 1.6)
    return stack(c, o, fill, outline=(110, 40, 0), ow=5, glow=[((255, 150, 30), 24, .4)], bev=5)


@style("hearts", "💖 ልብ ዳራ", "art")
def s_hearts(c):
    def bg():
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        for _ in range(48):
            cx, cy, sz = c.rnd.randrange(W), c.rnd.randrange(H), c.rnd.randint(14, 48)
            pts = []
            for k in range(40):
                t = k / 40 * 2 * math.pi
                x = 16 * math.sin(t) ** 3
                y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
                pts.append((cx + x * sz / 16, cy + y * sz / 16))
            d.polygon(pts, fill=c.rnd.randint(90, 255))
        h = A(img.filter(ImageFilter.GaussianBlur(1.5)))
        return over(bg_grad([(0, (110, 10, 60)), (1, (50, 4, 40))]), col((255, 90, 160)), h * .6)
    o = c.bg(bg)
    fill = ramp(c.yb, [(0, (255, 235, 245)), (1, (255, 140, 190))])
    return stack(c, o, fill, outline=(255, 255, 255), ow=5, glow=[((255, 80, 160), 26, .6)], bev=3)


# ------------------------------------------------------------------- render
def render(text, style_key, photo_bytes=None):
    c = Ctx(text, photo_bytes)
    arr = STYLES[style_key][2](c)
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf
