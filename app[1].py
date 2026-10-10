"""Amharic Text Style - web app (Flask).  Same style engine as the Telegram bot (styles.py).

Run locally:   pip install -r requirements.txt   then   python app.py     ->  http://localhost:5000
Render:        see DEPLOY.txt  (start command: gunicorn app:app ...)
"""
import io
import os
import re
import threading
import time
from collections import defaultdict, deque

from flask import Flask, Response, abort, jsonify, render_template, request
from PIL import Image, ImageOps

from styles import CATEGORIES, STYLES, render

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024          # 6 MB upload limit
Image.MAX_IMAGE_PIXELS = 40_000_000                         # protects against decompression bombs

MAX_LEN = 40
SITE_NAME = os.environ.get("SITE_NAME", "Amharic Text Style")
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "").strip()
ADSENSE_CLIENT = os.environ.get("ADSENSE_CLIENT", "").strip()        # e.g. ca-pub-1234567890123456
AD_SLOT_TOP = os.environ.get("AD_SLOT_TOP", "").strip()
AD_SLOT_BOTTOM = os.environ.get("AD_SLOT_BOTTOM", "").strip()

# English names for the UI (the style engine itself keeps the Amharic labels for the bot).
CATEGORY_EN = {
    "light": "Light & Neon", "metal": "Metal & Stone", "nature": "Fire & Nature",
    "sky": "Sky & Space", "art": "Art & Drawing",
}
STYLE_LIST = [
    {"key": k, "cat": v[1], "name": k.replace("_", " ").title()} for k, v in STYLES.items()
]
CATS = [{"key": k, "name": CATEGORY_EN.get(k, k.title())} for k, _ in CATEGORIES]

_render_slots = threading.BoundedSemaphore(2)               # at most 2 renders at once (small server)
_hits = defaultdict(deque)                                  # simple per-IP rate limit
RATE_LIMIT, RATE_WINDOW = 20, 60                            # 20 images per minute per IP


def _rate_limited(ip):
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


def _client_ip():
    return (request.headers.get("X-Forwarded-For", request.remote_addr or "") or "").split(",")[0].strip()


@app.context_processor
def inject_globals():
    return dict(site_name=SITE_NAME, adsense_client=ADSENSE_CLIENT, slot_top=AD_SLOT_TOP,
                slot_bottom=AD_SLOT_BOTTOM, contact_email=CONTACT_EMAIL)


@app.after_request
def headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return resp


# ---------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html", styles=STYLE_LIST, cats=CATS, max_len=MAX_LEN)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/healthz")
def healthz():
    return "ok"


# ---------------------------------------------------------------- API
@app.route("/api/render", methods=["POST"])
def api_render():
    if _rate_limited(_client_ip()):
        return jsonify(error="Too many requests. Please wait a minute and try again."), 429

    text = (request.form.get("text") or "").strip()
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    style = request.form.get("style") or ""
    if not text:
        return jsonify(error="Please type some text first."), 400
    if len(text) > MAX_LEN:
        return jsonify(error=f"Please use at most {MAX_LEN} characters."), 400
    if style not in STYLES:
        return jsonify(error="Unknown style."), 400

    photo_bytes = None
    f = request.files.get("photo")
    if f and f.filename:
        try:
            img = Image.open(f.stream)
            img.verify()                                    # is it really an image?
            f.stream.seek(0)
            img = ImageOps.exif_transpose(Image.open(f.stream)).convert("RGB")
            img.thumbnail((2000, 2000))
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=90)
            photo_bytes = out.getvalue()                    # kept in memory only, never saved
        except Exception:
            return jsonify(error="That file is not a valid image (use JPG or PNG)."), 400

    if not _render_slots.acquire(timeout=40):
        return jsonify(error="The server is busy. Please try again in a moment."), 503
    try:
        buf = render(text, style, photo_bytes)
    except Exception:
        app.logger.exception("render failed")
        return jsonify(error="Could not create the image. Try different text or another style."), 500
    finally:
        _render_slots.release()

    resp = Response(buf.getvalue(), mimetype="image/png")
    resp.headers["Cache-Control"] = "no-store"
    return resp


# ---------------------------------------------------------------- SEO / ads files
@app.route("/robots.txt")
def robots():
    return Response(f"User-agent: *\nAllow: /\nSitemap: {request.url_root}sitemap.xml\n", mimetype="text/plain")


@app.route("/sitemap.xml")
def sitemap():
    urls = "".join(f"<url><loc>{request.url_root.rstrip('/')}{p}</loc></url>"
                   for p in ("/", "/about", "/privacy", "/contact"))
    xml = f'<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>'
    return Response(xml, mimetype="application/xml")


@app.route("/ads.txt")
def ads_txt():
    m = re.fullmatch(r"ca-pub-(\d{10,20})", ADSENSE_CLIENT)
    if not m:
        abort(404)
    return Response(f"google.com, pub-{m.group(1)}, DIRECT, f08c47fec0942fa0\n", mimetype="text/plain")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
