"""Regenerate the demo GIFs in images/ from data/tesco_support.csv.

Each frame is the real embedscope chart with a selection preset in the spec, rendered
to PNG by vl-convert, so the GIFs always match the current look of the package.

    python -m pip install -e . vl-convert-python pillow scikit-learn umap-learn
    python scripts/make_demo_gifs.py
"""

import copy
import io
import pathlib

import numpy as np
import pandas as pd
import vl_convert as vlc
from PIL import Image
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from umap import UMAP

from embedscope import plot_text, plot_text_comparison

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORDS = ["voucher", "deliver", "refund"]
DELIVERY_BOX = ((-5.4, 1.2), (-3.4, 3.2))  # (x0, y0), (x1, y1) in UMAP coordinates
VOUCHER_BOX = ((-5.8, -3.8), (4.8, 6.9))  # (x range), (y range)


def load_texts():
    df = pd.read_csv(ROOT / "data" / "tesco_support.csv")
    df = df.loc[lambda d: ~d["text"].str.contains("https")]
    # The PNG renderer has no emoji font, so drop non-BMP characters and decode &amp;.
    clean = df["text"].str.replace(r"[^\u0000-￿]", "", regex=True)
    return list(clean.str.replace("&amp;", "&").sample(2000, random_state=42))


def embed(texts):
    def tfidf():
        return TfidfVectorizer(stop_words="english", min_df=2)

    umap = UMAP(metric="cosine", n_neighbors=10, min_dist=0.05, random_state=42)
    X1 = make_pipeline(tfidf(), umap).fit_transform(texts)
    X2 = make_pipeline(tfidf(), TruncatedSVD(random_state=42)).fit_transform(texts)
    return X1, X2


def render(base, brush=None, legend=None):
    spec = copy.deepcopy(base)
    for p in spec["params"]:
        if p["name"] == "brush" and brush:
            p["value"] = {"x": list(brush[0]), "y": list(brush[1])}
        if p["name"] == "legend" and legend:
            p["value"] = [{"color": c} for c in legend]
    png = vlc.vegalite_to_png(spec, scale=1)
    return Image.open(io.BytesIO(png)).convert("RGB")


def drag(box, steps=6):
    """Brush boxes growing from one corner to the full box, like a mouse drag."""
    (x0, y0), (x1, y1) = box
    return [
        ((x0, x0 + (x1 - x0) * f), (y0, y0 + (y1 - y0) * f))
        for f in np.linspace(0.15, 1, steps)
    ]


def brush_tour(base):
    frames = [(render(base), 1500)]
    frames += [(render(base, b), 120) for b in drag(DELIVERY_BOX)]
    frames[-1] = (frames[-1][0], 2600)
    frames.append((render(base, VOUCHER_BOX), 2600))
    return frames


def save(frames, path):
    """frames: list of (image, milliseconds). Pads to one size and writes a looping GIF."""
    w = max(im.width for im, _ in frames) + 16
    h = max(im.height for im, _ in frames) + 16
    out = []
    for im, _ in frames:
        canvas = Image.new("RGB", (w, h), "white")
        canvas.paste(im, (8, 8))
        out.append(canvas.quantize(colors=256, method=Image.Quantize.MAXCOVERAGE))
    durations = [ms for _, ms in frames]
    out[0].save(
        path,
        save_all=True,
        append_images=out[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"wrote {path.relative_to(ROOT)} ({w}x{h}, {len(out)} frames)")


def main():
    texts = load_texts()
    X1, X2 = embed(texts)
    (ROOT / "images").mkdir(exist_ok=True)

    base = plot_text(X1, texts, color_words=WORDS).to_dict()
    frames = brush_tour(base)
    frames.append((render(base, legend=["voucher"]), 2200))
    frames.append((render(base, legend=["deliver"]), 2200))
    save(frames, ROOT / "images" / "demo.gif")

    base = plot_text_comparison(X1, X2, texts, top_words=0).to_dict()
    save(brush_tour(base), ROOT / "images" / "demo-compare.gif")


if __name__ == "__main__":
    main()
