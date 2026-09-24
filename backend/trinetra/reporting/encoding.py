import base64
import io

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURE_BACKGROUND = "#0a0e1a"


def img_to_b64(img, max_side=800):
    """Grayscale or BGR array as a PNG data URL, shrunk to max_side."""
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        k = max_side / max(h, w)
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode()


def fig_to_b64(fig, dpi=120):
    """Matplotlib figure as a PNG data URL; the figure is closed."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=FIGURE_BACKGROUND, edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
