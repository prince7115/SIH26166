"""LROC NAC EDR products (PDS3, attached label)."""
import re

import numpy as np

# planetaryimage still calls NumPy APIs that NumPy 2 removed; restore them before importing it.
for _alias, _real in (("product", np.prod), ("cumproduct", np.cumprod),
                      ("alltrue", np.all), ("sometrue", np.any)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _real)
_fromstring = np.fromstring


def _fromstring_binary(string, dtype=float, count=-1, *, sep="", like=None):
    if isinstance(string, bytes) and sep == "":
        return np.frombuffer(string, dtype=dtype, count=count)
    return _fromstring(string, dtype=dtype, count=count, sep=sep)


np.fromstring = _fromstring_binary

from planetaryimage import PDS3Image  # noqa: E402


def read_header(img_path, max_bytes=65536):
    """KEY = VALUE pairs of the attached label, up to END (multi-line values are skipped)."""
    with open(img_path, "rb") as f:
        head = f.read(max_bytes).decode("latin-1", errors="replace")
    head = re.split(r"\r?\nEND\s*\r?\n", head, maxsplit=1)[0]
    out = {}
    for line in head.splitlines():
        m = re.match(r"^\s*([A-Z0-9_:^]+)\s*=\s*(.+?)\s*$", line)
        if m and m.group(1) not in out:
            out[m.group(1)] = m.group(2).strip().strip('"')
    return out


def read_image(img_path):
    """First band of the image; NAC's 8-bit samples come back signed and are reinterpreted as unsigned."""
    image = np.asarray(PDS3Image.open(img_path).image)
    if image.ndim == 3:
        image = image[0]
    if image.dtype == np.int8:
        image = image.view(np.uint8)
    return image
