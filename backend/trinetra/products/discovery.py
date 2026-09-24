"""Finding product files under data/raw/<kind>/<pair_id>/."""
import json
import os
import zipfile

_SKIP_DIRS = {"calibrated", "browse", ".ipynb_checkpoints", "__pycache__"}
RASTER_EXTS = (".img", ".qub")


def _walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in _SKIP_DIRS]
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def find_first(root, exts, recursive=True):
    """First file (in sorted path order) under root with one of the extensions, or None."""
    if not os.path.isdir(root):
        return None
    exts = {e.lower() for e in exts}
    files = _walk(root) if recursive else (
        os.path.join(root, f) for f in sorted(os.listdir(root))
        if os.path.isfile(os.path.join(root, f)))
    for p in sorted(files):
        if os.path.splitext(p)[1].lower() in exts:
            return p
    return None


def product_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def _localise(path, data_raw):
    """LINK.json paths were written on Colab (/content/drive/.../data/raw/...). If the path does
    not exist here, re-root everything after 'data/raw/' onto this machine's data_raw."""
    if os.path.exists(path):
        return path
    norm = path.replace("\\", "/")
    marker = "/data/raw/"
    if marker in norm:
        candidate = os.path.join(data_raw, *norm.split(marker, 1)[1].split("/"))
        if os.path.exists(candidate):
            return candidate
    return path


def resolve_pair_dir(data_raw, kind, pair_id):
    """(raster, xml label) for data/raw/<kind>/<pair_id>.

    A LINK.json in the folder points at a product stored for another pair, so a shared product
    is kept once. A downloaded .zip is extracted in place when no raster is present yet.
    """
    d = os.path.join(data_raw, kind, pair_id)
    link = os.path.join(d, "LINK.json")
    if os.path.exists(link):
        with open(link) as f:
            target = json.load(f)
        img, xml = _localise(target["img"], data_raw), _localise(target["xml"], data_raw)
        missing = [p for p in (img, xml) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{kind}/{pair_id}/LINK.json points at files that are not there: "
                                    + ", ".join(missing))
        return img, xml

    img = find_first(d, RASTER_EXTS)
    if img is None:
        archive = find_first(d, [".zip"])
        if archive:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(os.path.dirname(archive))
            img = find_first(d, RASTER_EXTS)
    xml = find_first(d, [".xml"])
    if img is None or xml is None:
        raise FileNotFoundError(f"Need a raster and a .xml label under {d} (found img={img}, xml={xml}).")
    return img, xml


def has_product(data_raw, kind, pair_id):
    """True if data/raw/<kind>/<pair_id> holds a raster, a .zip or a LINK.json."""
    d = os.path.join(data_raw, kind, pair_id)
    return (os.path.exists(os.path.join(d, "LINK.json"))
            or find_first(d, [*RASTER_EXTS, ".zip"]) is not None)


def pair_dirs(data_raw, kind):
    """Sorted pair folders (pairNN_*) under data/raw/<kind>."""
    d = os.path.join(data_raw, kind)
    if not os.path.isdir(d):
        return []
    return [p for p in sorted(os.listdir(d)) if p.startswith("pair")]
