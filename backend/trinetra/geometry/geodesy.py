"""Lunar geodesy and the four-corner pixel <-> lat/lon model."""
import numpy as np

MOON_RADIUS_KM = 1737.4
KM_PER_DEG_LAT = 2.0 * np.pi * MOON_RADIUS_KM / 360.0
CORNERS = ("UL", "UR", "LL", "LR")


def km_per_deg_lon(lat_deg):
    return KM_PER_DEG_LAT * float(np.cos(np.radians(lat_deg)))


def unwrap_lon(lon, ref):
    """Longitude shifted by whole turns to lie within 180 degrees of ref."""
    lon = np.asarray(lon, dtype=np.float64)
    return lon - 360.0 * np.round((lon - float(ref)) / 360.0)


def corner_affine(corners_deg, shape, lon_ref):
    """Least-squares affine through the four footprint corners.

    Returns (M, t) with [lat, lon] = M @ [row, col] + t, longitudes unwrapped around lon_ref.
    """
    n_lines, n_samples = shape
    pix = np.array([[0, 0], [0, n_samples - 1],
                    [n_lines - 1, 0], [n_lines - 1, n_samples - 1]], dtype=np.float64)
    lat = np.array([corners_deg[k]["lat"] for k in CORNERS], dtype=np.float64)
    lon = np.array([float(unwrap_lon(corners_deg[k]["lon"], lon_ref)) for k in CORNERS])
    A = np.column_stack([pix[:, 0], pix[:, 1], np.ones(4)])
    coef_lat, *_ = np.linalg.lstsq(A, lat, rcond=None)
    coef_lon, *_ = np.linalg.lstsq(A, lon, rcond=None)
    return np.array([coef_lat[:2], coef_lon[:2]]), np.array([coef_lat[2], coef_lon[2]])


def latlon_to_pixel(corners_deg, shape):
    """Function (lat, lon) -> (row, col) inverting the product's corner model."""
    lon_ref = float(corners_deg["UL"]["lon"])
    M, t = corner_affine(corners_deg, shape, lon_ref)
    M_inv = np.linalg.inv(M)

    def inverse(lat, lon):
        lon_u = float(unwrap_lon(lon, lon_ref))
        row, col = M_inv @ (np.array([lat, lon_u], dtype=np.float64) - t)
        return row, col
    return inverse
