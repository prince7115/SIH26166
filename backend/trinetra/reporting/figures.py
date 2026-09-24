"""Matplotlib figures for the preprocessing step and the result viewer."""
import numpy as np

from .encoding import fig_to_b64, plt


def _title(ax, text, fontsize=9):
    ax.set_title(text, color="white", fontsize=fontsize)


def preprocessing_figure(ohrc_pc, ref_pc, ohrc_sg, ref_sg, ref_name):
    """Step 5: plain stretches on top, matcher inputs (CLAHE, histogram-matched) below."""
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].imshow(ohrc_pc, cmap="gray")
    _title(ax[0, 0], "OHRC plain")
    ax[0, 1].imshow(ref_pc, cmap="gray")
    _title(ax[0, 1], f"{ref_name} plain")
    ax[1, 0].imshow(ohrc_sg, cmap="gray")
    _title(ax[1, 0], "OHRC CLAHE → SuperGlue")
    ax[1, 1].imshow(ref_sg, cmap="gray")
    _title(ax[1, 1], f"{ref_name} CLAHE + hist-matched")
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    return fig_to_b64(fig)


def overlay_figure(warped, ref, ref_name):
    """Warped OHRC, reference, and a checkerboard of the two."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(warped, cmap="gray")
    _title(ax[0], f"OHRC warped → {ref_name} frame")
    ax[1].imshow(ref, cmap="gray")
    _title(ax[1], f"{ref_name} reference")
    blk = max(8, min(warped.shape) // 20)
    yy, xx = np.mgrid[0:warped.shape[0], 0:warped.shape[1]]
    checker = np.where((((yy // blk) + (xx // blk)) % 2).astype(bool), warped, ref)
    ax[2].imshow(checker, cmap="gray")
    _title(ax[2], "Checkerboard")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    return fig_to_b64(fig)


def correspondence_figure(ohrc_u8, ref_u8, src, dst, max_lines=200):
    """OHRC and reference side by side with the first max_lines tie points joined."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    h1, h2 = ohrc_u8.shape[0], ref_u8.shape[0]
    if h1 != h2:
        h = max(h1, h2)
        if h1 < h:
            ohrc_u8 = np.pad(ohrc_u8, ((0, h - h1), (0, 0)), mode="constant")
        if h2 < h:
            ref_u8 = np.pad(ref_u8, ((0, h - h2), (0, 0)), mode="constant")
    ax.imshow(np.hstack([ohrc_u8, ref_u8]), cmap="gray")
    offset = ohrc_u8.shape[1]
    for i in range(min(max_lines, len(src))):
        ax.plot([src[i, 0], dst[i, 0] + offset], [src[i, 1], dst[i, 1]], "c-", lw=0.3, alpha=0.5)
    ax.plot(src[:max_lines, 0], src[:max_lines, 1], "r.", ms=2)
    ax.plot(dst[:max_lines, 0] + offset, dst[:max_lines, 1], "g.", ms=2)
    _title(ax, f"{len(src)} correspondences", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return fig_to_b64(fig)
