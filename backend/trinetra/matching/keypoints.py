"""SuperPoint keypoints matched with SuperGlue or LightGlue (Hugging Face checkpoints)."""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForKeypointMatching

from . import empty_matches
from ..imaging.resample import pad_to_square

CHECKPOINTS = {
    "superglue": ("SuperGlue", "magic-leap-community/superglue_outdoor"),
    "lightglue": ("LightGlue", "ETH-CVG/lightglue_superpoint"),
}
_loaded = {}   # (name, device) -> (processor, model): weights load once per process


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class KeypointMatcher:
    def __init__(self, name, device=None):
        self.label = CHECKPOINTS[name][0]
        self.device = device or default_device()
        key = (name, self.device)
        if key not in _loaded:
            checkpoint = CHECKPOINTS[name][1]
            _loaded[key] = (AutoImageProcessor.from_pretrained(checkpoint),
                            AutoModelForKeypointMatching.from_pretrained(checkpoint).to(self.device).eval())
        self.processor, self.model = _loaded[key]

    def match(self, a_u8, b_u8, threshold=0.15, size=1024):
        """Correspondences between two grayscale images: (points_a, points_b, scores), (x, y)."""
        if a_u8.size == 0 or b_u8.size == 0:
            return empty_matches()
        a_sq, ax, ay = pad_to_square(a_u8)
        b_sq, bx, by = pad_to_square(b_u8)
        images = [_to_rgb(a_sq), _to_rgb(b_sq)]
        self.processor.size = {"height": int(size), "width": int(size)}
        inputs = self.processor(images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            raw = self.model(**inputs)
        sizes = [[(im.height, im.width) for im in images]]
        out = self.processor.post_process_keypoint_matching(raw, sizes, threshold=threshold)[0]
        k0 = out["keypoints0"].float().cpu().numpy().reshape(-1, 2) - np.array([ax, ay], np.float32)
        k1 = out["keypoints1"].float().cpu().numpy().reshape(-1, 2) - np.array([bx, by], np.float32)
        sc = out["matching_scores"].float().cpu().numpy().reshape(-1)
        keep = ((k0[:, 0] >= 0) & (k0[:, 1] >= 0) & (k0[:, 0] < a_u8.shape[1]) & (k0[:, 1] < a_u8.shape[0]) &
                (k1[:, 0] >= 0) & (k1[:, 1] >= 0) & (k1[:, 0] < b_u8.shape[1]) & (k1[:, 1] < b_u8.shape[0]))
        return k0[keep], k1[keep], sc[keep]


def _to_rgb(gray_u8):
    return Image.fromarray(np.stack([gray_u8] * 3, axis=-1))
