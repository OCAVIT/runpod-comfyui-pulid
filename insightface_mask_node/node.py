"""InsightFace-based face detection → mask extractor for ComfyUI.

Detects faces using InsightFace (buffalo_l / antelopev2), classifies
gender, and returns a binary mask for the selected face. Works much
more reliably than GroundingDINO for distinguishing male/female faces.

Nodes:
  - InsightFaceMaskExtractor: detect face by gender → MASK
"""

import numpy as np
import torch
from PIL import Image

import comfy.model_management

# Lazy-load singleton
_face_analyzer = None


def _get_analyzer():
    global _face_analyzer
    if _face_analyzer is None:
        from insightface.app import FaceAnalysis

        _face_analyzer = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return _face_analyzer


class InsightFaceMaskExtractor:
    """Detect a face by gender and return a mask covering it."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gender": (["female", "male"],),
                "expand": (
                    "FLOAT",
                    {"default": 1.6, "min": 1.0, "max": 3.0, "step": 0.1},
                ),
                "face_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "extract"
    CATEGORY = "face"

    def extract(self, image, gender, expand, face_index):
        # image: [B, H, W, C] float32 0-1
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h, w = img_np.shape[:2]

        analyzer = _get_analyzer()
        faces = analyzer.get(img_np)

        # Filter by gender: insightface gender — 0=female, 1=male
        target_gender = 0 if gender == "female" else 1
        matched = [f for f in faces if f.gender == target_gender]

        # Sort by bbox area (largest first) for consistent indexing
        matched.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

        mask = torch.zeros((1, h, w), dtype=torch.float32)

        if face_index < len(matched):
            face = matched[face_index]
            x1, y1, x2, y2 = face.bbox.astype(int)

            # Expand bbox for hair/neck coverage
            bw, bh = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            new_w = int(bw * expand)
            new_h = int(bh * expand)
            nx1 = max(0, cx - new_w // 2)
            ny1 = max(0, cy - new_h // 2)
            nx2 = min(w, cx + new_w // 2)
            ny2 = min(h, cy + new_h // 2)

            mask[0, ny1:ny2, nx1:nx2] = 1.0

        return (mask,)


NODE_CLASS_MAPPINGS = {
    "InsightFaceMaskExtractor": InsightFaceMaskExtractor,
}
