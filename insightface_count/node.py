"""InsightFace face counter for ComfyUI.

Detects faces in generated image, counts them, and reports gender.
Used to decide whether to apply ReActor face swap post-generation.

Result is embedded in the image filename prefix so it can be parsed
from RunPod output without extra API calls.

Filename format: "fc{count}_f{females}_m{males}"
Example: "fc2_f1_m1" = 2 faces, 1 female, 1 male
"""

import numpy as np
import torch

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


class InsightFaceCount:
    """Count faces and genders in an image using InsightFace."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("face_info",)
    FUNCTION = "count"
    CATEGORY = "face"
    OUTPUT_NODE = True

    def count(self, image):
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)

        analyzer = _get_analyzer()
        faces = analyzer.get(img_np)

        face_count = len(faces)
        females = sum(1 for f in faces if f.gender == 0)
        males = sum(1 for f in faces if f.gender == 1)

        # Format: "fc2_f1_m1" = 2 faces, 1 female, 1 male
        face_info = f"fc{face_count}_f{females}_m{males}"

        return (face_info,)


NODE_CLASS_MAPPINGS = {
    "InsightFaceCount": InsightFaceCount,
}
