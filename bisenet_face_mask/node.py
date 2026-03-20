"""BiSeNet face parsing mask for ComfyUI.

Creates a pixel-perfect face mask using BiSeNet (19-class face segmentation).
Includes: skin, eyes, eyebrows, nose, mouth, ears.
Excludes: hair, hat, clothes, neck, background.

Used to blend face-swapped images with originals — only the face region
is taken from the swap, hair stays original.

Usage in workflow:
  1. ReActorFaceSwap → swapped image
  2. BiSeNetFaceMask(swapped) → face-only mask
  3. Composite: (swapped * mask) + (original * (1-mask))
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management

# Lazy singleton
_parse_model = None


def _get_model(device):
    global _parse_model
    if _parse_model is None:
        import sys
        import os
        # Add ReActor to path for r_facelib imports
        reactor_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "ComfyUI-ReActor"
        )
        # Try multiple paths
        for p in [reactor_path, "/comfyui/custom_nodes/ComfyUI-ReActor"]:
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        from r_facelib.parsing import init_parsing_model
        _parse_model = init_parsing_model(model_name="parsenet", device=device)
    return _parse_model


# BiSeNet/ParseNet 19-class mapping:
#  0=background, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye,
#  6=eye_glasses, 7=l_ear, 8=r_ear, 9=earring, 10=nose,
#  11=mouth, 12=u_lip, 13=l_lip, 14=neck, 15=necklace,
#  16=cloth, 17=hair, 18=hat
FACE_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}  # face only
# Excluded: 0(bg), 14(neck), 15(necklace), 16(cloth), 17(hair), 18(hat)


class BiSeNetFaceMask:
    """Generate pixel-perfect face mask excluding hair."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_radius": (
                    "INT",
                    {"default": 15, "min": 0, "max": 100, "step": 1},
                ),
                "expand": (
                    "INT",
                    {"default": 5, "min": -20, "max": 50, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "parse"
    CATEGORY = "face"

    def parse(self, image, blur_radius, expand):
        device = comfy.model_management.get_torch_device()
        model = _get_model(device)

        # image: [B, H, W, C] float32 0-1
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        h, w = img_np.shape[:2]

        # Detect faces with InsightFace to get face regions
        # For each face: crop, parse, create mask
        from insightface.app import FaceAnalysis
        global _face_analyzer
        try:
            _face_analyzer
        except NameError:
            _face_analyzer = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            _face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

        faces = _face_analyzer.get(img_np)
        full_mask = np.zeros((h, w), dtype=np.float32)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            # Expand bbox for parsing context
            bw, bh = x2 - x1, y2 - y1
            pad = int(max(bw, bh) * 0.5)
            fx1 = max(0, x1 - pad)
            fy1 = max(0, y1 - pad)
            fx2 = min(w, x2 + pad)
            fy2 = min(h, y2 + pad)

            face_crop = img_np[fy1:fy2, fx1:fx2]
            face_input = cv2.resize(face_crop, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_input = face_input.astype(np.float32) / 255.0
            face_input = torch.from_numpy(face_input).permute(2, 0, 1).unsqueeze(0)
            face_input = (face_input - 0.5) / 0.5  # normalize
            face_input = face_input.to(device)

            with torch.no_grad():
                out = model(face_input)[0]
            out = out.argmax(dim=1).squeeze().cpu().numpy()

            # Create binary mask for face classes only
            parse_mask = np.zeros(out.shape, dtype=np.float32)
            for cls in FACE_CLASSES:
                parse_mask[out == cls] = 1.0

            # Expand/shrink mask
            if expand > 0:
                kernel = np.ones((expand * 2 + 1, expand * 2 + 1), np.uint8)
                parse_mask = cv2.dilate(parse_mask, kernel, iterations=1)
            elif expand < 0:
                kernel = np.ones((abs(expand) * 2 + 1, abs(expand) * 2 + 1), np.uint8)
                parse_mask = cv2.erode(parse_mask, kernel, iterations=1)

            # Blur for smooth edges
            if blur_radius > 0:
                ksize = blur_radius * 2 + 1
                parse_mask = cv2.GaussianBlur(parse_mask, (ksize, ksize), blur_radius / 2)

            # Resize back to face crop size and place in full mask
            parse_mask = cv2.resize(parse_mask, (fx2 - fx1, fy2 - fy1))
            full_mask[fy1:fy2, fx1:fx2] = np.maximum(
                full_mask[fy1:fy2, fx1:fx2], parse_mask
            )

        mask_tensor = torch.from_numpy(full_mask).unsqueeze(0)  # [1, H, W]
        return (mask_tensor,)


class BiSeNetFaceBlend:
    """Blend swapped face onto original using BiSeNet face mask.

    Takes original + swapped images, creates face-only mask via BiSeNet,
    composites: face from swapped, everything else (hair, bg) from original.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "swapped": ("IMAGE",),
                "blur_radius": (
                    "INT",
                    {"default": 15, "min": 0, "max": 100, "step": 1},
                ),
                "expand": (
                    "INT",
                    {"default": 5, "min": -20, "max": 50, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "blend"
    CATEGORY = "face"

    def blend(self, original, swapped, blur_radius, expand):
        # Get face mask from the SWAPPED image (face regions to keep from swap)
        mask_node = BiSeNetFaceMask()
        (mask,) = mask_node.parse(swapped, blur_radius, expand)

        # mask: [1, H, W], images: [B, H, W, C]
        m = mask.unsqueeze(-1)  # [1, H, W, 1]
        result = swapped * m + original * (1 - m)

        return (result, mask)


NODE_CLASS_MAPPINGS = {
    "BiSeNetFaceMask": BiSeNetFaceMask,
    "BiSeNetFaceBlend": BiSeNetFaceBlend,
}
