# RunPod Serverless ComfyUI worker
# Flux Dev fp8 + PuLID + FaceDetailer + ReActor + RIFE
#
# Face consistency pipeline:
#   1. Generate scene with Flux (no PulID)
#   2. FaceDetailer: YOLO face detect → SAM segment → inpaint with PulID
#   3. Each face gets its own PulID identity pass
#
# Build: docker build --platform linux/amd64 -t comfyui-flux-face .

FROM runpod/worker-comfyui:5.7.1-flux1-dev-fp8

# ── System tools ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip build-essential python3-dev g++ && \
    rm -rf /var/lib/apt/lists/*

# ══════════════════════════════════════════════════════════════════
# SECTION 1: Custom nodes (clone + install requirements)
# IMPORTANT: insightface from GitHub LAST (overrides PyPI version)
# ══════════════════════════════════════════════════════════════════

# ── PuLID-Flux (face identity during generation/inpainting) ──────
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/sipie800/ComfyUI-PuLID-Flux-Enhanced.git && \
    cd ComfyUI-PuLID-Flux-Enhanced && \
    pip install --no-cache-dir -r requirements.txt

# ── Impact Pack (FaceDetailer) ────────────────────────────────────
# Install ALL deps from requirements.txt EXCEPT sam2 (fails to compile)
RUN pip install --no-cache-dir \
    ultralytics segment-anything scikit-image piexif \
    opencv-python-headless scipy numpy dill matplotlib transformers
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    cd ComfyUI-Impact-Pack && \
    touch impact/config/__init__.py 2>/dev/null; \
    mkdir -p /comfyui/models/onnx
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git

# ── ReActor (face swap fallback, SFW version) ────────────────────
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Gourieff/ComfyUI-ReActor.git && \
    cd ComfyUI-ReActor && \
    pip install --no-cache-dir -r requirements.txt

# ── VideoHelperSuite (RIFE video I/O) ────────────────────────────
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip install --no-cache-dir -r requirements.txt

# ── RIFE Frame Interpolation ─────────────────────────────────────
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    cd ComfyUI-Frame-Interpolation && \
    pip install --no-cache-dir -r requirements-no-cupy.txt

# ── CuPy for RIFE GPU acceleration ──────────────────────────────
RUN pip install --no-cache-dir cupy-cuda12x

# ══════════════════════════════════════════════════════════════════
# SECTION 2: Fix Python packages (AFTER all nodes installed)
# ══════════════════════════════════════════════════════════════════

# ── InsightFace from GitHub — MUST be LAST ───────────────────────
RUN pip install --no-cache-dir --force-reinstall \
    "insightface @ git+https://github.com/deepinsight/insightface.git@master#subdirectory=python-package"

# ── Ensure onnxruntime-gpu (not CPU) ─────────────────────────────
RUN pip uninstall -y onnxruntime 2>/dev/null; \
    pip install --no-cache-dir --force-reinstall onnxruntime-gpu

# ══════════════════════════════════════════════════════════════════
# SECTION 3: Models download
# ══════════════════════════════════════════════════════════════════

# ── PuLID Flux model (~1.1 GB) ───────────────────────────────────
RUN mkdir -p /comfyui/models/pulid && \
    wget -q -O /comfyui/models/pulid/pulid_flux_v0.9.0.safetensors \
    "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.0.safetensors"

# ── InsightFace models (PuLID + ReActor shared) ──────────────────
RUN mkdir -p /comfyui/models/insightface/models && \
    cd /tmp && \
    wget -q -O antelopev2.zip \
    "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip" && \
    unzip -o antelopev2.zip -d /comfyui/models/insightface/models/ && \
    rm antelopev2.zip && \
    mkdir -p /root/.insightface/models && \
    wget -q -O buffalo_l.zip \
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" && \
    unzip -o buffalo_l.zip -d /root/.insightface/models/ && \
    rm buffalo_l.zip

# ── Face swap models (ReActor) ───────────────────────────────────
RUN wget -q -O /comfyui/models/insightface/inswapper_128.onnx \
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx" && \
    mkdir -p /comfyui/models/hyperswap && \
    wget -q -O /comfyui/models/hyperswap/hyperswap_1c_256.onnx \
    "https://huggingface.co/facefusion/models-3.3.0/resolve/main/hyperswap_1c_256.onnx"

# ── Face restore + detection models ─────────────────────────────
RUN mkdir -p /comfyui/models/facerestore_models && \
    wget -q -O /comfyui/models/facerestore_models/GFPGANv1.4.pth \
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" && \
    mkdir -p /comfyui/models/ultralytics/bbox && \
    wget -q -O /comfyui/models/ultralytics/bbox/face_yolov8m.pt \
    "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt" && \
    mkdir -p /comfyui/models/sams && \
    wget -q -O /comfyui/models/sams/sam_vit_b_01ec64.pth \
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

# ── EVA-CLIP (pre-cache for PuLID cold start) ────────────────────
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('QuanSun/EVA-CLIP', 'EVA02_CLIP_L_336_psz14_s6B.pt', \
    cache_dir='/root/.cache/huggingface')" 2>/dev/null || \
    echo "EVA-CLIP pre-download skipped"

# ── RIFE models ──────────────────────────────────────────────────
RUN mkdir -p /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife && \
    wget -q -O /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife47.pth \
    "https://huggingface.co/wavespeed/misc/resolve/main/rife/rife47.pth" && \
    wget -q -O /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth \
    "https://huggingface.co/hfmaster/models-moved/resolve/main/rife/rife49.pth"

# ══════════════════════════════════════════════════════════════════
# SECTION 4: Patches + verification
# ══════════════════════════════════════════════════════════════════

# ── Patch handler: VHS_VideoCombine "gifs" → "images" ────────────
COPY patch_handler.py /tmp/patch_handler.py
RUN python3 /tmp/patch_handler.py && rm /tmp/patch_handler.py

# ── Verify installation ──────────────────────────────────────────
COPY verify_install.py /tmp/verify_install.py
RUN python3 /tmp/verify_install.py && rm /tmp/verify_install.py
