# RunPod Serverless ComfyUI worker with Flux Dev fp8 + PuLID-Flux + RIFE
# Auto-built by RunPod from GitHub repo.
#
# Base: official RunPod worker-comfyui with Flux support
# Added: PuLID-Flux (face consistency) + RIFE (frame interpolation for 60fps)
#
# Build: docker build --platform linux/amd64 -t comfyui-flux-pulid-rife .

FROM runpod/worker-comfyui:5.7.1-flux1-dev-fp8

# ── Install system tools (not in base image) ────────────────────
# build-essential + python3-dev needed to compile insightface from GitHub source
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget unzip build-essential python3-dev && \
    rm -rf /var/lib/apt/lists/*

# ── PuLID-Flux custom node (Enhanced version with attn_mask fix) ──────
# https://github.com/sipie800/ComfyUI-PuLID-Flux-Enhanced
# Fixes "forward_orig() got an unexpected keyword argument 'attn_mask'" error
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/sipie800/ComfyUI-PuLID-Flux-Enhanced.git && \
    cd ComfyUI-PuLID-Flux-Enhanced && \
    pip install --no-cache-dir -r requirements.txt

# ── InsightFace from GitHub (NOT PyPI!) ──────────────────────────
# PyPI insightface has FaceAnalysis.__init__(self, name, root) — NO **kwargs.
# GitHub version has FaceAnalysis.__init__(self, name, root, allowed_modules, **kwargs).
# PuLID-Flux passes providers= kwarg → needs **kwargs support.
# --force-reinstall overrides the PyPI version installed by PuLID-Flux requirements.
RUN pip install --no-cache-dir --force-reinstall \
    "insightface @ git+https://github.com/deepinsight/insightface.git@master#subdirectory=python-package"

# ── PuLID Flux model (~1.1 GB) ─────────────────────────────────
RUN mkdir -p /comfyui/models/pulid && \
    wget -q -O /comfyui/models/pulid/pulid_flux_v0.9.0.safetensors \
    "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.0.safetensors"

# ── InsightFace AntelopeV2 models (face detection/recognition) ─
# Zip contains antelopev2/*.onnx — extract to models/ (NOT models/antelopev2/)
RUN mkdir -p /comfyui/models/insightface/models && \
    cd /tmp && \
    wget -q -O antelopev2.zip \
    "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip" && \
    unzip -o antelopev2.zip -d /comfyui/models/insightface/models/ && \
    ls /comfyui/models/insightface/models/antelopev2/ && \
    rm antelopev2.zip

# ── EVA-CLIP (pre-cache for fast cold start) ────────────────────
RUN mkdir -p /root/.cache/huggingface && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('QuanSun/EVA-CLIP', 'EVA02_CLIP_L_336_psz14_s6B.pt', \
    cache_dir='/root/.cache/huggingface')" 2>/dev/null || \
    echo "EVA-CLIP pre-download skipped (will auto-download on first run)"

# ── VideoHelperSuite (VHS_LoadVideo, VHS_VideoCombine) ────────────
# Required for RIFE: loads video as frames, combines frames back to video
# https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    pip install --no-cache-dir -r requirements.txt

# ── RIFE Frame Interpolation ────────────────────────────────────
# https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    cd ComfyUI-Frame-Interpolation && \
    pip install --no-cache-dir -r requirements-no-cupy.txt

# ── CuPy for RIFE GPU acceleration (CUDA 12.x) ─────────────────
RUN pip install --no-cache-dir cupy-cuda12x

# ── Pre-download RIFE models (from HuggingFace — GitHub URLs are dead) ──
RUN mkdir -p /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife && \
    wget -q -O /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife47.pth \
    "https://huggingface.co/jasonot/mycomfyui/resolve/main/rife47.pth" && \
    wget -q -O /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth \
    "https://huggingface.co/jasonot/mycomfyui/resolve/main/rife49.pth" && \
    ls -lh /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/

# ── Verify installation ──────────────────────────────────────────
COPY verify_install.py /tmp/verify_install.py
RUN python3 /tmp/verify_install.py && rm /tmp/verify_install.py
