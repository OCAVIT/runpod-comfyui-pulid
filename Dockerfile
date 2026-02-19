# RunPod Serverless ComfyUI worker with Flux Dev fp8 + PuLID-Flux + RIFE
# Auto-built by RunPod from GitHub repo.
#
# Base: official RunPod worker-comfyui with Flux support
# Added: PuLID-Flux (face consistency) + RIFE (frame interpolation for 60fps)
#
# Build: docker build --platform linux/amd64 -t comfyui-flux-pulid-rife .

FROM runpod/worker-comfyui:5.7.1-flux1-dev-fp8

# ── Install system tools (not in base image) ────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends wget unzip && \
    rm -rf /var/lib/apt/lists/*

# ── PuLID-Flux custom node ──────────────────────────────────────
# https://github.com/balazik/ComfyUI-PuLID-Flux
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/balazik/ComfyUI-PuLID-Flux.git && \
    cd ComfyUI-PuLID-Flux && \
    pip install --no-cache-dir -r requirements.txt

# ── InsightFace + onnxruntime CPU (face analysis for PuLID) ─────
# onnxruntime (CPU) for ONNX face models — does NOT conflict with PyTorch CUDA.
# Do NOT install onnxruntime-gpu — it overwrites PyTorch CUDA and breaks ComfyUI.
RUN pip install --no-cache-dir --no-deps insightface && \
    pip install --no-cache-dir onnxruntime prettytable easydict albumentations

# ── PuLID Flux model (~1.1 GB) ─────────────────────────────────
RUN mkdir -p /comfyui/models/pulid && \
    wget -q -O /comfyui/models/pulid/pulid_flux_v0.9.0.safetensors \
    "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.0.safetensors"

# ── InsightFace AntelopeV2 models (face detection/recognition) ─
RUN mkdir -p /comfyui/models/insightface/models/antelopev2 && \
    cd /tmp && \
    wget -q -O antelopev2.zip \
    "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip" && \
    unzip -o antelopev2.zip -d /comfyui/models/insightface/models/antelopev2/ && \
    rm antelopev2.zip

# ── EVA-CLIP (pre-cache for fast cold start) ────────────────────
RUN mkdir -p /root/.cache/huggingface && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('QuanSun/EVA-CLIP', 'EVA02_CLIP_L_336_psz14_s6B.pt', \
    cache_dir='/root/.cache/huggingface')" 2>/dev/null || \
    echo "EVA-CLIP pre-download skipped (will auto-download on first run)"

# ── RIFE Frame Interpolation ────────────────────────────────────
# https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    cd ComfyUI-Frame-Interpolation && \
    pip install --no-cache-dir -r requirements-no-cupy.txt

# ── CuPy for RIFE GPU acceleration (CUDA 12.x) ─────────────────
RUN pip install --no-cache-dir cupy-cuda12x

# ── Pre-download RIFE models (avoid cold-start download) ────────
RUN mkdir -p /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife && \
    wget -q -O /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife47.pth \
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/rife47.pth" && \
    wget -q -O /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth \
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/rife49.pth"

# ── Verify installation ────────────────────────────────────────
RUN echo "=== Checking PuLID ===" && \
    ls /comfyui/models/pulid/pulid_flux_v0.9.0.safetensors && \
    ls /comfyui/custom_nodes/ComfyUI-PuLID-Flux/ && \
    echo "=== Checking onnxruntime ===" && \
    python -c "import onnxruntime; print('  onnxruntime', onnxruntime.__version__)" && \
    echo "=== Checking RIFE ===" && \
    ls /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/ && \
    echo "=== Checking Python imports ===" && \
    python -c "import insightface; print('  insightface OK')" && \
    python -c "import cupy; print('  cupy OK')" || true && \
    echo "=== All components installed ==="
