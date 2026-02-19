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

# ── Patch: remove 'providers' kwarg (not supported in latest insightface)
RUN cd /comfyui/custom_nodes/ComfyUI-PuLID-Flux && \
    sed -i "s/, providers=\[provider + 'ExecutionProvider',\]//" pulidflux.py

# ── InsightFace (face analysis for PuLID) ───────────────────────
# Install normally (with deps: onnxruntime, scikit-learn, scipy, opencv).
# insightface pulls onnxruntime (CPU) — does NOT conflict with PyTorch CUDA.
# Do NOT install onnxruntime-gpu separately — that breaks ComfyUI.
RUN pip install --no-cache-dir insightface

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

# ── RIFE Frame Interpolation ────────────────────────────────────
# https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    cd ComfyUI-Frame-Interpolation && \
    pip install --no-cache-dir -r requirements-no-cupy.txt

# ── CuPy for RIFE GPU acceleration (CUDA 12.x) ─────────────────
RUN pip install --no-cache-dir cupy-cuda12x

# ── Pre-download RIFE models (optional — will auto-download on first run)
RUN mkdir -p /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife && \
    (wget -q -O /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife47.pth \
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/rife47.pth" || \
    echo "rife47.pth download failed — will download on first use") && \
    (wget -q -O /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth \
    "https://github.com/styler00dollar/VSGAN-tensorrt-docker/releases/download/models/rife49.pth" || \
    echo "rife49.pth download failed — will download on first use")

# ── Verify installation (WILL FAIL BUILD if anything is wrong) ──
RUN echo "=== Debug insightface ===" && \
    python -c "\
import os, glob, onnxruntime, insightface; \
print('insightface version:', insightface.__version__); \
print('onnxruntime version:', onnxruntime.__version__); \
print('onnxruntime providers:', onnxruntime.get_available_providers()); \
model_dir = '/comfyui/models/insightface/models/antelopev2'; \
onnx_files = sorted(glob.glob(os.path.join(model_dir, '*.onnx'))); \
print(f'Found {len(onnx_files)} ONNX files in {model_dir}'); \
for f in onnx_files: \
    try: \
        sess = onnxruntime.InferenceSession(f, providers=['CPUExecutionProvider']); \
        print(f'  OK: {os.path.basename(f)} inputs={[i.name for i in sess.get_inputs()]}'); \
    except Exception as e: \
        print(f'  FAIL: {os.path.basename(f)} -> {e}'); \
print('--- Now trying FaceAnalysis ---'); \
from insightface.app import FaceAnalysis; \
import insightface.app.face_analysis as fa; \
print('FaceAnalysis source:', fa.__file__); \
import inspect; \
sig = inspect.signature(FaceAnalysis.__init__); \
print('FaceAnalysis.__init__ params:', list(sig.parameters.keys())); \
" && \
    echo "=== PuLID + RIFE ===" && \
    ls /comfyui/models/pulid/pulid_flux_v0.9.0.safetensors && \
    ls /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/ && \
    echo "=== CHECKS DONE ==="
