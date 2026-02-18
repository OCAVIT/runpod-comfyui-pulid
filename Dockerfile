# RunPod Serverless ComfyUI worker with Flux Dev fp8 + PuLID-Flux
# Auto-built by RunPod from GitHub repo.
#
# Base: official RunPod worker-comfyui with Flux support
# Added: PuLID-Flux custom nodes + models for face consistency
#
# Build: docker build --platform linux/amd64 -t comfyui-flux-pulid .

FROM runpod/worker-comfyui:3.3.1-flux

# ── PuLID-Flux custom node ──────────────────────────────────────
# https://github.com/balazik/ComfyUI-PuLID-Flux
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/balazik/ComfyUI-PuLID-Flux.git && \
    cd ComfyUI-PuLID-Flux && \
    pip install --no-cache-dir -r requirements.txt

# ── InsightFace + ONNX (face analysis for PuLID) ───────────────
RUN pip install --no-cache-dir insightface onnxruntime-gpu facexlib

# ── PuLID Flux model (~1.1 GB) ─────────────────────────────────
RUN mkdir -p /comfyui/models/pulid && \
    wget -q --show-progress -O /comfyui/models/pulid/pulid_flux_v0.9.0.safetensors \
    "https://huggingface.co/guozinan/PuLID/resolve/main/pulid_flux_v0.9.0.safetensors"

# ── InsightFace AntelopeV2 models (face detection/recognition) ─
RUN mkdir -p /comfyui/models/insightface/models/antelopev2 && \
    cd /tmp && \
    wget -q --show-progress -O antelopev2.zip \
    "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip" && \
    unzip -o antelopev2.zip -d /comfyui/models/insightface/models/antelopev2/ && \
    rm antelopev2.zip

# ── EVA-CLIP (auto-downloads on first run, but pre-cache for fast cold start)
# The model EVA02_CLIP_L_336_psz14_s6B.pt will be fetched by PuLID on first use.
# Pre-downloading to avoid cold start delay:
RUN mkdir -p /root/.cache/huggingface && \
    python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download('QuanSun/EVA-CLIP', 'EVA02_CLIP_L_336_psz14_s6B.pt', \
    cache_dir='/root/.cache/huggingface')" 2>/dev/null || \
    echo "EVA-CLIP pre-download skipped (will auto-download on first run)"

# ── Verify installation ────────────────────────────────────────
RUN python -c "import insightface; print('InsightFace OK')" && \
    ls /comfyui/models/pulid/pulid_flux_v0.9.0.safetensors && \
    ls /comfyui/custom_nodes/ComfyUI-PuLID-Flux/nodes.py && \
    echo '=== All PuLID components installed ==='
