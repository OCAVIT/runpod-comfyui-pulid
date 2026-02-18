# RunPod ComfyUI + Flux Dev + PuLID-Flux + RIFE

Custom RunPod Serverless worker image for **YouTube Factory** project.

## What's included

| Component | Description |
|-----------|-------------|
| ComfyUI | Base from `runpod/worker-comfyui:3.3.1-flux` |
| Flux Dev fp8 | Pre-installed in base image |
| **PuLID-Flux** | Face consistency ([balazik/ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)) |
| PuLID model | `pulid_flux_v0.9.0.safetensors` (~1.1 GB) |
| InsightFace | AntelopeV2 face detection/recognition |
| EVA-CLIP | Pre-cached for fast cold start |
| **RIFE** | Frame interpolation for 60fps ([Fannovel16/ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)) |
| RIFE models | `rife47.pth` + `rife49.pth` pre-downloaded |
| CuPy | GPU acceleration for RIFE (CUDA 12.x) |

## Deployment

This repo is connected to RunPod for automatic Docker builds:

1. Push to `master` branch → RunPod auto-builds
2. New serverless endpoint uses the updated image

## Manual build (optional)

```bash
docker build --platform linux/amd64 -t comfyui-flux-pulid-rife .
docker tag comfyui-flux-pulid-rife ocavit/comfyui-flux-pulid-rife:v1
docker push ocavit/comfyui-flux-pulid-rife:v1
```

## PuLID workflow nodes

- `PulidFluxModelLoader` — loads `pulid_flux_v0.9.0.safetensors`
- `PulidFluxInsightFaceLoader` — face analysis
- `PulidFluxEvaClipLoader` — EVA-CLIP encoder
- `ApplyPulidFlux` — applies face identity to generation

## RIFE workflow nodes

- `RIFE VFI` — frame interpolation (2x, 4x multiplier)

### RIFE VFI parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ckpt_name` | Model file (`rife47.pth` or `rife49.pth`) | `rife47.pth` |
| `frames` | Input IMAGE tensor | required |
| `multiplier` | Frame rate multiplier (2 = 30→60fps) | 2 |
| `fast_mode` | No effect on rife4.5+ | `true` |
| `ensemble` | Ensemble averaging (better quality) | `true` |
| `scale_factor` | Resolution scale (lower = less VRAM) | `1.0` |
| `clear_cache_after_n_frames` | Prevent OOM on long sequences | 10 |

## GPU requirements

- Minimum: RTX 4090 (24 GB VRAM)
- Flux Dev fp8 (~12 GB) + PuLID (~4 GB) + RIFE (~2 GB)
- Container disk: 30 GB (models + cache)
