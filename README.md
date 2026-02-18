# RunPod ComfyUI + Flux Dev + PuLID-Flux

Custom RunPod Serverless worker image for **YouTube Factory** project.

## What's included

| Component | Description |
|-----------|-------------|
| ComfyUI | Base from `runpod/worker-comfyui:3.3.1-flux` |
| Flux Dev fp8 | Pre-installed in base image |
| **PuLID-Flux** | Face consistency custom node ([balazik/ComfyUI-PuLID-Flux](https://github.com/balazik/ComfyUI-PuLID-Flux)) |
| PuLID model | `pulid_flux_v0.9.0.safetensors` (~1.1 GB) |
| InsightFace | AntelopeV2 face detection/recognition |
| EVA-CLIP | Pre-cached for fast cold start |

## Deployment

This repo is connected to RunPod for automatic Docker builds:

1. Push to `main` branch → RunPod auto-builds
2. New serverless endpoint uses the updated image

## Manual build (optional)

```bash
docker build --platform linux/amd64 -t comfyui-flux-pulid .
docker tag comfyui-flux-pulid ocavit/comfyui-flux-pulid:v1
docker push ocavit/comfyui-flux-pulid:v1
```

## PuLID workflow nodes

Nodes available after deployment:

- `PulidFluxModelLoader` — loads `pulid_flux_v0.9.0.safetensors`
- `PulidFluxInsightFaceLoader` — face analysis
- `PulidFluxEvaClipLoader` — EVA-CLIP encoder
- `ApplyPulidFlux` — applies face identity to generation

## GPU requirements

- Minimum: RTX 4090 (24 GB VRAM)
- Flux Dev fp8 (~12 GB) + PuLID overhead (~4 GB)
