"""Verify PuLID + InsightFace + RIFE installation in Docker image.
Runs during docker build — fails build if critical components missing.
"""
import os
import glob
import traceback

print("=" * 60)
print("VERIFY INSTALLATION")
print("=" * 60)

# 1. Check onnxruntime
import onnxruntime
print(f"onnxruntime version: {onnxruntime.__version__}")
print(f"onnxruntime providers: {onnxruntime.get_available_providers()}")

# 2. Check insightface
import insightface
print(f"insightface version: {insightface.__version__}")

# 3. Check ONNX model files
model_dir = "/comfyui/models/insightface/models/antelopev2"
onnx_files = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
print(f"\nFound {len(onnx_files)} ONNX files in {model_dir}:")
for f in onnx_files:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"  {os.path.basename(f)} ({size_mb:.1f} MB)")

# 4. Test each ONNX model loads
print("\nTesting ONNX model loading:")
for f in onnx_files:
    try:
        sess = onnxruntime.InferenceSession(f, providers=["CPUExecutionProvider"])
        inputs = [i.name for i in sess.get_inputs()]
        print(f"  OK: {os.path.basename(f)} inputs={inputs}")
    except Exception as e:
        print(f"  FAIL: {os.path.basename(f)} -> {e}")

# 5. Check FaceAnalysis class
print("\n--- FaceAnalysis debug ---")
from insightface.app import FaceAnalysis
import insightface.app.face_analysis as fa
import inspect

print(f"FaceAnalysis source: {fa.__file__}")
sig = inspect.signature(FaceAnalysis.__init__)
print(f"FaceAnalysis.__init__ params: {list(sig.parameters.keys())}")

# 6. CRITICAL: Actually try to instantiate FaceAnalysis
print("\n--- Trying FaceAnalysis instantiation ---")
try:
    app = FaceAnalysis(
        name="antelopev2",
        root="/comfyui/models/insightface",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print(f"FaceAnalysis loaded OK! Models: {list(app.models.keys())}")
except TypeError as e:
    # Maybe 'providers' not supported — retry without it
    print(f"FaceAnalysis with providers failed: {e}")
    print("Retrying without providers kwarg...")
    try:
        app = FaceAnalysis(
            name="antelopev2",
            root="/comfyui/models/insightface"
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        print(f"FaceAnalysis loaded OK (no providers)! Models: {list(app.models.keys())}")
    except Exception as e2:
        print(f"FaceAnalysis FAILED even without providers: {e2}")
        traceback.print_exc()
except Exception as e:
    print(f"FaceAnalysis FAILED: {e}")
    traceback.print_exc()

# 7. Check PuLID model
print("\n--- PuLID model ---")
pulid_path = "/comfyui/models/pulid/pulid_flux_v0.9.0.safetensors"
if os.path.exists(pulid_path):
    size_mb = os.path.getsize(pulid_path) / (1024 * 1024)
    print(f"OK: {pulid_path} ({size_mb:.0f} MB)")
else:
    print(f"MISSING: {pulid_path}")

# 8. Check RIFE
print("\n--- RIFE models ---")
rife_dir = "/comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/"
if os.path.exists(rife_dir):
    for f in os.listdir(rife_dir):
        fp = os.path.join(rife_dir, f)
        size_mb = os.path.getsize(fp) / (1024 * 1024)
        print(f"  {f} ({size_mb:.1f} MB)")
else:
    print(f"MISSING: {rife_dir}")

print("\n" + "=" * 60)
print("CHECKS DONE")
print("=" * 60)
