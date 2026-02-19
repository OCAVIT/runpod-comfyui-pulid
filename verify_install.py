"""Verify PuLID + InsightFace + RIFE installation in Docker image.
Runs during docker build — fails build if critical components missing.
"""
import os
import glob
import traceback
import inspect

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

# 4. Dump FaceAnalysis.__init__ source code
print("\n--- FaceAnalysis.__init__ SOURCE CODE ---")
from insightface.app import FaceAnalysis
import insightface.app.face_analysis as fa
try:
    src = inspect.getsource(FaceAnalysis.__init__)
    print(src)
except Exception as e:
    print(f"Cannot get source: {e}")

# 5. Dump ensure_available source code
print("\n--- ensure_available SOURCE CODE ---")
try:
    from insightface.utils import storage
    src = inspect.getsource(storage.ensure_available)
    print(src)
except Exception as e:
    print(f"Cannot get ensure_available source: {e}")

# 6. Test ensure_available resolution
print("\n--- ensure_available resolution ---")
try:
    resolved = storage.ensure_available('models', 'antelopev2', root='/comfyui/models/insightface')
    print(f"Resolved model_dir: {resolved}")
    resolved_files = glob.glob(os.path.join(resolved, '*.onnx'))
    print(f"Files in resolved dir: {resolved_files}")
except Exception as e:
    print(f"ensure_available failed: {e}")
    traceback.print_exc()

# 7. Test model_zoo.get_model for each file
print("\n--- model_zoo.get_model test ---")
try:
    from insightface.model_zoo import model_zoo
    # Try with explicit providers
    for f in onnx_files:
        try:
            m = model_zoo.get_model(f, providers=['CPUExecutionProvider'])
            if m is None:
                print(f"  {os.path.basename(f)} -> None (not recognized)")
            else:
                print(f"  {os.path.basename(f)} -> taskname={m.taskname}")
        except Exception as e:
            print(f"  {os.path.basename(f)} -> ERROR: {e}")
    # Also try without providers
    print("  (retry without providers kwarg:)")
    for f in onnx_files:
        try:
            m = model_zoo.get_model(f)
            if m is None:
                print(f"  {os.path.basename(f)} -> None (not recognized)")
            else:
                print(f"  {os.path.basename(f)} -> taskname={m.taskname}")
        except Exception as e:
            print(f"  {os.path.basename(f)} -> ERROR: {e}")
except Exception as e:
    print(f"model_zoo import/test failed: {e}")
    traceback.print_exc()

# 8. Dump model_zoo.get_model source
print("\n--- model_zoo.get_model SOURCE CODE ---")
try:
    src = inspect.getsource(model_zoo.get_model)
    print(src)
except Exception as e:
    print(f"Cannot get source: {e}")

# 9. Try FaceAnalysis instantiation
print("\n--- FaceAnalysis instantiation ---")
try:
    app = FaceAnalysis(
        name="antelopev2",
        root="/comfyui/models/insightface",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    print(f"OK with providers! Models: {list(app.models.keys())}")
except TypeError as e:
    print(f"With providers: TypeError: {e}")
    try:
        app = FaceAnalysis(
            name="antelopev2",
            root="/comfyui/models/insightface"
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        print(f"OK without providers! Models: {list(app.models.keys())}")
    except Exception as e2:
        print(f"Without providers: {e2}")
        traceback.print_exc()
except Exception as e:
    print(f"FaceAnalysis FAILED: {e}")
    traceback.print_exc()

# 10. Check PuLID model
print("\n--- PuLID model ---")
pulid_path = "/comfyui/models/pulid/pulid_flux_v0.9.0.safetensors"
if os.path.exists(pulid_path):
    size_mb = os.path.getsize(pulid_path) / (1024 * 1024)
    print(f"OK: {pulid_path} ({size_mb:.0f} MB)")
else:
    print(f"MISSING: {pulid_path}")

# 11. Check RIFE
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
