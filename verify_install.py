"""Verify PuLID + InsightFace + RIFE installation in Docker image."""
import os
import glob
import traceback
import inspect

print("=" * 60)
print("VERIFY INSTALLATION")
print("=" * 60)

# 1. Check insightface version and FaceAnalysis signature
import insightface
print(f"insightface version: {insightface.__version__}")

from insightface.app import FaceAnalysis
sig = inspect.signature(FaceAnalysis.__init__)
params = list(sig.parameters.keys())
print(f"FaceAnalysis.__init__ params: {params}")

has_kwargs = any(
    p.kind == inspect.Parameter.VAR_KEYWORD
    for p in sig.parameters.values()
)
print(f"Has **kwargs: {has_kwargs}")

if not has_kwargs:
    print("WARNING: FaceAnalysis missing **kwargs — providers won't be passed!")
    print("This means PuLID-Flux will FAIL. Need GitHub version of insightface.")

# 2. Check onnxruntime
import onnxruntime
print(f"\nonnxruntime version: {onnxruntime.__version__}")
print(f"onnxruntime providers: {onnxruntime.get_available_providers()}")

# 3. Check ONNX model files
model_dir = "/comfyui/models/insightface/models/antelopev2"
onnx_files = sorted(glob.glob(os.path.join(model_dir, "*.onnx")))
print(f"\nFound {len(onnx_files)} ONNX files in {model_dir}:")
for f in onnx_files:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f"  {os.path.basename(f)} ({size_mb:.1f} MB)")

# 4. Test model_zoo.get_model for each file (should work with CPU)
print("\nmodel_zoo.get_model test:")
try:
    from insightface.model_zoo import model_zoo
    for f in onnx_files:
        try:
            m = model_zoo.get_model(f, providers=['CPUExecutionProvider'])
            if m is None:
                print(f"  {os.path.basename(f)} -> None (not recognized)")
            else:
                print(f"  {os.path.basename(f)} -> taskname={m.taskname}")
        except Exception as e:
            print(f"  {os.path.basename(f)} -> ERROR: {e}")
except Exception as e:
    print(f"model_zoo import failed: {e}")

# 5. Check PuLID-Flux node is not patched (providers kwarg should be present)
print("\n--- PuLID-Flux pulidflux.py check ---")
pulid_src = "/comfyui/custom_nodes/ComfyUI-PuLID-Flux/pulidflux.py"
if os.path.exists(pulid_src):
    with open(pulid_src) as fp:
        content = fp.read()
    if "providers=" in content:
        print("OK: providers= kwarg present in pulidflux.py")
    else:
        print("WARNING: providers= kwarg MISSING — sed patch was applied but shouldn't be!")
else:
    print(f"MISSING: {pulid_src}")

# 6. Check PuLID model
print("\n--- PuLID model ---")
pulid_path = "/comfyui/models/pulid/pulid_flux_v0.9.0.safetensors"
if os.path.exists(pulid_path):
    size_mb = os.path.getsize(pulid_path) / (1024 * 1024)
    print(f"OK: {pulid_path} ({size_mb:.0f} MB)")
else:
    print(f"MISSING: {pulid_path}")

# 7. Check RIFE
print("\n--- RIFE models ---")
rife_dir = "/comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/"
if os.path.exists(rife_dir):
    for f in os.listdir(rife_dir):
        fp = os.path.join(rife_dir, f)
        size_mb = os.path.getsize(fp) / (1024 * 1024)
        print(f"  {f} ({size_mb:.1f} MB)")
else:
    print(f"MISSING: {rife_dir}")

# 8. Summary
print("\n" + "=" * 60)
if has_kwargs and len(onnx_files) >= 4:
    print("ALL CHECKS PASSED")
else:
    issues = []
    if not has_kwargs:
        issues.append("insightface missing **kwargs")
    if len(onnx_files) < 4:
        issues.append(f"only {len(onnx_files)} ONNX files (need 4+)")
    print(f"ISSUES: {', '.join(issues)}")
print("=" * 60)
