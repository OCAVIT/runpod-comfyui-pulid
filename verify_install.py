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

# 5. Check PuLID-Flux-Enhanced node (with attn_mask fix)
print("\n--- PuLID-Flux-Enhanced check ---")
pulid_src = "/comfyui/custom_nodes/ComfyUI-PuLID-Flux-Enhanced/pulidflux.py"
if os.path.exists(pulid_src):
    with open(pulid_src) as fp:
        content = fp.read()
    # Check for attn_mask fix (should accept **kwargs or attn_mask parameter)
    if "attn_mask" in content or "**kwargs" in content:
        print(f"OK: {pulid_src} found with attn_mask support")
    else:
        print(f"WARNING: attn_mask support unclear in {pulid_src}")
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

# 7. Check VideoHelperSuite
print("\n--- VideoHelperSuite ---")
vhs_dir = "/comfyui/custom_nodes/ComfyUI-VideoHelperSuite"
if os.path.exists(vhs_dir):
    print(f"OK: {vhs_dir} installed")
else:
    print(f"MISSING: {vhs_dir}")

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

# 9. Verify RIFE model files are valid (not empty / truncated)
print("\n--- RIFE model validation ---")
rife_ok = True
for model_name in ["rife47.pth", "rife49.pth"]:
    model_path = os.path.join(rife_dir, model_name) if os.path.exists(rife_dir) else ""
    if model_path and os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        if size_mb > 1:
            print(f"  OK: {model_name} ({size_mb:.1f} MB)")
        else:
            print(f"  BAD: {model_name} too small ({size_mb:.2f} MB) — download likely failed")
            rife_ok = False
    else:
        print(f"  MISSING: {model_name}")
        rife_ok = False

# 10. Summary
print("\n" + "=" * 60)
if has_kwargs and len(onnx_files) >= 4 and rife_ok:
    print("ALL CHECKS PASSED")
else:
    issues = []
    if not has_kwargs:
        issues.append("insightface missing **kwargs")
    if len(onnx_files) < 4:
        issues.append(f"only {len(onnx_files)} ONNX files (need 4+)")
    if not rife_ok:
        issues.append("RIFE models missing or corrupted")
    print(f"ISSUES: {', '.join(issues)}")
print("=" * 60)
