"""Patch RunPod worker handler to support VHS_VideoCombine 'gifs' output key.

Problem: VHS_VideoCombine returns output under "gifs" key, but the RunPod
worker-comfyui handler only checks for "images" key. This causes video outputs
to be silently dropped, returning {'images': [], 'status': 'success_no_images'}.

Fix: Normalize "gifs" → "images" before the handler processes output nodes.
"""

import os
import glob

# Known handler paths in runpod/worker-comfyui Docker images
SEARCH_PATHS = [
    "/rp_handler.py",
    "/handler.py",
    "/src/rp_handler.py",
    "/src/handler.py",
    "/app/rp_handler.py",
    "/app/handler.py",
    "/app/src/rp_handler.py",
]

# Also search by glob
SEARCH_GLOBS = [
    "/*handler*.py",
    "/src/*handler*.py",
    "/app/*handler*.py",
    "/app/src/*handler*.py",
]


def find_handler():
    """Find the RunPod handler file."""
    # Try exact paths first
    for path in SEARCH_PATHS:
        if os.path.exists(path):
            with open(path) as f:
                if '"images" in node_output' in f.read():
                    return path

    # Try globs
    for pattern in SEARCH_GLOBS:
        for path in glob.glob(pattern):
            if path.endswith(".py"):
                with open(path) as f:
                    if '"images" in node_output' in f.read():
                        return path

    return None


def patch_handler(filepath):
    """Patch the handler to also process 'gifs' output key from VHS nodes."""
    with open(filepath) as f:
        content = f.read()

    # Already patched?
    if '"gifs" in node_output' in content:
        print(f"  Already patched: {filepath}")
        return True

    # Strategy: find the output processing loop and add gifs normalization
    # The handler iterates: for node_id, node_output in outputs.items()
    # and checks: if "images" in node_output
    #
    # We add normalization right before: if gifs exists, copy to images

    target = 'if "images" in node_output'

    if target not in content:
        print(f"  ERROR: Pattern not found in {filepath}")
        return False

    # Add normalization before the "images" check
    patch = (
        '# VHS_VideoCombine uses "gifs" key for video output — normalize to "images"\n'
        '            if "gifs" in node_output and "images" not in node_output:\n'
        '                node_output["images"] = node_output["gifs"]\n'
        '            if "images" in node_output'
    )

    content = content.replace(target, patch, 1)

    with open(filepath, "w") as f:
        f.write(content)

    print(f"  Patched successfully: {filepath}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Patching RunPod handler for VHS video output support")
    print("=" * 60)

    handler = find_handler()
    if handler:
        print(f"Found handler: {handler}")
        success = patch_handler(handler)
        if not success:
            print("WARNING: Patch failed, video output may not work")
    else:
        print("WARNING: Handler not found. Searched:")
        for p in SEARCH_PATHS:
            print(f"  {p} — {'exists' if os.path.exists(p) else 'not found'}")
        print("Video output from VHS_VideoCombine may not work.")

    print("=" * 60)
