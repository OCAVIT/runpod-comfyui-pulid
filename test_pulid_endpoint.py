#!/usr/bin/env python3
"""Test PuLID face consistency on RunPod endpoint.

Usage:
    python test_pulid_endpoint.py <ENDPOINT_ID>

Generates:
1. Portrait reference (512x512) with a face
2. Two scenes (1344x768) using the same face via PuLID

If PuLID works, both scenes will have the SAME face as the reference.
"""
import sys
import os
import base64
import time
import requests
from pathlib import Path

ENDPOINT_ID = sys.argv[1] if len(sys.argv) > 1 else "i0n2m0ow6c7a2n"
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

if not RUNPOD_API_KEY:
    print("ERROR: Set RUNPOD_API_KEY environment variable")
    sys.exit(1)

API_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"
HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

def generate_image(prompt, width, height, face_image_b64=None):
    """Generate image via RunPod ComfyUI endpoint."""
    # Minimal workflow: text2img or PuLID-based
    if face_image_b64:
        # PuLID workflow (simplified - actual workflow depends on your ComfyUI setup)
        workflow = {
            "prompt": {
                # This is a PLACEHOLDER - you need to replace with actual ComfyUI workflow JSON
                # that uses PulidFluxInsightFaceLoader + ApplyPulid nodes
                "text_prompt": prompt,
                "width": width,
                "height": height,
                "face_reference": face_image_b64
            }
        }
    else:
        # Standard Flux workflow (no PuLID)
        workflow = {
            "prompt": {
                "text_prompt": prompt,
                "width": width,
                "height": height
            }
        }

    payload = {"input": {"workflow": workflow}}

    print(f"Generating: {prompt[:50]}... ({width}x{height})")
    resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=300)

    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code} {resp.text}")
        return None

    result = resp.json()

    # Extract base64 image from result
    # (Actual path depends on your ComfyUI output node)
    if "output" in result and "images" in result["output"]:
        return result["output"]["images"][0]  # base64 string

    print(f"WARNING: No image in result: {result}")
    return None

def save_image(b64_data, filename):
    """Save base64 image to file."""
    img_data = base64.b64decode(b64_data)
    Path(filename).write_bytes(img_data)
    print(f"Saved: {filename} ({len(img_data)} bytes)")

def main():
    print("=" * 60)
    print("PuLID Face Consistency Test")
    print("=" * 60)
    print(f"Endpoint: {ENDPOINT_ID}")
    print()

    # Step 1: Generate portrait reference (no PuLID, just standard Flux)
    print("Step 1: Generate portrait reference...")
    portrait_prompt = "A portrait photo of a middle-aged man with short brown hair and blue eyes, looking at camera, professional lighting, 8k, high detail"
    portrait_b64 = generate_image(portrait_prompt, 512, 512)

    if not portrait_b64:
        print("FAILED: Could not generate portrait")
        sys.exit(1)

    save_image(portrait_b64, "portrait_reference.png")

    # Step 2: Generate scene 1 with PuLID (same face)
    print("\nStep 2: Generate scene 1 with PuLID...")
    scene1_prompt = "The same man standing in a forest, wearing a jacket, cinematic lighting, 8k"
    scene1_b64 = generate_image(scene1_prompt, 1344, 768, face_image_b64=portrait_b64)

    if scene1_b64:
        save_image(scene1_b64, "scene1_pulid.png")
    else:
        print("WARNING: Scene 1 generation failed")

    # Step 3: Generate scene 2 with PuLID (same face)
    print("\nStep 3: Generate scene 2 with PuLID...")
    scene2_prompt = "The same man in a modern office, sitting at desk, professional environment, 8k"
    scene2_b64 = generate_image(scene2_prompt, 1344, 768, face_image_b64=portrait_b64)

    if scene2_b64:
        save_image(scene2_b64, "scene2_pulid.png")
    else:
        print("WARNING: Scene 2 generation failed")

    # Summary
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("Check generated images:")
    print("  1. portrait_reference.png - original face")
    print("  2. scene1_pulid.png - should have SAME face")
    print("  3. scene2_pulid.png - should have SAME face")
    print()
    print("If faces are identical → PuLID works! ✓")
    print("If faces are different → PuLID failed ✗")

if __name__ == "__main__":
    main()
