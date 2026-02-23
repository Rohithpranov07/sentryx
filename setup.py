#!/usr/bin/env python3
"""
setup.py — SENTRY-X one-command setup

Detects your environment and installs the right packages.
Run once before starting the server.

Usage:
    python setup.py
"""

import subprocess
import sys
import platform

def run(cmd, check=True):
    print(f"  → {cmd}")
    result = subprocess.run(cmd, shell=True, check=check)
    return result.returncode == 0

def main():
    print("=" * 55)
    print("  SENTRY-X — Environment Setup")
    print("=" * 55)

    os_name = platform.system()
    py_version = sys.version
    print(f"\n  OS: {os_name}")
    print(f"  Python: {py_version[:6]}")

    # ── Detect Apple Silicon ──────────────────────────────────────────────────
    is_apple_silicon = (
        os_name == "Darwin" and
        platform.processor() == "arm"
    )

    if is_apple_silicon:
        print("  Chip: Apple Silicon (MPS available)")
    elif os_name == "Darwin":
        print("  Chip: Intel Mac (CPU only)")
    else:
        print("  Chip: Non-Mac")

    print("\n  Installing dependencies...\n")

    # ── Core packages (all platforms) ─────────────────────────────────────────
    core = [
        "fastapi==0.111.0",
        "uvicorn[standard]==0.29.0",
        "python-multipart==0.0.9",
        "Pillow==10.3.0",
        "imagehash==4.3.1",
        "numpy==1.26.4",
        "pydantic==2.7.1",
        "aiofiles==23.2.1",
        "python-dotenv==1.0.1",
        "rich==13.7.1",
        "timm==1.0.3",
    ]

    # ── PyTorch: Mac needs specific build ─────────────────────────────────────
    if os_name == "Darwin":
        # Mac: standard pip torch includes MPS support since 2.0
        torch_pkg = "torch==2.3.0 torchvision==0.18.0"
        print("  [Mac] Installing PyTorch with MPS support...")
    else:
        # Linux/Windows with CUDA
        torch_pkg = "torch torchvision --index-url https://download.pytorch.org/whl/cu121"
        print("  [Linux/Windows] Installing PyTorch with CUDA support...")

    success = run(f"{sys.executable} -m pip install {torch_pkg}")
    if not success:
        print("  ⚠ PyTorch install failed — trying CPU-only fallback...")
        run(f"{sys.executable} -m pip install torch torchvision")

    # ── Install core packages ─────────────────────────────────────────────────
    run(f"{sys.executable} -m pip install {' '.join(core)}")

    # ── Optional: opencv ──────────────────────────────────────────────────────
    print("\n  Installing optional CV packages...")
    run(f"{sys.executable} -m pip install opencv-python-headless==4.9.0.80", check=False)

    # ── Create data directory ─────────────────────────────────────────────────
    import os
    os.makedirs("data/samples", exist_ok=True)
    print("\n  Created ./data/samples/")

    print("\n" + "=" * 55)
    print("  ✅ Setup complete!")
    print("=" * 55)
    print("""
  Next steps:
  
  1. Run validation tests:
       python tests/test_pipeline.py

  2. Start the API server:
       uvicorn app.main:app --reload --port 8000

  3. Open interactive docs:
       http://localhost:8000/docs

  4. Test with curl:
       curl -X POST http://localhost:8000/v1/analyze \\
         -F "file=@your_image.jpg" \\
         -F "platform_id=my_platform"
""")

if __name__ == "__main__":
    main()
