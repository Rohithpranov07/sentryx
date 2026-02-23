"""
tests/test_pipeline.py

Quick validation that the full pipeline works before running the server.
Run with: python tests/test_pipeline.py

Tests:
  1. Device detection
  2. Model loads successfully
  3. Fingerprinting works
  4. Ledger init, write, read
  5. End-to-end inference on a synthetic test image
"""

import sys
import os
import time
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import print as rprint

console = Console()


def make_test_image(mode: str = "real") -> Image.Image:
    """
    Create a synthetic test image.
    'real'  → naturalistic texture
    'fake'  → unnaturally smooth (like GAN output)
    """
    img = Image.new("RGB", (512, 512))
    pixels = np.zeros((512, 512, 3), dtype=np.uint8)

    if mode == "real":
        # Add natural-looking noise
        noise = np.random.normal(128, 30, (512, 512, 3)).clip(0, 255).astype(np.uint8)
        pixels = noise
    else:
        # Unnaturally smooth — uniform with very slight variation
        pixels[:] = [180, 140, 120]
        pixels += np.random.randint(0, 3, (512, 512, 3), dtype=np.uint8)

    return Image.fromarray(pixels)


def run_tests():
    console.rule("[bold blue]SENTRY-X Pipeline Validation[/bold blue]")
    results = []

    # ── Test 1: Device detection ──────────────────────────────────────────────
    console.print("\n[cyan]Test 1: Device detection[/cyan]")
    try:
        from utils.device import DEVICE
        console.print(f"  ✅ Device: [green]{DEVICE}[/green]")
        results.append(("Device Detection", "PASS", str(DEVICE)))
    except Exception as e:
        console.print(f"  ❌ FAILED: {e}")
        results.append(("Device Detection", "FAIL", str(e)))

    # ── Test 2: Model loading ─────────────────────────────────────────────────
    console.print("\n[cyan]Test 2: Model loading (may take 10-30s first time)[/cyan]")
    try:
        t = time.time()
        from models.detector import detector, classify_risk
        elapsed = time.time() - t
        console.print(f"  ✅ Model loaded in [green]{elapsed:.2f}s[/green]")
        results.append(("Model Load", "PASS", f"{elapsed:.2f}s"))
    except Exception as e:
        console.print(f"  ❌ FAILED: {e}")
        results.append(("Model Load", "FAIL", str(e)))
        return results  # Can't continue without model

    # ── Test 3: Fingerprinting ────────────────────────────────────────────────
    console.print("\n[cyan]Test 3: Fingerprinting[/cyan]")
    try:
        from utils.fingerprint import generate_fingerprints
        test_img = make_test_image("real")
        buf = io.BytesIO()
        test_img.save(buf, format="PNG")
        fp = generate_fingerprints(buf.getvalue(), test_img)
        assert len(fp["sha256"]) == 64
        assert len(fp["phash"]) > 0
        assert fp["fingerprint_id"].startswith("FP-")
        console.print(f"  ✅ Fingerprint ID: [green]{fp['fingerprint_id']}[/green]")
        console.print(f"  ✅ SHA256: {fp['sha256'][:20]}...")
        results.append(("Fingerprinting", "PASS", fp["fingerprint_id"]))
    except Exception as e:
        console.print(f"  ❌ FAILED: {e}")
        results.append(("Fingerprinting", "FAIL", str(e)))

    # ── Test 4: Ledger ────────────────────────────────────────────────────────
    console.print("\n[cyan]Test 4: Provenance ledger[/cyan]")
    try:
        from utils.ledger import init_ledger, register_threat, lookup_sha256, get_stats
        init_ledger()

        # Write a fake threat
        test_fp = {
            "sha256": "a" * 64,
            "phash": "0" * 16,
            "dhash": "0" * 16,
            "fingerprint_id": "FP-TEST000001",
        }
        test_verdict = {
            "risk_level": "red",
            "verdict": "Test Threat",
            "action": "block",
            "color_code": "#ef4444",
            "confidence": 0.95,
            "forensic_signals": ["test signal"],
        }
        register_threat(test_fp, test_verdict, "test.jpg", "test_platform")

        # Read it back
        entry = lookup_sha256("a" * 64)
        assert entry is not None
        assert entry["risk_level"] == "red"

        stats = get_stats()
        console.print(f"  ✅ Write + read verified")
        console.print(f"  ✅ Ledger stats: {stats}")
        results.append(("Ledger", "PASS", f"{stats['total_fingerprints']} entries"))
    except Exception as e:
        console.print(f"  ❌ FAILED: {e}")
        results.append(("Ledger", "FAIL", str(e)))

    # ── Test 5: End-to-end inference ──────────────────────────────────────────
    console.print("\n[cyan]Test 5: End-to-end inference[/cyan]")
    for mode in ["real", "fake"]:
        try:
            img = make_test_image(mode)
            t = time.time()
            confidence, signals = detector.predict(img)
            elapsed = (time.time() - t) * 1000
            risk = classify_risk(confidence)

            label = f"[green]{risk['verdict']}[/green]" if risk["risk_level"] == "green" else f"[red]{risk['verdict']}[/red]"
            console.print(f"  ✅ [{mode.upper()}] → {label} | confidence={confidence:.3f} | {elapsed:.0f}ms")
            console.print(f"       Signals: {signals[0]}")
            results.append((f"Inference ({mode})", "PASS", f"conf={confidence:.3f} | {elapsed:.0f}ms"))
        except Exception as e:
            console.print(f"  ❌ [{mode}] FAILED: {e}")
            results.append((f"Inference ({mode})", "FAIL", str(e)))

    # ── Summary table ─────────────────────────────────────────────────────────
    console.rule("\n[bold]Test Summary[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Test", style="cyan", width=25)
    table.add_column("Result", width=8)
    table.add_column("Detail", style="dim")

    for name, status, detail in results:
        color = "green" if status == "PASS" else "red"
        table.add_row(name, f"[{color}]{status}[/{color}]", detail)

    console.print(table)

    passed = sum(1 for _, s, _ in results if s == "PASS")
    total = len(results)
    console.print(f"\n[bold]Result: {passed}/{total} tests passed[/bold]")

    if passed == total:
        console.print("\n[bold green]✅ All systems operational. Start the server with:[/bold green]")
        console.print("[white]  cd sentry-x && uvicorn app.main:app --reload --port 8000[/white]")
    else:
        console.print("\n[bold red]⚠ Some tests failed. Fix before starting server.[/bold red]")

    return results


if __name__ == "__main__":
    run_tests()
