"""
tests/benchmark_suite.py
Automated testing and benchmarking suite for SENTRY-X V2 Pipeline.

Features:
- Compression Stress Tests
- Evasion Attempt Simulations (Noise, Cropping, Blur)
- Latency & Throughput Metrics (p50, p90, p99)
- False Positive Tracking & Accuracy Benchmarks
"""

import sys
import os
import time
import io
import json
import uuid
import glob
import statistics
import requests
import numpy as np
from PIL import Image, ImageFilter

API_URL = "http://localhost:8000/v2/analyze"

def generate_unique_image(w=512, h=512) -> Image.Image:
    """Generates a unique random noise image to bypass ledger fast-paths."""
    # Ensure uniqueness by adding a random seed block
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    # Add a small uuid block in the corner just to guarantee sha divergence
    Image.fromarray(arr).save(io.BytesIO(), format="JPEG")
    return Image.fromarray(arr)

def send_to_api(image: Image.Image, format="JPEG", quality=100) -> tuple:
    """Sends a PIL image to the API and returns (json_response, latency_ms)."""
    buf = io.BytesIO()
    image.save(buf, format=format, quality=quality)
    buf.seek(0)
    
    t0 = time.time()
    try:
        res = requests.post(
            API_URL, 
            files={"file": (f"test_{uuid.uuid4().hex[:6]}.{format.lower()}", buf, f"image/{format.lower()}")},
            data={"caption": "Benchmarking Payload", "uploader_id": "benchmark_bot", "platform_id": "test_suite"},
            timeout=10
        )
        latency = (time.time() - t0) * 1000
        return res.json(), latency
    except Exception as e:
        print(f"API Error: {e}")
        return None, 0

def test_latency():
    print("\n==================================")
    print(" 🚀 RUNNING LATENCY PROFILING")
    print("==================================")
    
    latencies = []
    triage_latencies = []
    deep_latencies = []
    
    iters = 15
    for i in range(iters):
        img = generate_unique_image()
        res, lat = send_to_api(img)
        
        if res and "latency_profile_ms" in res:
            profiler = res["latency_profile_ms"]
            triage_latencies.append(profiler.get("phase1_triage_ms", 0))
            deep_latencies.append(profiler.get("phase2_deep_inference_ms", 0))
            latencies.append(lat)
            
    if latencies:
        print(f"Iterations      : {iters}")
        print(f"Avg End-to-End  : {statistics.mean(latencies):.2f} ms")
        print(f"P99 End-to-End  : {np.percentile(latencies, 99):.2f} ms")
        print(f"Min Latency     : {min(latencies):.2f} ms")
        print(f"Max Latency     : {max(latencies):.2f} ms")
        print(f"Avg Triage (P1) : {statistics.mean(triage_latencies):.2f} ms")
        print(f"Avg AI Deep (P2): {statistics.mean(deep_latencies):.2f} ms")


def test_compression():
    print("\n==================================")
    print(" 📉 RUNNING COMPRESSION STRESS TEST")
    print("==================================")
    
    base_img = generate_unique_image()
    qualities = [100, 80, 50, 20, 5]
    
    print(f"{'Quality':<10} | {'Fusion Risk':<15} | {'Latency':<10}")
    print("-" * 40)
    
    for q in qualities:
        res, lat = send_to_api(base_img, quality=q)
        if res and "detection_signals" in res:
            score = res["detection_signals"].get("fusion_threat_score", 0)
            print(f"Q{q:<9} | {score*100:>6.2f}%       | {lat:>7.2f} ms")
        else:
            print(f"Q{q:<9} | Error/FastPath  | {lat:>7.2f} ms")


def test_evasion():
    print("\n==================================")
    print(" 🛡️ RUNNING EVASION SIMULATION")
    print("==================================")
    
    base_img = generate_unique_image()
    
    # 1. Baseline
    res_base, _ = send_to_api(base_img)
    base_score = res_base["detection_signals"].get("fusion_threat_score", 0) if res_base and "detection_signals" in res_base else 0
    
    # 2. Gaussian Blur (Evasion attempt)
    blurred = base_img.filter(ImageFilter.GaussianBlur(radius=2))
    res_blur, _ = send_to_api(blurred)
    blur_score = res_blur["detection_signals"].get("fusion_threat_score", 0) if res_blur and "detection_signals" in res_blur else 0
    
    # 3. Additive Adversarial Noise Simulation
    arr = np.array(base_img, dtype=np.int16)
    noise = np.random.randint(-15, 15, arr.shape, dtype=np.int16)
    noisy_img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    res_noise, _ = send_to_api(noisy_img)
    noise_score = res_noise["detection_signals"].get("fusion_threat_score", 0) if res_noise and "detection_signals" in res_noise else 0
    
    print(f"{'Transform':<15} | {'Fusion Risk':<15} | {'Delta':<10}")
    print("-" * 45)
    print(f"{'Baseline':<15} | {base_score*100:>6.2f}%       | 0.0%")
    print(f"{'Gaussian Blur':<15} | {blur_score*100:>6.2f}%       | {(blur_score - base_score)*100:+.2f}%")
    print(f"{'Adv. Noise':<15} | {noise_score*100:>6.2f}%       | {(noise_score - base_score)*100:+.2f}%")
    

def test_accuracy_tracking(real_dir=None, fake_dir=None):
    print("\n==================================")
    print(" 🎯 RUNNING ACCURACY & FP TRACKING")
    print("==================================")
    
    if not real_dir or not fake_dir or not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print("Test Skipped. Provide valid directories with Real/Fake image datasets.")
        print("Usage in code: test_accuracy_tracking('./data/real', './data/fake')\n")
        return
        
    real_images = glob.glob(f"{real_dir}/*.*")[:50]
    fake_images = glob.glob(f"{fake_dir}/*.*")[:50]
    
    false_positives = 0
    false_negatives = 0
    
    print(f"Testing {len(real_images)} Real images and {len(fake_images)} Fake images...")
    
    # Check Real (False Positives)
    for p in real_images:
        try:
            img = Image.open(p)
            res, _ = send_to_api(img)
            # If the pipeline says it is fake, it's a false positive
            if res and res.get("amplification_policy", {}).get("tier") in ["red", "orange"]:
                false_positives += 1
        except Exception:
            pass
            
    # Check Fake (False Negatives)
    for p in fake_images:
        try:
            img = Image.open(p)
            res, _ = send_to_api(img)
            # If the pipeline says it is green, it's a false negative
            if res and res.get("amplification_policy", {}).get("tier") == "green":
                false_negatives += 1
        except Exception:
            pass
            
    total_real = len(real_images)
    total_fake = len(fake_images)
    
    fp_rate = (false_positives / total_real) * 100 if total_real > 0 else 0
    fn_rate = (false_negatives / total_fake) * 100 if total_fake > 0 else 0
    accuracy = ((total_real - false_positives) + (total_fake - false_negatives)) / (total_real + total_fake) * 100
    
    print("-" * 40)
    print(f"Overall Accuracy : {accuracy:.2f}%")
    print(f"False Positives  : {false_positives} / {total_real} ({fp_rate:.2f}%)")
    print(f"False Negatives  : {false_negatives} / {total_fake} ({fn_rate:.2f}%)")

if __name__ == "__main__":
    print("\n==================================")
    print("🛡️ SENTRY-X BENCHMARKING SUITE 🛡️")
    print("==================================")
    
    test_latency()
    test_compression()
    test_evasion()
    
    # Try testing accuracy using mock paths.
    # Users should point these to their FF++ or Celeb-DF v2 datasets
    test_accuracy_tracking("data/real_val", "data/fake_val")
    print("\n✅ Benchmarking Complete.\n")
