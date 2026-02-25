"""
tests/ensemble_benchmark.py

SENTRY-X V2.1 Ensemble Benchmark Suite (TASK 6)
=================================================

Comprehensive benchmarking of the multi-model ensemble detection:

1. Real images (camera photos with various characteristics)
2. AI-generated images (simulated synthetic patterns)
3. Compressed variants (JPEG quality 5-100)
4. Screenshot variants (mobile dimension laundering)

Metrics:
- Accuracy, Precision, Recall
- False Positive Rate, False Negative Rate
- Per-model agreement rates
- Latency profiling
- Threshold tuning recommendations

Runs locally without API server — directly invokes the pipeline.
"""

import sys
import os
import time
import io
import json
import statistics
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_real_image(variant: str = "natural", w: int = 512, h: int = 384) -> Image.Image:
    """
    Generate images with realistic camera-like characteristics.
    These should be classified as REAL.
    """
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    
    if variant == "natural":
        # Simulate natural photography: smooth gradients + sensor noise
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        
        # Natural sky-to-ground gradient
        base_r = (120 + 80 * yy).astype(np.uint8)
        base_g = (150 + 50 * yy - 30 * xx).astype(np.uint8)
        base_b = (200 - 60 * yy).astype(np.uint8)
        
        img[:, :, 0] = np.clip(base_r + np.random.normal(0, 12, (h, w)), 0, 255).astype(np.uint8)
        img[:, :, 1] = np.clip(base_g + np.random.normal(0, 12, (h, w)), 0, 255).astype(np.uint8)
        img[:, :, 2] = np.clip(base_b + np.random.normal(0, 12, (h, w)), 0, 255).astype(np.uint8)
    
    elif variant == "portrait":
        # Simulate portrait with face-like structure + bokeh
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Background blur (bokeh effect)
        bg = np.ones((h, w, 3), dtype=np.float32) * 60
        bg += np.random.normal(0, 5, (h, w, 3))
        
        # Face-like centered region with more texture
        face_mask = (dist < min(w, h) * 0.25).astype(np.float32)
        face_region = np.random.normal(180, 15, (h, w, 3))
        
        img = np.clip(bg * (1 - face_mask[:, :, None]) + face_region * face_mask[:, :, None], 0, 255).astype(np.uint8)
        # Add sensor noise
        img = np.clip(img.astype(np.int16) + np.random.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
    
    elif variant == "landscape":
        # Natural landscape with horizon
        for row in range(h):
            t = row / h
            r = int(135 + 100 * t + np.random.normal(0, 8))
            g = int(180 - 50 * t + np.random.normal(0, 8))
            b = int(220 - 140 * t + np.random.normal(0, 8))
            img[row, :, 0] = np.clip(r + np.random.normal(0, 10, w), 0, 255).astype(np.uint8)
            img[row, :, 1] = np.clip(g + np.random.normal(0, 10, w), 0, 255).astype(np.uint8)
            img[row, :, 2] = np.clip(b + np.random.normal(0, 10, w), 0, 255).astype(np.uint8)
    
    elif variant == "noisy_phone":
        # Low-light phone photo (high noise, low contrast)
        base = np.random.normal(80, 5, (h, w, 3))
        noise = np.random.normal(0, 20, (h, w, 3))
        img = np.clip(base + noise, 0, 255).astype(np.uint8)
    
    elif variant == "high_contrast":
        # High contrast scene (bright highlights, dark shadows)
        half_h = h // 2
        img[:half_h, :] = np.random.normal(220, 10, (half_h, w, 3)).clip(0, 255).astype(np.uint8)
        img[half_h:, :] = np.random.normal(30, 8, (h - half_h, w, 3)).clip(0, 255).astype(np.uint8)
        # Add transition noise
        transition = slice(half_h - 20, half_h + 20)
        img[transition, :] = np.random.normal(120, 25, (40, w, 3)).clip(0, 255).astype(np.uint8)
    
    return Image.fromarray(img, "RGB")


def generate_ai_image(variant: str = "smooth", w: int = 512, h: int = 512) -> Image.Image:
    """
    Generate images with AI-generated characteristics.
    These should be classified as FAKE.
    """
    if variant == "smooth":
        # AI-generated: unnaturally smooth gradients without sensor noise
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 0] = (128 + 127 * np.sin(xx * 3.14)).astype(np.uint8)
        img[:, :, 1] = (128 + 127 * np.cos(yy * 3.14)).astype(np.uint8)
        img[:, :, 2] = (128 + 127 * np.sin((xx + yy) * 2.0)).astype(np.uint8)
        # Very little noise (GAN hallmark)
        img = np.clip(img.astype(np.int16) + np.random.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    
    elif variant == "perfect_face":
        # AI face: too perfect, no natural imperfections
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Ultra-smooth skin tones
        skin_color = np.array([210, 180, 160])
        img = np.ones((h, w, 3), dtype=np.float32) * skin_color
        
        # Perfect circular symmetry (unnatural)
        falloff = np.exp(-(dist ** 2) / (2 * (min(w, h) * 0.3) ** 2))
        img = img * falloff[:, :, None] * 0.5 + img * 0.5
        
        # Minimal noise (GAN output)
        img = np.clip(img + np.random.normal(0, 0.8, img.shape), 0, 255).astype(np.uint8)
    
    elif variant == "checkerboard":
        # GAN artifact: subtle checkerboard in high frequencies
        img = np.random.normal(128, 15, (h, w, 3)).astype(np.float32)
        
        # Add checkerboard pattern (GAN upsampling artifact)
        checker = np.zeros((h, w))
        checker[::2, ::2] = 3.0
        checker[1::2, 1::2] = 3.0
        for c in range(3):
            img[:, :, c] += checker
        
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    elif variant == "diffusion":
        # Diffusion model characteristics: perfectly clean, unusual spectral profile
        # Create concentric circles pattern (common in diffusion)
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        img = np.zeros((h, w, 3), dtype=np.float32)
        ring_pattern = np.sin(dist * 0.15) * 50 + 128
        for c in range(3):
            img[:, :, c] = ring_pattern + np.random.normal(0, 2, (h, w))
        
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    elif variant == "deepfake_swap":
        # Face swap: inconsistent noise between face region and background
        # Background with normal noise
        bg = np.random.normal(100, 12, (h, w, 3)).clip(0, 255).astype(np.float32)
        
        # Face region with different noise level (from different source)
        center_y, center_x = h // 2, w // 2
        y_coords, x_coords = np.ogrid[:h, :w]
        dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        face_mask = (dist < min(w, h) * 0.3).astype(np.float32)
        
        face = np.random.normal(190, 3, (h, w, 3)).clip(0, 255).astype(np.float32)  # Unnaturally smooth
        
        img = bg * (1 - face_mask[:, :, None]) + face * face_mask[:, :, None]
        
        # Add boundary artifacts (typical of face swap)
        boundary = ((dist > min(w, h) * 0.27) & (dist < min(w, h) * 0.33)).astype(np.float32)
        img += boundary[:, :, None] * np.random.normal(0, 20, (h, w, 3))
        
        img = np.clip(img, 0, 255).astype(np.uint8)
    
    # AI images often have 1:1 aspect ratio
    return Image.fromarray(img, "RGB")


def compress_image(image: Image.Image, quality: int) -> Image.Image:
    """Simulate JPEG compression at specified quality."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def simulate_screenshot(image: Image.Image, device: str = "iphone") -> Image.Image:
    """Simulate screenshot by resizing to device dimensions."""
    device_dims = {
        "iphone_14": (1170, 2532),
        "iphone_15_pro": (1290, 2796),
        "android_fhd": (1080, 2340),
        "android_hd": (1080, 1920),
        "ipad": (2048, 2732),
    }
    dims = device_dims.get(device, (1080, 1920))
    return image.resize(dims, Image.LANCZOS)


def run_detection(image: Image.Image) -> Dict[str, Any]:
    """Run the ensemble pipeline on a single image."""
    from models.ensemble_detector import ensemble_detector
    from models.physiological_detector import physiological_detector
    from detection.evasion_detector import evasion_detector
    from models.verification_layer import verification_layer
    from models.dynamic_threshold import dynamic_threshold_engine, final_decision_engine
    
    t0 = time.time()
    
    # Ensemble detection
    ensemble_result = ensemble_detector.predict(image)
    ensemble_score = ensemble_result["ensemble_score"]
    agreement = ensemble_result["agreement"]
    
    # Physiological
    physio_data = physiological_detector.analyze(image)
    physio_conf = physio_data["physiological_confidence"]
    
    # Evasion
    evasion_data = evasion_detector.analyze(image)
    
    # Support adjustment
    combined = min(1.0, ensemble_score + physio_conf * 0.12 + evasion_data["evasion_risk"] * 0.08)
    
    # Verification
    verification_data = None
    if verification_layer.should_verify(combined):
        verification_data = verification_layer.verify(image, combined)
        combined = verification_data["verified_score"]
    
    # Dynamic thresholds
    thresholds = dynamic_threshold_engine.compute_thresholds(
        physio_data=physio_data,
        evasion_data=evasion_data,
        ensemble_agreement=agreement,
    )
    
    # Final decision
    risk = final_decision_engine.classify(
        ensemble_score=combined,
        thresholds=thresholds,
        agreement=agreement,
        verification_data=verification_data,
    )
    
    latency_ms = (time.time() - t0) * 1000
    
    return {
        "ensemble_score": combined,
        "risk_level": risk["risk_level"],
        "action": risk["action"],
        "verdict": risk["verdict"],
        "decision_path": risk.get("decision_path", "unknown"),
        "agreement": agreement,
        "individual_detectors": ensemble_result["individual_results"],
        "physio_conf": physio_conf,
        "verification": verification_data,
        "latency_ms": latency_ms,
    }


def classify_prediction(result: Dict, actual_label: str) -> str:
    """Classify a prediction as TP, TN, FP, or FN."""
    is_flagged = result["risk_level"] in ("red", "orange", "yellow")
    is_fake = actual_label == "fake"
    
    if is_fake and is_flagged:
        return "TP"
    elif not is_fake and not is_flagged:
        return "TN"
    elif not is_fake and is_flagged:
        return "FP"
    else:  # is_fake and not is_flagged
        return "FN"


def strict_classify(result: Dict, actual_label: str) -> str:
    """Stricter classification: only red/orange count as 'detected'."""
    is_flagged = result["risk_level"] in ("red", "orange")
    is_fake = actual_label == "fake"
    
    if is_fake and is_flagged:
        return "TP"
    elif not is_fake and not is_flagged:
        return "TN"
    elif not is_fake and is_flagged:
        return "FP"
    else:
        return "FN"


def compute_metrics(classifications: List[str]) -> Dict[str, float]:
    """Compute standard classification metrics."""
    tp = classifications.count("TP")
    tn = classifications.count("TN")
    fp = classifications.count("FP")
    fn = classifications.count("FN")
    total = len(classifications)
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "total": total,
    }


def run_benchmark():
    """Execute the full benchmark suite."""
    print("\n" + "=" * 70)
    print(" 🛡️  SENTRY-X V2.1 ENSEMBLE BENCHMARK SUITE ")
    print("=" * 70)
    print(f" Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Models: 3 (EfficientNet-B4, XceptionNet, CLIP-ViT)")
    print("=" * 70)
    
    all_results = []
    all_classifications = []
    all_strict_classifications = []
    latencies = []
    
    # ══════════════════════════════════════════════════════════════════════
    # TEST 1: Real Images (should be classified as SAFE)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print(" 📸 TEST 1: Real Images (100 samples)")
    print("-" * 60)
    
    real_variants = ["natural", "portrait", "landscape", "noisy_phone", "high_contrast"]
    real_count = 100
    real_results = []
    
    for i in range(real_count):
        variant = real_variants[i % len(real_variants)]
        img = generate_real_image(variant)
        result = run_detection(img)
        cls = classify_prediction(result, "real")
        strict_cls = strict_classify(result, "real")
        
        real_results.append(result)
        all_results.append({"label": "real", "variant": variant, "result": result})
        all_classifications.append(cls)
        all_strict_classifications.append(strict_cls)
        latencies.append(result["latency_ms"])
        
        if (i + 1) % 25 == 0:
            fps_so_far = sum(1 for r in real_results if r["risk_level"] in ("red", "orange")) 
            print(f"  Progress: {i+1}/{real_count} | FPs so far: {fps_so_far}")
    
    real_metrics = compute_metrics([classify_prediction(r, "real") for r in [all_results[i]["result"] for i in range(len(real_results))]])
    real_fps = sum(1 for r in real_results if r["risk_level"] in ("red", "orange"))
    print(f"  ✓ Real images FP rate: {real_fps}/{real_count} ({real_fps/real_count*100:.1f}%)")
    
    # ══════════════════════════════════════════════════════════════════════
    # TEST 2: AI-Generated Images (should be classified as FAKE)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print(" 🤖 TEST 2: AI-Generated Images (100 samples)")
    print("-" * 60)
    
    ai_variants = ["smooth", "perfect_face", "checkerboard", "diffusion", "deepfake_swap"]
    ai_count = 100
    ai_results = []
    
    for i in range(ai_count):
        variant = ai_variants[i % len(ai_variants)]
        img = generate_ai_image(variant)
        result = run_detection(img)
        cls = classify_prediction(result, "fake")
        strict_cls = strict_classify(result, "fake")
        
        ai_results.append(result)
        all_results.append({"label": "fake", "variant": variant, "result": result})
        all_classifications.append(cls)
        all_strict_classifications.append(strict_cls)
        latencies.append(result["latency_ms"])
        
        if (i + 1) % 25 == 0:
            detected = sum(1 for r in ai_results if r["risk_level"] in ("red", "orange", "yellow"))
            print(f"  Progress: {i+1}/{ai_count} | Detected so far: {detected}")
    
    ai_detected = sum(1 for r in ai_results if r["risk_level"] in ("red", "orange", "yellow"))
    print(f"  ✓ AI images detection rate: {ai_detected}/{ai_count} ({ai_detected/ai_count*100:.1f}%)")
    
    # ══════════════════════════════════════════════════════════════════════
    # TEST 3: Compressed Variants (50 samples)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print(" 📉 TEST 3: Compressed Variants (50 samples)")
    print("-" * 60)
    
    compressed_count = 50
    qualities = [80, 50, 30, 15, 5]
    compressed_results = []
    
    for i in range(compressed_count):
        # Mix of real and AI images, then compress
        if i < 25:
            img = generate_real_image(real_variants[i % len(real_variants)])
            label = "real"
        else:
            img = generate_ai_image(ai_variants[i % len(ai_variants)])
            label = "fake"
        
        quality = qualities[i % len(qualities)]
        compressed = compress_image(img, quality)
        result = run_detection(compressed)
        cls = classify_prediction(result, label)
        strict_cls = strict_classify(result, label)
        
        compressed_results.append({"label": label, "quality": quality, "result": result})
        all_classifications.append(cls)
        all_strict_classifications.append(strict_cls)
        latencies.append(result["latency_ms"])
    
    comp_real_fps = sum(1 for r in compressed_results[:25] if r["result"]["risk_level"] in ("red", "orange"))
    comp_fake_detected = sum(1 for r in compressed_results[25:] if r["result"]["risk_level"] in ("red", "orange", "yellow"))
    print(f"  ✓ Compressed real FP rate: {comp_real_fps}/25 ({comp_real_fps/25*100:.1f}%)")
    print(f"  ✓ Compressed fake detection: {comp_fake_detected}/25 ({comp_fake_detected/25*100:.1f}%)")
    
    # ══════════════════════════════════════════════════════════════════════
    # TEST 4: Screenshot Variants (30 samples)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 60)
    print(" 📱 TEST 4: Screenshot Variants (30 samples)")
    print("-" * 60)
    
    screenshot_count = 30
    devices = ["iphone_14", "iphone_15_pro", "android_fhd", "android_hd", "ipad"]
    screenshot_results = []
    
    for i in range(screenshot_count):
        if i < 15:
            img = generate_real_image(real_variants[i % len(real_variants)])
            label = "real"
        else:
            img = generate_ai_image(ai_variants[i % len(ai_variants)])
            label = "fake"
        
        device = devices[i % len(devices)]
        screenshotted = simulate_screenshot(img, device)
        result = run_detection(screenshotted)
        cls = classify_prediction(result, label)
        strict_cls = strict_classify(result, label)
        
        screenshot_results.append({"label": label, "device": device, "result": result})
        all_classifications.append(cls)
        all_strict_classifications.append(strict_cls)
        latencies.append(result["latency_ms"])
    
    ss_real_fps = sum(1 for r in screenshot_results[:15] if r["result"]["risk_level"] in ("red", "orange"))
    ss_fake_detected = sum(1 for r in screenshot_results[15:] if r["result"]["risk_level"] in ("red", "orange", "yellow"))
    print(f"  ✓ Screenshot real FP rate: {ss_real_fps}/15 ({ss_real_fps/15*100:.1f}%)")
    print(f"  ✓ Screenshot fake detection: {ss_fake_detected}/15 ({ss_fake_detected/15*100:.1f}%)")
    
    # ══════════════════════════════════════════════════════════════════════
    # RESULTS SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(" 📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    
    # Overall metrics (lenient: yellow counts as detected)
    overall_metrics = compute_metrics(all_classifications)
    print("\n── Overall Metrics (lenient: yellow+ = detected) ──")
    print(f"  Accuracy       : {overall_metrics['accuracy']*100:.2f}%")
    print(f"  Precision      : {overall_metrics['precision']*100:.2f}%")
    print(f"  Recall         : {overall_metrics['recall']*100:.2f}%")
    print(f"  F1 Score       : {overall_metrics['f1_score']*100:.2f}%")
    print(f"  FP Rate        : {overall_metrics['false_positive_rate']*100:.2f}%")
    print(f"  FN Rate        : {overall_metrics['false_negative_rate']*100:.2f}%")
    print(f"  TP={overall_metrics['true_positives']} TN={overall_metrics['true_negatives']} FP={overall_metrics['false_positives']} FN={overall_metrics['false_negatives']}")
    
    # Strict metrics (only red/orange count as detected)
    strict_metrics = compute_metrics(all_strict_classifications)
    print("\n── Strict Metrics (red/orange = detected) ──")
    print(f"  Accuracy       : {strict_metrics['accuracy']*100:.2f}%")
    print(f"  Precision      : {strict_metrics['precision']*100:.2f}%")
    print(f"  Recall         : {strict_metrics['recall']*100:.2f}%")
    print(f"  F1 Score       : {strict_metrics['f1_score']*100:.2f}%")
    print(f"  FP Rate        : {strict_metrics['false_positive_rate']*100:.2f}%")
    print(f"  FN Rate        : {strict_metrics['false_negative_rate']*100:.2f}%")
    
    # Latency metrics
    print("\n── Latency Metrics ──")
    if latencies:
        print(f"  Mean Latency   : {statistics.mean(latencies):.1f} ms")
        print(f"  Median Latency : {statistics.median(latencies):.1f} ms")
        print(f"  P95 Latency    : {np.percentile(latencies, 95):.1f} ms")
        print(f"  P99 Latency    : {np.percentile(latencies, 99):.1f} ms")
        print(f"  Min Latency    : {min(latencies):.1f} ms")
        print(f"  Max Latency    : {max(latencies):.1f} ms")
    
    # Per-test breakdown
    print("\n── Per-Test Breakdown ──")
    print(f"  {'Test':<25} | {'FP/FN Rate':<15} | {'Status'}")
    print(f"  {'-'*25} | {'-'*15} | {'-'*20}")
    print(f"  {'Real Images (100)':<25} | FP: {real_fps:>3}/100     | {'✅ GOOD' if real_fps < 10 else '⚠️  HIGH FP'}")
    print(f"  {'AI Images (100)':<25} | FN: {100-ai_detected:>3}/100     | {'✅ GOOD' if ai_detected > 70 else '⚠️  LOW RECALL'}")
    print(f"  {'Compressed Real (25)':<25} | FP: {comp_real_fps:>3}/25      | {'✅ GOOD' if comp_real_fps < 5 else '⚠️  HIGH FP'}")
    print(f"  {'Compressed Fake (25)':<25} | FN: {25-comp_fake_detected:>3}/25      | {'✅ GOOD' if comp_fake_detected > 15 else '⚠️  LOW RECALL'}")
    print(f"  {'Screenshot Real (15)':<25} | FP: {ss_real_fps:>3}/15      | {'✅ GOOD' if ss_real_fps < 3 else '⚠️  HIGH FP'}")
    print(f"  {'Screenshot Fake (15)':<25} | FN: {15-ss_fake_detected:>3}/15      | {'✅ GOOD' if ss_fake_detected > 10 else '⚠️  LOW RECALL'}")
    
    # Model agreement analysis
    print("\n── Model Agreement Analysis ──")
    agreement_stats = {"0": 0, "1": 0, "2": 0, "3": 0}
    for r in all_results:
        result = r["result"]
        flagging = result["agreement"]["models_flagging_medium"]
        agreement_stats[str(min(3, flagging))] += 1
    
    total_checked = len(all_results)
    for k, v in agreement_stats.items():
        print(f"  {k} models flagging : {v:>4} ({v/total_checked*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print(" ✅ Benchmark Complete")
    print("=" * 70)
    
    # Save results to file
    report = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": "2.1",
        "overall_metrics_lenient": overall_metrics,
        "overall_metrics_strict": strict_metrics,
        "latency": {
            "mean_ms": round(statistics.mean(latencies), 1) if latencies else 0,
            "p95_ms": round(float(np.percentile(latencies, 95)), 1) if latencies else 0,
            "p99_ms": round(float(np.percentile(latencies, 99)), 1) if latencies else 0,
        },
        "per_test": {
            "real_fp_rate": real_fps / real_count,
            "ai_detection_rate": ai_detected / ai_count,
            "compressed_real_fp": comp_real_fps / 25,
            "compressed_fake_detection": comp_fake_detected / 25,
            "screenshot_real_fp": ss_real_fps / 15,
            "screenshot_fake_detection": ss_fake_detected / 15,
        },
    }
    
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n 📄 Full report saved to: {report_path}")
    
    return report


if __name__ == "__main__":
    run_benchmark()
