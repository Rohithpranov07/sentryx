# SENTRY-X V2: Testing & Benchmarking Suite

This testing suite continuously audits SENTRY-X against adversarial manipulation, tracking false positives, pipeline latency, and the breakdown of network-level degradations like high-compression environments.

## Running the Suite

Ensure the SENTRY-X pipeline is currently running (`uvicorn app.main:app --port 8000`).
Run the primary automated test script:

```bash
python tests/benchmark_suite.py
```

### 1. Latency & Throughput Metrics (`test_latency()`)

This process hits the local V2 Analysis endpoint repeatedly with `512x512` randomized matrices to ensure cryptographic signatures don't trigger the "Fast-Path" Ledger lookup.
It aggregates our native `latency_profile_ms` into:

- **P99 End-to-End Latency:** Total user-experienced latency.
- **Average Triage (Phase 1):** Milliseconds taken to compute cryptographic Phashes.
- **Average Deep Inference (Phase 2):** Milliseconds taken for VRAM/GPU execution of the multimodal tensors.

### 2. Compression Stress Testing (`test_compression()`)

Simulates how platforms degrade media textures over network bottlenecks. This script runs iterative passes (Q100, Q80, Q50, Q20, Q5) and reports the exact divergence inside the `fusion_threat_score`.
If performance decays heavily below Q30, the underlying `EfficientNet/Detector` module requires adversarial retraining on simulated WhatsApp/Instagram artifacts.

### 3. Evasion Simulations (`test_evasion()`)

Malicious actors purposefully attempt to bypass computer vision analysis by manipulating frequencies. This test applies:

- **Gaussian Blurring:** Heavy focal distortion to hide latent generation patterns.
- **Additive Adversarial Noise:** High-frequency pixel modification meant to confuse softmax layer classifiers.
  The suite outputs exactly how much `Delta %` the fusion risk diverges under attack.

### 4. Real-World Accuracy & False Positive Tracking (`test_accuracy_tracking()`)

To benchmark SENTRY-X on actual human datasets to measure commercial False Positives (FPs), open `benchmark_suite.py` and modify the target directories at the bottom of the script:

```python
# Place actual datasets inside tests/data/
test_accuracy_tracking("data/real_val", "data/fake_val")
```

**Recommended Commercial Datasets for SENTRY-X Verification:**

1. **FaceForensics++ (FF++)**
2. **Celeb-DF V2**
3. **Deepfake Detection Challenge (DFDC) dataset**

The test maps every image through the engine and calculates the physical `amplification_policy` tier. A `red` policy returned on a real image counts as a strict **False Positive**.
