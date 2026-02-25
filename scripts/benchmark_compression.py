"""
scripts/benchmark_compression.py

Tests SENTRY-X spatial model robustness under varying degrees 
of social media compression (JPEG Q-factors).
Simulates how platforms like WhatsApp, Twitter, and Instagram degrade textures 
and measures if the Deepfake can still be detected accurately under heavy artifacts.
"""
import io
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PIL import Image
from models.detector import detector

def run_compression_simulation(test_image_path: str):
    print("=========================================================")
    print(f"[SENTRY-X Benchmarker] Testing Compression Resilience")
    print(f"Target: {test_image_path}")
    print("=========================================================\n")
    
    try:
        base_img = Image.open(test_image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    qualities = [100, 90, 70, 50, 30, 10]
    
    print(f"{'JPEG Quality':<15} | {'Risk Confidence':<18} | {'Delta':<10} | {'Latency (ms)':<15}")
    print("-" * 65)
    
    baseline_conf = None
    
    for q in qualities:
        # Simulate compression
        buffer = io.BytesIO()
        base_img.save(buffer, format="JPEG", quality=q)
        buffer.seek(0)
        compressed_img = Image.open(buffer).convert("RGB")
        
        t0 = time.time()
        conf, sigs = detector.predict(compressed_img)
        latency = (time.time() - t0) * 1000
        
        if baseline_conf is None:
            baseline_conf = conf
            delta_str = "0.0%"
        else:
            delta = conf - baseline_conf
            delta_str = f"{delta*100:+.2f}%"
            
        print(f"{q:<15} | {conf*100:>6.2f}%              | {delta_str:<10} | {latency:>7.2f} ms")

if __name__ == "__main__":
    import sys
    # Supply dummy4.jpg for testing
    img_path = sys.argv[1] if len(sys.argv) > 1 else "../dummy4.jpg"
    run_compression_simulation(img_path)
