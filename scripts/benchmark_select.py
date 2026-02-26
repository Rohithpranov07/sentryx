import os
import time
import json
import torch
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm

import timm
from transformers import AutoModelForImageClassification, AutoImageProcessor
import google.generativeai as genai

# ── 1. MODEL INFERENCE WRAPPERS ──────────────────────────────────────────────

class DetectorWrapper:
    def __init__(self, name: str, source: str):
        self.name = name
        self.source = source
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    def load(self):
        raise NotImplementedError
        
    def predict_image(self, image: Image.Image) -> Dict[str, float]:
        """
        Returns:
            {
                "prob_ai": float,
                "prob_real": float,
                "raw_score": float
            }
        """
        raise NotImplementedError

class ViTDeepfakeDetector(DetectorWrapper):
    def __init__(self):
        super().__init__("ViT_Deepfake", "prithivMLmods/Deep-Fake-Detector-v2-Model")
        
    def load(self):
        print(f"Loading {self.name} on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(self.source)
        self.model = AutoModelForImageClassification.from_pretrained(self.source).to(self.device)
        self.model.eval()
        
    def predict_image(self, image: Image.Image) -> Dict[str, float]:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            # Output logic: label 1 is usually Deepfake for this model
            prob_ai = probs[1].item()
            prob_real = probs[0].item()
            
        return {"prob_ai": prob_ai, "prob_real": prob_real, "raw_score": prob_ai}


class UmMMaybeAIDetector(DetectorWrapper):
    def __init__(self):
        super().__init__("ViT_Synthetic_Art", "umm-maybe/AI-image-detector")
        
    def load(self):
        print(f"Loading {self.name} on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(self.source)
        self.model = AutoModelForImageClassification.from_pretrained(self.source).to(self.device)
        self.model.eval()
        
    def predict_image(self, image: Image.Image) -> Dict[str, float]:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            # label 0 = artificial, label 1 = human
            prob_ai = probs[0].item()
            prob_real = probs[1].item()
            
        return {"prob_ai": prob_ai, "prob_real": prob_real, "raw_score": prob_ai}


class EfficientNetB4Detector(DetectorWrapper):
    def __init__(self):
        super().__init__("EfficientNet_B4_Deepfake", "timm/tf_efficientnet_b4_ns")
        
    def load(self):
        print(f"Loading {self.name} on {self.device}...")
        # Since we just want to test inference structure, we load a pretrained efficientnet
        # and mock a binary head since a true dedicated deepfake efficientnet requires a specific HF repo.
        self.model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=2).to(self.device)
        self.model.eval()
        
        # TIMM data config
        data_config = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        
    def predict_image(self, image: Image.Image) -> Dict[str, float]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=-1)[0]
            prob_ai = probs[1].item()
            prob_real = probs[0].item()
            
        return {"prob_ai": prob_ai, "prob_real": prob_real, "raw_score": prob_ai}


class GeminiFailsafeDetector(DetectorWrapper):
    def __init__(self):
        super().__init__("Gemini_2.5_Flash", "google/gemini-2.5-flash")
        
    def load(self):
        from dotenv import load_dotenv
        load_dotenv(override=True)
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        print(f"Loading {self.name} via API...")
        
    def predict_image(self, image: Image.Image) -> Dict[str, float]:
        prompt = (
            "You are a forensic analyst. Is this image authentic or AI-generated? "
            "Reply strictly with a single number between 0.0 (real) and 1.0 (AI)."
        )
        try:
            res = self.model.generate_content([prompt, image])
            import re
            m = re.search(r"0\.\d+|1\.0|0\.0", res.text.strip())
            prob_ai = float(m.group()) if m else 0.5
        except:
            prob_ai = 0.5
            
        return {"prob_ai": prob_ai, "prob_real": 1 - prob_ai, "raw_score": prob_ai}

# ── 2. DATASET CREATION ──────────────────────────────────────────────────────

def create_mock_dataset(base_dir: str, num_real=10, num_ai=10):
    """
    Creates a deterministic physical evaluation dataset including variants
    (Instagram compressed, Screenshot). Minimal counts used for script proof.
    """
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    # Generate Base "Real" Images
    for i in range(num_real):
        samples.append({"id": f"real_{i}", "label": "real", "base_color": "green"})
    # Generate Base "AI" Images
    for i in range(num_ai):
        samples.append({"id": f"ai_{i}", "label": "ai", "base_color": "mediumpurple"})
        
    dataset_records = []
        
    for s in samples:
        lbl = s["label"]
        folder = out_dir / lbl
        folder.mkdir(exist_ok=True)
        
        # Base Image (Original)
        img = Image.new("RGB", (600, 600), color=s["base_color"])
        draw = ImageDraw.Draw(img)
        draw.text((300, 300), s["id"], fill="white")
        
        orig_path = folder / f"{s['id']}_original.png"
        img.save(orig_path)
        
        # Variant 1: Instagram Compressed
        ig_path = folder / f"{s['id']}_ig.jpg"
        img.resize((1080, 1080)).save(ig_path, format="JPEG", quality=70)
        
        # Variant 2: Screenshot
        ss_path = folder / f"{s['id']}_screenshot.png"
        ss_img = Image.new("RGB", (1170, 2532), color="white")
        ss_img.paste(img.resize((1170, 1170)), (0, 300))
        ss_img.save(ss_path)
        
        dataset_records.append({
            "id": s["id"],
            "label": 0 if lbl == "real" else 1, # 0=real, 1=AI
            "variants": [
                {"type": "original", "path": str(orig_path)},
                {"type": "instagram_compressed", "path": str(ig_path)},
                {"type": "screenshot", "path": str(ss_path)}
            ]
        })
        
    return dataset_records
        
# ── 3. BENCHMARK ENGINE ──────────────────────────────────────────────────────

def evaluate_model(model: DetectorWrapper, records: List[Dict]):
    print(f"\nEvaluating: {model.name}")
    try:
        model.load()
    except Exception as e:
        print(f"  [ERROR] Failed to load {model.name}: {e}")
        return None
        
    y_true = []
    y_pred_probs = []
    
    # Test on all deterministic variants (original, compressed, screenshot)
    for row in tqdm(records, desc=f"Evaluating {model.name}"):
        label = row["label"]
        for variant in row["variants"]:
            path = variant["path"] 
            img = Image.open(path).convert("RGB")
            
            try:
                preds = model.predict_image(img)
                y_true.append(label)
                y_pred_probs.append(preds["prob_ai"])
            except Exception as e:
                print(f"  [Warning] Inference failed for {path}: {e}")
                
            if model.name.startswith("Gemini"):
                time.sleep(1.5) # Avoid API rate limits
            
    if not y_true:
        return None
        
    y_true = np.array(y_true)
    y_probs = np.array(y_pred_probs)
    y_preds = (y_probs >= 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_preds)
    prec = precision_score(y_true, y_preds, zero_division=0)
    rec = recall_score(y_true, y_preds, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print("\n  Classification Results:")
    print(f"  Accuracy:  {acc:.2%}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  FPR:       {fpr:.2%} (Real called AI)")
    print(f"  FNR:       {fnr:.2%} (AI called Real)")
    print(f"  Conf Mat:  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    status = "ACCEPT"
    reason = "Meets benchmark criteria."
    if acc < 0.75:
        status = "DISCARD"
        reason = "Accuracy < 75%"
    elif fpr > 0.20:
        status = "DISCARD"
        reason = "FPR > 20% (Too many false positives)"
    elif fnr > 0.20:
        status = "DISCARD"
        reason = "FNR > 20% (Misses too many AI artifacts)"
        
    print(f"  Status:    [{status}] -> {reason}")
    return {"accuracy": acc, "fpr": fpr, "fnr": fnr, "status": status, "reason": reason}

if __name__ == "__main__":
    print("SENTRY-X Deterministic Model Evaluation Framework")
    print("=" * 60)
    
    # 1. Prepare controlled proxy set
    eval_dir = "sentryx_eval_dataset_sample"
    print(f"Generating deterministic variants at '{eval_dir}'...")
    records = create_mock_dataset(eval_dir, num_real=15, num_ai=15)
    
    detectors = [
        ViTDeepfakeDetector(),
        UmMMaybeAIDetector(),
        EfficientNetB4Detector(), 
        GeminiFailsafeDetector()
    ]
    
    results = {}
    for d in detectors:
        res = evaluate_model(d, records)
        if res:
            results[d.name] = res
            
    print("\n" + "=" * 60)
    print("FINAL SELECTION SUMMARY")
    print("=" * 60)
    for model_name, st in results.items():
        print(f"{model_name:<25}: {st['status']:<10} | {st['reason']}")
