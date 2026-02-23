"""
utils/fingerprint.py

Generates two types of fingerprints for every media file:

1. SHA256 (cryptographic) — exact match identity. 
   Used to detect bit-for-bit reuploads.

2. pHash (perceptual) — content-aware similarity.
   Used to detect re-encoded or slightly modified reuploads
   that would evade exact-match detection.

Both are stored in the provenance ledger together.
"""

import hashlib
import imagehash
from PIL import Image
from pathlib import Path


def sha256_file(file_bytes: bytes) -> str:
    """Cryptographic hash of raw file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def phash_image(image: Image.Image, hash_size: int = 16) -> str:
    """
    Perceptual hash of image content.
    hash_size=16 gives a 256-bit hash — good balance of sensitivity
    and tolerance to minor re-encoding.
    
    Two images with Hamming distance <= 10 are considered
    perceptually similar (same content, different encoding).
    """
    return str(imagehash.phash(image, hash_size=hash_size))


def dhash_image(image: Image.Image) -> str:
    """
    Difference hash — captures gradient structure.
    More stable than pHash under JPEG compression attacks.
    Used as secondary fingerprint for adversarial robustness.
    """
    return str(imagehash.dhash(image))


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Compute Hamming distance between two perceptual hashes.
    Distance 0 = identical content.
    Distance <= 10 = probably same image, different encoding.
    Distance > 10 = different images.
    """
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return h1 - h2


def generate_fingerprints(file_bytes: bytes, image: Image.Image) -> dict:
    """
    Generate all fingerprints for a given media file.
    Returns a dict suitable for ledger storage and API response.
    """
    sha = sha256_file(file_bytes)
    ph = phash_image(image)
    dh = dhash_image(image)

    return {
        "sha256": sha,
        "phash": ph,
        "dhash": dh,
        # Short fingerprint ID shown in UI (first 12 chars of sha256)
        "fingerprint_id": f"FP-{sha[:12].upper()}",
    }
