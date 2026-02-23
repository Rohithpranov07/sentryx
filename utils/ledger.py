"""
utils/ledger.py

Local SQLite provenance ledger — the PoC stand-in for blockchain storage.

In production this would write to a Polygon L2 smart contract.
For the PoC, SQLite gives us:
  - Immutable append-only semantics (we never DELETE or UPDATE)
  - Queryable by sha256, phash, or fingerprint_id
  - Hamming-distance similarity search for near-duplicate detection
  - Persistent across server restarts

This is explicitly framed as "local provenance node" in the API response,
not a real blockchain. Honesty > marketing.
"""

import sqlite3
import os
import json
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from utils.fingerprint import hamming_distance

load_dotenv()

DB_PATH = os.getenv("LEDGER_DB_PATH", "./data/ledger.db")


def _get_conn() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_ledger():
    """Create ledger table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS threat_ledger (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint_id  TEXT NOT NULL,
            sha256          TEXT NOT NULL UNIQUE,
            phash           TEXT NOT NULL,
            dhash           TEXT NOT NULL,
            risk_level      TEXT NOT NULL,
            verdict         TEXT NOT NULL,
            confidence      REAL NOT NULL,
            filename        TEXT,
            platform_id     TEXT,
            timestamp       REAL NOT NULL,
            metadata        TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sha256 ON threat_ledger(sha256)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_phash  ON threat_ledger(phash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_risk   ON threat_ledger(risk_level)")
    conn.commit()
    conn.close()


def lookup_sha256(sha256: str) -> Optional[dict]:
    """Exact match lookup by cryptographic hash."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM threat_ledger WHERE sha256 = ?", (sha256,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def lookup_similar(phash: str, max_distance: int = 10) -> Optional[dict]:
    """
    Perceptual similarity search.
    Scans all stored phashes and returns the closest match
    if within Hamming distance threshold.
    
    Note: For production at scale this would use a VP-tree
    or LSH index. SQLite full-scan is fine for PoC.
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM threat_ledger WHERE risk_level IN ('red', 'orange')"
    ).fetchall()
    conn.close()

    best_match = None
    best_distance = max_distance + 1

    for row in rows:
        try:
            dist = hamming_distance(phash, row["phash"])
            if dist <= max_distance and dist < best_distance:
                best_distance = dist
                best_match = dict(row)
                best_match["hamming_distance"] = dist
        except Exception:
            continue

    return best_match


def register_threat(fingerprints: dict, verdict: dict, filename: str = None, platform_id: str = None) -> dict:
    """
    Write a RED or ORANGE verdict to the permanent ledger.
    Skips if SHA256 already exists (idempotent).
    Returns the ledger entry.
    """
    conn = _get_conn()
    timestamp = time.time()

    try:
        conn.execute("""
            INSERT OR IGNORE INTO threat_ledger
            (fingerprint_id, sha256, phash, dhash, risk_level, verdict, confidence, filename, platform_id, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fingerprints["fingerprint_id"],
            fingerprints["sha256"],
            fingerprints["phash"],
            fingerprints["dhash"],
            verdict["risk_level"],
            verdict["verdict"],
            verdict["confidence"],
            filename,
            platform_id,
            timestamp,
            json.dumps({"forensic_signals": verdict.get("forensic_signals", [])}),
        ))
        conn.commit()
    finally:
        conn.close()

    return {
        "registered": True,
        "fingerprint_id": fingerprints["fingerprint_id"],
        "timestamp": timestamp,
        "ledger_type": "local_provenance_node",
        "note": "Production deployment writes to Polygon L2 smart contract.",
    }


def get_stats() -> dict:
    """Summary stats for the /health endpoint."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM threat_ledger").fetchone()[0]
    by_risk = conn.execute(
        "SELECT risk_level, COUNT(*) as count FROM threat_ledger GROUP BY risk_level"
    ).fetchall()
    conn.close()

    return {
        "total_fingerprints": total,
        "by_risk_level": {row["risk_level"]: row["count"] for row in by_risk},
    }
