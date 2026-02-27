"""
Sentinel — Provenance Signing Utility

HMAC-based signing for raw events to ensure chain-of-custody integrity.
Every raw event gets a provenance hash at ingestion time.
Verification can detect any tampering post-ingestion.
"""
import hashlib
import hmac
import json
import os
from datetime import datetime
from typing import Any, Optional

# Default signing key — in production, load from HSM or secrets manager
_SIGNING_KEY = os.environ.get("SENTINEL_SIGNING_KEY", "sentinel-dev-key-2026").encode()


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file for integrity verification."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def compute_payload_hash(payload: dict[str, Any]) -> str:
    """Compute SHA256 hash of a JSON payload (deterministic serialization)."""
    # Sort keys for deterministic serialization
    canonical = json.dumps(payload, sort_keys=True, default=str).encode()
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def sign_raw_event(
    event_type: str,
    source_id: str,
    timestamp: str,
    raw_payload: dict[str, Any],
    file_hash: Optional[str] = None,
    signer: str = "system",
) -> str:
    """
    Generate HMAC signature for a raw event.
    
    The signature covers:
    - event type
    - source ID
    - timestamp
    - canonical payload hash
    - file hash (if applicable)
    
    This ensures any modification to the raw event is detectable.
    """
    message_parts = [
        event_type,
        source_id,
        str(timestamp),
        compute_payload_hash(raw_payload),
    ]
    if file_hash:
        message_parts.append(file_hash)

    message = "|".join(message_parts).encode()

    signature = hmac.new(_SIGNING_KEY, message, hashlib.sha256).hexdigest()
    return f"hmac-sha256:{signature}"


def verify_raw_event(
    event_type: str,
    source_id: str,
    timestamp: str,
    raw_payload: dict[str, Any],
    provenance_hash: str,
    file_hash: Optional[str] = None,
) -> bool:
    """
    Verify the HMAC signature of a raw event.
    Returns True if the event has not been tampered with.
    """
    expected = sign_raw_event(
        event_type=event_type,
        source_id=source_id,
        timestamp=timestamp,
        raw_payload=raw_payload,
        file_hash=file_hash,
    )
    return hmac.compare_digest(expected, provenance_hash)


def sign_evidence_package(
    file_path: str,
    raw_event_ids: list[str],
    derived_event_ids: list[str],
    signer: str = "system",
) -> dict[str, str]:
    """
    Sign an evidence package for export.
    Returns signature and metadata.
    """
    file_hash = compute_file_hash(file_path)
    message = "|".join([
        file_hash,
        ",".join(sorted(raw_event_ids)),
        ",".join(sorted(derived_event_ids)),
        signer,
    ]).encode()

    signature = hmac.new(_SIGNING_KEY, message, hashlib.sha256).hexdigest()

    return {
        "file_hash": file_hash,
        "signature": f"hmac-sha256:{signature}",
        "signer": signer,
        "signed_at": datetime.utcnow().isoformat(),
        "raw_event_count": len(raw_event_ids),
        "derived_event_count": len(derived_event_ids),
    }
