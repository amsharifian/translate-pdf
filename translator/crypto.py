"""Light encryption helpers for API keys stored in SQLite.

Uses Fernet symmetric encryption with a machine-local key.
The key file is created once and lives alongside the database.
"""
from __future__ import annotations

from pathlib import Path

from cryptography.fernet import Fernet

_KEY_PATH = Path("jobs/.key")


def _get_or_create_key() -> bytes:
    if _KEY_PATH.exists():
        return _KEY_PATH.read_bytes()
    _KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    _KEY_PATH.write_bytes(key)
    _KEY_PATH.chmod(0o600)
    return key


def encrypt(plaintext: str) -> str:
    if not plaintext:
        return plaintext
    f = Fernet(_get_or_create_key())
    return f.encrypt(plaintext.encode()).decode()


def decrypt(token: str) -> str:
    if not token:
        return token
    try:
        f = Fernet(_get_or_create_key())
        return f.decrypt(token.encode()).decode()
    except Exception:
        # Fallback: the value may have been stored before encryption was added
        return token
