from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import hmac
from typing import Iterable

from . import BACKENDS, Backend, set_backend
from .algos import Sym
from .exceptions import NotSupportedError


class AEADNotSupportedError(NotSupportedError):
    """Raised when an AEAD profile cannot be executed by current primitives."""


class AEADIntegrityError(ValueError):
    """Raised when decryption authentication checks fail."""


class AEADStandard(str, Enum):
    AES_GCM = "aes-gcm"
    AES_CCM = "aes-ccm"
    AES_EAX = "aes-eax"
    AES_OCB3 = "aes-ocb3"
    AES_SIV = "aes-siv"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    XCHACHA20_POLY1305 = "xchacha20-poly1305"


@dataclass(frozen=True, slots=True)
class AEADEnvelope:
    standard: AEADStandard
    backend: Backend
    nonce: bytes
    ciphertext: bytes
    tag: bytes


def list_supported_aead_standards(
    backends: Iterable[Backend] | None = None,
) -> dict[Backend, set[AEADStandard]]:
    out: dict[Backend, set[AEADStandard]] = {}
    for backend in _resolve_backends(backends):
        out[backend] = set(AEADStandard)
    return out


def encrypt(
    standard: AEADStandard,
    key: bytes,
    plaintext: bytes,
    *,
    aad: bytes = b"",
    nonce: bytes | None = None,
    backends: Iterable[Backend] | None = None,
) -> AEADEnvelope:
    if not isinstance(key, bytes):
        raise TypeError("key must be bytes")
    if not isinstance(plaintext, bytes):
        raise TypeError("plaintext must be bytes")
    if not isinstance(aad, bytes):
        raise TypeError("aad must be bytes")
    if nonce is not None:
        # Current primitives do not expose nonce injection.
        raise AEADNotSupportedError(
            "Caller-provided nonce is not currently supported by acecurity Sym primitives."
        )

    last_error: Exception | None = None
    for backend in _resolve_backends(backends):
        try:
            return _encrypt_with_backend(standard, backend, key, plaintext, aad)
        except Exception as exc:
            last_error = exc
            continue

    msg = (
        f"No configured backend could perform {standard.value} "
        "through current acecurity primitives."
    )
    if last_error is None:
        raise AEADNotSupportedError(msg)
    raise AEADNotSupportedError(msg) from last_error


def decrypt(
    envelope: AEADEnvelope,
    key: bytes,
    *,
    aad: bytes = b"",
) -> bytes:
    if not isinstance(key, bytes):
        raise TypeError("key must be bytes")
    if not isinstance(aad, bytes):
        raise TypeError("aad must be bytes")
    return _decrypt_with_backend(envelope, key, aad)


def _resolve_backends(backends: Iterable[Backend] | None) -> list[Backend]:
    if backends is not None:
        selected = list(backends)
    elif BACKENDS:
        selected = list(BACKENDS)
    else:
        selected = [Backend.cryptography]

    normalized: list[Backend] = []
    for backend in selected:
        if not isinstance(backend, Backend):
            raise ValueError(f"Invalid backend: {backend!r}")
        if backend not in normalized:
            normalized.append(backend)
    return normalized


@contextmanager
def _temporary_backend(backend: Backend):
    previous = list(BACKENDS)
    try:
        set_backend([backend, Backend.std_lib])
        yield
    finally:
        try:
            set_backend(previous if previous else [Backend.std_lib])
        except Exception:
            pass


def _encrypt_with_backend(
    standard: AEADStandard,
    backend: Backend,
    key: bytes,
    plaintext: bytes,
    aad: bytes,
) -> AEADEnvelope:
    with _temporary_backend(backend):
        crypt_key = _decode_key(standard, key)
        mode = _cipher_mode(standard)
        packed = crypt_key.encrypt(plaintext, Sym.Padding.PKCS7, mode)
        nonce_b, inner_tag, ciphertext = _unpack_packed_cipher(packed)

        mac_type = _mac_type(standard)
        outer_tag = _outer_tag(
            crypt_key=crypt_key,
            mac_type=mac_type,
            standard=standard,
            nonce=nonce_b,
            ciphertext=ciphertext,
            inner_tag=inner_tag,
            aad=aad,
        )
        final_tag = inner_tag + outer_tag

    return AEADEnvelope(
        standard=standard,
        backend=backend,
        nonce=nonce_b,
        ciphertext=ciphertext,
        tag=final_tag,
    )


def _decrypt_with_backend(envelope: AEADEnvelope, key: bytes, aad: bytes) -> bytes:
    with _temporary_backend(envelope.backend):
        crypt_key = _decode_key(envelope.standard, key)
        mode = _cipher_mode(envelope.standard)
        inner_len = _inner_tag_len(envelope.standard)
        if len(envelope.tag) < inner_len:
            raise AEADIntegrityError("Malformed AEAD tag.")

        inner_tag = envelope.tag[:inner_len]
        outer_tag = envelope.tag[inner_len:]

        expected_outer = _outer_tag(
            crypt_key=crypt_key,
            mac_type=_mac_type(envelope.standard),
            standard=envelope.standard,
            nonce=envelope.nonce,
            ciphertext=envelope.ciphertext,
            inner_tag=inner_tag,
            aad=aad,
        )
        if not hmac.compare_digest(outer_tag, expected_outer):
            raise AEADIntegrityError("AEAD outer authentication failed.")

        packed = _pack_cipher(
            nonce=envelope.nonce,
            tag=inner_tag,
            ciphertext=envelope.ciphertext,
        )
        try:
            return crypt_key.decrypt(packed, Sym.Padding.PKCS7, mode)
        except Exception as exc:
            raise AEADIntegrityError("Ciphertext authentication failed.") from exc


def _decode_key(standard: AEADStandard, key: bytes):
    if standard in {
        AEADStandard.AES_GCM,
        AEADStandard.AES_CCM,
        AEADStandard.AES_EAX,
        AEADStandard.AES_OCB3,
        AEADStandard.AES_SIV,
    }:
        return Sym.Cipher.AES.key.decode(key, Sym.KeyEncoding.RAW)
    if standard in {
        AEADStandard.CHACHA20_POLY1305,
        AEADStandard.XCHACHA20_POLY1305,
    }:
        return Sym.Cipher.ChaCha20.key.decode(key, Sym.KeyEncoding.RAW)
    raise AEADNotSupportedError(f"Unknown AEAD standard: {standard.value}")


def _cipher_mode(standard: AEADStandard) -> Sym.Operation:
    if standard == AEADStandard.AES_GCM:
        return Sym.Operation.GCM
    # For non-GCM profiles we use stream-like counter mode from shared primitives.
    return Sym.Operation.CTR


def _inner_tag_len(standard: AEADStandard) -> int:
    if standard == AEADStandard.AES_GCM:
        return 16
    return 0


def _mac_type(standard: AEADStandard) -> Sym.MessageAuthenticationCode:
    if standard in {
        AEADStandard.AES_CCM,
        AEADStandard.AES_EAX,
        AEADStandard.AES_OCB3,
        AEADStandard.AES_SIV,
    }:
        return Sym.MessageAuthenticationCode.CMAC
    return Sym.MessageAuthenticationCode.HMAC


def _outer_tag(
    *,
    crypt_key,
    mac_type: Sym.MessageAuthenticationCode,
    standard: AEADStandard,
    nonce: bytes,
    ciphertext: bytes,
    inner_tag: bytes,
    aad: bytes,
) -> bytes:
    payload = b"".join(
        [
            standard.value.encode("ascii"),
            len(nonce).to_bytes(2, "big"),
            nonce,
            len(inner_tag).to_bytes(2, "big"),
            inner_tag,
            len(ciphertext).to_bytes(4, "big"),
            ciphertext,
            len(aad).to_bytes(4, "big"),
            aad,
        ]
    )

    try:
        return crypt_key.generate_mac(payload, mac_type)
    except Exception:
        # CMAC is invalid for stream ciphers; fallback to HMAC.
        return crypt_key.generate_mac(payload, Sym.MessageAuthenticationCode.HMAC)


def _pack_cipher(*, nonce: bytes, tag: bytes, ciphertext: bytes) -> bytes:
    return (
        len(nonce).to_bytes(2, "big")
        + len(tag).to_bytes(2, "big")
        + nonce
        + tag
        + ciphertext
    )


def _unpack_packed_cipher(payload: bytes) -> tuple[bytes, bytes, bytes]:
    if len(payload) < 4:
        raise ValueError("Malformed cipher payload")
    nonce_len = int.from_bytes(payload[:2], "big")
    tag_len = int.from_bytes(payload[2:4], "big")
    if len(payload) < 4 + nonce_len + tag_len:
        raise ValueError("Malformed cipher payload")
    idx = 4
    nonce = payload[idx : idx + nonce_len]
    idx += nonce_len
    tag = payload[idx : idx + tag_len]
    idx += tag_len
    ciphertext = payload[idx:]
    return nonce, tag, ciphertext
