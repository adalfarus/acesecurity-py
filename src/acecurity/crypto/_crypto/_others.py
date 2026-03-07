from __future__ import annotations

import hmac
from typing import Optional

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf import concatkdf, hkdf, pbkdf2, scrypt, x963kdf

try:
    from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
except Exception:  # pragma: no cover - version dependent import
    Argon2id = None


# --- Hashes --- #


def _generic_hash(algorithm, data: bytes, digest_size: Optional[int] = None) -> bytes:
    if algorithm in (hashes.SHAKE128, hashes.SHAKE256, hashes.BLAKE2b, hashes.BLAKE2s):
        digest_size = digest_size or 32
        h = hashes.Hash(algorithm(digest_size), backend=default_backend())
    else:
        h = hashes.Hash(algorithm(), backend=default_backend())
    h.update(data)
    return h.finalize()


def _verify_hash(data: bytes, expected: bytes, algo_fn, *args, **kwargs) -> bool:
    return hmac.compare_digest(algo_fn(data, *args, **kwargs), expected)


def hash_sha1(data: bytes) -> bytes:
    return _generic_hash(hashes.SHA1, data)


def hash_verify_sha1(data: bytes, expected: bytes) -> bool:
    return _verify_hash(data, expected, hash_sha1)


# SHA2

def hash_sha256(data: bytes) -> bytes:
    return _generic_hash(hashes.SHA256, data)


def hash_verify_sha256(data: bytes, expected: bytes) -> bool:
    return _verify_hash(data, expected, hash_sha256)


def hash_sha384(data: bytes) -> bytes:
    return _generic_hash(hashes.SHA384, data)


def hash_verify_sha384(data: bytes, expected: bytes) -> bool:
    return _verify_hash(data, expected, hash_sha384)


def hash_sha512(data: bytes) -> bytes:
    return _generic_hash(hashes.SHA512, data)


def hash_verify_sha512(data: bytes, expected: bytes) -> bool:
    return _verify_hash(data, expected, hash_sha512)


def hash_sha224(data: bytes) -> bytes:
    return _generic_hash(hashes.SHA224, data)


def hash_verify_sha224(data: bytes, expected: bytes) -> bool:
    return _verify_hash(data, expected, hash_sha224)


def hash_sha512_224(data: bytes) -> bytes:
    return _generic_hash(hashes.SHA512_224, data)


def hash_verify_sha512_224(data: bytes, expected: bytes) -> bool:
    return _verify_hash(data, expected, hash_sha512_224)


def hash_sha512_256(data: bytes) -> bytes:
    return _generic_hash(hashes.SHA512_256, data)


def hash_verify_sha512_256(data: bytes, expected: bytes) -> bool:
    return _verify_hash(data, expected, hash_sha512_256)


# SHA3

def hash_sha3_256(data: bytes) -> bytes:
    return _generic_hash(hashes.SHA3_256, data)


def hash_verify_sha3_256(data: bytes, expected: bytes) -> bool:
    return _verify_hash(data, expected, hash_sha3_256)


def hash_sha3_512(data: bytes) -> bytes:
    return _generic_hash(hashes.SHA3_512, data)


def hash_verify_sha3_512(data: bytes, expected: bytes) -> bool:
    return _verify_hash(data, expected, hash_sha3_512)


def hash_sha3_224(data: bytes) -> bytes:
    return _generic_hash(hashes.SHA3_224, data)


def hash_verify_sha3_224(data: bytes, expected: bytes) -> bool:
    return _verify_hash(data, expected, hash_sha3_224)


def hash_sha3_384(data: bytes) -> bytes:
    return _generic_hash(hashes.SHA3_384, data)


def hash_verify_sha3_384(data: bytes, expected: bytes) -> bool:
    return _verify_hash(data, expected, hash_sha3_384)


# SHAKE

def hash_sha3_shake_128(data: bytes, digest_size: int = 32) -> bytes:
    return _generic_hash(hashes.SHAKE128, data, digest_size)


def hash_verify_sha3_shake_128(data: bytes, expected: bytes, digest_size: int = 32) -> bool:
    return _verify_hash(data, expected, hash_sha3_shake_128, digest_size)


def hash_sha3_shake_256(data: bytes, digest_size: int = 32) -> bytes:
    return _generic_hash(hashes.SHAKE256, data, digest_size)


def hash_verify_sha3_shake_256(data: bytes, expected: bytes, digest_size: int = 32) -> bool:
    return _verify_hash(data, expected, hash_sha3_shake_256, digest_size)


# Others

def hash_blake2b(data: bytes, digest_size: int = 64) -> bytes:
    return _generic_hash(hashes.BLAKE2b, data, digest_size)


def hash_verify_blake2b(data: bytes, expected: bytes, digest_size: int = 64) -> bool:
    return _verify_hash(data, expected, hash_blake2b, digest_size)


def hash_blake2s(data: bytes, digest_size: int = 32) -> bytes:
    return _generic_hash(hashes.BLAKE2s, data, digest_size)


def hash_verify_blake2s(data: bytes, expected: bytes, digest_size: int = 32) -> bool:
    return _verify_hash(data, expected, hash_blake2s, digest_size)


def hash_md5(data: bytes) -> bytes:
    return _generic_hash(hashes.MD5, data)


def hash_verify_md5(data: bytes, expected: bytes) -> bool:
    return _verify_hash(data, expected, hash_md5)


def hash_sm3(data: bytes) -> bytes:
    return _generic_hash(hashes.SM3, data)


def hash_verify_sm3(data: bytes, expected: bytes) -> bool:
    return _verify_hash(data, expected, hash_sm3)


# --- KDFs --- #

algo_map = {
    "sha1": hashes.SHA1,
    "sha224": hashes.SHA224,
    "sha256": hashes.SHA256,
    "sha384": hashes.SHA384,
    "sha512": hashes.SHA512,
    "sha512_224": hashes.SHA512_224,
    "sha512_256": hashes.SHA512_256,
    "sha3_224": hashes.SHA3_224,
    "sha3_256": hashes.SHA3_256,
    "sha3_384": hashes.SHA3_384,
    "sha3_512": hashes.SHA3_512,
    "blake2b": hashes.BLAKE2b,
    "blake2s": hashes.BLAKE2s,
    "sm3": hashes.SM3,
    "md5": hashes.MD5,
}


def derive_pbkdf2hmac(password: bytes, salt: bytes, length: int, iterations: int, algorithm: str) -> bytes:
    algo_cls = algo_map.get(algorithm)
    if algo_cls is None:
        raise ValueError(f"Passed invalid/unsupported algorithm '{algorithm}'")
    return pbkdf2.PBKDF2HMAC(
        algorithm=algo_cls(),
        length=length,
        salt=salt,
        iterations=iterations,
        backend=default_backend(),
    ).derive(password)


def derive_scrypt(password: bytes, salt: bytes, length: int, n: int, r: int, p: int) -> bytes:
    return scrypt.Scrypt(salt=salt, length=length, n=n, r=r, p=p, backend=default_backend()).derive(password)


def derive_hkdf(password: bytes, salt: bytes, info: bytes, length: int, algorithm: str) -> bytes:
    algo_cls = algo_map.get(algorithm)
    if algo_cls is None:
        raise ValueError(f"Passed invalid/unsupported algorithm '{algorithm}'")
    return hkdf.HKDF(
        algorithm=algo_cls(),
        length=length,
        salt=salt,
        info=info,
        backend=default_backend(),
    ).derive(password)


def derive_x963(password: bytes, length: int, sharedinfo: bytes | None, algorithm: str) -> bytes:
    algo_cls = algo_map.get(algorithm)
    if algo_cls is None:
        raise ValueError(f"Passed invalid/unsupported algorithm '{algorithm}'")
    return x963kdf.X963KDF(
        algorithm=algo_cls(),
        length=length,
        sharedinfo=sharedinfo,
        backend=default_backend(),
    ).derive(password)


def derive_concatkdf(password: bytes, length: int, otherinfo: bytes | None, algorithm: str) -> bytes:
    algo_cls = algo_map.get(algorithm)
    if algo_cls is None:
        raise ValueError(f"Passed invalid/unsupported algorithm '{algorithm}'")
    return concatkdf.ConcatKDFHash(
        algorithm=algo_cls(),
        length=length,
        otherinfo=otherinfo,
        backend=default_backend(),
    ).derive(password)


def derive_argon2(
    password: bytes,
    salt: bytes,
    length: int,
    time_cost: int,
    memory_cost: int,
    parallelism: int,
    variant: str,
) -> bytes:
    if variant != "id":
        raise ValueError("cryptography Argon2 backend only supports variant='id'")
    if Argon2id is None:
        raise ValueError("Argon2id is unavailable in installed cryptography version")

    try:
        kdf = Argon2id(
            salt=salt,
            length=length,
            iterations=time_cost,
            lanes=parallelism,
            memory_cost=memory_cost,
        )
    except TypeError:
        # Compatibility fallback if constructor signature differs.
        kdf = Argon2id(
            salt=salt,
            length=length,
            time_cost=time_cost,
            parallelism=parallelism,
            memory_cost=memory_cost,
        )
    return kdf.derive(password)


def derive_kmac128(password: bytes, length: int, key: bytes = b"", customization: bytes = b"") -> bytes:
    kmac128 = getattr(hashes, "KMAC128", None)
    if kmac128 is None:
        raise ValueError("KMAC128 is unavailable in installed cryptography version")
    h = hashes.Hash(kmac128(key=key, length=length, customization_string=customization), backend=default_backend())
    h.update(password)
    return h.finalize()


def derive_kmac256(password: bytes, length: int, key: bytes = b"", customization: bytes = b"") -> bytes:
    kmac256 = getattr(hashes, "KMAC256", None)
    if kmac256 is None:
        raise ValueError("KMAC256 is unavailable in installed cryptography version")
    h = hashes.Hash(kmac256(key=key, length=length, customization_string=customization), backend=default_backend())
    h.update(password)
    return h.finalize()
