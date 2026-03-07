from __future__ import annotations

import hmac
from typing import Callable

from Cryptodome.Hash import (
    BLAKE2b,
    BLAKE2s,
    MD5,
    RIPEMD160,
    SHA1,
    SHA224,
    SHA256,
    SHA384,
    SHA3_224,
    SHA3_256,
    SHA3_384,
    SHA3_512,
    SHA512,
    SHA512_224,
    SHA512_256,
    SHAKE128,
    SHAKE256,
)
from Cryptodome.Protocol.KDF import HKDF, PBKDF1, PBKDF2, scrypt

try:
    from Cryptodome.Hash import KMAC128, KMAC256
except Exception:  # pragma: no cover
    KMAC128 = None
    KMAC256 = None

_HASH_FNS: dict[str, Callable[..., bytes]] = {}


def _reg(name: str):
    def _wrap(fn: Callable[..., bytes]):
        _HASH_FNS[name] = fn
        return fn

    return _wrap


@_reg("sha1")
def hash_sha1(data: bytes) -> bytes:
    return SHA1.new(data).digest()


@_reg("sha224")
def hash_sha224(data: bytes) -> bytes:
    return SHA224.new(data).digest()


@_reg("sha256")
def hash_sha256(data: bytes) -> bytes:
    return SHA256.new(data).digest()


@_reg("sha384")
def hash_sha384(data: bytes) -> bytes:
    return SHA384.new(data).digest()


@_reg("sha512")
def hash_sha512(data: bytes) -> bytes:
    return SHA512.new(data).digest()


@_reg("sha512_224")
def hash_sha512_224(data: bytes) -> bytes:
    return SHA512_224.new(data).digest()


@_reg("sha512_256")
def hash_sha512_256(data: bytes) -> bytes:
    return SHA512_256.new(data).digest()


@_reg("sha3_224")
def hash_sha3_224(data: bytes) -> bytes:
    return SHA3_224.new(data).digest()


@_reg("sha3_256")
def hash_sha3_256(data: bytes) -> bytes:
    return SHA3_256.new(data).digest()


@_reg("sha3_384")
def hash_sha3_384(data: bytes) -> bytes:
    return SHA3_384.new(data).digest()


@_reg("sha3_512")
def hash_sha3_512(data: bytes) -> bytes:
    return SHA3_512.new(data).digest()


@_reg("sha3_shake_128")
def hash_sha3_shake_128(data: bytes, length: int = 32) -> bytes:
    return SHAKE128.new(data).read(length)


@_reg("sha3_shake_256")
def hash_sha3_shake_256(data: bytes, length: int = 32) -> bytes:
    return SHAKE256.new(data).read(length)


@_reg("blake2b")
def hash_blake2b(data: bytes, digest_size: int = 64) -> bytes:
    return BLAKE2b.new(data=data, digest_bytes=digest_size).digest()


@_reg("blake2s")
def hash_blake2s(data: bytes, digest_size: int = 32) -> bytes:
    return BLAKE2s.new(data=data, digest_bits=digest_size * 8).digest()


@_reg("md5")
def hash_md5(data: bytes) -> bytes:
    return MD5.new(data).digest()


@_reg("ripemd160")
def hash_ripemd160(data: bytes) -> bytes:
    return RIPEMD160.new(data).digest()


def _verify(name: str, data: bytes, expected: bytes, *args) -> bool:
    fn = _HASH_FNS[name]
    return hmac.compare_digest(fn(data, *args), expected)


def hash_verify_sha1(data: bytes, expected: bytes) -> bool:
    return _verify("sha1", data, expected)


def hash_verify_sha224(data: bytes, expected: bytes) -> bool:
    return _verify("sha224", data, expected)


def hash_verify_sha256(data: bytes, expected: bytes) -> bool:
    return _verify("sha256", data, expected)


def hash_verify_sha384(data: bytes, expected: bytes) -> bool:
    return _verify("sha384", data, expected)


def hash_verify_sha512(data: bytes, expected: bytes) -> bool:
    return _verify("sha512", data, expected)


def hash_verify_sha512_224(data: bytes, expected: bytes) -> bool:
    return _verify("sha512_224", data, expected)


def hash_verify_sha512_256(data: bytes, expected: bytes) -> bool:
    return _verify("sha512_256", data, expected)


def hash_verify_sha3_224(data: bytes, expected: bytes) -> bool:
    return _verify("sha3_224", data, expected)


def hash_verify_sha3_256(data: bytes, expected: bytes) -> bool:
    return _verify("sha3_256", data, expected)


def hash_verify_sha3_384(data: bytes, expected: bytes) -> bool:
    return _verify("sha3_384", data, expected)


def hash_verify_sha3_512(data: bytes, expected: bytes) -> bool:
    return _verify("sha3_512", data, expected)


def hash_verify_sha3_shake_128(data: bytes, expected: bytes, length: int = 32) -> bool:
    return _verify("sha3_shake_128", data, expected, length)


def hash_verify_sha3_shake_256(data: bytes, expected: bytes, length: int = 32) -> bool:
    return _verify("sha3_shake_256", data, expected, length)


def hash_verify_blake2b(data: bytes, expected: bytes, digest_size: int = 64) -> bool:
    return _verify("blake2b", data, expected, digest_size)


def hash_verify_blake2s(data: bytes, expected: bytes, digest_size: int = 32) -> bool:
    return _verify("blake2s", data, expected, digest_size)


def hash_verify_md5(data: bytes, expected: bytes) -> bool:
    return _verify("md5", data, expected)


def hash_verify_ripemd160(data: bytes, expected: bytes) -> bool:
    return _verify("ripemd160", data, expected)


_HASH_MODS = {
    "sha1": SHA1,
    "sha224": SHA224,
    "sha256": SHA256,
    "sha384": SHA384,
    "sha512": SHA512,
    "sha512_224": SHA512_224,
    "sha512_256": SHA512_256,
    "sha3_224": SHA3_224,
    "sha3_256": SHA3_256,
    "sha3_384": SHA3_384,
    "sha3_512": SHA3_512,
    "md5": MD5,
    "ripemd160": RIPEMD160,
}


def _get_hash_mod(name: str):
    mod = _HASH_MODS.get(name)
    if mod is None:
        raise ValueError(f"Unsupported hash algorithm '{name}'")
    return mod


def derive_pbkdf2hmac(password: bytes, salt: bytes, length: int, iterations: int, algorithm: str) -> bytes:
    return PBKDF2(password, salt, dkLen=length, count=iterations, hmac_hash_module=_get_hash_mod(algorithm))


def derive_pbkdf1(password: bytes, salt: bytes, length: int, iterations: int, algorithm: str) -> bytes:
    return PBKDF1(password, salt, dkLen=length, count=iterations, hashAlgo=_get_hash_mod(algorithm))


def derive_scrypt(password: bytes, salt: bytes, length: int, n: int, r: int, p: int) -> bytes:
    return scrypt(password, salt=salt, key_len=length, N=n, r=r, p=p)


def derive_hkdf(password: bytes, salt: bytes, info: bytes, length: int, algorithm: str) -> bytes:
    return HKDF(password, key_len=length, salt=salt, hashmod=_get_hash_mod(algorithm), context=info)


def derive_x963(password: bytes, length: int, otherinfo: bytes, algorithm: str) -> bytes:
    # ANSI X9.63 style KDF approximation using counter || Z || sharedinfo.
    return derive_concatkdf(password, length, otherinfo, algorithm)


def derive_concatkdf(password: bytes, length: int, otherinfo: bytes, algorithm: str) -> bytes:
    hash_mod = _get_hash_mod(algorithm)
    out = b""
    counter = 1
    while len(out) < length:
        h = hash_mod.new()
        h.update(counter.to_bytes(4, "big"))
        h.update(password)
        h.update(otherinfo)
        out += h.digest()
        counter += 1
    return out[:length]


def derive_kmac128(password: bytes, length: int, key: bytes = b"", customization: bytes = b"") -> bytes:
    if KMAC128 is None:
        raise ValueError("KMAC128 is unavailable in installed pycryptodomex")
    kmac = KMAC128.new(key=key, mac_len=length, custom=customization)
    kmac.update(password)
    return kmac.digest()


def derive_kmac256(password: bytes, length: int, key: bytes = b"", customization: bytes = b"") -> bytes:
    if KMAC256 is None:
        raise ValueError("KMAC256 is unavailable in installed pycryptodomex")
    kmac = KMAC256.new(key=key, mac_len=length, custom=customization)
    kmac.update(password)
    return kmac.digest()
