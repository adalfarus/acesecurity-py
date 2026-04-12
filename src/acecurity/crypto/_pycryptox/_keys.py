from __future__ import annotations

import base64
import hashlib
import os
import typing as _ty

from Cryptodome.Cipher import AES, ARC2, ARC4, Blowfish, CAST, ChaCha20, DES, DES3, PKCS1_OAEP, PKCS1_v1_5, Salsa20
from Cryptodome.Hash import CMAC, HMAC, SHA256
from Cryptodome.Protocol import DH
from Cryptodome.PublicKey import DSA, ECC, RSA
from Cryptodome.Signature import DSS, eddsa, pkcs1_15, pss
from Cryptodome.Util.Padding import pad, unpad

try:
    from Cryptodome.Cipher import Camellia
except Exception:  # pragma: no cover
    Camellia = None

try:
    from Cryptodome.Cipher import IDEA
except Exception:  # pragma: no cover
    IDEA = None

try:
    from Cryptodome.Cipher import SEED
except Exception:  # pragma: no cover
    SEED = None

try:
    from Cryptodome.Cipher import SM4
except Exception:  # pragma: no cover
    SM4 = None

from .._definitions import (
    Backend,
    ECCCurve,
    ECCType,
    _AES_KEYTYPE,
    _ARC2_KEYTYPE,
    _ARC4_KEYTYPE,
    _Blowfish_KEYTYPE,
    _CAST5_KEYTYPE,
    _Camellia_KEYTYPE,
    _ChaCha20_KEYTYPE,
    _DES_KEYTYPE,
    _DSA_KEYPAIRTYPE,
    _ECC_KEYPAIRTYPE,
    _IDEA_KEYTYPE,
    _RSA_KEYPAIRTYPE,
    _Salsa20_KEYTYPE,
    _SEED_KEYTYPE,
    _SM4_KEYTYPE,
    _TripleDES_KEYTYPE,
)
from ..algos._asym import KeyEncoding as ASymKeyEncoding, KeyFormat as ASymKeyFormat, Padding as ASymPadding
from ..algos._sym import (
    KeyEncoding as SymKeyEncoding,
    MessageAuthenticationCode as MAC,
    Operation as SymOperation,
    Padding as SymPadding,
)
from ..exceptions import NotSupportedError as _NotSupportedError

_BLOCK_CIPHERS: dict[type, tuple[_ty.Any, tuple[int, ...] | None]] = {
    _AES_KEYTYPE: (AES, (16, 24, 32)),
    _TripleDES_KEYTYPE: (DES3, (24,)),
    _Blowfish_KEYTYPE: (Blowfish, None),
    _CAST5_KEYTYPE: (CAST, None),
    _DES_KEYTYPE: (DES, (8,)),
    _ARC2_KEYTYPE: (ARC2, None),
}
if Camellia is not None:
    _BLOCK_CIPHERS[_Camellia_KEYTYPE] = (Camellia, None)
if IDEA is not None:
    _BLOCK_CIPHERS[_IDEA_KEYTYPE] = (IDEA, (16,))
if SEED is not None:
    _BLOCK_CIPHERS[_SEED_KEYTYPE] = (SEED, (16,))
if SM4 is not None:
    _BLOCK_CIPHERS[_SM4_KEYTYPE] = (SM4, (16,))

_STREAM_CIPHERS: dict[type, tuple[_ty.Any, int | None]] = {
    _ARC4_KEYTYPE: (ARC4, None),
    _ChaCha20_KEYTYPE: (ChaCha20, 32),
    _Salsa20_KEYTYPE: (Salsa20, None),
}


def _derive_key_from_password(password: bytes | str, length: int, salt: bytes) -> bytes:
    if isinstance(password, str):
        password = password.encode("utf-8")
    return hashlib.pbkdf2_hmac("sha256", password, salt, 100_000, dklen=length)


def _encode_bytes(data: bytes, encoding: SymKeyEncoding) -> bytes:
    if encoding == SymKeyEncoding.RAW:
        return data
    if encoding == SymKeyEncoding.BASE64:
        return base64.b64encode(data)
    if encoding == SymKeyEncoding.HEX:
        return data.hex().encode("ascii")
    if encoding == SymKeyEncoding.BASE32:
        return base64.b32encode(data)
    raise _NotSupportedError(f"Unsupported key encoding: {encoding}")


def _decode_bytes(data: bytes, encoding: SymKeyEncoding) -> bytes:
    if encoding == SymKeyEncoding.RAW:
        return data
    if encoding == SymKeyEncoding.BASE64:
        return base64.b64decode(data)
    if encoding == SymKeyEncoding.HEX:
        return bytes.fromhex(data.decode("ascii"))
    if encoding == SymKeyEncoding.BASE32:
        return base64.b32decode(data)
    raise _NotSupportedError(f"Unsupported key encoding: {encoding}")


def _pack(parts: dict[str, bytes]) -> bytes:
    iv = parts.get("iv", b"")
    tag = parts.get("tag", b"")
    ct = parts["ciphertext"]
    return len(iv).to_bytes(2, "big") + len(tag).to_bytes(2, "big") + iv + tag + ct


def _unpack(payload: bytes) -> dict[str, bytes]:
    if len(payload) < 4:
        raise ValueError("cipher payload too short")
    iv_len = int.from_bytes(payload[:2], "big")
    tag_len = int.from_bytes(payload[2:4], "big")
    if len(payload) < 4 + iv_len + tag_len:
        raise ValueError("cipher payload malformed")
    base = 4
    iv = payload[base:base + iv_len]
    base += iv_len
    tag = payload[base:base + tag_len]
    ct = payload[base + tag_len:]
    out = {"ciphertext": ct}
    if iv_len:
        out["iv"] = iv
    if tag_len:
        out["tag"] = tag
    return out


class _SymBase:
    backend = Backend.pycryptodomex
    __concrete__ = True

    _ciphermod: _ty.Any
    _stream: bool = False
    _fixed_len: int | None = None
    _allow_gcm: bool = False

    def __init__(self, key_size: int, pwd: bytes | str | None) -> None:
        key_len = self._fixed_len if self._fixed_len is not None else key_size // 8
        self.key_size = key_len * 8

        if pwd is None:
            self._key = os.urandom(key_len)
        else:
            pwd_b = pwd.encode("utf-8") if isinstance(pwd, str) else pwd
            if len(pwd_b) == key_len:
                self._key = pwd_b
            else:
                self._key = _derive_key_from_password(pwd_b, key_len, hashlib.sha256(pwd_b).digest()[:16])

    @classmethod
    def decode(cls, key: bytes, encoding: SymKeyEncoding) -> _ty.Self:
        raw = _decode_bytes(key, encoding)
        size_bits = len(raw) * 8
        try:
            obj = cls(raw)  # for pwd-only constructors
        except TypeError:
            obj = cls(size_bits, raw)
        return obj

    def encode(self, encoding: SymKeyEncoding) -> bytes:
        return _encode_bytes(self._key, encoding)

    def _block_size(self) -> int:
        if self._stream:
            return 1
        return self._ciphermod.block_size

    def _pad(self, data: bytes, padding: SymPadding) -> bytes:
        if padding == SymPadding.PKCS7:
            return pad(data, self._block_size(), style="pkcs7")
        if padding == SymPadding.ANSIX923:
            return pad(data, self._block_size(), style="x923")
        raise _NotSupportedError(f"Unsupported padding {padding}")

    def _unpad(self, data: bytes, padding: SymPadding) -> bytes:
        if padding == SymPadding.PKCS7:
            return unpad(data, self._block_size(), style="pkcs7")
        if padding == SymPadding.ANSIX923:
            return unpad(data, self._block_size(), style="x923")
        raise _NotSupportedError(f"Unsupported padding {padding}")

    def _cipher_for_encrypt(self, mode: SymOperation):
        if self._stream:
            if self._ciphermod is ARC4:
                return self._ciphermod.new(self._key), {}
            if self._ciphermod is ChaCha20:
                nonce = os.urandom(8)
                return self._ciphermod.new(key=self._key, nonce=nonce), {"iv": nonce}
            if self._ciphermod is Salsa20:
                nonce = os.urandom(8)
                return self._ciphermod.new(key=self._key, nonce=nonce), {"iv": nonce}

        if mode == SymOperation.ECB:
            return self._ciphermod.new(self._key, self._ciphermod.MODE_ECB), {}
        if mode == SymOperation.CBC:
            iv = os.urandom(self._ciphermod.block_size)
            return self._ciphermod.new(self._key, self._ciphermod.MODE_CBC, iv=iv), {"iv": iv}
        if mode == SymOperation.CFB:
            iv = os.urandom(self._ciphermod.block_size)
            return self._ciphermod.new(self._key, self._ciphermod.MODE_CFB, iv=iv, segment_size=self._ciphermod.block_size * 8), {"iv": iv}
        if mode == SymOperation.OFB:
            iv = os.urandom(self._ciphermod.block_size)
            return self._ciphermod.new(self._key, self._ciphermod.MODE_OFB, iv=iv), {"iv": iv}
        if mode == SymOperation.CTR:
            nonce = os.urandom(max(1, self._ciphermod.block_size // 2))
            return self._ciphermod.new(self._key, self._ciphermod.MODE_CTR, nonce=nonce), {"iv": nonce}
        if mode == SymOperation.GCM:
            if not self._allow_gcm:
                raise _NotSupportedError("GCM is only supported for AES")
            nonce = os.urandom(12)
            return self._ciphermod.new(self._key, self._ciphermod.MODE_GCM, nonce=nonce), {"iv": nonce}
        raise _NotSupportedError(f"Unsupported operation mode {mode}")

    def _cipher_for_decrypt(self, mode: SymOperation, parts: dict[str, bytes]):
        iv = parts.get("iv")
        tag = parts.get("tag")

        if self._stream:
            if self._ciphermod is ARC4:
                return self._ciphermod.new(self._key)
            if self._ciphermod is ChaCha20:
                if iv is None:
                    raise ValueError("ChaCha20 requires nonce")
                return self._ciphermod.new(key=self._key, nonce=iv)
            if self._ciphermod is Salsa20:
                if iv is None:
                    raise ValueError("Salsa20 requires nonce")
                return self._ciphermod.new(key=self._key, nonce=iv)

        if mode == SymOperation.ECB:
            return self._ciphermod.new(self._key, self._ciphermod.MODE_ECB)
        if mode == SymOperation.CBC:
            if iv is None:
                raise ValueError("CBC requires IV")
            return self._ciphermod.new(self._key, self._ciphermod.MODE_CBC, iv=iv)
        if mode == SymOperation.CFB:
            if iv is None:
                raise ValueError("CFB requires IV")
            return self._ciphermod.new(self._key, self._ciphermod.MODE_CFB, iv=iv, segment_size=self._ciphermod.block_size * 8)
        if mode == SymOperation.OFB:
            if iv is None:
                raise ValueError("OFB requires IV")
            return self._ciphermod.new(self._key, self._ciphermod.MODE_OFB, iv=iv)
        if mode == SymOperation.CTR:
            if iv is None:
                raise ValueError("CTR requires nonce")
            return self._ciphermod.new(self._key, self._ciphermod.MODE_CTR, nonce=iv)
        if mode == SymOperation.GCM:
            if not self._allow_gcm:
                raise _NotSupportedError("GCM is only supported for AES")
            if iv is None or tag is None:
                raise ValueError("GCM requires nonce and tag")
            return self._ciphermod.new(self._key, self._ciphermod.MODE_GCM, nonce=iv)
        raise _NotSupportedError(f"Unsupported operation mode {mode}")

    def encrypt(self, plain_bytes: bytes, padding: SymPadding, mode: SymOperation, /, *, auto_pack: bool = True) -> bytes | dict[str, bytes]:
        if self._stream and mode is not SymOperation.CTR:
            raise _NotSupportedError("Stream ciphers support only CTR-like mode in this API")

        cipher, meta = self._cipher_for_encrypt(mode)
        payload = plain_bytes if self._stream or mode not in (SymOperation.ECB, SymOperation.CBC) else self._pad(plain_bytes, padding)
        ciphertext = cipher.encrypt(payload)
        parts = {"ciphertext": ciphertext, **meta}
        if mode == SymOperation.GCM and hasattr(cipher, "digest"):
            parts["tag"] = cipher.digest()
        return _pack(parts) if auto_pack else parts

    def decrypt(self, cipher_bytes_or_dict: bytes | dict[str, bytes], padding: SymPadding, mode: SymOperation, /, *, auto_pack: bool = True) -> bytes:
        if self._stream and mode is not SymOperation.CTR:
            raise _NotSupportedError("Stream ciphers support only CTR-like mode in this API")

        parts = _unpack(cipher_bytes_or_dict) if auto_pack else _ty.cast(dict[str, bytes], cipher_bytes_or_dict)
        cipher = self._cipher_for_decrypt(mode, parts)
        plaintext = cipher.decrypt(parts["ciphertext"])

        if mode == SymOperation.GCM:
            cipher.verify(parts["tag"])
        if not self._stream and mode in (SymOperation.ECB, SymOperation.CBC):
            plaintext = self._unpad(plaintext, padding)
        return plaintext

    def generate_mac(self, data: bytes, auth_type: MAC) -> bytes:
        if auth_type == MAC.HMAC:
            h = HMAC.new(self._key, digestmod=SHA256)
            h.update(data)
            return h.digest()
        if auth_type == MAC.CMAC:
            if self._stream:
                raise _NotSupportedError("CMAC is only supported for block ciphers")
            c = CMAC.new(self._key, ciphermod=self._ciphermod)
            c.update(data)
            return c.digest()
        raise _NotSupportedError(f"Unsupported MAC type {auth_type}")

    def verify_mac(self, data: bytes, mac: bytes, auth_type: MAC) -> bool:
        return self.generate_mac(data, auth_type) == mac


class _AES_KEY(_SymBase, _AES_KEYTYPE):
    _ciphermod = AES
    _allow_gcm = True


class _ChaCha20_KEY(_SymBase, _ChaCha20_KEYTYPE):
    _ciphermod = ChaCha20
    _stream = True
    _fixed_len = 32

    def __init__(self, pwd: bytes | str | None = None) -> None:
        super().__init__(256, pwd)


class _TripleDES_KEY(_SymBase, _TripleDES_KEYTYPE):
    _ciphermod = DES3
    _fixed_len = 24

    def __init__(self, key_size: _ty.Literal[192], pwd: bytes | str | None = None) -> None:
        super().__init__(key_size, pwd)


class _Blowfish_KEY(_SymBase, _Blowfish_KEYTYPE):
    _ciphermod = Blowfish


class _CAST5_KEY(_SymBase, _CAST5_KEYTYPE):
    _ciphermod = CAST


class _ARC4_KEY(_SymBase, _ARC4_KEYTYPE):
    _ciphermod = ARC4
    _stream = True

    def __init__(self, pwd: bytes | str | None = None) -> None:
        super().__init__(128, pwd)


if Camellia is not None:
    class _Camellia_KEY(_SymBase, _Camellia_KEYTYPE):
        _ciphermod = Camellia
else:
    class _Camellia_KEY(_Camellia_KEYTYPE): __concrete__ = False


if IDEA is not None:
    class _IDEA_KEY(_SymBase, _IDEA_KEYTYPE):
        _ciphermod = IDEA
        _fixed_len = 16

        def __init__(self, pwd: bytes | str | None = None) -> None:
            super().__init__(128, pwd)
else:
    class _IDEA_KEY(_IDEA_KEYTYPE): __concrete__ = False


if SEED is not None:
    class _SEED_KEY(_SymBase, _SEED_KEYTYPE):
        _ciphermod = SEED
        _fixed_len = 16

        def __init__(self, pwd: bytes | str | None = None) -> None:
            super().__init__(128, pwd)
else:
    class _SEED_KEY(_SEED_KEYTYPE): __concrete__ = False


if SM4 is not None:
    class _SM4_KEY(_SymBase, _SM4_KEYTYPE):
        _ciphermod = SM4
        _fixed_len = 16

        def __init__(self, pwd: bytes | str | None = None) -> None:
            super().__init__(128, pwd)
else:
    class _SM4_KEY(_SM4_KEYTYPE): __concrete__ = False


class _DES_KEY(_SymBase, _DES_KEYTYPE):
    _ciphermod = DES
    _fixed_len = 8

    def __init__(self, pwd: bytes | str | None = None) -> None:
        super().__init__(64, pwd)


class _ARC2_KEY(_SymBase, _ARC2_KEYTYPE):
    _ciphermod = ARC2


class _Salsa20_KEY(_SymBase, _Salsa20_KEYTYPE):
    _ciphermod = Salsa20
    _stream = True


class _RSA_KEYPAIR(_RSA_KEYPAIRTYPE):
    backend = Backend.pycryptodomex
    __concrete__ = True

    def __init__(self, key_size: int, pwd: bytes | str | None = None) -> None:
        self._private_key = RSA.generate(key_size)
        self._public_key = self._private_key.publickey()

    @classmethod
    def decode_private_key(cls, data: bytes, format_: ASymKeyFormat, encoding: ASymKeyEncoding, password: bytes | None = None) -> _ty.Self:
        key = RSA.import_key(data, passphrase=password)
        obj = cls(key.size_in_bits())
        obj._private_key = key
        obj._public_key = key.publickey()
        return obj

    @classmethod
    def decode_public_key(cls, data: bytes, format_: ASymKeyFormat, encoding: ASymKeyEncoding) -> _ty.Self:
        key = RSA.import_key(data)
        obj = cls(2048)
        obj._private_key = None
        obj._public_key = key
        return obj

    def encode_private_key(self, format_: ASymKeyFormat, encoding: ASymKeyEncoding, password: bytes | None = None) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")
        fmt = "PEM" if encoding == ASymKeyEncoding.PEM else "DER"
        if format_ == ASymKeyFormat.PKCS8:
            if password is None:
                return self._private_key.export_key(format=fmt, pkcs=8)
            return self._private_key.export_key(format=fmt, pkcs=8, passphrase=password, protection="PBKDF2WithHMAC-SHA1AndAES256-CBC")
        if format_ == ASymKeyFormat.PKCS1:
            return self._private_key.export_key(format=fmt, pkcs=1, passphrase=password)
        raise _NotSupportedError(f"Unsupported RSA private format: {format_}")

    def encode_public_key(self, format_: ASymKeyFormat, encoding: ASymKeyEncoding) -> bytes:
        if self._public_key is None:
            raise ValueError("No public key present")
        if format_ == ASymKeyFormat.OPENSSH:
            return self._public_key.export_key(format="OpenSSH")
        fmt = "PEM" if encoding == ASymKeyEncoding.PEM else "DER"
        return self._public_key.export_key(format=fmt, pkcs=1 if format_ == ASymKeyFormat.PKCS1 else 8)

    def encrypt(self, plain_bytes: bytes, padding: ASymPadding) -> bytes:
        if padding == ASymPadding.OAEP:
            return PKCS1_OAEP.new(self._public_key).encrypt(plain_bytes)
        if padding == ASymPadding.PKCShash1v15:
            return PKCS1_v1_5.new(self._public_key).encrypt(plain_bytes)
        raise _NotSupportedError(f"Unsupported RSA encryption padding: {padding}")

    def decrypt(self, cipher_bytes: bytes, padding: ASymPadding) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")
        if padding == ASymPadding.OAEP:
            return PKCS1_OAEP.new(self._private_key).decrypt(cipher_bytes)
        if padding == ASymPadding.PKCShash1v15:
            sentinel = os.urandom(32)
            out = PKCS1_v1_5.new(self._private_key).decrypt(cipher_bytes, sentinel)
            if out == sentinel:
                raise ValueError("RSA PKCS1v15 decryption failed")
            return out
        raise _NotSupportedError(f"Unsupported RSA encryption padding: {padding}")

    def sign(self, data: bytes, padding: ASymPadding) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")
        h = SHA256.new(data)
        if padding == ASymPadding.PSS:
            return pss.new(self._private_key).sign(h)
        if padding == ASymPadding.PKCShash1v15:
            return pkcs1_15.new(self._private_key).sign(h)
        raise _NotSupportedError(f"Unsupported RSA signing padding: {padding}")

    def sign_verify(self, data: bytes, signature: bytes, padding: ASymPadding) -> bool:
        h = SHA256.new(data)
        try:
            if padding == ASymPadding.PSS:
                pss.new(self._public_key).verify(h, signature)
            elif padding == ASymPadding.PKCShash1v15:
                pkcs1_15.new(self._public_key).verify(h, signature)
            else:
                raise _NotSupportedError(f"Unsupported RSA signing padding: {padding}")
            return True
        except Exception:
            return False


class _DSA_KEYPAIR(_DSA_KEYPAIRTYPE):
    backend = Backend.pycryptodomex
    __concrete__ = True

    def __init__(self, key_size: int, pwd: bytes | str | None = None) -> None:
        self._private_key = DSA.generate(key_size)
        self._public_key = self._private_key.publickey()

    @classmethod
    def decode_private_key(cls, data: bytes, format_: ASymKeyFormat, encoding: ASymKeyEncoding, password: bytes | None = None) -> _ty.Self:
        key = DSA.import_key(data, passphrase=password)
        obj = cls(key.p.bit_length())
        obj._private_key = key
        obj._public_key = key.publickey()
        return obj

    @classmethod
    def decode_public_key(cls, data: bytes, format_: ASymKeyFormat, encoding: ASymKeyEncoding) -> _ty.Self:
        key = DSA.import_key(data)
        obj = cls(2048)
        obj._private_key = None
        obj._public_key = key
        return obj

    def encode_private_key(self, format_: ASymKeyFormat, encoding: ASymKeyEncoding, password: bytes | None = None) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")
        fmt = "PEM" if encoding == ASymKeyEncoding.PEM else "DER"
        return self._private_key.export_key(format=fmt, passphrase=password, pkcs8=True)

    def encode_public_key(self, format_: ASymKeyFormat, encoding: ASymKeyEncoding) -> bytes:
        fmt = "PEM" if encoding == ASymKeyEncoding.PEM else "DER"
        return self._public_key.export_key(format=fmt)

    def sign(self, data: bytes, padding: ASymPadding) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")
        signer = DSS.new(self._private_key, "fips-186-3")
        return signer.sign(SHA256.new(data))

    def sign_verify(self, data: bytes, signature: bytes, padding: ASymPadding) -> bool:
        verifier = DSS.new(self._public_key, "fips-186-3")
        try:
            verifier.verify(SHA256.new(data), signature)
            return True
        except Exception:
            return False


_CURVE_MAP = {
    ECCCurve.SECP192R1: "P-192",
    ECCCurve.SECP224R1: "P-224",
    ECCCurve.SECP256R1: "P-256",
    ECCCurve.SECP384R1: "P-384",
    ECCCurve.SECP521R1: "P-521",
    ECCCurve.SECP256K1: "secp256k1",
}


class _ECC_KEYPAIR(_ECC_KEYPAIRTYPE):
    backend = Backend.pycryptodomex
    __concrete__ = True

    def __init__(self, ecc_type: ECCType = ECCType.ECDSA, ecc_curve: ECCCurve | None = ECCCurve.SECP256R1, pwd: bytes | str | None = None) -> None:
        self._ecc_type = ecc_type
        self._ecc_curve = ecc_curve

        if ecc_type == ECCType.ECDSA:
            curve = _CURVE_MAP.get(ecc_curve or ECCCurve.SECP256R1)
            if curve is None:
                raise _NotSupportedError(f"Unsupported ECC curve for pycryptodomex: {ecc_curve}")
            self._private_key = ECC.generate(curve=curve)
        elif ecc_type == ECCType.Ed25519:
            self._private_key = ECC.generate(curve="Ed25519")
        elif ecc_type == ECCType.Ed448:
            self._private_key = ECC.generate(curve="Ed448")
        elif ecc_type == ECCType.X25519:
            self._private_key = ECC.generate(curve="Curve25519")
        elif ecc_type == ECCType.X448:
            self._private_key = ECC.generate(curve="Curve448")
        else:
            raise _NotSupportedError(f"Unsupported ECC type {ecc_type}")
        self._public_key = self._private_key.public_key()

    @classmethod
    def decode_private_key(cls, data: bytes, format_: ASymKeyFormat, encoding: ASymKeyEncoding, password: bytes | None = None) -> _ty.Self:
        key = ECC.import_key(data, passphrase=password)
        curve_name = key.curve
        obj = cls(ECCType.ECDSA, ECCCurve.SECP256R1)
        obj._private_key = key
        obj._public_key = key.public_key()
        if curve_name in ("Ed25519",):
            obj._ecc_type = ECCType.Ed25519
            obj._ecc_curve = None
        elif curve_name in ("Ed448",):
            obj._ecc_type = ECCType.Ed448
            obj._ecc_curve = None
        elif curve_name in ("Curve25519",):
            obj._ecc_type = ECCType.X25519
            obj._ecc_curve = None
        elif curve_name in ("Curve448",):
            obj._ecc_type = ECCType.X448
            obj._ecc_curve = None
        else:
            obj._ecc_type = ECCType.ECDSA
            obj._ecc_curve = next((k for k, v in _CURVE_MAP.items() if v == curve_name), ECCCurve.SECP256R1)
        return obj

    @classmethod
    def decode_public_key(cls, data: bytes, format_: ASymKeyFormat, encoding: ASymKeyEncoding) -> _ty.Self:
        key = ECC.import_key(data)
        obj = cls(ECCType.ECDSA, ECCCurve.SECP256R1)
        obj._private_key = None
        obj._public_key = key
        curve_name = key.curve
        if curve_name == "Ed25519":
            obj._ecc_type = ECCType.Ed25519
            obj._ecc_curve = None
        elif curve_name == "Ed448":
            obj._ecc_type = ECCType.Ed448
            obj._ecc_curve = None
        elif curve_name == "Curve25519":
            obj._ecc_type = ECCType.X25519
            obj._ecc_curve = None
        elif curve_name == "Curve448":
            obj._ecc_type = ECCType.X448
            obj._ecc_curve = None
        else:
            obj._ecc_type = ECCType.ECDSA
            obj._ecc_curve = next((k for k, v in _CURVE_MAP.items() if v == curve_name), ECCCurve.SECP256R1)
        return obj

    def encode_private_key(self, format_: ASymKeyFormat, encoding: ASymKeyEncoding, password: bytes | None = None) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")
        fmt = "PEM" if encoding == ASymKeyEncoding.PEM else "DER"
        if encoding == ASymKeyEncoding.RAW:
            raise _NotSupportedError("RAW private ECC export is unsupported")
        return self._private_key.export_key(format=fmt, passphrase=password)

    def encode_public_key(self, format_: ASymKeyFormat, encoding: ASymKeyEncoding) -> bytes:
        if encoding == ASymKeyEncoding.RAW:
            return self._public_key.export_key(format="raw")
        if format_ == ASymKeyFormat.OPENSSH:
            return self._public_key.export_key(format="OpenSSH")
        fmt = "PEM" if encoding == ASymKeyEncoding.PEM else "DER"
        return self._public_key.export_key(format=fmt)

    def sign(self, data: bytes, padding: ASymPadding) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")
        if self._ecc_type == ECCType.ECDSA:
            signer = DSS.new(self._private_key, "fips-186-3")
            return signer.sign(SHA256.new(data))
        if self._ecc_type in (ECCType.Ed25519, ECCType.Ed448):
            signer = eddsa.new(self._private_key, mode="rfc8032")
            return signer.sign(data)
        raise _NotSupportedError(f"{self._ecc_type} does not support signing")

    def sign_verify(self, data: bytes, signature: bytes, padding: ASymPadding) -> bool:
        try:
            if self._ecc_type == ECCType.ECDSA:
                verifier = DSS.new(self._public_key, "fips-186-3")
                verifier.verify(SHA256.new(data), signature)
                return True
            if self._ecc_type in (ECCType.Ed25519, ECCType.Ed448):
                verifier = eddsa.new(self._public_key, mode="rfc8032")
                verifier.verify(data, signature)
                return True
            raise _NotSupportedError(f"{self._ecc_type} does not support signature verification")
        except Exception:
            return False

    def key_exchange(self, peer_public_key: _ty.Self) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")
        if self._ecc_type not in (ECCType.ECDSA, ECCType.X25519, ECCType.X448):
            raise _NotSupportedError(f"{self._ecc_type} does not support key exchange")
        try:
            return DH.key_agreement(static_priv=self._private_key, static_pub=peer_public_key._public_key, kdf=lambda x: x)
        except Exception as exc:
            raise _NotSupportedError("ECC key exchange not available for this key type in pycryptodomex") from exc
