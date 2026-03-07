from __future__ import annotations

import base64
import os
import typing as _ty

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import cmac, hashes, hmac, padding as sym_padding, serialization
from cryptography.hazmat.primitives.asymmetric import (
    dsa,
    ec,
    ed25519,
    ed448,
    padding as asym_padding,
    rsa,
    x25519,
    x448,
)
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.constant_time import bytes_eq
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

try:
    from cryptography.hazmat.decrepit.ciphers import algorithms as decrepit_algorithms
except Exception:  # pragma: no cover - version dependent import
    decrepit_algorithms = None

from .._definitions import (
    Backend,
    ECCCurve,
    ECCType,
    _AES_KEYLITERAL,
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
    _SEED_KEYTYPE,
    _SM4_KEYTYPE,
    _TripleDES_KEYTYPE,
)
from ..algos._asym import (
    KeyEncoding as ASymKeyEncoding,
    KeyFormat as ASymKeyFormat,
    Padding as ASymPadding,
)
from ..algos._sym import (
    KeyEncoding as SymKeyEncoding,
    MessageAuthenticationCode as MAC,
    Operation as SymOperation,
    Padding as SymPadding,
)
from ..exceptions import NotSupportedError as _NotSupportedError

_OPERATION_MAP = {
    SymOperation.ECB: lambda iv=None: modes.ECB(),
    SymOperation.CBC: lambda iv: modes.CBC(iv),
    SymOperation.CFB: lambda iv: modes.CFB(iv),
    SymOperation.OFB: lambda iv: modes.OFB(iv),
    SymOperation.CTR: lambda iv: modes.CTR(iv),
    SymOperation.GCM: lambda iv, tag=None: modes.GCM(iv, tag) if tag else modes.GCM(iv),
}

_ARC4_TYPES: tuple[type, ...] = tuple(
    t
    for t in (
        getattr(algorithms, "ARC4", None),
        getattr(decrepit_algorithms, "ARC4", None) if decrepit_algorithms is not None else None,
    )
    if isinstance(t, type)
)


def _derive_key_from_password(password: str | bytes, length: int, salt: bytes) -> bytes:
    if isinstance(password, str):
        password = password.encode("utf-8")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=100_000,
        backend=default_backend(),
    )
    return kdf.derive(password)


def _decode_bytes(data: bytes, encoding: SymKeyEncoding | ASymKeyEncoding) -> bytes:
    if encoding in (SymKeyEncoding.RAW, ASymKeyEncoding.RAW):
        return data
    if encoding in (SymKeyEncoding.BASE64,):
        return base64.b64decode(data)
    if encoding in (SymKeyEncoding.HEX,):
        return bytes.fromhex(data.decode("ascii"))
    if encoding in (SymKeyEncoding.BASE32,):
        return base64.b32decode(data)
    return data


def _encode_bytes(data: bytes, encoding: SymKeyEncoding | ASymKeyEncoding) -> bytes:
    if encoding in (SymKeyEncoding.RAW, ASymKeyEncoding.RAW):
        return data
    if encoding in (SymKeyEncoding.BASE64,):
        return base64.b64encode(data)
    if encoding in (SymKeyEncoding.HEX,):
        return data.hex().encode("ascii")
    if encoding in (SymKeyEncoding.BASE32,):
        return base64.b32encode(data)
    return data


class _SymmetricKeyBase:
    backend = Backend.cryptography_alpha
    __concrete__ = True
    _ALG_CLS: _ty.Any = None
    _SUPPORTS_GCM: bool = False
    _FIXED_KEY_BYTES: int | None = None
    _PWD_ONLY_INIT: bool = False

    def __init__(self, key_size: int, pwd: _ty.Optional[bytes | str]) -> None:
        self.key_size = key_size
        key_len = key_size // 8
        if self._FIXED_KEY_BYTES is not None:
            key_len = self._FIXED_KEY_BYTES
            self.key_size = key_len * 8

        if pwd is None:
            self._key = os.urandom(key_len)
            return

        pwd_bytes = pwd.encode("utf-8") if isinstance(pwd, str) else pwd
        if len(pwd_bytes) == key_len:
            self._key = pwd_bytes
            return

        salt_hash = hashes.Hash(hashes.SHA256(), backend=default_backend())
        salt_hash.update(pwd_bytes)
        self._key = _derive_key_from_password(pwd_bytes, key_len, salt_hash.finalize()[:16])

    @classmethod
    def decode(cls, key: bytes, encoding: SymKeyEncoding) -> _ty.Self:
        if not isinstance(key, bytes):
            raise ValueError("key needs to be of type 'bytes'")
        if not isinstance(encoding, SymKeyEncoding):
            raise ValueError("encoding needs to be of type Sym.KeyEncoding")

        raw = _decode_bytes(key, encoding)
        if cls._FIXED_KEY_BYTES is not None and len(raw) != cls._FIXED_KEY_BYTES:
            raise ValueError(f"Expected {cls._FIXED_KEY_BYTES} bytes for this key type")

        key_bits = len(raw) * 8
        obj = cls(raw) if cls._PWD_ONLY_INIT else cls(key_bits, raw)
        obj._algorithm()  # Validate key length via backend algorithm constructor.
        return obj

    def encode(self, encoding: SymKeyEncoding) -> bytes:
        if not isinstance(encoding, SymKeyEncoding):
            raise ValueError("encoding needs to be of type Sym.KeyEncoding")
        return _encode_bytes(self._key, encoding)

    def _algorithm(self):
        if self._ALG_CLS is None:
            raise _NotSupportedError(f"{type(self).__name__} has no backend algorithm")
        return self._ALG_CLS(self._key)

    def _is_stream_cipher(self) -> bool:
        stream_types: tuple[type, ...] = _ARC4_TYPES + (algorithms.ChaCha20,)
        return isinstance(self._algorithm(), stream_types)

    @staticmethod
    def _needs_padding(mode: SymOperation) -> bool:
        return mode in (SymOperation.ECB, SymOperation.CBC)

    @staticmethod
    def _pack_cipher_parts(parts: dict[str, bytes]) -> bytes:
        iv = parts.get("iv", b"")
        tag = parts.get("tag", b"")
        ct = parts["ciphertext"]
        return len(iv).to_bytes(2, "big") + len(tag).to_bytes(2, "big") + iv + tag + ct

    @staticmethod
    def _unpack_cipher_parts(payload: bytes) -> dict[str, bytes]:
        if len(payload) < 4:
            raise ValueError("cipher payload too short")
        iv_len = int.from_bytes(payload[:2], "big")
        tag_len = int.from_bytes(payload[2:4], "big")
        if len(payload) < 4 + iv_len + tag_len:
            raise ValueError("cipher payload malformed")

        iv_start = 4
        tag_start = iv_start + iv_len
        ct_start = tag_start + tag_len

        out = {"ciphertext": payload[ct_start:]}
        if iv_len:
            out["iv"] = payload[iv_start:tag_start]
        if tag_len:
            out["tag"] = payload[tag_start:ct_start]
        return out

    def encrypt(
        self,
        plain_bytes: bytes,
        padding: SymPadding,
        mode: SymOperation,
        /,
        *,
        auto_pack: bool = True,
    ) -> bytes | dict[str, bytes]:
        if not isinstance(plain_bytes, bytes):
            raise ValueError("plain_bytes needs to be of type 'bytes'")
        if not isinstance(padding, SymPadding):
            raise ValueError("padding needs to be of type Sym.Padding")
        if not isinstance(mode, SymOperation):
            raise ValueError("mode needs to be of type Sym.Operation")

        algorithm = self._algorithm()
        payload = plain_bytes
        iv: bytes | None = None
        tag: bytes | None = None

        if isinstance(algorithm, algorithms.ChaCha20):
            if mode is not SymOperation.CTR:
                raise _NotSupportedError("ChaCha20 supports only CTR-style operation in this API")
            if padding is not SymPadding.PKCS7:
                raise _NotSupportedError("ChaCha20 does not support block padding")
            iv = os.urandom(16)
            cipher = Cipher(algorithms.ChaCha20(self._key, iv), mode=None, backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(payload) + encryptor.finalize()
            out = {"ciphertext": ciphertext, "iv": iv}
            return self._pack_cipher_parts(out) if auto_pack else out

        if _ARC4_TYPES and isinstance(algorithm, _ARC4_TYPES):
            if mode is not SymOperation.CTR:
                raise _NotSupportedError("ARC4 supports only CTR-style operation in this API")
            if padding is not SymPadding.PKCS7:
                raise _NotSupportedError("ARC4 does not support block padding")
            cipher = Cipher(algorithm, mode=None, backend=default_backend())
            encryptor = cipher.encryptor()
            ciphertext = encryptor.update(payload) + encryptor.finalize()
            out = {"ciphertext": ciphertext}
            return self._pack_cipher_parts(out) if auto_pack else out

        if self._needs_padding(mode):
            if padding == SymPadding.PKCS7:
                padder = sym_padding.PKCS7(algorithm.block_size).padder()
                payload = padder.update(payload) + padder.finalize()
            elif padding == SymPadding.ANSIX923:
                padder = sym_padding.ANSIX923(algorithm.block_size).padder()
                payload = padder.update(payload) + padder.finalize()
            else:
                raise _NotSupportedError(f"Unsupported padding: {padding}")

        if mode == SymOperation.ECB:
            crypt_mode = _OPERATION_MAP[mode]()
        elif mode == SymOperation.GCM:
            if not self._SUPPORTS_GCM:
                raise _NotSupportedError("GCM is supported only for AES in this backend")
            iv = os.urandom(12)
            crypt_mode = _OPERATION_MAP[mode](iv)
        else:
            iv = os.urandom(algorithm.block_size // 8)
            crypt_mode = _OPERATION_MAP[mode](iv)

        encryptor = Cipher(algorithm, crypt_mode, backend=default_backend()).encryptor()
        ciphertext = encryptor.update(payload) + encryptor.finalize()
        if mode == SymOperation.GCM:
            tag = encryptor.tag

        out = {"ciphertext": ciphertext}
        if iv is not None:
            out["iv"] = iv
        if tag is not None:
            out["tag"] = tag
        return self._pack_cipher_parts(out) if auto_pack else out

    def decrypt(
        self,
        cipher_bytes_or_dict: bytes | dict[str, bytes],
        padding: SymPadding,
        mode: SymOperation,
        /,
        *,
        auto_pack: bool = True,
    ) -> bytes:
        if not isinstance(padding, SymPadding):
            raise ValueError("padding needs to be of type Sym.Padding")
        if not isinstance(mode, SymOperation):
            raise ValueError("mode needs to be of type Sym.Operation")

        parts = (
            self._unpack_cipher_parts(cipher_bytes_or_dict)
            if auto_pack
            else _ty.cast(dict[str, bytes], cipher_bytes_or_dict)
        )
        if auto_pack and not isinstance(cipher_bytes_or_dict, bytes):
            raise ValueError("Expected packed ciphertext as bytes")
        if not auto_pack and not isinstance(cipher_bytes_or_dict, dict):
            raise ValueError("Expected unpacked ciphertext as dict[str, bytes]")

        algorithm = self._algorithm()
        ciphertext = parts.get("ciphertext", b"")
        iv = parts.get("iv")
        tag = parts.get("tag")

        if isinstance(algorithm, algorithms.ChaCha20):
            if mode is not SymOperation.CTR:
                raise _NotSupportedError("ChaCha20 supports only CTR-style operation in this API")
            if iv is None:
                raise ValueError("ChaCha20 decryption requires iv")
            cipher = Cipher(algorithms.ChaCha20(self._key, iv), mode=None, backend=default_backend())
            decryptor = cipher.decryptor()
            return decryptor.update(ciphertext) + decryptor.finalize()

        if _ARC4_TYPES and isinstance(algorithm, _ARC4_TYPES):
            if mode is not SymOperation.CTR:
                raise _NotSupportedError("ARC4 supports only CTR-style operation in this API")
            cipher = Cipher(algorithm, mode=None, backend=default_backend())
            decryptor = cipher.decryptor()
            return decryptor.update(ciphertext) + decryptor.finalize()

        if mode == SymOperation.ECB:
            crypt_mode = _OPERATION_MAP[mode]()
        elif mode == SymOperation.GCM:
            if not self._SUPPORTS_GCM:
                raise _NotSupportedError("GCM is supported only for AES in this backend")
            if iv is None or tag is None:
                raise ValueError("GCM decryption requires iv and tag")
            crypt_mode = _OPERATION_MAP[mode](iv, tag)
        else:
            if iv is None:
                raise ValueError(f"{mode} decryption requires iv")
            crypt_mode = _OPERATION_MAP[mode](iv)

        decryptor = Cipher(algorithm, crypt_mode, backend=default_backend()).decryptor()
        plain = decryptor.update(ciphertext) + decryptor.finalize()

        if self._needs_padding(mode):
            if padding == SymPadding.PKCS7:
                unpadder = sym_padding.PKCS7(algorithm.block_size).unpadder()
                plain = unpadder.update(plain) + unpadder.finalize()
            elif padding == SymPadding.ANSIX923:
                unpadder = sym_padding.ANSIX923(algorithm.block_size).unpadder()
                plain = unpadder.update(plain) + unpadder.finalize()
            else:
                raise _NotSupportedError(f"Unsupported padding: {padding}")

        return plain

    def generate_mac(self, data: bytes, auth_type: MAC) -> bytes:
        if not isinstance(data, bytes):
            raise ValueError("data needs to be bytes")
        if not isinstance(auth_type, MAC):
            raise ValueError("auth_type needs to be Sym.MessageAuthenticationCode")

        if auth_type == MAC.HMAC:
            h = hmac.HMAC(self._key, hashes.SHA256(), backend=default_backend())
            h.update(data)
            return h.finalize()

        if auth_type == MAC.CMAC:
            algorithm = self._algorithm()
            if (_ARC4_TYPES and isinstance(algorithm, _ARC4_TYPES)) or isinstance(
                algorithm, algorithms.ChaCha20
            ):
                raise _NotSupportedError("CMAC is only supported for block ciphers")
            c = cmac.CMAC(type(algorithm)(self._key), backend=default_backend())
            c.update(data)
            return c.finalize()

        raise _NotSupportedError(f"Unsupported MAC type: {auth_type}")

    def verify_mac(self, data: bytes, mac: bytes, auth_type: MAC) -> bool:
        expected = self.generate_mac(data, auth_type)
        return bool(bytes_eq(expected, mac))

    def __repr__(self) -> str:
        return f"<{type(self).__name__} key_size={self.key_size}>"


class _AES_KEY(_SymmetricKeyBase, _AES_KEYTYPE):
    _ALG_CLS = algorithms.AES
    _SUPPORTS_GCM = True

    def __init__(self, key_size: _AES_KEYLITERAL, pwd: _ty.Optional[bytes | str]) -> None:
        super().__init__(key_size, pwd)


class _ChaCha20_KEY(_SymmetricKeyBase, _ChaCha20_KEYTYPE):
    _ALG_CLS = algorithms.ChaCha20
    _FIXED_KEY_BYTES = 32
    _PWD_ONLY_INIT = True

    def __init__(self, pwd: _ty.Optional[bytes | str] = None) -> None:
        super().__init__(256, pwd)

    def _algorithm(self):
        # Only used for type/capability checks; the real nonce is generated per encryption call.
        return algorithms.ChaCha20(self._key, b"\x00" * 16)


if getattr(algorithms, "Camellia", None) is not None:
    class _Camellia_KEY(_SymmetricKeyBase, _Camellia_KEYTYPE):
        _ALG_CLS = algorithms.Camellia

        def __init__(self, key_size: int, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(key_size, pwd)
else:
    class _Camellia_KEY(_Camellia_KEYTYPE): __concrete__ = False


if getattr(algorithms, "SM4", None) is not None:
    class _SM4_KEY(_SymmetricKeyBase, _SM4_KEYTYPE):
        _ALG_CLS = algorithms.SM4
        _FIXED_KEY_BYTES = 16
        _PWD_ONLY_INIT = True

        def __init__(self, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(128, pwd)
else:
    class _SM4_KEY(_SM4_KEYTYPE): __concrete__ = False


if getattr(algorithms, "TripleDES", None) is not None:
    class _TripleDES_KEY(_SymmetricKeyBase, _TripleDES_KEYTYPE):
        _ALG_CLS = algorithms.TripleDES
        _FIXED_KEY_BYTES = 24

        def __init__(self, key_size: _ty.Literal[192], pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(key_size, pwd)
elif decrepit_algorithms is not None and getattr(decrepit_algorithms, "TripleDES", None) is not None:
    class _TripleDES_KEY(_SymmetricKeyBase, _TripleDES_KEYTYPE):
        _ALG_CLS = decrepit_algorithms.TripleDES
        _FIXED_KEY_BYTES = 24

        def __init__(self, key_size: _ty.Literal[192], pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(key_size, pwd)
else:
    class _TripleDES_KEY(_TripleDES_KEYTYPE): __concrete__ = False


if getattr(algorithms, "Blowfish", None) is not None:
    class _Blowfish_KEY(_SymmetricKeyBase, _Blowfish_KEYTYPE):
        _ALG_CLS = algorithms.Blowfish

        def __init__(self, key_size: int, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(key_size, pwd)
elif decrepit_algorithms is not None and getattr(decrepit_algorithms, "Blowfish", None) is not None:
    class _Blowfish_KEY(_SymmetricKeyBase, _Blowfish_KEYTYPE):
        _ALG_CLS = decrepit_algorithms.Blowfish

        def __init__(self, key_size: int, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(key_size, pwd)
else:
    class _Blowfish_KEY(_Blowfish_KEYTYPE): __concrete__ = False


if getattr(algorithms, "CAST5", None) is not None:
    class _CAST5_KEY(_SymmetricKeyBase, _CAST5_KEYTYPE):
        _ALG_CLS = algorithms.CAST5

        def __init__(self, key_size: int, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(key_size, pwd)
elif decrepit_algorithms is not None and getattr(decrepit_algorithms, "CAST5", None) is not None:
    class _CAST5_KEY(_SymmetricKeyBase, _CAST5_KEYTYPE):
        _ALG_CLS = decrepit_algorithms.CAST5

        def __init__(self, key_size: int, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(key_size, pwd)
else:
    class _CAST5_KEY(_CAST5_KEYTYPE): __concrete__ = False


if getattr(algorithms, "ARC4", None) is not None:
    class _ARC4_KEY(_SymmetricKeyBase, _ARC4_KEYTYPE):
        _ALG_CLS = algorithms.ARC4
        _PWD_ONLY_INIT = True

        def __init__(self, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(128, pwd)
elif decrepit_algorithms is not None and getattr(decrepit_algorithms, "ARC4", None) is not None:
    class _ARC4_KEY(_SymmetricKeyBase, _ARC4_KEYTYPE):
        _ALG_CLS = decrepit_algorithms.ARC4
        _PWD_ONLY_INIT = True

        def __init__(self, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(128, pwd)
else:
    class _ARC4_KEY(_ARC4_KEYTYPE): __concrete__ = False


if getattr(algorithms, "IDEA", None) is not None:
    class _IDEA_KEY(_SymmetricKeyBase, _IDEA_KEYTYPE):
        _ALG_CLS = algorithms.IDEA
        _FIXED_KEY_BYTES = 16
        _PWD_ONLY_INIT = True

        def __init__(self, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(128, pwd)
elif decrepit_algorithms is not None and getattr(decrepit_algorithms, "IDEA", None) is not None:
    class _IDEA_KEY(_SymmetricKeyBase, _IDEA_KEYTYPE):
        _ALG_CLS = decrepit_algorithms.IDEA
        _FIXED_KEY_BYTES = 16
        _PWD_ONLY_INIT = True

        def __init__(self, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(128, pwd)
else:
    class _IDEA_KEY(_IDEA_KEYTYPE): __concrete__ = False


if getattr(algorithms, "SEED", None) is not None:
    class _SEED_KEY(_SymmetricKeyBase, _SEED_KEYTYPE):
        _ALG_CLS = algorithms.SEED
        _FIXED_KEY_BYTES = 16
        _PWD_ONLY_INIT = True

        def __init__(self, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(128, pwd)
elif decrepit_algorithms is not None and getattr(decrepit_algorithms, "SEED", None) is not None:
    class _SEED_KEY(_SymmetricKeyBase, _SEED_KEYTYPE):
        _ALG_CLS = decrepit_algorithms.SEED
        _FIXED_KEY_BYTES = 16
        _PWD_ONLY_INIT = True

        def __init__(self, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(128, pwd)
else:
    class _SEED_KEY(_SEED_KEYTYPE): __concrete__ = False


if getattr(algorithms, "DES", None) is not None:
    class _DES_KEY(_SymmetricKeyBase, _DES_KEYTYPE):
        _ALG_CLS = algorithms.DES
        _FIXED_KEY_BYTES = 8
        _PWD_ONLY_INIT = True

        def __init__(self, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(64, pwd)
elif decrepit_algorithms is not None and getattr(decrepit_algorithms, "DES", None) is not None:
    class _DES_KEY(_SymmetricKeyBase, _DES_KEYTYPE):
        _ALG_CLS = decrepit_algorithms.DES
        _FIXED_KEY_BYTES = 8
        _PWD_ONLY_INIT = True

        def __init__(self, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(64, pwd)
else:
    class _DES_KEY(_DES_KEYTYPE): __concrete__ = False


if getattr(algorithms, "ARC2", None) is not None:
    class _ARC2_KEY(_SymmetricKeyBase, _ARC2_KEYTYPE):
        _ALG_CLS = algorithms.ARC2

        def __init__(self, key_size: int = 128, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(key_size, pwd)
elif decrepit_algorithms is not None and getattr(decrepit_algorithms, "ARC2", None) is not None:
    class _ARC2_KEY(_SymmetricKeyBase, _ARC2_KEYTYPE):
        _ALG_CLS = decrepit_algorithms.ARC2

        def __init__(self, key_size: int = 128, pwd: _ty.Optional[bytes | str] = None) -> None:
            super().__init__(key_size, pwd)
else:
    class _ARC2_KEY(_ARC2_KEYTYPE): __concrete__ = False


def _asym_encoding(encoding: ASymKeyEncoding) -> serialization.Encoding:
    if encoding == ASymKeyEncoding.PEM:
        return serialization.Encoding.PEM
    if encoding == ASymKeyEncoding.DER:
        return serialization.Encoding.DER
    if encoding == ASymKeyEncoding.RAW:
        return serialization.Encoding.Raw
    raise _NotSupportedError(f"Unsupported encoding: {encoding}")


def _private_format(format_: ASymKeyFormat, is_rsa: bool, is_ec: bool) -> serialization.PrivateFormat:
    if format_ == ASymKeyFormat.PKCS8:
        return serialization.PrivateFormat.PKCS8
    if format_ == ASymKeyFormat.PKCS1:
        if not is_rsa:
            raise _NotSupportedError("PKCS1 private encoding is RSA-only")
        return serialization.PrivateFormat.TraditionalOpenSSL
    if format_ == ASymKeyFormat.SEC1:
        if not is_ec:
            raise _NotSupportedError("SEC1 private encoding is EC-only")
        return serialization.PrivateFormat.TraditionalOpenSSL
    raise _NotSupportedError(f"Unsupported private key format: {format_}")


def _public_format(format_: ASymKeyFormat, encoding: ASymKeyEncoding, is_rsa: bool, is_ec: bool) -> serialization.PublicFormat:
    if format_ == ASymKeyFormat.OPENSSH:
        return serialization.PublicFormat.OpenSSH
    if format_ == ASymKeyFormat.PKCS1:
        if not is_rsa:
            raise _NotSupportedError("PKCS1 public encoding is RSA-only")
        return serialization.PublicFormat.PKCS1
    if format_ == ASymKeyFormat.SEC1:
        if not is_ec:
            raise _NotSupportedError("SEC1 public encoding is EC-only")
        if encoding == ASymKeyEncoding.RAW:
            return serialization.PublicFormat.CompressedPoint
        return serialization.PublicFormat.SubjectPublicKeyInfo
    return serialization.PublicFormat.SubjectPublicKeyInfo


def _load_private_key(data: bytes, encoding: ASymKeyEncoding, password: bytes | None):
    if encoding == ASymKeyEncoding.PEM:
        return serialization.load_pem_private_key(data, password=password, backend=default_backend())
    if encoding == ASymKeyEncoding.DER:
        return serialization.load_der_private_key(data, password=password, backend=default_backend())
    raise _NotSupportedError("RAW private key decode is unsupported for this key type")


def _load_public_key(data: bytes, encoding: ASymKeyEncoding):
    if encoding == ASymKeyEncoding.PEM:
        return serialization.load_pem_public_key(data, backend=default_backend())
    if encoding == ASymKeyEncoding.DER:
        return serialization.load_der_public_key(data, backend=default_backend())
    raise _NotSupportedError("RAW public key decode is unsupported for this key type")


def _rsa_padding(padding: ASymPadding):
    if padding == ASymPadding.OAEP:
        return asym_padding.OAEP(
            mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        )
    if padding == ASymPadding.PKCShash1v15:
        return asym_padding.PKCS1v15()
    raise _NotSupportedError(f"Unsupported RSA padding: {padding}")


def _rsa_sign_padding(padding: ASymPadding):
    if padding == ASymPadding.PSS:
        return asym_padding.PSS(
            mgf=asym_padding.MGF1(hashes.SHA256()),
            salt_length=asym_padding.PSS.MAX_LENGTH,
        )
    if padding == ASymPadding.PKCShash1v15:
        return asym_padding.PKCS1v15()
    raise _NotSupportedError(f"Unsupported RSA signing padding: {padding}")


_CURVE_MAP: dict[ECCCurve, _ty.Type[ec.EllipticCurve]] = {
    ECCCurve.SECP192R1: ec.SECP192R1,
    ECCCurve.SECP224R1: ec.SECP224R1,
    ECCCurve.SECP256K1: ec.SECP256K1,
    ECCCurve.SECP256R1: ec.SECP256R1,
    ECCCurve.SECP384R1: ec.SECP384R1,
    ECCCurve.SECP521R1: ec.SECP521R1,
    ECCCurve.SECT163K1: ec.SECT163K1,
    ECCCurve.SECT163R2: ec.SECT163R2,
    ECCCurve.SECT233K1: ec.SECT233K1,
    ECCCurve.SECT233R1: ec.SECT233R1,
    ECCCurve.SECT283K1: ec.SECT283K1,
    ECCCurve.SECT283R1: ec.SECT283R1,
    ECCCurve.SECT409K1: ec.SECT409K1,
    ECCCurve.SECT409R1: ec.SECT409R1,
    ECCCurve.SECT571K1: ec.SECT571K1,
    ECCCurve.SECT571R1: ec.SECT571R1,
}


class _RSA_KEYPAIR(_RSA_KEYPAIRTYPE):
    backend = Backend.cryptography_alpha
    __concrete__ = True

    def __init__(self, key_size: int, pwd: _ty.Optional[bytes | str] = None) -> None:
        if isinstance(pwd, str):
            pwd = pwd.encode("utf-8")
        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size, backend=default_backend())
        self._public_key = self._private_key.public_key()

    @classmethod
    def decode_private_key(
        cls,
        data: bytes,
        format_: ASymKeyFormat,
        encoding: ASymKeyEncoding,
        password: bytes | None = None,
    ) -> _ty.Self:
        key = _load_private_key(data, encoding, password)
        if not isinstance(key, rsa.RSAPrivateKey):
            raise ValueError("Provided key is not an RSA private key")
        obj = cls(key.key_size)
        obj._private_key = key
        obj._public_key = key.public_key()
        return obj

    @classmethod
    def decode_public_key(
        cls, data: bytes, format_: ASymKeyFormat, encoding: ASymKeyEncoding
    ) -> _ty.Self:
        key = _load_public_key(data, encoding)
        if not isinstance(key, rsa.RSAPublicKey):
            raise ValueError("Provided key is not an RSA public key")
        obj = cls(2048)
        obj._private_key = None
        obj._public_key = key
        return obj

    def encode_private_key(
        self,
        format_: ASymKeyFormat,
        encoding: ASymKeyEncoding,
        password: bytes | None = None,
    ) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")
        enc = _asym_encoding(encoding)
        private_format = _private_format(format_, is_rsa=True, is_ec=False)
        protection = (
            serialization.NoEncryption()
            if password is None
            else serialization.BestAvailableEncryption(password)
        )
        return self._private_key.private_bytes(enc, private_format, protection)

    def encode_public_key(self, format_: ASymKeyFormat, encoding: ASymKeyEncoding) -> bytes:
        if self._public_key is None:
            raise ValueError("No public key present")
        enc = _asym_encoding(encoding)
        public_format = _public_format(format_, encoding, is_rsa=True, is_ec=False)
        return self._public_key.public_bytes(enc, public_format)

    def encrypt(self, plain_bytes: bytes, padding: ASymPadding) -> bytes:
        if self._public_key is None:
            raise ValueError("No public key present")
        return self._public_key.encrypt(plain_bytes, _rsa_padding(padding))

    def decrypt(self, cipher_bytes: bytes, padding: ASymPadding) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")
        return self._private_key.decrypt(cipher_bytes, _rsa_padding(padding))

    def sign(self, data: bytes, padding: ASymPadding) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")
        return self._private_key.sign(data, _rsa_sign_padding(padding), hashes.SHA256())

    def sign_verify(self, data: bytes, signature: bytes, padding: ASymPadding) -> bool:
        if self._public_key is None:
            raise ValueError("No public key present")
        try:
            self._public_key.verify(signature, data, _rsa_sign_padding(padding), hashes.SHA256())
            return True
        except InvalidSignature:
            return False

    def __repr__(self) -> str:
        return f"<RSA private={self._private_key is not None}>"


class _DSA_KEYPAIR(_DSA_KEYPAIRTYPE):
    backend = Backend.cryptography_alpha
    __concrete__ = True

    def __init__(self, key_size: int, pwd: _ty.Optional[bytes | str] = None) -> None:
        self._private_key = dsa.generate_private_key(key_size=key_size, backend=default_backend())
        self._public_key = self._private_key.public_key()

    @classmethod
    def decode_private_key(
        cls,
        data: bytes,
        format_: ASymKeyFormat,
        encoding: ASymKeyEncoding,
        password: bytes | None = None,
    ) -> _ty.Self:
        key = _load_private_key(data, encoding, password)
        if not isinstance(key, dsa.DSAPrivateKey):
            raise ValueError("Provided key is not a DSA private key")
        obj = cls(key.key_size)
        obj._private_key = key
        obj._public_key = key.public_key()
        return obj

    @classmethod
    def decode_public_key(
        cls, data: bytes, format_: ASymKeyFormat, encoding: ASymKeyEncoding
    ) -> _ty.Self:
        key = _load_public_key(data, encoding)
        if not isinstance(key, dsa.DSAPublicKey):
            raise ValueError("Provided key is not a DSA public key")
        obj = cls(2048)
        obj._private_key = None
        obj._public_key = key
        return obj

    def encode_private_key(
        self,
        format_: ASymKeyFormat,
        encoding: ASymKeyEncoding,
        password: bytes | None = None,
    ) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")
        if format_ == ASymKeyFormat.PKCS1:
            raise _NotSupportedError("PKCS1 is RSA-only")
        enc = _asym_encoding(encoding)
        protection = (
            serialization.NoEncryption()
            if password is None
            else serialization.BestAvailableEncryption(password)
        )
        return self._private_key.private_bytes(
            enc,
            serialization.PrivateFormat.PKCS8,
            protection,
        )

    def encode_public_key(self, format_: ASymKeyFormat, encoding: ASymKeyEncoding) -> bytes:
        if self._public_key is None:
            raise ValueError("No public key present")
        if format_ == ASymKeyFormat.PKCS1:
            raise _NotSupportedError("PKCS1 is RSA-only")
        enc = _asym_encoding(encoding)
        public_format = _public_format(format_, encoding, is_rsa=False, is_ec=False)
        return self._public_key.public_bytes(enc, public_format)

    def sign(self, data: bytes, padding: ASymPadding) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")
        return self._private_key.sign(data, hashes.SHA256())

    def sign_verify(self, data: bytes, signature: bytes, padding: ASymPadding) -> bool:
        if self._public_key is None:
            raise ValueError("No public key present")
        try:
            self._public_key.verify(signature, data, hashes.SHA256())
            return True
        except InvalidSignature:
            return False

    def __repr__(self) -> str:
        return f"<DSA private={self._private_key is not None}>"


class _ECC_KEYPAIR(_ECC_KEYPAIRTYPE):
    backend = Backend.cryptography_alpha
    __concrete__ = True

    def __init__(
        self,
        ecc_type: ECCType = ECCType.ECDSA,
        ecc_curve: ECCCurve | None = ECCCurve.SECP256R1,
        pwd: _ty.Optional[bytes | str] = None,
    ) -> None:
        self._ecc_type = ecc_type
        self._ecc_curve = ecc_curve
        self._private_key: _ty.Any
        self._public_key: _ty.Any

        if ecc_type == ECCType.ECDSA:
            curve_cls = _CURVE_MAP.get(ecc_curve or ECCCurve.SECP256R1)
            if curve_cls is None:
                raise ValueError(f"Unsupported ECC curve: {ecc_curve}")
            self._private_key = ec.generate_private_key(curve_cls(), backend=default_backend())
        elif ecc_type == ECCType.Ed25519:
            self._private_key = ed25519.Ed25519PrivateKey.generate()
        elif ecc_type == ECCType.Ed448:
            self._private_key = ed448.Ed448PrivateKey.generate()
        elif ecc_type == ECCType.X25519:
            self._private_key = x25519.X25519PrivateKey.generate()
        elif ecc_type == ECCType.X448:
            self._private_key = x448.X448PrivateKey.generate()
        else:
            raise ValueError(f"Unsupported ECC type: {ecc_type}")

        self._public_key = self._private_key.public_key()

    @classmethod
    def decode_private_key(
        cls,
        data: bytes,
        format_: ASymKeyFormat,
        encoding: ASymKeyEncoding,
        password: bytes | None = None,
    ) -> _ty.Self:
        key = _load_private_key(data, encoding, password)
        obj = cls(ECCType.ECDSA, ECCCurve.SECP256R1)
        if isinstance(key, ec.EllipticCurvePrivateKey):
            curve_name = key.curve.name.lower()
            curve = next((c for c, cc in _CURVE_MAP.items() if cc().name.lower() == curve_name), ECCCurve.SECP256R1)
            obj._ecc_type = ECCType.ECDSA
            obj._ecc_curve = curve
        elif isinstance(key, ed25519.Ed25519PrivateKey):
            obj._ecc_type = ECCType.Ed25519
            obj._ecc_curve = None
        elif isinstance(key, ed448.Ed448PrivateKey):
            obj._ecc_type = ECCType.Ed448
            obj._ecc_curve = None
        elif isinstance(key, x25519.X25519PrivateKey):
            obj._ecc_type = ECCType.X25519
            obj._ecc_curve = None
        elif isinstance(key, x448.X448PrivateKey):
            obj._ecc_type = ECCType.X448
            obj._ecc_curve = None
        else:
            raise ValueError("Provided key is not a supported ECC private key")

        obj._private_key = key
        obj._public_key = key.public_key()
        return obj

    @classmethod
    def decode_public_key(
        cls, data: bytes, format_: ASymKeyFormat, encoding: ASymKeyEncoding
    ) -> _ty.Self:
        if encoding == ASymKeyEncoding.RAW:
            raise _NotSupportedError("RAW public decode requires type context for ECC")
        key = _load_public_key(data, encoding)
        obj = cls(ECCType.ECDSA, ECCCurve.SECP256R1)
        obj._private_key = None
        obj._public_key = key

        if isinstance(key, ec.EllipticCurvePublicKey):
            obj._ecc_type = ECCType.ECDSA
            curve_name = key.curve.name.lower()
            obj._ecc_curve = next((c for c, cc in _CURVE_MAP.items() if cc().name.lower() == curve_name), ECCCurve.SECP256R1)
        elif isinstance(key, ed25519.Ed25519PublicKey):
            obj._ecc_type = ECCType.Ed25519
            obj._ecc_curve = None
        elif isinstance(key, ed448.Ed448PublicKey):
            obj._ecc_type = ECCType.Ed448
            obj._ecc_curve = None
        elif isinstance(key, x25519.X25519PublicKey):
            obj._ecc_type = ECCType.X25519
            obj._ecc_curve = None
        elif isinstance(key, x448.X448PublicKey):
            obj._ecc_type = ECCType.X448
            obj._ecc_curve = None
        else:
            raise ValueError("Provided key is not a supported ECC public key")

        return obj

    def encode_private_key(
        self,
        format_: ASymKeyFormat,
        encoding: ASymKeyEncoding,
        password: bytes | None = None,
    ) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")

        if encoding == ASymKeyEncoding.RAW:
            if hasattr(self._private_key, "private_bytes_raw"):
                return self._private_key.private_bytes_raw()
            raise _NotSupportedError("RAW private encoding unsupported for this ECC type")

        enc = _asym_encoding(encoding)
        is_ec = self._ecc_type == ECCType.ECDSA
        priv_format = _private_format(format_, is_rsa=False, is_ec=is_ec)
        protection = (
            serialization.NoEncryption()
            if password is None
            else serialization.BestAvailableEncryption(password)
        )
        return self._private_key.private_bytes(enc, priv_format, protection)

    def encode_public_key(self, format_: ASymKeyFormat, encoding: ASymKeyEncoding) -> bytes:
        if self._public_key is None:
            raise ValueError("No public key present")

        if encoding == ASymKeyEncoding.RAW:
            if hasattr(self._public_key, "public_bytes_raw"):
                return self._public_key.public_bytes_raw()
            if self._ecc_type == ECCType.ECDSA:
                return self._public_key.public_bytes(
                    serialization.Encoding.X962,
                    serialization.PublicFormat.CompressedPoint,
                )
            raise _NotSupportedError("RAW public encoding unsupported for this ECC type")

        enc = _asym_encoding(encoding)
        pub_format = _public_format(
            format_, encoding, is_rsa=False, is_ec=(self._ecc_type == ECCType.ECDSA)
        )
        return self._public_key.public_bytes(enc, pub_format)

    def sign(self, data: bytes, padding: ASymPadding) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")

        if self._ecc_type == ECCType.ECDSA:
            return self._private_key.sign(data, ec.ECDSA(hashes.SHA256()))
        if self._ecc_type in (ECCType.Ed25519, ECCType.Ed448):
            return self._private_key.sign(data)
        raise _NotSupportedError(f"{self._ecc_type} does not support signing")

    def sign_verify(self, data: bytes, signature: bytes, padding: ASymPadding) -> bool:
        if self._public_key is None:
            raise ValueError("No public key present")

        try:
            if self._ecc_type == ECCType.ECDSA:
                self._public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
            elif self._ecc_type in (ECCType.Ed25519, ECCType.Ed448):
                self._public_key.verify(signature, data)
            else:
                raise _NotSupportedError(f"{self._ecc_type} does not support signature verification")
            return True
        except InvalidSignature:
            return False

    def key_exchange(self, peer_public_key: _ty.Self) -> bytes:
        if self._private_key is None:
            raise ValueError("No private key present")

        if self._ecc_type == ECCType.ECDSA:
            if not isinstance(self._private_key, ec.EllipticCurvePrivateKey):
                raise ValueError("Invalid private key for ECDSA key exchange")
            if not isinstance(peer_public_key._public_key, ec.EllipticCurvePublicKey):
                raise ValueError("Peer key must be an EC public key")
            return self._private_key.exchange(ec.ECDH(), peer_public_key._public_key)

        if self._ecc_type == ECCType.X25519:
            if not isinstance(peer_public_key._public_key, x25519.X25519PublicKey):
                raise ValueError("Peer key must be an X25519 public key")
            return self._private_key.exchange(peer_public_key._public_key)

        if self._ecc_type == ECCType.X448:
            if not isinstance(peer_public_key._public_key, x448.X448PublicKey):
                raise ValueError("Peer key must be an X448 public key")
            return self._private_key.exchange(peer_public_key._public_key)

        raise _NotSupportedError(f"{self._ecc_type} does not support key exchange")

    def __repr__(self) -> str:
        return f"<ECC type={self._ecc_type}>"
