from __future__ import annotations

import pytest

pytest.importorskip("cryptography")

from ..crypto import Backend, set_backend
from ..crypto.algos import Asym, HashAlgorithm, KeyDerivationFunction, Sym
from ..crypto.exceptions import NotSupportedError


@pytest.fixture()
def crypto_backend() -> None:
    set_backend([Backend.cryptography_alpha, Backend.std_lib])


def test_backend_wires_core_primitives(crypto_backend: None) -> None:
    assert Sym.Cipher.AES.key.__concrete__ is True
    assert Sym.Cipher.ChaCha20.key.__concrete__ is True
    assert Asym.Cipher.RSA.keypair.__concrete__ is True
    assert Asym.Cipher.DSA.keypair.__concrete__ is True
    assert Asym.Cipher.ECC.keypair.__concrete__ is True


def test_hash_roundtrips_include_sha1(crypto_backend: None) -> None:
    for hasher in (HashAlgorithm.SHA1, HashAlgorithm.SHA2.SHA256, HashAlgorithm.SHA3.SHA256):
        digest = hasher.hash(b"abc")
        assert hasher.verify(b"abc", digest)


def test_kdf_derive_lengths(crypto_backend: None) -> None:
    salt = b"0123456789abcdef"

    assert len(KeyDerivationFunction.PBKDF2HMAC.derive(b"pw", salt=salt, length=16)) == 16
    assert len(KeyDerivationFunction.Scrypt.derive(b"pw", salt=salt, length=16, n=2**10)) == 16
    assert len(KeyDerivationFunction.HKDF.derive(b"pw", salt=salt, length=16)) == 16
    assert len(KeyDerivationFunction.X963.derive(b"pw", otherinfo=b"ctx", length=16)) == 16
    assert len(KeyDerivationFunction.ConcatKDF.derive(b"pw", otherinfo=b"ctx", length=16)) == 16


def test_argon2_variant_guard(crypto_backend: None) -> None:
    with pytest.raises(ValueError, match="variant"):
        KeyDerivationFunction.ARGON2.derive(
            b"pw",
            salt=b"0123456789abcdef",
            length=16,
            variant="i",
        )


def test_kmac_optional_behavior(crypto_backend: None) -> None:
    try:
        out = KeyDerivationFunction.KMAC128.derive(b"pw", length=16)
        assert isinstance(out, bytes)
        assert len(out) == 16
    except ValueError as exc:
        assert "KMAC128" in str(exc)


def test_aes_modes_roundtrip_and_tamper_detection(crypto_backend: None) -> None:
    key = Sym.Cipher.AES.key.new(128)
    plaintext = b"secret payload"

    for mode in (Sym.Operation.ECB, Sym.Operation.CBC, Sym.Operation.CFB, Sym.Operation.OFB, Sym.Operation.CTR):
        ciphertext = key.encrypt(plaintext, Sym.Padding.PKCS7, mode)
        assert key.decrypt(ciphertext, Sym.Padding.PKCS7, mode) == plaintext

    packed_gcm = key.encrypt(plaintext, Sym.Padding.PKCS7, Sym.Operation.GCM)
    assert key.decrypt(packed_gcm, Sym.Padding.PKCS7, Sym.Operation.GCM) == plaintext

    tampered = bytearray(packed_gcm)
    tampered[-1] ^= 0x01
    with pytest.raises(Exception):
        key.decrypt(bytes(tampered), Sym.Padding.PKCS7, Sym.Operation.GCM)


def test_aes_encodings_roundtrip(crypto_backend: None) -> None:
    key = Sym.Cipher.AES.key.new(256)
    plain = b"roundtrip"

    for enc in (Sym.KeyEncoding.RAW, Sym.KeyEncoding.BASE64, Sym.KeyEncoding.HEX, Sym.KeyEncoding.BASE32):
        encoded = key.encode(enc)
        decoded = Sym.Cipher.AES.key.decode(encoded, enc)
        ct = decoded.encrypt(plain, Sym.Padding.PKCS7, Sym.Operation.ECB)
        assert decoded.decrypt(ct, Sym.Padding.PKCS7, Sym.Operation.ECB) == plain


def test_chacha20_roundtrip_and_rejects_block_mode(crypto_backend: None) -> None:
    key = Sym.Cipher.ChaCha20.key.new()
    plain = b"stream-data"

    ct = key.encrypt(plain, Sym.Padding.PKCS7, Sym.Operation.CTR)
    assert key.decrypt(ct, Sym.Padding.PKCS7, Sym.Operation.CTR) == plain

    with pytest.raises(NotSupportedError):
        key.encrypt(plain, Sym.Padding.PKCS7, Sym.Operation.CBC)


def test_mac_generation_and_verification(crypto_backend: None) -> None:
    key = Sym.Cipher.AES.key.new(128)
    data = b"mac-data"

    hmac_mac = key.generate_mac(data, Sym.MessageAuthenticationCode.HMAC)
    assert key.verify_mac(data, hmac_mac, Sym.MessageAuthenticationCode.HMAC)

    cmac_mac = key.generate_mac(data, Sym.MessageAuthenticationCode.CMAC)
    assert key.verify_mac(data, cmac_mac, Sym.MessageAuthenticationCode.CMAC)


def test_rsa_sign_encrypt_and_serialization(crypto_backend: None) -> None:
    key = Asym.Cipher.RSA.keypair.new(2048)
    data = b"hello"

    signature = key.sign(data, Asym.Padding.PSS)
    assert key.sign_verify(data, signature, Asym.Padding.PSS)

    ciphertext = key.encrypt(data, Asym.Padding.OAEP)
    assert key.decrypt(ciphertext, Asym.Padding.OAEP) == data

    pub = key.encode_public_key(Asym.KeyFormat.PKCS1, Asym.KeyEncoding.PEM)
    pub_only = Asym.Cipher.RSA.keypair.decode_public_key(pub, Asym.KeyFormat.PKCS1, Asym.KeyEncoding.PEM)
    assert pub_only.sign_verify(data, signature, Asym.Padding.PSS)


def test_dsa_sign_and_verify(crypto_backend: None) -> None:
    key = Asym.Cipher.DSA.keypair.new(2048)
    data = b"hello"

    signature = key.sign(data, Asym.Padding.PSS)
    assert key.sign_verify(data, signature, Asym.Padding.PSS)


def test_ecc_ecdsa_and_key_exchange(crypto_backend: None) -> None:
    a = Asym.Cipher.ECC.keypair.ecdsa_key(Asym.Cipher.ECC.Curve.SECP256R1)
    b = Asym.Cipher.ECC.keypair.ecdsa_key(Asym.Cipher.ECC.Curve.SECP256R1)
    data = b"hello"

    signature = a.sign(data, Asym.Padding.PSS)
    assert a.sign_verify(data, signature, Asym.Padding.PSS)

    secret_a = a.key_exchange(b)
    secret_b = b.key_exchange(a)
    assert secret_a == secret_b


def test_ecc_ed25519_and_x25519_capabilities(crypto_backend: None) -> None:
    ed = Asym.Cipher.ECC.keypair.optimized_key(Asym.Cipher.ECC.Type.Ed25519)
    data = b"hello"

    signature = ed.sign(data, Asym.Padding.PSS)
    assert ed.sign_verify(data, signature, Asym.Padding.PSS)
    with pytest.raises(NotSupportedError):
        ed.key_exchange(ed)

    x1 = Asym.Cipher.ECC.keypair.optimized_key(Asym.Cipher.ECC.Type.X25519)
    x2 = Asym.Cipher.ECC.keypair.optimized_key(Asym.Cipher.ECC.Type.X25519)
    assert x1.key_exchange(x2) == x2.key_exchange(x1)


def test_backend_fallback_with_stdlib(crypto_backend: None) -> None:
    digest = HashAlgorithm.SHA2.SHA256.hash(b"hello")
    assert HashAlgorithm.SHA2.SHA256.verify(b"hello", digest)
