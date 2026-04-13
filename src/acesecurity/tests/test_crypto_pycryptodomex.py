from __future__ import annotations

import pytest

pytest.importorskip("Cryptodome")

from ..crypto import Backend, set_backend
from ..crypto.algos import Asym, HashAlgorithm, KeyDerivationFunction, Sym
from ..crypto.exceptions import NotSupportedError


@pytest.fixture()
def pycrypto_backend() -> None:
    set_backend([Backend.pycryptodomex_alpha, Backend.std_lib])


def test_backend_wires_core(pycrypto_backend: None) -> None:
    assert Sym.Cipher.AES.key.__concrete__ is True
    assert Sym.Cipher.ChaCha20.key.__concrete__ is True
    assert Asym.Cipher.RSA.keypair.__concrete__ is True


def test_hash_roundtrip(pycrypto_backend: None) -> None:
    for hasher in (HashAlgorithm.SHA1, HashAlgorithm.SHA2.SHA256, HashAlgorithm.BLAKE2.BLAKE2s):
        h = hasher.hash(b"abc")
        assert hasher.verify(b"abc", h)


def test_kdf_roundtrip(pycrypto_backend: None) -> None:
    salt = b"0123456789abcdef"
    assert len(KeyDerivationFunction.PBKDF2HMAC.derive(b"pw", salt=salt, length=16)) == 16
    assert len(KeyDerivationFunction.Scrypt.derive(b"pw", salt=salt, length=16, n=2**10)) == 16
    assert len(KeyDerivationFunction.HKDF.derive(b"pw", salt=salt, length=16)) == 16


def test_aes_gcm_and_cbc(pycrypto_backend: None) -> None:
    key = Sym.Cipher.AES.key.new(128)
    pt = b"hello"

    gcm = key.encrypt(pt, Sym.Padding.PKCS7, Sym.Operation.GCM)
    assert key.decrypt(gcm, Sym.Padding.PKCS7, Sym.Operation.GCM) == pt

    cbc = key.encrypt(pt, Sym.Padding.PKCS7, Sym.Operation.CBC)
    assert key.decrypt(cbc, Sym.Padding.PKCS7, Sym.Operation.CBC) == pt


def test_chacha20_roundtrip(pycrypto_backend: None) -> None:
    key = Sym.Cipher.ChaCha20.key.new()
    pt = b"hello stream"
    ct = key.encrypt(pt, Sym.Padding.PKCS7, Sym.Operation.CTR)
    assert key.decrypt(ct, Sym.Padding.PKCS7, Sym.Operation.CTR) == pt



def test_mac_support(pycrypto_backend: None) -> None:
    key = Sym.Cipher.AES.key.new(128)
    data = b"auth"
    m = key.generate_mac(data, Sym.MessageAuthenticationCode.HMAC)
    assert key.verify_mac(data, m, Sym.MessageAuthenticationCode.HMAC)


def test_rsa_sign_encrypt(pycrypto_backend: None) -> None:
    key = Asym.Cipher.RSA.keypair.new(2048)
    msg = b"hello"

    sig = key.sign(msg, Asym.Padding.PSS)
    assert key.sign_verify(msg, sig, Asym.Padding.PSS)

    ct = key.encrypt(msg, Asym.Padding.OAEP)
    assert key.decrypt(ct, Asym.Padding.OAEP) == msg


def test_dsa_sign(pycrypto_backend: None) -> None:
    key = Asym.Cipher.DSA.keypair.new(2048)
    msg = b"hello"
    sig = key.sign(msg, Asym.Padding.PSS)
    assert key.sign_verify(msg, sig, Asym.Padding.PSS)


def test_ecc_sign_and_key_exchange(pycrypto_backend: None) -> None:
    a = Asym.Cipher.ECC.keypair.ecdsa_key(Asym.Cipher.ECC.Curve.SECP256R1)
    b = Asym.Cipher.ECC.keypair.ecdsa_key(Asym.Cipher.ECC.Curve.SECP256R1)

    msg = b"hello"
    sig = a.sign(msg, Asym.Padding.PSS)
    assert a.sign_verify(msg, sig, Asym.Padding.PSS)

    try:
        sa = a.key_exchange(b)
        sb = b.key_exchange(a)
        assert sa == sb
    except NotSupportedError:
        pytest.skip("ECC key exchange not available in this pycryptodomex build")


def test_fallback_hash(pycrypto_backend: None) -> None:
    h = HashAlgorithm.SHA2.SHA256.hash(b"hello")
    assert HashAlgorithm.SHA2.SHA256.verify(b"hello", h)
