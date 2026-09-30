"""Load a private, verified index bundle when a deployment has no local index."""

import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

import requests
import streamlit as st

from core.config import EMBEDDING_MODEL
from vectorstore.faiss_store import FAISSStore, ROOT

MAX_BUNDLE_BYTES = 100 * 1024 * 1024


def _setting(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or os.getenv(name, ""))
    except Exception:
        return os.getenv(name, "")


def _download_bundle(url: str, token: str, expected_hash: str) -> bytes:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("A URL do índice deve usar HTTPS")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
        raise ValueError("SHA-256 do índice ausente ou inválido")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = requests.get(url, headers=headers, stream=True, allow_redirects=False, timeout=(10, 60))
    except requests.RequestException:
        raise OSError("Não foi possível obter o índice privado") from None
    try:
        if response.status_code != 200:
            raise OSError("Não foi possível obter o índice privado")
        data = bytearray()
        digest = hashlib.sha256()
        try:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                data.extend(chunk)
                digest.update(chunk)
                if len(data) > MAX_BUNDLE_BYTES:
                    raise ValueError("Pacote do índice excede o limite")
        except requests.RequestException:
            raise OSError("Não foi possível obter o índice privado") from None
        if digest.hexdigest().lower() != expected_hash.lower():
            raise ValueError("SHA-256 do índice não confere")
        return bytes(data)
    finally:
        response.close()


def _install_bundle(bundle: bytes, root: Path) -> str:
    root = Path(root)
    versions = root / "versions"
    versions.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        names = archive.namelist()
        if "CURRENT" not in names:
            raise ValueError("Pacote sem ponteiro CURRENT")
        version = archive.read("CURRENT").decode("ascii").strip()
        if not re.fullmatch(r"[0-9a-f]{32}", version):
            raise ValueError("Versão inválida no pacote")
        expected = {"CURRENT", *(f"versions/{version}/{name}" for name in
                                 ("faiss.index", "metadata.jsonl", "manifest.json"))}
        if set(names) != expected or len(names) != len(expected):
            raise ValueError("Arquivos inesperados ou ausentes no pacote")
        if sum(info.file_size for info in archive.infolist()) > MAX_BUNDLE_BYTES:
            raise ValueError("Pacote expandido excede o limite")
        manifest = json.loads(archive.read(f"versions/{version}/manifest.json"))
        if manifest.get("index_version") != version or manifest.get("embedding_model") != EMBEDDING_MODEL:
            raise ValueError("Manifesto do índice incompatível")
        with tempfile.TemporaryDirectory(prefix="staging-", dir=versions) as temporary:
            staged = Path(temporary)
            for name in ("faiss.index", "metadata.jsonl", "manifest.json"):
                (staged / name).write_bytes(archive.read(f"versions/{version}/{name}"))
            FAISSStore._validate_files(staged, manifest)
            target = versions / version
            if target.exists():
                FAISSStore._validate_files(target, manifest)
            else:
                os.replace(staged, target)
    pointer = root / "CURRENT.tmp"
    pointer.write_text(version, encoding="ascii")
    os.replace(pointer, root / "CURRENT")
    return version


def ensure_index_available(root: Path = ROOT) -> str:
    """Use an existing index, or install one from private HTTPS storage."""
    root = Path(root)
    pointer = root / "CURRENT"
    if pointer.is_file():
        return pointer.read_text(encoding="ascii").strip()
    url = _setting("LEXRAG_INDEX_BUNDLE_URL")
    if not url:
        raise FileNotFoundError("Índice local e URL do pacote privado ausentes")
    bundle = _download_bundle(url, _setting("LEXRAG_INDEX_BUNDLE_TOKEN"),
                              _setting("LEXRAG_INDEX_BUNDLE_SHA256"))
    return _install_bundle(bundle, root)
