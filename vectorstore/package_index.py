"""Create a private deploy bundle from the current local FAISS index."""

import hashlib
import zipfile
from pathlib import Path

from vectorstore.faiss_store import FAISSStore, ROOT


def package_index(root: Path = ROOT, output: Path | None = None) -> tuple[Path, str]:
    root = Path(root)
    output = Path(output) if output else root / "index-bundle.zip"
    store = FAISSStore().load(root)
    version = store.manifest["index_version"]
    files = ["CURRENT", *(f"versions/{version}/{name}" for name in
                           ("faiss.index", "metadata.jsonl", "manifest.json"))]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(root / name, arcname=name)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return output, digest


if __name__ == "__main__":
    path, sha256 = package_index()
    print(f"Pacote privado: {path}")
    print(f"SHA-256: {sha256}")
