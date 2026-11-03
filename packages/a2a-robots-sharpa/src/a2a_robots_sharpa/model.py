"""Download the Sharpa Wave MuJoCo model (Apache-2.0) at a pinned commit.

The meshes are about 24 MB, so they are fetched on first use and cached
instead of being committed to this repository.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = "sharpa-robotics/sharpa-urdf-usd-xml"
COMMIT = "0d19cac602f46456b819e4b6a2c09a74982c9a3e"
# sha256 over the sorted "path sha256" lines of every extracted file.
DIGEST = "e6e654c254a8d954e5574da66d77067f78a3962e5a1ef4dd86f4af6321429816"
MODEL_FILE = "dual_sharpa_wave.xml"

_KEEP = (
    "LICENSE.txt",
    "NOTICE.txt",
    "wave_01/left_sharpa_wave/meshes/",
    "wave_01/right_sharpa_wave/meshes/",
)
_XML = "wave_01/dual_sharpa_wave/dual_sharpa_wave.xml"


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "a2a-robots" / f"sharpa-wave-{COMMIT[:12]}"


def fetch_model(cache_dir: str | Path | None = None) -> Path:
    """Return the path to ``dual_sharpa_wave.xml``, downloading it if needed.

    The XML is placed next to the ``left_sharpa_wave`` and ``right_sharpa_wave``
    mesh folders so its relative mesh paths resolve.
    """
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    xml = root / MODEL_FILE
    if xml.is_file() and _digest(root) == DIGEST:
        return xml
    url = f"https://codeload.github.com/{REPO}/tar.gz/{COMMIT}"
    with urllib.request.urlopen(url, timeout=120) as response:
        archive = response.read()
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=root.parent, prefix=".sharpa-"))
    try:
        _extract(archive, staging)
        digest = _digest(staging)
        if digest != DIGEST:
            raise RuntimeError(f"Sharpa model digest mismatch: got {digest}, expected {DIGEST}")
        shutil.rmtree(root, ignore_errors=True)
        staging.rename(root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return xml


def _extract(archive: bytes, dest: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # Strip the "<repo>-<commit>/" prefix GitHub adds.
            path = member.name.split("/", 1)[1] if "/" in member.name else ""
            if path == _XML:
                target = dest / MODEL_FILE
            elif path.startswith(_KEEP) and ".." not in path.split("/"):
                target = dest / path.removeprefix("wave_01/")
            else:
                continue
            source = tar.extractfile(member)
            assert source is not None
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())


def _digest(root: Path) -> str:
    lines = sorted(
        f"{p.relative_to(root).as_posix()} {hashlib.sha256(p.read_bytes()).hexdigest()}"
        for p in root.rglob("*")
        if p.is_file()
    )
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()
