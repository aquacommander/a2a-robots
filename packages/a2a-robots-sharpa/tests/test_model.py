import io
import tarfile

import pytest

from a2a_robots_sharpa import model


def archive(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(f"sharpa-urdf-usd-xml-{model.COMMIT}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


SOURCE = {
    "LICENSE.txt": b"license",
    "NOTICE.txt": b"notice",
    "README.md": b"not kept",
    "wave_01/dual_sharpa_wave/dual_sharpa_wave.xml": b"<mujoco/>",
    "wave_01/dual_sharpa_wave/dual_sharpa_wave.urdf": b"not kept",
    "wave_01/left_sharpa_wave/meshes/a.STL": b"left",
    "wave_01/right_sharpa_wave/meshes/b.STL": b"right",
    "wave_01/left_sharpa_wave/meshes/../../../escape.txt": b"outside",
}


def test_extract_keeps_the_xml_next_to_the_mesh_folders(tmp_path):
    model._extract(archive(SOURCE), tmp_path)
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert files == [
        "LICENSE.txt",
        "NOTICE.txt",
        "dual_sharpa_wave.xml",
        "left_sharpa_wave/meshes/a.STL",
        "right_sharpa_wave/meshes/b.STL",
    ]
    assert not (tmp_path.parent / "escape.txt").exists()


def test_fetch_rejects_a_download_with_the_wrong_digest(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(model.urllib.request, "urlopen", lambda *a, **k: Response(archive(SOURCE)))
    with pytest.raises(RuntimeError, match="digest mismatch"):
        model.fetch_model(tmp_path / "cache")
    assert not (tmp_path / "cache").exists()
    assert list(tmp_path.iterdir()) == []


def test_fetch_uses_a_valid_cache_without_downloading(tmp_path, monkeypatch):
    model._extract(archive(SOURCE), tmp_path)
    monkeypatch.setattr(model, "DIGEST", model._digest(tmp_path))

    def no_network(*args, **kwargs):
        raise AssertionError("should not download")

    monkeypatch.setattr(model.urllib.request, "urlopen", no_network)
    assert model.fetch_model(tmp_path) == tmp_path / "dual_sharpa_wave.xml"
