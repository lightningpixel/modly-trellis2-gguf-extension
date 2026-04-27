"""
Trellis.2 GGUF — extension setup script.

Creates an isolated venv and installs all required dependencies:
  - PyTorch (version selected by GPU SM / CUDA driver)
  - Custom CUDA wheels from https://pozzettiandrea.github.io/cuda-wheels/
      cumesh, nvdiffrast, nvdiffrec_render, flex_gemm, o_voxel
  - triton-windows (Windows only)
  - Python packages from requirements (gguf, meshlib, rembg, trimesh, …)
  - trellis2_gguf source package (from ComfyUI-Trellis2-GGUF GitHub)
  - Patches: flexible_dual_grid.py, remeshing.py

Called by Modly at extension install time with:
    python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":124}'
"""

import io
import json
import platform
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

_COMFYUI_TRELLIS2_ZIP = "https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF/archive/refs/heads/main.zip"
_WHEELS_INDEX_BASE    = "https://pozzettiandrea.github.io/cuda-wheels/"

# cumesh and flex-gemm are hard dependencies; the others are optional (texture baking)
# Keys are index URL names (hyphens); values are the pip-installable wheel names
_CUDA_WHEELS_REQUIRED = ["cumesh", "flex-gemm"]
_CUDA_WHEELS_OPTIONAL = ["nvdiffrast", "o-voxel"]
_CUDA_WHEELS = _CUDA_WHEELS_REQUIRED + _CUDA_WHEELS_OPTIONAL

# Standard Python packages
_PY_PACKAGES = [
    "Pillow",
    "numpy",
    "scipy",
    "trimesh",
    "pymeshlab",
    "meshlib",
    "opencv-python-headless",
    "gguf",
    "sdnq",
    "open3d",
    "rectpack",
    "requests",
    "huggingface_hub",
    "transformers==5.2.0",
    "accelerate",
    "einops",
    "easydict",
]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _pip(venv: Path, *args: str) -> None:
    is_win  = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def _python(venv: Path) -> Path:
    is_win = platform.system() == "Windows"
    return venv / ("Scripts/python.exe" if is_win else "bin/python")


def _site_packages(venv: Path) -> Path:
    exe = _python(venv)
    out = subprocess.check_output(
        [str(exe), "-c",
         "import site; print([p for p in site.getsitepackages() if 'site-packages' in p][0])"],
        text=True,
    ).strip()
    return Path(out)


def _get_torch_version(venv: Path) -> str:
    """Return the installed PyTorch version string, e.g. '2.7.0'."""
    exe = _python(venv)
    try:
        out = subprocess.check_output(
            [str(exe), "-c", "import torch; print(torch.__version__.split('+')[0])"],
            text=True,
        ).strip()
        return out
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Custom CUDA wheels (pozzettiandrea.github.io)                                #
# --------------------------------------------------------------------------- #

def _find_wheel_url(lib_name: str, python_tag: str, platform_tag: str, torch_ver: str) -> str | None:
    """
    Fetch the pozzettiandrea.github.io wheel index for lib_name and return
    the best matching wheel URL for (python_tag, platform_tag, torch_ver).

    Tries the exact torch version first, then falls back to the closest
    lower version available on the index.
    Returns None if the index page is unreachable or no match found.
    """
    index_url = f"{_WHEELS_INDEX_BASE}{lib_name}/"
    try:
        with urllib.request.urlopen(index_url, timeout=30) as resp:
            html = resp.read().decode("utf-8")
    except Exception as exc:
        print(f"[setup] WARNING: Could not fetch wheel index for {lib_name}: {exc}")
        return None

    # Extract all .whl href links
    links = re.findall(r'href=["\']([^"\']*\.whl)["\']', html)
    if not links:
        links = re.findall(r'([\w\-\.]+\.whl)', html)

    def _abs(link: str) -> str:
        if link.startswith("http"):
            return link
        return f"{index_url}{link.split('/')[-1]}"

    # Normalise torch version: "2.7.0" -> "27" (index uses "torch27", not "torch270")
    parts = torch_ver.split(".")
    major, minor = int(parts[0]), int(parts[1])
    tv_tag = f"{major}{minor}"

    def _candidates(maj: int, min_: int) -> list[str]:
        # Index display names use "torch26", GitHub URLs use "torch2.6"
        tags = [f"torch{maj}{min_}", f"torch{maj}.{min_}"]
        return [
            _abs(link) for link in links
            if python_tag in link.split("/")[-1]
            and platform_tag in link.split("/")[-1]
            and any(t in link.split("/")[-1] for t in tags)
        ]

    # Try exact version first, then fall back to lower minor versions
    for m in range(minor, -1, -1):
        matches = _candidates(major, m)
        if matches:
            if m != minor:
                print(f"[setup] NOTE: No wheel for torch{tv_tag} — using torch{major}{m} fallback.")
            return matches[0]

    return None


def _install_cuda_wheels(venv: Path, gpu_sm: int) -> None:
    """Download and install custom CUDA wheels from pozzettiandrea.github.io."""
    is_win = platform.system() == "Windows"
    python_tag   = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform_tag = "win_amd64" if is_win else "linux_x86_64"
    torch_ver    = _get_torch_version(venv)

    print(f"[setup] Installing CUDA wheels (python={python_tag}, platform={platform_tag}, torch={torch_ver}) …")

    for lib in _CUDA_WHEELS:
        print(f"[setup] Finding wheel for {lib} …")
        url = _find_wheel_url(lib, python_tag, platform_tag, torch_ver)
        if url is None:
            if lib in _CUDA_WHEELS_REQUIRED:
                print(
                    f"[setup] ERROR: No compatible wheel found for {lib} (torch {torch_ver}).\n"
                    f"[setup]   This is a required dependency — the extension will fail to load.\n"
                    f"[setup]   Check https://pozzettiandrea.github.io/cuda-wheels/{lib}/ for available wheels."
                )
            else:
                print(f"[setup] WARNING: No wheel found for {lib} — texture baking will be unavailable.")
            continue
        print(f"[setup] Installing {lib} from {url} …")
        try:
            _pip(venv, "install", url)
            print(f"[setup] {lib} installed.")
        except subprocess.CalledProcessError as exc:
            if lib in _CUDA_WHEELS_REQUIRED:
                print(f"[setup] ERROR: Failed to install {lib} ({exc}). The extension will not load.")
            else:
                print(f"[setup] WARNING: Failed to install {lib} ({exc}). Texture baking may be unavailable.")


# --------------------------------------------------------------------------- #
# triton-windows (Windows only)                                                #
# --------------------------------------------------------------------------- #

def _install_triton_windows(venv: Path, torch_ver: str, gpu_sm: int = 0) -> None:
    """Install triton-windows matching the PyTorch version.

    Blackwell GPUs (SM 12.x) require triton-windows >= 3.3.1 which bundles
    ptxas 12.8 and adds SM 12.x support (triton-lang PR #8498).
    """
    if platform.system() != "Windows":
        return

    # Version constraints from ComfyUI-Trellis2-GGUF install.py
    tv = tuple(int(x) for x in torch_ver.split(".")[:2])
    if tv >= (2, 10):
        triton_spec = "triton-windows<3.7"
    elif tv >= (2, 9):
        triton_spec = "triton-windows<3.6"
    elif tv >= (2, 8):
        triton_spec = "triton-windows<3.5"
    elif tv >= (2, 7):
        if gpu_sm >= 100:
            # Blackwell: 3.3.1+ bundles ptxas 12.8 with SM 12.x support
            triton_spec = "triton-windows>=3.3.1,<3.4"
        else:
            triton_spec = "triton-windows<3.4"
    else:
        triton_spec = "triton-windows"

    print(f"[setup] Installing {triton_spec} …")
    try:
        _pip(venv, "install", triton_spec)
        print("[setup] triton-windows installed.")
    except subprocess.CalledProcessError as exc:
        print(f"[setup] WARNING: triton-windows install failed ({exc}). Some CUDA kernels may not work.")


# --------------------------------------------------------------------------- #
# trellis2_gguf source                                                         #
# --------------------------------------------------------------------------- #

def _install_trellis2_gguf(venv: Path) -> None:
    """
    Download ComfyUI-Trellis2-GGUF from GitHub and extract:
      - trellis2_gguf/   -> site-packages/trellis2_gguf/
      - patch/           -> site-packages/trellis2_gguf_patch/

    Also applies the patches to the corresponding installed packages
    (flexible_dual_grid.py in spconv, remeshing.py in trellis2_gguf).
    """
    sp    = _site_packages(venv)
    dest  = sp / "trellis2_gguf"

    if dest.exists():
        print("[setup] trellis2_gguf already installed, skipping.")
        return

    print("[setup] Downloading ComfyUI-Trellis2-GGUF source from GitHub …")
    try:
        with urllib.request.urlopen(_COMFYUI_TRELLIS2_ZIP, timeout=300) as resp:
            data = resp.read()
    except Exception as exc:
        raise RuntimeError(f"[setup] Could not download trellis2_gguf source: {exc}") from exc

    zip_root = "ComfyUI-Trellis2-GGUF-main/"
    pkg_prefix   = f"{zip_root}trellis2_gguf/"
    patch_prefix = f"{zip_root}patch/"

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # ── trellis2_gguf package ──────────────────────────────────────── #
        for member in zf.namelist():
            if not member.startswith(pkg_prefix):
                continue
            rel    = member[len(zip_root):]          # "trellis2_gguf/..."
            target = sp / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))

        # ── patch files ────────────────────────────────────────────────── #
        patch_dest = sp / "trellis2_gguf_patch"
        for member in zf.namelist():
            if not member.startswith(patch_prefix):
                continue
            rel    = member[len(patch_prefix):]       # e.g. "flexible_dual_grid.py"
            if not rel or member.endswith("/"):
                continue
            target = patch_dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(member))

    print(f"[setup] trellis2_gguf installed to {sp}.")

    # ── Apply patches ──────────────────────────────────────────────────── #
    _apply_patches(sp, patch_dest)


def _install_comfyui_gguf(venv: Path) -> None:
    """
    Download ops.py / dequant.py / loader.py from city96/ComfyUI-GGUF into the
    path that trellis2_gguf's _setup_native_gguf() searches:
      <venv>/Lib/ComfyUI-GGUF/
    Without these files the GGUF dequant falls back to a CPU implementation.
    """
    sp       = _site_packages(venv)
    gguf_dir = sp.parent.parent / "Lib" / "ComfyUI-GGUF"   # <venv>/Lib/ComfyUI-GGUF
    _FILES   = ["ops.py", "dequant.py", "loader.py"]

    if all((gguf_dir / f).exists() for f in _FILES):
        print("[setup] ComfyUI-GGUF already installed, skipping.")
        return

    gguf_dir.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/"
    print(f"[setup] Installing ComfyUI-GGUF (city96) GGUF ops to {gguf_dir} …")
    for fname in _FILES:
        dest = gguf_dir / fname
        if dest.exists():
            continue
        try:
            with urllib.request.urlopen(base + fname, timeout=60) as resp:
                dest.write_bytes(resp.read())
            print(f"[setup]   downloaded {fname}")
        except Exception as exc:
            print(f"[setup] WARNING: could not download ComfyUI-GGUF/{fname}: {exc}")
    print("[setup] ComfyUI-GGUF installed.")


def _apply_patches(sp: Path, patch_dir: Path) -> None:
    """
    Replace specific files in installed packages with the patched versions.

    Known patches:
      flexible_dual_grid.py -> overwrites the same file inside spconv/modules/
      remeshing.py          -> overwrites the same file inside trellis2_gguf/
    """
    if not patch_dir.exists():
        return

    patch_map = {
        "flexible_dual_grid.py": _find_in_site(sp, "flexible_dual_grid.py"),
        # Exclude cumesh/remeshing.py — cumesh ships its own correct version;
        # overwriting it with the trellis2_gguf patch breaks cumesh remesh.
        "remeshing.py": [p for p in _find_in_site(sp, "remeshing.py")
                         if "cumesh" not in p.parts],
    }

    for patch_name, targets in patch_map.items():
        src = patch_dir / patch_name
        if not src.exists():
            continue
        for tgt in targets:
            try:
                tgt.write_bytes(src.read_bytes())
                print(f"[setup] Patched {tgt.relative_to(sp)}")
            except Exception as exc:
                print(f"[setup] WARNING: Could not patch {tgt}: {exc}")


def _find_in_site(sp: Path, filename: str) -> list[Path]:
    """Return all occurrences of filename under site-packages."""
    return list(sp.rglob(filename))


# --------------------------------------------------------------------------- #
# Main setup                                                                   #
# --------------------------------------------------------------------------- #

def setup(python_exe: str, ext_dir: Path, gpu_sm: int, cuda_version: int = 0) -> None:
    venv = ext_dir / "venv"

    print(f"[setup] Creating venv at {venv} …")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ── PyTorch — select build based on GPU architecture / CUDA driver ── #
    if gpu_sm >= 100 or cuda_version >= 128:
        # Blackwell (RTX 50xx, B100…) — SM 12.x kernels require PyTorch 2.7+
        torch_pkgs  = ["torch==2.7.0", "torchvision==0.22.0"]
        torch_index = "https://download.pytorch.org/whl/cu128"
        print(f"[setup] GPU SM {gpu_sm}, CUDA {cuda_version} -> PyTorch 2.7 + CUDA 12.8 (Blackwell)")
    elif gpu_sm == 0 or gpu_sm >= 70:
        # Volta / Turing / Ampere / Ada / Hopper
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        torch_index = "https://download.pytorch.org/whl/cu126"
        print(f"[setup] GPU SM {gpu_sm} -> PyTorch 2.6 + CUDA 12.6")
    else:
        # Pascal (SM 6.x) — last PyTorch with SM 6.1 support
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        torch_index = "https://download.pytorch.org/whl/cu118"
        print(f"[setup] GPU SM {gpu_sm} (legacy) -> PyTorch 2.5 + CUDA 11.8")

    print("[setup] Installing PyTorch …")
    _pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ── Core Python dependencies ─────────────────────────────────────── #
    print("[setup] Installing core Python dependencies …")
    _pip(venv, "install", *_PY_PACKAGES)

    # ── rembg (background removal) ───────────────────────────────────── #
    print("[setup] Installing rembg …")
    if gpu_sm >= 70:
        _pip(venv, "install", "rembg[gpu]")
    else:
        _pip(venv, "install", "rembg", "onnxruntime")

    # ── Custom CUDA wheels (cumesh, nvdiffrast, flex_gemm, …) ─────────── #
    _install_cuda_wheels(venv, gpu_sm)

    # ── triton-windows ────────────────────────────────────────────────── #
    torch_ver = _get_torch_version(venv)
    if torch_ver:
        _install_triton_windows(venv, torch_ver, gpu_sm)

    # ── trellis2_gguf source ──────────────────────────────────────────── #
    _install_trellis2_gguf(venv)

    # ── ComfyUI-GGUF (city96) — native GGUF dequant on GPU ───────────── #
    _install_comfyui_gguf(venv)

    print("[setup] Done. Venv ready at:", venv)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        # PowerShell may pass the surrounding single quotes as part of the string
        raw = sys.argv[1].strip("'\"")
        args = json.loads(raw)
        setup(
            python_exe   = args["python_exe"],
            ext_dir      = Path(args["ext_dir"]),
            gpu_sm       = int(args.get("gpu_sm",       86)),
            cuda_version = int(args.get("cuda_version",  0)),
        )
    elif len(sys.argv) >= 4:
        setup(
            python_exe   = sys.argv[1],
            ext_dir      = Path(sys.argv[2]),
            gpu_sm       = int(sys.argv[3]),
            cuda_version = int(sys.argv[4]) if len(sys.argv) >= 5 else 0,
        )
    else:
        print("Usage (positional): python setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]")
        print('Usage (JSON)      : python setup.py "{\"python_exe\":\"...\",\"ext_dir\":\"...\",\"gpu_sm\":86,\"cuda_version\":128}"')
        sys.exit(1)
