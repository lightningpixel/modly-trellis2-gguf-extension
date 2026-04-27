"""
Trellis.2 GGUF extension for Modly.

Model   : https://huggingface.co/Aero-Ex/Trellis2-GGUF
Wrapper : https://github.com/Aero-Ex/ComfyUI-Trellis2-GGUF

Single Generate node : image -> geometry GLB.

Pipeline stages:
  1. Background removal via rembg
  2. Sparse-structure diffusion  (ss_steps)
  3. Shape SLaT diffusion        (slat_steps)
  4. Geometry export via cumesh remesh -> GLB

Model weights are downloaded once to <models>/trellis2/generate/ and reused.
"""
from __future__ import annotations

import io
import random
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled  # noqa: F401

_EXTENSION_DIR = Path(__file__).parent

# HuggingFace model repo
_HF_REPO = "Aero-Ex/Trellis2-GGUF"

# Files to download: all GGUF variants + JSON configs + Vision encoder + decoders/encoders
# (skip BF16/FP8 safetensors which add ~80 GB to the download; skip texture weights)
_HF_ALLOW_PATTERNS = [
    "pipeline.json",
    "Vision/**",
    "decoders/**/*.json",
    "decoders/**/*.safetensors",
    "encoders/**/*.json",
    "encoders/**/*.safetensors",
    "refiner/*.json",
    "refiner/*.gguf",
    "shape/*.json",
    "shape/*.gguf",
]

# Sampler params — matched to ComfyUI reference workflow (Q4 high-quality results)
_SS_CFG            = 6.5
_SS_RESCALE        = 0.20
_SS_INTERVAL       = [0.1, 1.0]
_SS_RESCALE_T      = 4.0

_SLAT_CFG          = 6.5
_SLAT_RESCALE      = 0.20
_SLAT_INTERVAL     = [0.1, 1.0]
_SLAT_RESCALE_T    = 4.0

# Allow the sparse structure stage to generate up to this many tokens.
# 49152 (old default) truncates complex objects — 999999 lets the model
# use as many voxels as it needs for full detail.
_MAX_NUM_TOKENS    = 150000


class Trellis2GGUFGenerator(BaseGenerator):
    MODEL_ID     = "trellis2"
    DISPLAY_NAME = "Trellis.2 GGUF"
    VRAM_GB      = 8

    # ------------------------------------------------------------------ #
    # Shared weights directory                                            #
    # ------------------------------------------------------------------ #

    @property
    def _weights_dir(self) -> Path:
        """
        model_dir = <models>/trellis2/{generate|refine}
        weights   = <models>/trellis2/  (one level up)
        """
        if self.model_dir.name in ("generate", "refine"):
            return self.model_dir.parent
        return self.model_dir

    # ------------------------------------------------------------------ #
    # Download                                                            #
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        return (self._weights_dir / "pipeline.json").exists()

    def _auto_download(self) -> None:
        import os
        from huggingface_hub import snapshot_download

        repo   = self.hf_repo or _HF_REPO
        target = self._weights_dir
        target.mkdir(parents=True, exist_ok=True)

        token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN") or None

        print(f"[Trellis2GGUFGenerator] Downloading {repo} -> {target} ...")
        print("[Trellis2GGUFGenerator] (Only GGUF weights + config files, ~3-8 GB depending on quantisations.)")
        snapshot_download(
            repo_id=repo,
            local_dir=str(target),
            allow_patterns=_HF_ALLOW_PATTERNS,
            token=token,
        )
        print("[Trellis2GGUFGenerator] Download complete.")

    # ------------------------------------------------------------------ #
    # Load / Unload                                                       #
    # ------------------------------------------------------------------ #

    def _ensure_venv_on_path(self) -> None:
        """Add the extension venv's site-packages to sys.path if not already present."""
        import sys
        import platform

        venv = _EXTENSION_DIR / "venv"
        if not venv.exists():
            return

        if platform.system() == "Windows":
            sp = venv / "Lib" / "site-packages"
        else:
            lib = venv / "lib"
            candidates = sorted(lib.glob("python3*/site-packages")) if lib.exists() else []
            if not candidates:
                return
            sp = candidates[-1]

        sp_str = str(sp)
        if sp.exists() and sp_str not in sys.path:
            sys.path.insert(0, sp_str)
            print(f"[Trellis2GGUFGenerator] Added venv site-packages to sys.path: {sp_str}")

    def load(self) -> None:
        if self._model is not None:
            return

        self._ensure_venv_on_path()

        if not self.is_downloaded():
            self._auto_download()

        self._ensure_trellis2_gguf()

        # gguf_quant is not available at load time (no UI params yet),
        # so default to Q5_K_M.  generate() will reload if the user
        # picks a different quantisation on first inference.
        self._load_pipeline("Q5_K_M")

    def _prepare_dinov3_dir(self) -> str | None:
        """
        Make Vision/ loadable by DINOv3ViTModel.from_pretrained:
          - creates model.safetensors (hardlink to original .safetensors)
          - writes config.json derived from safetensors key shapes if missing
        Returns path to Vision/ if ready, None on failure.
        """
        import os, json, shutil
        vision_dir = self._weights_dir / "Vision"
        if not vision_dir.exists():
            return None

        src = vision_dir / "dinov3-vitl16-pretrain-lvd1689m.safetensors"
        if not src.exists():
            return None

        # HuggingFace from_pretrained needs the weights file named model.safetensors
        dst = vision_dir / "model.safetensors"
        if not dst.exists():
            try:
                os.link(str(src), str(dst))
            except OSError:
                shutil.copy2(str(src), str(dst))

        # Write config.json if missing (derived from safetensors key shapes)
        config_json = vision_dir / "config.json"
        if not config_json.exists():
            cfg = {
                "model_type": "dinov3_vit",
                "architectures": ["DINOv3ViTModel"],
                "image_size": 224, "patch_size": 16, "num_channels": 3,
                "hidden_size": 1024, "intermediate_size": 4096,
                "num_hidden_layers": 24, "num_attention_heads": 16,
                "hidden_act": "gelu", "attention_dropout": 0.0,
                "layer_norm_eps": 1e-5, "layerscale_value": 1.0,
                "drop_path_rate": 0.0, "use_gated_mlp": False,
                "rope_theta": 100.0,
                "query_bias": True, "key_bias": False, "value_bias": True,
                "proj_bias": True, "mlp_bias": True,
                "num_register_tokens": 4,
                "pos_embed_shift": None, "pos_embed_jitter": None,
                "pos_embed_rescale": 2.0, "apply_layernorm": True,
                "reshape_hidden_states": True,
                "out_features": ["stage24"], "out_indices": [24],
                "stage_names": ["stem"] + [f"stage{i}" for i in range(1, 25)],
                "transformers_version": "5.2.0",
            }
            config_json.write_text(json.dumps(cfg, indent=2))
            print(f"[Trellis2] Wrote DINOv3 config.json to {vision_dir}")

        return str(vision_dir)

    @staticmethod
    def _find_system_ptxas(min_cuda_ver: tuple[int, int] = (0, 0)) -> str | None:
        """Return the path to the best ptxas.exe >= min_cuda_ver, or None.

        Search order:
          1. Standard CUDA Toolkit path (C:\\Program Files\\NVIDIA GPU Computing Toolkit\\...)
          2. CUDA_PATH environment variable
          3. triton-windows bundled ptxas (version detected via 'ptxas --version')
        """
        import glob as _glob
        import os as _os
        import re as _re
        import subprocess as _sp

        def _ver_from_path(path: str) -> tuple[int, int]:
            m = _re.search(r'CUDA[/\\]v(\d+)\.(\d+)', path, _re.IGNORECASE)
            return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

        def _ver_from_binary(path: str) -> tuple[int, int]:
            try:
                out = _sp.check_output(
                    [path, "--version"], stderr=_sp.STDOUT, text=True, timeout=10,
                )
                m = _re.search(r'release (\d+)\.(\d+)', out)
                if m:
                    return (int(m.group(1)), int(m.group(2)))
            except Exception:
                pass
            return (0, 0)

        # (version, path) pairs
        candidates: list[tuple[tuple[int, int], str]] = []

        # 1. Standard CUDA Toolkit location
        for p in _glob.glob(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\ptxas.exe"
        ):
            v = _ver_from_path(p)
            if v != (0, 0):
                candidates.append((v, p))

        # 2. CUDA_PATH env var
        cuda_path = _os.environ.get("CUDA_PATH", "")
        if cuda_path:
            p = _os.path.join(cuda_path, "bin", "ptxas.exe")
            if _os.path.isfile(p):
                v = _ver_from_path(p) or _ver_from_binary(p)
                candidates.append((v, p))

        # 3. triton-windows bundled ptxas (inside the triton package dir)
        try:
            import triton as _triton
            _triton_dir = Path(_triton.__file__).parent
            for _p in _triton_dir.rglob("ptxas.exe"):
                _v = _ver_from_binary(str(_p))
                candidates.append((_v, str(_p)))
        except Exception:
            pass

        # 4. ptxas on the system PATH (triton-windows may install it to venv/Scripts)
        try:
            import shutil as _shutil
            _ptxas_path = _shutil.which("ptxas.exe") or _shutil.which("ptxas")
            if _ptxas_path:
                _v = _ver_from_binary(_ptxas_path)
                candidates.append((_v, _ptxas_path))
        except Exception:
            pass

        eligible = [(v, p) for v, p in candidates if v >= min_cuda_ver]
        if not eligible:
            return None
        eligible.sort(key=lambda x: x[0], reverse=True)
        return eligible[0][1]

    def _apply_blackwell_patch(self, torch) -> None:
        """
        triton-windows 3.3.x bundles a ptxas that predates Blackwell (SM 12.x).
        It generates PTX correctly but the bundled ptxas cannot assemble it into
        a SM 12.x cubin -> "no kernel image available".

        Fix strategy (tried in order):
          1. Point TRITON_PTXAS_PATH at the user's CUDA Toolkit ptxas (12.8+),
             which does support SM 12.x.  Triton uses it automatically.
          2. Switch trellis2_gguf sparse-conv backend away from flex_gemm if an
             alternative backend is registered (avoids Triton entirely).
          3. Spoof Triton's SM detection to SM 9.0 so it generates SM 9.0 PTX;
             the CUDA driver may PTX-JIT it to native SM 12.x (last resort,
             not guaranteed to work).
        """
        import os as _os

        sm_major, sm_minor = torch.cuda.get_device_capability()
        if sm_major < 12:
            return

        print(f"[Trellis2] Blackwell GPU detected (SM {sm_major}.{sm_minor}).")

        # ── 1. System ptxas (CUDA Toolkit 12.8+) ─────────────────────────── #
        # sm_120 / sm_120a (Blackwell) require ptxas >= 12.8; CUDA 12.6 will crash
        # with "Value 'sm_120a' is not defined for option 'gpu-name'".
        if "TRITON_PTXAS_PATH" not in _os.environ:
            ptxas = self._find_system_ptxas(min_cuda_ver=(12, 8))
            if ptxas:
                _os.environ["TRITON_PTXAS_PATH"] = ptxas
                print(f"[Trellis2] Blackwell: TRITON_PTXAS_PATH={ptxas}")
            else:
                # Also log any ptxas that were found but are too old (diagnostic help)
                old_ptxas = self._find_system_ptxas(min_cuda_ver=(0, 0))
                if old_ptxas:
                    print(f"[Trellis2] Blackwell: found ptxas at '{old_ptxas}' but it is "
                          "older than 12.8 and does not support SM 12.x. "
                          "Install CUDA Toolkit 12.8+ for native Blackwell support.")
                else:
                    print("[Trellis2] Blackwell: no ptxas found anywhere (CUDA Toolkit, "
                          "CUDA_PATH, or triton-windows package). "
                          "Install CUDA Toolkit 12.8+ for native Blackwell support.")
        else:
            print(f"[Trellis2] Blackwell: TRITON_PTXAS_PATH already set to "
                  f"{_os.environ['TRITON_PTXAS_PATH']}")

        # ── 1b. Isolate Triton kernel cache for Blackwell ─────────────────── #
        # Kernels compiled for sm_90 (via the SM spoof below) are cached in
        # ~/.triton/cache/.  After updating triton-windows, those stale sm_90
        # cubins would be reused (same kernel hash, different arch) and fail on
        # sm_120.  Point Triton at a separate Blackwell-specific cache dir so it
        # always compiles fresh with whatever ptxas is available.
        if "TRITON_CACHE_DIR" not in _os.environ:
            try:
                _bw_cache = Path(_os.path.expanduser("~")) / ".triton" / "cache-sm120"
                _bw_cache.mkdir(parents=True, exist_ok=True)
                _os.environ["TRITON_CACHE_DIR"] = str(_bw_cache)
                print(f"[Trellis2] Blackwell: TRITON_CACHE_DIR={_bw_cache}")
            except Exception as _ce:
                print(f"[Trellis2] Blackwell: could not set TRITON_CACHE_DIR ({_ce}).")

        # ── 2. Try non-Triton sparse-conv backend ─────────────────────────── #
        try:
            import trellis2_gguf.modules.sparse.conv.conv as _conv_mod
            import trellis2_gguf.config as _t2cfg
            current = getattr(_t2cfg, "CONV", "flex_gemm")
            alts = [k for k in getattr(_conv_mod, "_backends", {}) if k != current]
            if alts:
                _t2cfg.CONV = alts[0]
                print(f"[Trellis2] Blackwell: switched sparse-conv backend "
                      f"'{current}' -> '{alts[0]}'")
                return
        except Exception as _e:
            print(f"[Trellis2] Blackwell: backend switch unavailable ({_e}).")

        # ── 2b. Warn if triton-windows version predates Blackwell support ──── #
        # triton-windows >= 3.3.1 bundles ptxas 12.8 with SM 12.x support.
        # Older versions (3.3.0.postN < post14, plain 3.3.0) do not.
        # If the user has an old version, suggest a Repair to upgrade it.
        try:
            import triton as _triton_ver
            import re as _re
            _m = _re.match(r"(\d+)\.(\d+)\.(\d+)", _triton_ver.__version__)
            if _m:
                _tv = tuple(int(x) for x in _m.groups())
                if _tv < (3, 3, 1):
                    print(
                        f"[Trellis2] Blackwell WARNING: triton-windows {_triton_ver.__version__} "
                        "is older than 3.3.1 and likely bundles a ptxas that does not support SM 12.x. "
                        "Run a Repair in Modly to upgrade triton-windows to >=3.3.1."
                    )
        except Exception:
            pass

        # ── 3. Spoof SM to 9.0 (last resort) ─────────────────────────────── #
        # In Triton 3.3.x the GPU arch fed to ptxas comes from get_current_target(),
        # not get_device_capability().  Patch both the Triton driver class method and
        # torch.cuda.get_device_capability (the root source) so that Triton emits
        # sm_90 PTX — CUDA driver JIT-compiles it to native sm_12x at runtime.
        _spoofed = False
        try:
            import triton.backends.nvidia.driver as _nv_drv

            # Approach A: patch get_current_target (Triton 3.3.x primary path)
            for _attr in dir(_nv_drv):
                _obj = getattr(_nv_drv, _attr)
                if isinstance(_obj, type) and hasattr(_obj, "get_current_target"):
                    _orig_gct = _obj.get_current_target
                    def _gct_spoof(self, *_a, **_kw):
                        tgt = _orig_gct(self, *_a, **_kw)
                        try:
                            if tgt is not None and int(getattr(tgt, "arch", 0)) >= 120:
                                return type(tgt)(tgt.backend, 90, tgt.warp_size)
                        except Exception:
                            pass
                        return tgt
                    _obj.get_current_target = _gct_spoof
                    print("[Trellis2] Blackwell: Triton get_current_target spoofed to SM 9.0.")
                    _spoofed = True
                    break

            # Approach B: older Triton — patch get_device_capability on the class
            if not _spoofed:
                for _attr in dir(_nv_drv):
                    _obj = getattr(_nv_drv, _attr)
                    if isinstance(_obj, type) and hasattr(_obj, "get_device_capability"):
                        _orig_cap = _obj.get_device_capability
                        def _cap_spoof(self, device=0):
                            major, minor = _orig_cap(self, device)
                            return (9, 0) if major >= 12 else (major, minor)
                        _obj.get_device_capability = _cap_spoof
                        print("[Trellis2] Blackwell: Triton get_device_capability spoofed to SM 9.0.")
                        _spoofed = True
                        break
                if not _spoofed and hasattr(_nv_drv, "get_device_capability"):
                    _orig_fn = _nv_drv.get_device_capability
                    def _fn_spoof(device=0):
                        major, minor = _orig_fn(device)
                        return (9, 0) if major >= 12 else (major, minor)
                    _nv_drv.get_device_capability = _fn_spoof
                    print("[Trellis2] Blackwell: Triton module get_device_capability spoofed to SM 9.0.")
                    _spoofed = True

        except Exception as _e:
            print(f"[Trellis2] Blackwell: Triton driver patch failed ({_e}).")

        # Approach C: patch torch.cuda.get_device_capability — the root source that
        # all Triton paths ultimately read.  Applied regardless of A/B success so
        # any future Triton code path is also covered.
        try:
            import torch.cuda as _torch_cuda
            _orig_tgdc = _torch_cuda.get_device_capability
            def _torch_gdc_spoof(device=None):
                cap = _orig_tgdc(device)
                return (9, 0) if (cap and cap[0] >= 12) else cap
            _torch_cuda.get_device_capability = _torch_gdc_spoof
            torch.cuda.get_device_capability = _torch_gdc_spoof
            print("[Trellis2] Blackwell: torch.cuda.get_device_capability spoofed to SM 9.0.")
            _spoofed = True
        except Exception as _e:
            print(f"[Trellis2] Blackwell: torch patch failed ({_e}).")

        if not _spoofed:
            print("[Trellis2] Blackwell: all SM-spoof patches failed. "
                  "Install CUDA Toolkit 12.8+ for native Blackwell support.")

    def _probe_cumesh_remesh(self) -> None:
        """
        Test cumesh.remeshing.remesh_narrow_band_dc with a tiny mesh.

        setup.py used to overwrite cumesh/remeshing.py with the trellis2_gguf patch,
        replacing the original function and leaving hashmap_vox undefined.
        If that's the case, try to locate hashmap_vox in cumesh's own namespace
        (or its C extension) and inject it — fixing existing installs without a Repair.
        Sets self._cumesh_remesh_ok so _export_geometry skips the expensive BVH
        build when remesh is known-broken.
        """
        import torch as _t
        self._cumesh_remesh_ok = False
        try:
            import cumesh as _c
            import cumesh.remeshing as _crm

            def _probe():
                # Unit cube — 8 verts, 12 tris (cuBVH requires >= 8 triangles)
                vp = _t.tensor([
                    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
                    [0,0,1],[1,0,1],[1,1,1],[0,1,1],
                ], dtype=_t.float32, device="cuda")
                fp = _t.tensor([
                    [0,2,1],[0,3,2],  # bottom
                    [4,5,6],[4,6,7],  # top
                    [0,1,5],[0,5,4],  # front
                    [2,3,7],[2,7,6],  # back
                    [0,4,7],[0,7,3],  # left
                    [1,2,6],[1,6,5],  # right
                ], dtype=_t.int32, device="cuda")
                bvh    = _c.cuBVH(vp, fp)
                center = vp.mean(0)
                scale  = (vp.max(0).values - vp.min(0).values).max().item() * 2.0
                _crm.remesh_narrow_band_dc(
                    vp, fp,
                    center=center, scale=scale,
                    resolution=8, band=1, project_back=0.5, bvh=bvh,
                )

            try:
                _probe()
                self._cumesh_remesh_ok = True
                print("[Trellis2] cumesh remesh: OK")
                return
            except NameError as ne:
                # Extract the missing symbol name from the NameError message
                msg = str(ne)
                sym = msg.split("'")[1] if "'" in msg else msg.split()[-1].strip("'\"")
                print(f"[Trellis2] cumesh remesh broken ({ne}) — attempting runtime fix...")

                # Search cumesh top-level and cumesh._C for the missing symbol
                fixed = False
                for ns_name, ns_obj in [("cumesh", _c)]:
                    for attr in dir(ns_obj):
                        if sym.lower() in attr.lower():
                            setattr(_crm, sym, getattr(ns_obj, attr))
                            print(f"[Trellis2] Injected {ns_name}.{attr} -> cumesh.remeshing.{sym}")
                            fixed = True
                            break
                    if fixed:
                        break

                if not fixed:
                    try:
                        import cumesh._C as _cc
                        # Exact match first, then partial
                        if hasattr(_cc, sym):
                            setattr(_crm, sym, getattr(_cc, sym))
                            fixed = True
                            print(f"[Trellis2] Injected cumesh._C.{sym} -> cumesh.remeshing.{sym}")
                        else:
                            for attr in dir(_cc):
                                if sym.lower() in attr.lower():
                                    setattr(_crm, sym, getattr(_cc, attr))
                                    fixed = True
                                    print(f"[Trellis2] Injected cumesh._C.{attr} -> cumesh.remeshing.{sym}")
                                    break
                    except ImportError:
                        pass

                if not fixed:
                    print(
                        "[Trellis2] Could not auto-fix cumesh remesh. "
                        "Click Repair on the Models page to reinstall the extension."
                    )
                    return

                # Re-test after injection
                try:
                    _probe()
                    self._cumesh_remesh_ok = True
                    print("[Trellis2] cumesh remesh: fixed and verified OK")
                except Exception as e2:
                    print(f"[Trellis2] cumesh remesh still broken after fix ({e2}). Click Repair.")

        except Exception as exc:
            print(f"[Trellis2] cumesh remesh probe failed ({exc})")

    def _load_pipeline(self, gguf_quant: str) -> None:
        import os
        from trellis2_gguf.pipelines import Trellis2ImageTo3DPipeline
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Trellis2GGUFGenerator] Loading pipeline (GGUF {gguf_quant}) on {device} ...")

        if device == "cuda":
            self._apply_blackwell_patch(torch)
            self._probe_cumesh_remesh()

        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            str(self._weights_dir),
            keep_models_loaded=False,  # free sub-models between stages to save VRAM
            enable_gguf=True,
            gguf_quant=gguf_quant,
            precision="bf16",
        )

        # The pipeline code always overrides image_cond_model's model_name to a
        # ComfyUI-style path under folder_paths.models_dir that doesn't exist in
        # our layout.  Fix it to point at the Vision/ directory we already have.
        if hasattr(pipeline, "_pretrained_args"):
            ic = pipeline._pretrained_args.get("image_cond_model", {})
            args = ic.get("args", {})
            model_name = args.get("model_name", "")
            if model_name and not os.path.isdir(model_name):
                local_dir = self._prepare_dinov3_dir()
                if local_dir:
                    args["model_name"] = local_dir
                    print(f"[Trellis2] DINOv3 -> {local_dir}")
                else:
                    # Last resort: let HF download from hub
                    args["model_name"] = "facebook/dinov3-vitl16-pretrain-lvd1689m"
                    print("[Trellis2] DINOv3 -> HuggingFace download fallback")

        # DINOv3ViTModel.from_pretrained loads weights on CPU by default, but
        # DinoV3FeatureExtractor.__call__ always sends the input tensor to CUDA.
        # Patch extract_features at the class level so the model follows the input.
        if device == "cuda":
            try:
                from trellis2_gguf.modules.image_feature_extractor import DinoV3FeatureExtractor
                _orig_extract = DinoV3FeatureExtractor.extract_features
                def _extract_on_input_device(self, image: "torch.Tensor"):
                    self.model.to(image.device)
                    return _orig_extract(self, image)
                DinoV3FeatureExtractor.extract_features = _extract_on_input_device
                print("[Trellis2] Patched DinoV3FeatureExtractor.extract_features for CUDA")
            except Exception as exc:
                print(f"[Trellis2] Warning: could not patch DinoV3FeatureExtractor: {exc}")

        # from_pretrained hardcodes pipeline._device = 'cpu' — override it.
        pipeline._device = device
        print(f"[Trellis2] Pipeline device set to: {device}")

        self._model      = pipeline
        self._device     = device
        self._gguf_quant = gguf_quant
        print(f"[Trellis2GGUFGenerator] Ready.")

    def unload(self) -> None:
        self._device     = None
        self._gguf_quant = None
        super().unload()

    # ------------------------------------------------------------------ #
    # Entry point                                                         #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        if self.model_dir.name == "refine":
            return self._run_refine(image_bytes, params, progress_cb, cancel_event)
        return self._run_generate(image_bytes, params, progress_cb, cancel_event)

    def _run_generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch

        # --- Reload if quantisation changed since last call ---
        gguf_quant = str(params.get("gguf_quant", "Q5_K_M"))
        if self._model is not None and getattr(self, "_gguf_quant", None) != gguf_quant:
            print(f"[Trellis2GGUFGenerator] gguf_quant changed -> reloading ({gguf_quant})")
            self.unload()
            self._ensure_trellis2_gguf()
            self._load_pipeline(gguf_quant)

        pipeline_type     = str(params.get("pipeline_type",     "1024_cascade"))
        ss_steps          = int(params.get("ss_steps",          25))
        slat_steps        = int(params.get("slat_steps",        25))
        fg_ratio          = float(params.get("foreground_ratio", 0.85))
        remesh_resolution = int(params.get("remesh_resolution", 768))
        seed              = self._resolve_seed(params)

        # --- Pre-process image ---
        self._report(progress_cb, 5, "Removing background...")
        image_pil = self._preprocess(image_bytes, fg_ratio)
        self._check_cancelled(cancel_event)

        # --- Run pipeline ---
        self._report(progress_cb, 10, "Running Trellis.2 pipeline...")
        stop_evt = threading.Event()
        if progress_cb:
            total_steps = ss_steps + slat_steps
            threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 10, 84, "Diffusion stages...", stop_evt, float(max(total_steps, 1))),
                daemon=True,
            ).start()

        try:
            with torch.no_grad():
                mesh_with_voxel = self._model.run(
                    image=image_pil,
                    seed=seed,
                    pipeline_type=pipeline_type,
                    max_num_tokens=_MAX_NUM_TOKENS,
                    sparse_structure_sampler_params={
                        "steps":              ss_steps,
                        "guidance_strength":  _SS_CFG,
                        "guidance_rescale":   _SS_RESCALE,
                        "guidance_interval":  _SS_INTERVAL,
                        "rescale_t":          _SS_RESCALE_T,
                    },
                    shape_slat_sampler_params={
                        "steps":              slat_steps,
                        "guidance_strength":  _SLAT_CFG,
                        "guidance_rescale":   _SLAT_RESCALE,
                        "guidance_interval":  _SLAT_INTERVAL,
                        "rescale_t":          _SLAT_RESCALE_T,
                    },
                    generate_texture_slat=False,
                )[0]
        except RuntimeError as _exc:
            _msg = str(_exc)
            if "no kernel image" in _msg or ("CUDA error" in _msg and "kernel" in _msg.lower()):
                raise RuntimeError(
                    "GPU kernel error — your RTX 50-series (Blackwell) GPU requires a "
                    "ptxas compiler that supports SM 12.x.\n\n"
                    "Fix: install CUDA Toolkit 12.8 or newer from "
                    "https://developer.nvidia.com/cuda-downloads\n"
                    "then restart Modly. The toolkit's ptxas will be picked up automatically.\n\n"
                    f"Original error: {_msg}"
                ) from _exc
            raise
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        # --- Export ---
        self._report(progress_cb, 92, "Exporting mesh...")
        import torch as _torch_gc; _torch_gc.cuda.empty_cache()
        path = self._export_geometry(mesh_with_voxel, remesh_resolution)

        self._report(progress_cb, 100, "Done")
        return path

    def _run_refine(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """Texture an existing GLB mesh using Trellis2's native SLaT texture pipeline."""
        import torch
        import trimesh

        mesh_path = str(params.get("mesh_path", "")).strip()
        if not mesh_path:
            raise ValueError("mesh_path is required for the Texture Mesh node")

        texture_resolution = int(params.get("texture_resolution", 1024))
        texture_size       = int(params.get("texture_size",       2048))
        texture_steps      = int(params.get("texture_steps",      12))
        texture_guidance   = float(params.get("texture_guidance", 1.0))
        fg_ratio           = float(params.get("foreground_ratio", 0.85))
        seed               = self._resolve_seed(params)

        # Resolve workspace-relative or /workspace/... paths to absolute filesystem paths.
        # The workflow runner passes a relative path ("Workflows/file.glb") for workspace
        # meshes; the Generate page UI passes "/workspace/Default/file.glb".
        # outputs_dir = WORKSPACE_DIR/collection, so its parent is WORKSPACE_DIR.
        workspace_dir = self.outputs_dir.parent
        mesh_p = Path(mesh_path)
        if mesh_path.startswith("/workspace/"):
            mesh_path = str(workspace_dir / mesh_path[len("/workspace/"):])
        elif not mesh_p.is_absolute():
            mesh_path = str(workspace_dir / mesh_path)

        # --- Load input mesh ---
        self._report(progress_cb, 3, "Loading mesh...")
        mesh = trimesh.load(mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Could not load a valid mesh from: {mesh_path}")
        print(f"[Trellis2GGUFGenerator] Loaded mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        # --- Pre-process image ---
        # force_cpu=True: the Trellis2 pipeline is already loaded in VRAM.
        # rembg's onnxruntime CUDA provider crashes with error 700 under VRAM
        # pressure, corrupting the PyTorch CUDA context for all subsequent calls.
        self._report(progress_cb, 8, "Removing background...")
        image_pil = self._preprocess(image_bytes, fg_ratio, force_cpu=True)
        self._check_cancelled(cancel_event)

        # --- Run texture pipeline ---
        self._report(progress_cb, 12, "Encoding shape SLaT...")
        stop_evt = threading.Event()
        if progress_cb:
            threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 12, 90, "SLaT texture diffusion...", stop_evt, float(texture_steps)),
                daemon=True,
            ).start()

        try:
            with torch.no_grad():
                out_mesh, _, _ = self._model.texture_mesh(
                    mesh=mesh,
                    image=image_pil,
                    seed=seed,
                    tex_slat_sampler_params={
                        "steps":             texture_steps,
                        "guidance_strength": texture_guidance,
                    },
                    resolution=texture_resolution,
                    texture_size=texture_size,
                )
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        # --- Export ---
        self._report(progress_cb, 92, "Exporting textured mesh...")
        import torch as _torch_gc; _torch_gc.cuda.empty_cache()

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_textured.glb"
        path = self.outputs_dir / name
        out_mesh.export(str(path))

        self._report(progress_cb, 100, "Done")
        return path

    # ------------------------------------------------------------------ #
    # Export — geometry only (always available)                           #
    # ------------------------------------------------------------------ #

    def _export_geometry(self, mesh_with_voxel, remesh_resolution: int = 768) -> Path:
        """Export raw vertices + faces as GLB, no texture."""
        import trimesh
        import numpy as np

        verts = mesh_with_voxel.vertices
        faces = mesh_with_voxel.faces

        if hasattr(verts, "cpu"):
            verts = verts.cpu().numpy()
        if hasattr(faces, "cpu"):
            faces = faces.cpu().numpy()

        verts = np.asarray(verts, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)

        # Robust cleanup: same pipeline as textured export (cumesh > manual fallback).
        # flexible_dual_grid_to_mesh produces ~50% inward-facing quads; cumesh's
        # unify_face_orientations propagates consistent winding across the manifold,
        # which is far more reliable than the centroid heuristic for complex shapes.
        _cumesh_ok = False
        try:
            import torch as _torch
            import cumesh as _cumesh
            from cumesh import CuMesh
            _vt = _torch.from_numpy(verts).float().cuda().contiguous()
            _ft = _torch.from_numpy(faces).int().cuda().contiguous()

            cm = CuMesh()
            cm.init(_vt, _ft)

            # Holes in the Flexible Dual Grid mesh are structural: a face is only created
            # when all 4 voxel neighbors of an intersected edge exist in the sparse structure.
            # Missing boundary voxels = holes that fill_holes cannot reliably fix.
            #
            # Solution: narrow-band dual contouring remesh (same as to_glb remesh=True in
            # o_voxel/postprocess.py). Builds a BVH on the holey mesh, then runs dual
            # contouring on a narrow band around the surface to reconstruct clean topology.
            try:
                if not getattr(self, "_cumesh_remesh_ok", True):
                    raise RuntimeError("cumesh remesh known-broken — skipping to fallback")
                # Free pipeline VRAM before remesh — generation leaves tensors in cache
                _torch.cuda.empty_cache()
                bvh = _cumesh.cuBVH(_vt, _ft)
                aabb_min = _vt.min(dim=0).values
                aabb_max = _vt.max(dim=0).values
                center = (aabb_min + aabb_max) / 2.0
                scale = (aabb_max - aabb_min).max().item()
                resolution = remesh_resolution
                band = 2 if remesh_resolution >= 768 else 1
                remesh_scale = (resolution + 3 * band) / resolution * scale
                _rv, _rf = _cumesh.remeshing.remesh_narrow_band_dc(
                    _vt, _ft,
                    center=center,
                    scale=remesh_scale,
                    resolution=resolution,
                    band=band,
                    project_back=0.9,
                    bvh=bvh,
                )
                # Thin features (spikes, fins) may not be fully enclosed by dual
                # contouring — apply pymeshlab hole-filling before loading into CuMesh.
                try:
                    import pymeshlab as _pml
                    ms = _pml.MeshSet()
                    ms.add_mesh(_pml.Mesh(
                        vertex_matrix=_rv.cpu().numpy().astype(np.float64),
                        face_matrix=_rf.cpu().numpy(),
                    ))
                    ms.meshing_remove_duplicate_faces()
                    ms.meshing_repair_non_manifold_edges()
                    ms.meshing_close_holes(maxholesize=200)
                    _pm = ms.current_mesh()
                    _rv = _torch.from_numpy(_pm.vertex_matrix().astype(np.float32)).cuda().contiguous()
                    _rf = _torch.from_numpy(_pm.face_matrix().astype(np.int32)).cuda().contiguous()
                except Exception:
                    pass
                cm.init(_rv, _rf)
                print("[Trellis2GGUFGenerator] Remesh OK")
            except Exception as remesh_exc:
                # Root cause: setup.py's _apply_patches used to overwrite cumesh's
                # own remeshing.py with trellis2_gguf's patch, which references
                # hashmap_vox without importing it (NameError). Fixed in setup.py;
                # reinstalling the extension (Repair) restores cumesh's remeshing.py.
                print(f"[Trellis2GGUFGenerator] cumesh remesh unavailable ({remesh_exc}), using fallback...")
                _torch.cuda.empty_cache()
                # Try pymeshlab for hole-filling — handles structural holes better
                # than CuMesh.fill_holes for FDG meshes with missing boundary voxels.
                try:
                    import pymeshlab as _pml
                    ms = _pml.MeshSet()
                    ms.add_mesh(_pml.Mesh(
                        vertex_matrix=_vt.cpu().numpy().astype(np.float64),
                        face_matrix=_ft.cpu().numpy(),
                    ))
                    ms.meshing_remove_duplicate_faces()
                    ms.meshing_repair_non_manifold_edges()
                    ms.meshing_close_holes(maxholesize=100)
                    _pm = ms.current_mesh()
                    _vt = _torch.from_numpy(_pm.vertex_matrix().astype(np.float32)).cuda().contiguous()
                    _ft = _torch.from_numpy(_pm.face_matrix().astype(np.int32)).cuda().contiguous()
                    print("[Trellis2GGUFGenerator] pymeshlab repair applied")
                except Exception as pml_exc:
                    print(f"[Trellis2GGUFGenerator] pymeshlab unavailable ({pml_exc}), using fill_holes")
                cm = CuMesh()
                cm.init(_vt, _ft)
                cm.fill_holes(max_hole_perimeter=3e-2)
                cm.remove_duplicate_faces()
                cm.repair_non_manifold_edges()
                cm.remove_small_connected_components(1e-5)
                cm.fill_holes(max_hole_perimeter=0.1)
                cm.repair_non_manifold_edges()

            cm.unify_face_orientations()
            _vout, _fout = cm.read()
            verts = _vout.cpu().numpy().astype(np.float32)
            faces = _fout.cpu().numpy().astype(np.int32)
            _cumesh_ok = True
        except Exception as exc:
            print(f"[Trellis2GGUFGenerator] cumesh cleanup failed ({exc}), using centroid winding fix")
            v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
            face_normals   = np.cross(v1 - v0, v2 - v0)
            face_centroids = (v0 + v1 + v2) / 3.0
            mesh_center    = verts.mean(axis=0)
            dot = (face_normals * (face_centroids - mesh_center)).sum(axis=1)
            inward = dot < 0
            faces[inward] = faces[inward][:, ::-1]

        if _cumesh_ok:
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]
            fn = np.cross(v1 - v0, v2 - v0)
            fc = (v0 + v1 + v2) / 3.0
            mc = verts.mean(axis=0)
            dot = (fn * (fc - mc)).sum(axis=1)
            if np.mean(dot < 0) > 0.5:
                faces = faces[:, ::-1]

        # Convert from TRELLIS (Z-up, front at +Y) to GLB/Three.js (Y-up, front at +Z).
        # Steps: rotate -90° around X (Z-up -> Y-up), then 180° around new Y (front +Y -> +Z).
        # Net: x->-x, y<-z_trellis, z<-y_trellis  (det=+1, no winding change).
        _x = verts[:, 0].copy()
        _y = verts[:, 1].copy()
        _z = verts[:, 2].copy()
        verts[:, 0] = -_x
        verts[:, 1] = _z
        verts[:, 2] = _y

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name

        # process=True: removes degenerate (zero-area) and duplicate faces, merges duplicate
        # vertices at grid boundaries — the main source of small "dot" holes.
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        # doubleSided=True ensures any face whose normal still points inward is rendered
        # from both sides, preventing see-through artifacts.
        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(doubleSided=True)
        )
        mesh.export(str(path))
        return path

    # ------------------------------------------------------------------ #
    # Vendor / source setup                                               #
    # ------------------------------------------------------------------ #

    def _ensure_comfyui_gguf(self) -> None:
        """
        Ensure ComfyUI-GGUF (city96) files are present at the path that trellis2_gguf's
        _setup_native_gguf() searches.  Without these, GGUF dequant falls back to CPU.
        """
        import os, urllib.request
        from pathlib import Path

        utils_dir = Path(_EXTENSION_DIR) / "venv" / "Lib" / "site-packages" / "trellis2_gguf" / "utils"
        gguf_dir  = (utils_dir / ".." / ".." / ".." / "ComfyUI-GGUF").resolve()

        _FILES = ["ops.py", "dequant.py", "loader.py"]
        if all((gguf_dir / f).exists() for f in _FILES):
            return

        gguf_dir.mkdir(parents=True, exist_ok=True)
        base = "https://raw.githubusercontent.com/city96/ComfyUI-GGUF/main/"
        for fname in _FILES:
            dest = gguf_dir / fname
            if dest.exists():
                continue
            print(f"[Trellis2] Downloading ComfyUI-GGUF/{fname} …")
            try:
                urllib.request.urlretrieve(base + fname, str(dest))
            except Exception as exc:
                print(f"[Trellis2] Warning: could not download {fname}: {exc}")
        print(f"[Trellis2] ComfyUI-GGUF installed at {gguf_dir}")

    def _ensure_trellis2_gguf(self) -> None:
        """Assert that trellis2_gguf is importable from the venv (installed by setup.py)."""
        import sys
        import types
        import torch  # noqa — registers CUDA DLLs on Windows before any CUDA extension

        self._ensure_comfyui_gguf()

        # trellis2_gguf is a ComfyUI extension; stub ComfyUI-specific modules
        # so it can be used standalone.
        def _stub(name: str, **attrs):
            if name not in sys.modules:
                m = types.ModuleType(name)
                for k, v in attrs.items():
                    setattr(m, k, v)
                sys.modules[name] = m

        models_dir = str(self._weights_dir.parent)
        _stub("folder_paths",
              models_dir=models_dir,
              get_filename_list=lambda *a, **kw: [],
              get_full_path=lambda *a, **kw: None,
              get_input_directory=lambda: "",
              get_output_directory=lambda: "")

        class _ProgressBar:
            def __init__(self, total=100): pass
            def update(self, val=1): pass
            def update_absolute(self, val, total=None, preview=None): pass

        _stub("comfy.utils", ProgressBar=_ProgressBar)
        _stub("comfy", utils=sys.modules["comfy.utils"])

        # trellis2_gguf/models/__init__.py dynamically loads model_manager.py from
        # site-packages (a ComfyUI-specific file).  Pre-inject a stub so the file
        # lookup is skipped entirely.
        if "trellis2_model_manager" not in sys.modules:
            import glob as _glob
            import os as _os

            _search_root = models_dir  # e.g. D:\ModlyModels\models

            def _resolve_local_path(basename, enable_gguf=False, gguf_quant="Q8_0", precision=None):
                if enable_gguf:
                    pattern = _os.path.join(_search_root, "**", f"{basename}_{gguf_quant}.gguf")
                    hits = _glob.glob(pattern, recursive=True)
                    if hits:
                        model_file  = hits[0]
                        config_file = model_file.replace(f"_{gguf_quant}.gguf", ".json")
                        if not _os.path.exists(config_file):
                            config_file = _os.path.join(_os.path.dirname(model_file), basename + ".json")
                        return config_file, model_file, True
                suf     = f"_{precision}" if precision else ""
                pattern = _os.path.join(_search_root, "**", f"{basename}{suf}.safetensors")
                hits    = _glob.glob(pattern, recursive=True)
                matched_suf = suf
                if not hits:
                    pattern = _os.path.join(_search_root, "**", f"{basename}.safetensors")
                    hits    = _glob.glob(pattern, recursive=True)
                    matched_suf = ""
                if hits:
                    model_file  = hits[0]
                    config_file = model_file.replace(f"{matched_suf}.safetensors", ".json")
                    if not _os.path.exists(config_file):
                        config_file = _os.path.join(_os.path.dirname(model_file), basename + ".json")
                    return config_file, model_file, False
                raise FileNotFoundError(
                    f"[Trellis2] Cannot resolve model: {basename} "
                    f"(gguf={enable_gguf}, quant={gguf_quant}, precision={precision}) "
                    f"in {_search_root}"
                )

            mm = types.ModuleType("trellis2_model_manager")
            mm.resolve_local_path  = _resolve_local_path
            mm.ensure_model_files  = lambda: None
            sys.modules["trellis2_model_manager"] = mm
            print(f"[Trellis2] Injected trellis2_model_manager stub (search root: {_search_root})")

        # huggingface_hub >=0.24 validates repo IDs strictly and rejects
        # Windows absolute paths.  Patch it to allow local paths through.
        try:
            import huggingface_hub.utils._validators as _hf_val
            _orig_validate = _hf_val.validate_repo_id
            import os as _os
            def _patched_validate(repo_id, *args, **kwargs):
                if _os.path.isabs(repo_id) or _os.path.exists(repo_id):
                    return
                _orig_validate(repo_id, *args, **kwargs)
            _hf_val.validate_repo_id = _patched_validate
        except Exception:
            pass

        try:
            from trellis2_gguf.pipelines import Trellis2ImageTo3DPipeline  # noqa
        except ImportError as exc:
            raise RuntimeError(
                "[Trellis2GGUFGenerator] trellis2_gguf not found. "
                "Click Repair on the Models page to re-run setup.py."
            ) from exc

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _resolve_seed(self, params: dict) -> int:
        seed = int(params.get("seed", -1))
        return random.randint(0, 2**31 - 1) if seed == -1 else seed

    def _preprocess(self, image_bytes: bytes, fg_ratio: float, force_cpu: bool = False):
        """Background removal (rembg) + foreground crop.

        force_cpu=True avoids rembg using the CUDA onnxruntime provider, which
        corrupts the PyTorch CUDA context when the Trellis2 pipeline is already
        loaded and VRAM is under pressure (error 700 propagates to all subsequent
        torch.cuda calls).
        """
        import numpy as np
        from PIL import Image as PILImage

        image = PILImage.open(io.BytesIO(image_bytes)).convert("RGBA")

        try:
            import rembg
            if force_cpu:
                session = rembg.new_session(providers=["CPUExecutionProvider"])
                image   = rembg.remove(image, session=session)
            else:
                try:
                    session = rembg.new_session()
                    image   = rembg.remove(image, session=session)
                except Exception:
                    session = rembg.new_session(providers=["CPUExecutionProvider"])
                    image   = rembg.remove(image, session=session)
        except Exception as exc:
            print(f"[Trellis2GGUFGenerator] Background removal skipped: {exc}")

        # Composite on white background
        bg = PILImage.new("RGBA", image.size, (255, 255, 255, 255))
        bg.paste(image, mask=image.split()[3])
        image = bg.convert("RGB")

        return self._resize_foreground(image, fg_ratio)

    def _resize_foreground(self, image, ratio: float):
        import numpy as np
        from PIL import Image as PILImage

        arr  = np.array(image)
        mask = ~np.all(arr >= 250, axis=-1)
        if not mask.any():
            return image

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        fg     = image.crop((cmin, rmin, cmax + 1, rmax + 1))
        fw, fh = fg.size
        iw, ih = image.size
        scale  = ratio * min(iw, ih) / max(fw, fh)
        nw     = max(1, int(fw * scale))
        nh     = max(1, int(fh * scale))
        fg     = fg.resize((nw, nh), PILImage.LANCZOS)

        result = PILImage.new("RGB", (iw, ih), (255, 255, 255))
        result.paste(fg, ((iw - nw) // 2, (ih - nh) // 2))
        return result
