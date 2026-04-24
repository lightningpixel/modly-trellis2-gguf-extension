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

    def _apply_blackwell_patch(self, torch) -> None:
        """
        triton-windows 3.3.x (the version paired with torch 2.7) compiles Triton JIT
        kernels targeting SM 9.0 (Hopper) when it encounters an unknown SM like 12.x
        (Blackwell).  The resulting cubin cannot load on SM 12.x → "no kernel image".

        Workaround: patch Triton's nvidia driver to report SM 9.0 for Blackwell devices.
        The CUDA driver will then PTX-JIT the kernel to native SM 12.x code at first run
        (forward PTX compatibility).  Slightly slower first inference, but functional.

        Also attempts to switch trellis2_gguf's sparse-conv backend away from flex_gemm
        if an alternative is registered (e.g. spconv), which avoids Triton entirely.
        """
        sm_major, sm_minor = torch.cuda.get_device_capability()
        if sm_major < 12:
            return

        print(f"[Trellis2] Blackwell GPU detected (SM {sm_major}.{sm_minor}).")

        # ── 1. Try non-Triton sparse-conv backend first ──────────────────── #
        try:
            import trellis2_gguf.modules.sparse.conv.conv as _conv_mod
            import trellis2_gguf.config as _t2cfg
            current = getattr(_t2cfg, "CONV", "flex_gemm")
            alts = [k for k in getattr(_conv_mod, "_backends", {}) if k != current]
            if alts:
                _t2cfg.CONV = alts[0]
                print(f"[Trellis2] Blackwell: switched sparse-conv backend '{current}' -> '{alts[0]}'")
                return
        except Exception as _e:
            print(f"[Trellis2] Blackwell: backend switch unavailable ({_e}), trying Triton SM patch.")

        # ── 2. Patch Triton SM detection: report SM 9.0 for SM 12.x ──────── #
        # triton-windows 3.3 knows SM 9.0 (Hopper) and generates valid PTX for it.
        # CUDA's forward PTX compatibility then JIT-compiles to native SM 12.x.
        try:
            import triton.backends.nvidia.driver as _nv_drv
            _cls = None
            for attr in dir(_nv_drv):
                obj = getattr(_nv_drv, attr)
                if isinstance(obj, type) and hasattr(obj, "get_device_capability"):
                    _cls = obj
                    break
            if _cls is not None:
                _orig_cap = _cls.get_device_capability
                def _cap_hopper_fallback(self, device=0):
                    major, minor = _orig_cap(self, device)
                    if major >= 12:
                        return (9, 0)
                    return (major, minor)
                _cls.get_device_capability = _cap_hopper_fallback
                print("[Trellis2] Blackwell: Triton SM patched to 9.0 (PTX forward-compat).")
            else:
                # Newer triton may expose it as a module-level function
                if hasattr(_nv_drv, "get_device_capability"):
                    _orig_fn = _nv_drv.get_device_capability
                    def _fn_fallback(device=0):
                        major, minor = _orig_fn(device)
                        if major >= 12:
                            return (9, 0)
                        return (major, minor)
                    _nv_drv.get_device_capability = _fn_fallback
                    print("[Trellis2] Blackwell: Triton SM fn patched to 9.0.")
        except Exception as _e:
            print(f"[Trellis2] Blackwell: Triton SM patch failed ({_e}). "
                  "Generation may fail — a triton-windows update for SM 12.x is required.")

    def _load_pipeline(self, gguf_quant: str) -> None:
        import os
        from trellis2_gguf.pipelines import Trellis2ImageTo3DPipeline
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Trellis2GGUFGenerator] Loading pipeline (GGUF {gguf_quant}) on {device} ...")

        if device == "cuda":
            self._apply_blackwell_patch(torch)

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
        self._report(progress_cb, 8, "Removing background...")
        image_pil = self._preprocess(image_bytes, fg_ratio)
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
                cm.init(*_cumesh.remeshing.remesh_narrow_band_dc(
                    _vt, _ft,
                    center=center,
                    scale=remesh_scale,
                    resolution=resolution,
                    band=band,
                    project_back=0.9,
                    bvh=bvh,
                ))
                # No simplification after remesh — uniform triangles don't need it
                # and simplification on remeshed topology reintroduces artifacts.
                print("[Trellis2GGUFGenerator] Remesh OK")
            except Exception as remesh_exc:
                print(f"[Trellis2GGUFGenerator] Remesh unavailable ({remesh_exc}), falling back to fill_holes")
                # OOM leaves CUDA in corrupted state — reset before any further ops
                _torch.cuda.empty_cache()
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

    def _preprocess(self, image_bytes: bytes, fg_ratio: float):
        """Background removal (rembg) + foreground crop."""
        import numpy as np
        from PIL import Image as PILImage

        image = PILImage.open(io.BytesIO(image_bytes)).convert("RGBA")

        try:
            import rembg
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
