"""One combined input node for the Qwen pipeline (UI in js/qwen_inputs.js).

`QwenInputs`        reference image OR a 3D model from the library, an optional saved/imported pose and optional typed
                    reference images (outfit, background, same character, ...). Outputs a bundle + the prompt with reference notes.
`QwenInputsUnpack`  unpacks the bundle inside a subgraph.
Routes: /midfr/library, /midfr/input_images, /midfr/import_model (GLB/GLTF/OBJ/STL + their texture files), thumbnails."""
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time

import numpy as np
import torch
from aiohttp import web

import folder_paths
from comfy_api.latest import Types
from server import PromptServer

from .character_lib import LIB_DIR, _lib_names, _view_raster, _cam_basis
from .pipeline_nodes import POSE_DIR, _pose_names

REF_TYPES = {
    "Outfit": "Dress the character in the outfit shown in <image{n}>.",
    "Background": "Place the character in the scene / background shown in <image{n}>.",
    "Same Character": "<image{n}> shows the same character from another angle; keep her identity, face, hair and outfit consistent with it.",
    "Other Character": "<image{n}> shows another character; include them with the main character and keep their identity.",
    "Face": "Use the face and identity of the person in <image{n}>.",
    "Hairstyle": "Use the hairstyle and hair colour shown in <image{n}>.",
    "Object / Prop": "Include the object shown in <image{n}>.",
    "Style": "Match the art style, colours and lighting of <image{n}>.",
    "Lighting": "Light the scene like <image{n}>.",
}
MAX_REFS = 3
BLANK = lambda: torch.zeros(1, 64, 64, 3)


def _load_img(name):
    from PIL import Image, ImageOps
    p = folder_paths.get_annotated_filepath(name)
    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None]


def _file_stamp(name):
    try:
        return str(os.path.getmtime(folder_paths.get_annotated_filepath(name)))
    except Exception:
        return "0"


def _load_entry(name):
    d = os.path.join(LIB_DIR, name)
    f = torch.load(os.path.join(d, "mesh.pt"), map_location="cpu", weights_only=False)
    m = Types.MESH(f.pop("vertices"), f.pop("faces"))
    for k, v in f.items():
        setattr(m, k, v)
    cp = os.path.join(d, "character.pt")
    ch = torch.load(cp, map_location="cpu", weights_only=False) if os.path.isfile(cp) else None
    sp = os.path.join(d, "source.png")
    src = None
    if os.path.isfile(sp):
        from PIL import Image
        src = torch.from_numpy(np.asarray(Image.open(sp).convert("RGB"), dtype=np.float32) / 255.0)[None]
    meta = {}
    if os.path.isfile(os.path.join(d, "meta.json")):
        meta = json.load(open(os.path.join(d, "meta.json")))
    return m, ch, src, meta


class QwenInputs:
    CATEGORY = "invoke_inpainting/pipeline"
    FUNCTION = "run"
    RETURN_TYPES = ("QWEN_INPUTS", "IMAGE", "STRING")
    RETURN_NAMES = ("inputs", "image", "prompt")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"config": ("STRING", {"default": "{}", "multiline": False})},
                "optional": {"prompt": ("STRING", {"forceInput": True})}}

    @classmethod
    def IS_CHANGED(cls, config, prompt=""):
        try:
            c = json.loads(config)
        except Exception:
            return config
        h = hashlib.md5(config.encode()).hexdigest()
        for n in [c.get("image"), c.get("pose_image")] + [r.get("image") for r in c.get("refs", [])]:
            if n:
                h += _file_stamp(n)
        for d, k in ((LIB_DIR, "model"), (POSE_DIR, "pose")):
            if c.get(k):
                p = os.path.join(d, c[k], "mesh.pt" if d == LIB_DIR else "pose.pt")
                h += str(os.path.getmtime(p)) if os.path.isfile(p) else "0"
        return h + (prompt or "")

    def run(self, config, prompt=""):
        c = json.loads(config or "{}")
        b = dict(use_model=False, mesh=None, character=None, name="", needs_rig=False, needs_texture=False,
                 pose_image=BLANK(), use_pose=False, saved_pose=None, use_saved=False, pose_name="", skip_pose_save=True,
                 refs=[None] * MAX_REFS, angles=[None] * MAX_REFS, angle_prompt="")
        main = _load_img(c["image"]) if c.get("image") else None
        image = main
        if c.get("use_model"):
            if not c.get("model") or c["model"] not in _lib_names():
                raise ValueError("Qwen Inputs: 'Use 3D Model' is on but no model is selected")
            mesh, ch, src, meta = _load_entry(c["model"])
            b.update(use_model=True, mesh=mesh, character=ch, name=c["model"], needs_rig=ch is None,
                     needs_texture=getattr(mesh, "texture", None) is None)
            image = src if src is not None else (main if main is not None else BLANK())
            if b["needs_texture"] and main is None and src is None:
                print("[Qwen Inputs] imported model has no texture and no reference image: it will be untextured")
        if image is None:
            raise ValueError("Qwen Inputs: choose a reference image (or switch on 'Use 3D Model')")
        if c.get("use_pose"):
            if c.get("pose_from") == "saved":
                f = os.path.join(POSE_DIR, c.get("pose", ""), "pose.pt")
                if not os.path.isfile(f):
                    raise ValueError(f"Qwen Inputs: saved pose '{c.get('pose')}' not found")
                b.update(use_pose=True, use_saved=True, saved_pose=torch.load(f, map_location="cpu", weights_only=False), pose_name=c["pose"])
            else:
                if not c.get("pose_image"):
                    raise ValueError("Qwen Inputs: 'Use Saved Pose' is on but no pose image/pose is chosen")
                b.update(use_pose=True, pose_image=_load_img(c["pose_image"]), pose_name=os.path.splitext(os.path.basename(c["pose_image"]))[0],
                         skip_pose_save=not c.get("save_pose", True))
        notes, k, angles = [], 0, []
        if c.get("use_refs"):
            for r in c.get("refs", [])[:MAX_REFS]:
                if not r.get("image"):
                    continue
                t = r.get("type", "Outfit")
                slot = k + 2  # image_1 is the main reference
                b["refs"][k] = _load_img(r["image"])
                notes.append(REF_TYPES.get(t, "Use <image{n}> as a reference.").format(n=slot))
                if t == "Same Character":
                    b["angles"][len(angles)] = b["refs"][k]
                    angles.append(len(angles) + 2)
                k += 1
        if angles:
            b["angle_prompt"] = " ".join(f"<image{n}>" for n in angles) + (" show" if len(angles) > 1 else " shows") + " the same character from other angles; keep her identical."
        b["image"] = image
        b["prompt"] = ((prompt or "").strip() + (" " + " ".join(notes) if notes else "")).strip()
        return (b, image, b["prompt"])


class QwenInputsUnpack:
    CATEGORY = "invoke_inpainting/pipeline"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MESH", "MHR_POSE_DATA", "BOOLEAN", "STRING", "BOOLEAN", "BOOLEAN",
                    "IMAGE", "BOOLEAN", "MHR_POSE_DATA", "BOOLEAN", "STRING", "BOOLEAN",
                    "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "mesh", "character", "use_model", "name", "needs_rig", "needs_texture",
                    "pose_image", "use_pose", "saved_pose", "use_saved", "pose_name", "skip_pose_save",
                    "ref_1", "ref_2", "ref_3", "angle_1", "angle_2", "angle_3", "angle_prompt", "prompt")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"inputs": ("QWEN_INPUTS",)}}

    def run(self, inputs):
        b = inputs
        return (b["image"], b["mesh"], b["character"], b["use_model"], b["name"], b["needs_rig"], b["needs_texture"],
                b["pose_image"], b["use_pose"], b["saved_pose"], b["use_saved"], b["pose_name"], b["skip_pose_save"],
                *b["refs"], *b["angles"], b["angle_prompt"], b["prompt"])


# ----------------------------------------------------------------------------- 3D model import
def _rot_y(v, deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.stack([v[:, 0] * c + v[:, 2] * s, v[:, 1], -v[:, 0] * s + v[:, 2] * c], 1)


def _orient(v, up, facing):
    if up == "Z":
        v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)
    deg = {"+Z": 0, "-Z": 180, "+X": -90, "-X": 90}.get(facing, 0)
    return _rot_y(v, deg) if deg else v


def _auto_up(v):
    e = v.max(0) - v.min(0)
    return "Z" if e[2] > 1.5 * max(e[1], e[0]) else "Y"


def _read_model(files, main):
    from comfy_extras.mesh3d.fileio import gltf_read, mesh_file_read
    from comfy_extras.nodes_mesh_io import _merge_primitives
    data = files[main]
    ext = os.path.splitext(main)[1].lower()
    tmp = tempfile.mkdtemp(prefix="midfr_import_")
    try:
        for n, d in files.items():
            with open(os.path.join(tmp, os.path.basename(n)), "wb") as f:
                f.write(d)
        info = {}
        if ext in (".glb", ".gltf"):
            gltf, buffers, prims = gltf_read.load_gltf(data, tmp, lambda *a: None)
            if not prims:
                raise ValueError("no triangle geometry in the glTF")
            mi = [p["material"] for p in prims if p["material"] is not None]
            info = gltf_read.extract_material(gltf, buffers, tmp, mi[0] if mi else None, lambda *a: None) or {}
        elif ext == ".obj":
            prims = [mesh_file_read.load_obj(data)]
            txt = data.decode("utf-8", "ignore")
            mt = re.search(r"^mtllib\s+(.+)$", txt, re.M)
            names = {os.path.basename(n).lower(): n for n in files}
            mtl = names.get(os.path.basename(mt.group(1).strip()).lower()) if mt else None
            if mtl is None:
                mtl = next((n for n in files if n.lower().endswith(".mtl")), None)
            tex = None
            if mtl:
                m = re.search(r"^\s*map_Kd\s+(?:-\S+\s+\S+\s+)*(.+)$", files[mtl].decode("utf-8", "ignore"), re.M)
                if m:
                    tex = names.get(os.path.basename(m.group(1).strip().replace("\\", "/")).lower())
            if tex is None:
                tex = next((n for n in files if n.lower().endswith((".png", ".jpg", ".jpeg"))), None)
            if tex:
                from PIL import Image
                info = {"texture": np.asarray(Image.open(io.BytesIO(files[tex])).convert("RGB"), dtype=np.float32) / 255.0}
        elif ext == ".stl":
            prims = [mesh_file_read.load_stl(data)]
        else:
            raise ValueError(f"{ext} is not supported (GLB, GLTF, OBJ, STL). FBX / USDZ: convert to GLB first")
        return _merge_primitives(prims), info
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _thumb(mesh, path):
    import comfy.model_management
    from PIL import Image
    dev = comfy.model_management.get_torch_device()
    V = mesh.vertices[0].to(dev).float()
    Fc = mesh.faces[0].to(dev).long()
    lo, hi = V.min(0).values, V.max(0).values
    ctr = (lo + hi) / 2
    W, H = 384, 576
    import math
    tf = math.tan(math.radians(28) / 2)
    dist = max(float(hi[1] - lo[1]) / 2 / tf, float(((V - ctr)[:, [0, 2]]).norm(dim=1).max()) / (tf * W / H)) * 1.15
    pos, fwd, right, up = _cam_basis(ctr, 0.0, 5.0, dist)
    cam = dict(pos=pos.tolist(), fwd=fwd.tolist(), right=right.tolist(), up=up.tolist(), f=(H / 2) / tf, W=W, H=H)
    _, _, _, dep = _view_raster(V, Fc, cam, dev)
    have = torch.isfinite(dep)
    z = torch.where(have, dep, torch.zeros_like(dep))
    z0, z1 = z[have].min(), z[have].max()
    g = torch.where(have, 0.25 + 0.7 * (1 - (z - z0) / (z1 - z0).clamp(min=1e-6)), torch.full_like(dep, 0.08)).view(H, W)
    Image.fromarray((g.cpu().numpy() * 255).astype(np.uint8)).convert("RGB").save(path)


def import_model(files, name, up="auto", facing="+Z"):
    main = next((n for n in files if os.path.splitext(n)[1].lower() in (".glb", ".gltf", ".obj", ".stl")), None)
    if main is None:
        raise ValueError("choose a .glb, .gltf, .obj or .stl file (plus its texture files)")
    merged, info = _read_model(files, main)
    v = merged["vertices"].astype(np.float32)
    if up == "auto":
        up = _auto_up(v)
    v = _orient(v, up, facing)
    nrm = merged["normals"]
    if nrm is not None:
        nrm = _orient(nrm.astype(np.float32), up, facing)
    b = lambda a: torch.from_numpy(a)[None] if a is not None else None
    mesh = Types.MESH(vertices=b(v), faces=b(merged["faces"]), uvs=b(merged["uvs"]), vertex_colors=b(merged["colors"]), normals=b(nrm),
                      texture=b(info.get("texture")), metallic_roughness=b(info.get("metallic_roughness")),
                      normal_map=b(info.get("normal_map")), emissive=b(info.get("emissive")),
                      unlit=bool(info.get("unlit", False)), occlusion_in_mr=bool(info.get("occlusion_in_mr", False)),
                      material=info.get("material") or None)
    safe = "".join(c if (c.isalnum() or c in "-_ ") else "_" for c in (name or os.path.splitext(main)[0])).strip() or "model"
    d = os.path.join(LIB_DIR, safe)
    if os.path.isdir(d):
        d = f"{d}_{time.strftime('%H%M%S')}"
    os.makedirs(d, exist_ok=True)
    fields = {k: (x.detach().cpu() if torch.is_tensor(x) else x) for k, x in vars(mesh).items() if x is not None}
    torch.save(fields, os.path.join(d, "mesh.pt"))
    try:
        _thumb(mesh, os.path.join(d, "thumb.png"))
    except Exception as e:
        print(f"[import_model] thumbnail failed: {e}")
    json.dump({"name": os.path.basename(d), "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "imported": True, "source": main,
               "up": up, "facing": facing, "vertices": int(v.shape[0]), "faces": int(merged["faces"].shape[0]),
               "textured": mesh.texture is not None}, open(os.path.join(d, "meta.json"), "w"), indent=1)
    return os.path.basename(d), mesh.texture is not None


@PromptServer.instance.routes.post("/midfr/import_model")
async def _midfr_import(request):
    reader = await request.multipart()
    files, opts = {}, {}
    while True:
        part = await reader.next()
        if part is None:
            break
        if part.filename:
            files[os.path.basename(part.filename)] = bytes(await part.read())
        else:
            opts[part.name] = (await part.text()).strip()
    try:
        name, tex = import_model(files, opts.get("name", ""), opts.get("up", "auto"), opts.get("facing", "+Z"))
        return web.json_response({"name": name, "textured": tex})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=400)


@PromptServer.instance.routes.get("/midfr/library")
async def _midfr_library(request):
    out = []
    for n in _lib_names():
        meta = {}
        mp = os.path.join(LIB_DIR, n, "meta.json")
        if os.path.isfile(mp):
            try:
                meta = json.load(open(mp))
            except Exception:
                pass
        out.append({"name": n, "imported": bool(meta.get("imported")), "textured": bool(meta.get("textured", True)),
                    "rigged": os.path.isfile(os.path.join(LIB_DIR, n, "character.pt"))})
    return web.json_response(out)


@PromptServer.instance.routes.get("/midfr/input_images")
async def _midfr_inputs(request):
    d = folder_paths.get_input_directory()
    files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
    return web.json_response(sorted(folder_paths.filter_files_content_types(files, ["image"])))


NODE_CLASS_MAPPINGS = {"QwenInputs": QwenInputs, "QwenInputsUnpack": QwenInputsUnpack}
NODE_DISPLAY_NAME_MAPPINGS = {"QwenInputs": "Qwen Inputs", "QwenInputsUnpack": "Qwen Inputs (unpack)"}
