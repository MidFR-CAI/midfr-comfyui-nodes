"""Generated-character library: orbit cameras + multi-view texture projection for a MESH, and a saved-character library.

`MeshOrbitCameras`   orbit cameras around a (rigged, A-posed) mesh -> per-view control depth, masks and view prompts (as lists,
                     so a Qwen stage downstream runs once per view), plus the camera set.
`MeshProjectTexture` projects the generated views back onto the UV-unwrapped mesh (visibility + facing weighted blend,
                     seam-safe fill) -> textured MESH.
`CharacterSave`      stores mesh + fitted pose data + source/thumbnail images as one library entry.
`CharacterLoad`      loads an entry from the library (dropdown) for use as an input instead of an image."""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
import folder_paths
from comfy_api.latest._util.geometry_types import MESH
from comfy_extras.nodes_mesh_postprocess import _normalize_uvs_to_unit

from .shell_fit import _raster_depth, _nconv
from .character3d import _raster_ids

LIB_DIR = os.path.join(folder_paths.models_dir, "characters")

_VIEW_NAMES = {
    0: "from the front, facing the camera",
    45: "from the front-left three-quarter angle, face turned slightly toward the camera",
    90: "from her left side in profile, the camera sees her left side only",
    135: "from behind-left, the camera is behind her: mostly her back, hair and the back of her outfit are visible, her face is turned away and hidden",
    180: "from directly behind, the camera is behind her: only the BACK of her head and hair, her back, the back of the top, and the back of the jeans with back pockets are visible; her face and front are NOT visible",
    225: "from behind-right, the camera is behind her: mostly her back, hair and the back of her outfit are visible, her face is turned away and hidden",
    270: "from her right side in profile, the camera sees her right side only",
    315: "from the front-right three-quarter angle, face turned slightly toward the camera"}


def _view_name(yaw):
    y = int(round(yaw / 45.0)) * 45 % 360
    return _VIEW_NAMES.get(y, f"{yaw:.0f} degrees around")


def _mesh_arrays(mesh, dev):
    n = int(mesh.vertex_counts[0]) if mesh.vertex_counts is not None else mesh.vertices.shape[1]
    m = int(mesh.face_counts[0]) if mesh.face_counts is not None else mesh.faces.shape[1]
    V = mesh.vertices[0, :n].to(dev).float()
    Fc = mesh.faces[0, :m].to(dev).long()
    return V, Fc, n, m


def _cam_basis(center, yaw, pitch, dist):
    y, p = math.radians(yaw), math.radians(pitch)
    pos = center + dist * torch.tensor([math.sin(y) * math.cos(p), math.sin(p), math.cos(y) * math.cos(p)], device=center.device)
    fwd = (center - pos)
    fwd = fwd / fwd.norm()
    up = torch.tensor([0.0, 1.0, 0.0], device=center.device)
    right = torch.cross(fwd, up, dim=0)
    right = right / right.norm().clamp(min=1e-6)
    up2 = torch.cross(right, fwd, dim=0)
    return pos, fwd, right, up2


def _project(V, cam, dev):
    pos = torch.tensor(cam["pos"], device=dev)
    fwd, right, up = (torch.tensor(cam[k], device=dev) for k in ("fwd", "right", "up"))
    d = V - pos
    z = d @ fwd
    x = d @ right
    y = d @ up
    f = cam["f"]
    u = cam["W"] / 2 + f * x / z.clamp(min=1e-6)
    v = cam["H"] / 2 - f * y / z.clamp(min=1e-6)
    return u, v, z


def _view_raster(V, Fc, cam, dev):
    u, v, z = _project(V, cam, dev)
    H, W = cam["H"], cam["W"]
    dep = _raster_depth(u, v, z, Fc, H, W)
    return u, v, z, dep


class MeshOrbitCameras:
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "CHAR_CAMS")
    RETURN_NAMES = ("control_depth", "body_mask", "view_prompt", "cameras")
    OUTPUT_IS_LIST = (True, True, True, False)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mesh": ("MESH",),
            "views": ("INT", {"default": 8, "min": 1, "max": 16, "tooltip": "evenly spaced around the character, starting at the front"}),
            "pitch": ("FLOAT", {"default": 5.0, "min": -30.0, "max": 60.0, "step": 1.0}),
            "fov": ("FLOAT", {"default": 28.0, "min": 10.0, "max": 60.0, "step": 1.0, "tooltip": "vertical field of view (degrees)"}),
            "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 16}),
            "height": ("INT", {"default": 1248, "min": 256, "max": 2048, "step": 16}),
            "margin": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 1.6, "step": 0.01}),
            "extra_top_view": ("BOOLEAN", {"default": False}),
            "prompt_template": ("STRING", {"multiline": True, "default":
                "Redraw the character from <image1> as a full-body photograph {view}, matching the depth guide exactly. "
                "Keep exactly the same face, hair, glasses, jewellery, outfit, fabric, skin tone and body proportions as <image1>. "
                "Parts of the outfit not visible in <image1> continue the same design and colours. Plain light grey studio background.",
                "tooltip": "{view} is replaced with front / left side / back ... for each camera"}),
            "closeups": ("INT", {"default": 0, "min": 0, "max": 3, "tooltip": "extra zoomed head-and-bust views appended after the orbit: 1 = front, 2 = + front-left 3/4, 3 = + front-right 3/4. Sharp face and chest texture."}),
            "closeup_height": ("FLOAT", {"default": 0.30, "min": 0.12, "max": 0.6, "step": 0.01, "tooltip": "fraction of the character's height shown in a close-up frame, measured down from the top of the head"}),
            "closeup_template": ("STRING", {"multiline": True, "default":
                "Redraw the head and upper body of the character from <image1> as a sharp, highly detailed close-up portrait photograph {view}, matching the depth guide exactly. "
                "Keep exactly the same face, eyes, hair colour, glasses, jewellery, neckline, fabric and skin tone as <image1>. Plain light grey studio background."}),
        }}

    def run(self, mesh, views, pitch, fov, width, height, margin, extra_top_view, prompt_template, closeups=0, closeup_height=0.30, closeup_template=""):
        dev = comfy.model_management.get_torch_device()
        V, Fc, _, _ = _mesh_arrays(mesh, dev)
        lo, hi = V.min(0).values, V.max(0).values
        center = (lo + hi) / 2
        half_h = float((hi[1] - lo[1]) / 2)
        rad = float(((V - center)[:, [0, 2]]).norm(dim=1).max())
        tf = math.tan(math.radians(fov) / 2)
        th = tf * width / height
        dist = max(half_h / tf, rad / th) * margin + rad * 0.15
        f = (height / 2) / tf
        yaws = [i * 360.0 / views for i in range(views)]
        cams, depths, masks, prompts = [], [], [], []
        for yaw in yaws:
            pos, fwd, right, up = _cam_basis(center, yaw, pitch, dist)
            cam = dict(pos=pos.tolist(), fwd=fwd.tolist(), right=right.tolist(), up=up.tolist(), f=f, W=width, H=height,
                       yaw=yaw, dist=dist)
            u, v, z, dep = _view_raster(V, Fc, cam, dev)
            have = torch.isfinite(dep)
            m = have.view(height, width).float()
            zz = torch.where(have, dep, torch.zeros_like(dep))
            z0, z1 = zz[have].min(), zz[have].max()
            nd = 1.0 - (zz - z0) / (z1 - z0).clamp(min=1e-6)
            img = torch.where(have, 0.1 + 0.9 * nd, torch.zeros_like(nd)).view(height, width)
            depths.append(img[None, :, :, None].repeat(1, 1, 1, 3).cpu())
            masks.append(m[None].cpu())
            cams.append(cam)
            prompts.append(prompt_template.replace('{view}', _view_name(yaw)))
        if closeups > 0:
            # zoomed head-and-bust frames: aimed at the top `closeup_height` of the character, same image size as the orbit views
            total_h = float(hi[1] - lo[1])
            win = closeup_height * total_h
            top = V[V[:, 1] > hi[1] - 0.06 * total_h]
            tgt = center.clone()
            tgt[0], tgt[2] = top[:, 0].mean(), top[:, 2].mean()
            tgt[1] = hi[1] - win / 2
            cdist = (win / 2) / tf
            cf = (height / 2) / tf
            for yaw in ([0.0, 40.0, -40.0])[:closeups]:
                pos, fwd, right, up = _cam_basis(tgt, yaw, 3.0, cdist)
                cam = dict(pos=pos.tolist(), fwd=fwd.tolist(), right=right.tolist(), up=up.tolist(), f=cf, W=width, H=height,
                           yaw=yaw % 360.0, dist=cdist, closeup=True, weight=6.0)
                u, v, z, dep = _view_raster(V, Fc, cam, dev)
                have = torch.isfinite(dep)
                zz = torch.where(have, dep, torch.zeros_like(dep))
                z0, z1 = zz[have].min(), zz[have].max()
                nd = 1.0 - (zz - z0) / (z1 - z0).clamp(min=1e-6)
                img = torch.where(have, 0.1 + 0.9 * nd, torch.zeros_like(nd)).view(height, width)
                depths.append(img[None, :, :, None].repeat(1, 1, 1, 3).cpu())
                masks.append(have.view(height, width).float()[None].cpu())
                cams.append(cam)
                tpl = closeup_template.strip() or ("Redraw the head and upper body of the character from <image1> as a sharp close-up portrait {view}.")
                prompts.append(tpl.replace('{view}', _view_name(yaw)))
        return (depths, masks, prompts, {"cams": cams})


def _erode(m, px):
    if px <= 0:
        return m
    return -F.max_pool2d(-m, 2 * px + 1, 1, px)


class MeshProjectTexture:
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    INPUT_IS_LIST = True
    RETURN_TYPES = ("MESH", "IMAGE", "IMAGE")
    RETURN_NAMES = ("mesh", "texture", "coverage")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mesh": ("MESH", {"tooltip": "UV-unwrapped mesh (UnwrapMesh), same geometry the cameras came from"}),
            "views": ("IMAGE", {"tooltip": "one generated image per camera"}),
            "cameras": ("CHAR_CAMS",),
            "texture_size": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
            "facing_power": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 8.0, "step": 0.5, "tooltip": "higher = trust the view that faces a surface most"}),
            "min_facing": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.8, "step": 0.05}),
            "edge_erode": ("INT", {"default": 4, "min": 0, "max": 20, "tooltip": "pixels trimmed off each view's silhouette (avoids background bleed)"}),
            "flip_v": ("BOOLEAN", {"default": False}),
        }}

    def run(self, mesh, views, cameras, texture_size, facing_power, min_facing, edge_erode, flip_v):
        mesh, cams = mesh[0], cameras[0]["cams"]
        T, p, mf, er, fl = texture_size[0], facing_power[0], min_facing[0], edge_erode[0], flip_v[0]
        imgs = []
        for v in views:
            v = v if v.dim() == 4 else v[None]
            imgs += [v[i:i + 1, :, :, :3] for i in range(v.shape[0])]
        if len(imgs) != len(cams):
            raise ValueError(f"MeshProjectTexture: {len(imgs)} views but {len(cams)} cameras")
        dev = comfy.model_management.get_torch_device()
        V, Fc, n, _ = _mesh_arrays(mesh, dev)
        if mesh.uvs is None:
            raise ValueError("MeshProjectTexture: mesh has no UVs, run UnwrapMesh first")
        uv = torch.from_numpy(_normalize_uvs_to_unit(mesh.uvs[0, :n].detach().cpu().numpy())).to(dev)
        fv = V[Fc]
        fn = torch.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0], dim=1)
        fn = fn / fn.norm(dim=1, keepdim=True).clamp(min=1e-9)
        # texel -> triangle + barycentrics
        ux = uv[:, 0] * T
        vy = (uv[:, 1] if not fl else 1.0 - uv[:, 1]) * T
        zz = torch.ones(n, device=dev)
        ids, bary = _raster_ids(ux, vy, zz, Fc, T, T, torch.full((T * T,), 10.0, device=dev))
        have = ids >= 0
        pos = torch.zeros(T * T, 3, device=dev)
        nrm = torch.zeros(T * T, 3, device=dev)
        fid = ids[have]
        pos[have] = (fv[fid] * bary[have][:, :, None]).sum(1)
        nrm[have] = fn[fid]
        acc = torch.zeros(T * T, 3, device=dev)
        wsum = torch.zeros(T * T, device=dev)
        for img, cam in zip(imgs, cams):
            H, W = cam["H"], cam["W"]
            x = img.permute(0, 3, 1, 2).to(dev).float()
            if x.shape[-2:] != (H, W):
                x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
            u, v, z, dep = _view_raster(V, Fc, cam, dev)
            body0 = torch.isfinite(dep).view(1, 1, H, W).float()
            # trim the silhouette hard (er px) and fade samples out over the next 3*er px: near the outline a view is
            # grazing and its colour is often background; the face-on view of the same texel is trusted instead
            body = _erode(body0, er)
            if er > 0:
                ramp = sum(_erode(body0, er * k) for k in (1, 2, 3, 4)) / 4.0
                body = torch.minimum(body, ramp)
            cpos = torch.tensor(cam["pos"], device=dev)
            P = pos[have]
            dv = P - cpos
            zt = dv @ torch.tensor(cam["fwd"], device=dev)
            xt = dv @ torch.tensor(cam["right"], device=dev)
            yt = dv @ torch.tensor(cam["up"], device=dev)
            pu = W / 2 + cam["f"] * xt / zt.clamp(min=1e-6)
            pv = H / 2 - cam["f"] * yt / zt.clamp(min=1e-6)
            inside = (pu >= 0) & (pu < W - 1) & (pv >= 0) & (pv < H - 1) & (zt > 0.05)
            iu, iv = pu.clamp(0, W - 1).long(), pv.clamp(0, H - 1).long()
            dmap = torch.where(torch.isfinite(dep), dep, torch.full_like(dep, 1e6)).view(1, 1, H, W)
            dmap = -F.max_pool2d(-dmap, 5, 1, 2)  # conservative occluder map: thin gaps / holes in the mesh no longer let texels see through
            seen = dmap.view(-1)[iv * W + iu]
            tol = 0.006 * cam["dist"]
            vis = inside & (zt <= seen + tol)
            cosv = -(nrm[have] * (dv / dv.norm(dim=1, keepdim=True).clamp(min=1e-9))).sum(1)
            w = torch.where(vis & (cosv > mf), cosv.clamp(min=0) ** p, torch.zeros_like(cosv))
            g = torch.stack([pu / (W - 1) * 2 - 1, pv / (H - 1) * 2 - 1], 1)[None, None]
            col = F.grid_sample(x, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0].T
            bm = F.grid_sample(body, g, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0, 0]
            w = w * bm.clamp(0, 1)
            if cam.get("closeup"):
                # close-ups win where they have pixels to spare, fading out toward their frame edge so they blend into the body views
                edge = torch.minimum(torch.minimum(pu, W - 1 - pu), torch.minimum(pv, H - 1 - pv)) / (0.12 * min(W, H))
                w = w * cam.get("weight", 1.0) * edge.clamp(0, 1)
            # near the outline, samples whose colour is the local background are dropped (the generated figure can be a little
            # narrower than the mesh). Local background = diffusion of the pixels outside the mesh silhouette, so floor gradients work.
            bgm = _erode(1.0 - body0, 3)
            bg_est = torch.cat([_nconv(x[:, c:c + 1], bgm, 14.0) for c in range(3)], 1)
            band = (_erode(body0, 24) < 0.5).float()
            bgl = F.grid_sample(bg_est, g, mode="bilinear", padding_mode="border", align_corners=True)[0, :, 0].T
            bnd = F.grid_sample(band, g, mode="bilinear", padding_mode="border", align_corners=True)[0, 0, 0]
            dcol = (col - bgl).abs().max(1).values
            bgness = (1.0 - (dcol - 0.05) / 0.08).clamp(0, 1)
            w = w * (1.0 - bgness * (bnd > 0.5).float())
            idx = have.nonzero()[:, 0]
            cnt = lambda m: int(m.sum())
            print(f"[MeshProjectTexture] view yaw {cam['yaw']:.0f}: texels {len(w)}, inside {cnt(inside)}, visible {cnt(vis)}, facing {cnt(vis & (cosv > mf))}, "
                  f"w>0 {cnt(w > 1e-6)} (bm<=0 {cnt(bm <= 0)}, bg-rejected {cnt(bgness * (bnd > 0.5).float() > 0.5)})", flush=True)
            acc[idx] += col * w[:, None]
            wsum[idx] += w
            del x
        seenm = (wsum > 1e-6)
        print(f"[MeshProjectTexture] island texels {int(have.sum())}, covered {int((seenm & have).sum())} ({100.0 * float((seenm & have).sum()) / max(1, int(have.sum())):.1f}%)", flush=True)
        col = torch.where(seenm[:, None], acc / wsum.clamp(min=1e-9)[:, None], torch.zeros_like(acc))
        tex = col.view(T, T, 3).permute(2, 0, 1)[None]
        island = have.view(1, 1, T, T).float()
        valid = seenm.view(1, 1, T, T).float()
        out = tex.clone()
        for s in (3, 10, 40):  # progressively larger fills for texels no view saw
            f = torch.cat([_nconv(out[:, c:c + 1], valid, s) for c in range(3)], 1)
            out = torch.where(valid > 0, out, f)
            valid = torch.maximum(valid, (_nconv(valid, torch.ones_like(valid), s) > 0.01).float() * island)
        # gutter: grow island colours outward so bilinear sampling at chart edges never hits black
        full = island.clone()
        for s in (2, 4, 8):
            f = torch.cat([_nconv(out[:, c:c + 1], full, s) for c in range(3)], 1)
            out = torch.where(full > 0, out, f)
            full = torch.maximum(full, (_nconv(full, torch.ones_like(full), s) > 0.02).float())
        out = out.clamp(0, 1)
        newm = _copy_mesh(mesh)
        uvs = mesh.uvs.clone()
        uvs[0, :n] = uv.cpu()
        newm.uvs = uvs
        newm.texture = out.permute(0, 2, 3, 1).cpu()
        cov = (wsum.view(1, T, T, 1) > 1e-6).float().repeat(1, 1, 1, 3).cpu()
        return (newm, newm.texture, cov)


def _copy_mesh(m):
    import copy
    return copy.copy(m)


# ----------------------------------------------------------------------------- library
def _lib_names():
    if not os.path.isdir(LIB_DIR):
        return []
    return sorted(d for d in os.listdir(LIB_DIR) if os.path.isfile(os.path.join(LIB_DIR, d, "mesh.pt")))


def _img_save(t, path):
    from PIL import Image
    a = (t[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(a).save(path)


def _img_load(path):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a)[None]


class CharacterSave:
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("folder",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mesh": ("MESH", {"tooltip": "textured character mesh in its A-pose", "lazy": True}),
            "character": ("MHR_POSE_DATA", {"tooltip": "SAM 3D Body prediction of the A-pose image the mesh was made from", "lazy": True}),
            "name": ("STRING", {"default": "character"}),
            "overwrite": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "source_image": ("IMAGE", {"tooltip": "the A-pose image (kept as the identity/outfit reference)", "lazy": True}),
            "thumbnail": ("IMAGE", {"lazy": True}),
            "skip": ("BOOLEAN", {"default": False, "tooltip": "do nothing (and do not evaluate the inputs), e.g. when the character came from the library"}),
        }}

    def check_lazy_status(self, name, overwrite, mesh=None, character=None, source_image=None, thumbnail=None, skip=False):
        if skip:
            return []
        return [k for k, v in (("mesh", mesh), ("character", character), ("source_image", source_image), ("thumbnail", thumbnail)) if v is None]

    def run(self, name, overwrite, mesh=None, character=None, source_image=None, thumbnail=None, skip=False):
        if skip or mesh is None or character is None:
            return {"ui": {"text": ["skipped"]}, "result": ("",)}
        safe = "".join(c if (c.isalnum() or c in "-_ ") else "_" for c in name).strip() or "character"
        d = os.path.join(LIB_DIR, safe)
        if os.path.isdir(d) and not overwrite:
            d = f"{d}_{time.strftime('%H%M%S')}"
        os.makedirs(d, exist_ok=True)
        fields = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in vars(mesh).items() if v is not None}
        torch.save(fields, os.path.join(d, "mesh.pt"))
        torch.save(character, os.path.join(d, "character.pt"))
        thumb = thumbnail if thumbnail is not None else source_image
        if source_image is not None:
            _img_save(source_image, os.path.join(d, "source.png"))
        if thumb is not None:
            t = thumb[:1].permute(0, 3, 1, 2)
            h = 512
            t = F.interpolate(t, size=(h, max(1, int(t.shape[3] * h / t.shape[2]))), mode="bilinear", align_corners=False)
            _img_save(t.permute(0, 2, 3, 1), os.path.join(d, "thumb.png"))
        json.dump({"name": os.path.basename(d), "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "vertices": int(mesh.vertices.shape[1]), "faces": int(mesh.faces.shape[1]),
                   "textured": mesh.texture is not None}, open(os.path.join(d, "meta.json"), "w"), indent=1)
        return {"ui": {"text": [d]}, "result": (d,)}


class CharacterLoad:
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    RETURN_TYPES = ("MESH", "MHR_POSE_DATA", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("mesh", "character", "thumbnail", "source_image", "name")

    @classmethod
    def INPUT_TYPES(cls):
        names = _lib_names()
        return {"required": {"character": (names if names else ["(library is empty)"],)}}

    @classmethod
    def IS_CHANGED(cls, character):
        p = os.path.join(LIB_DIR, character, "mesh.pt")
        return os.path.getmtime(p) if os.path.isfile(p) else 0

    def run(self, character):
        d = os.path.join(LIB_DIR, character)
        if not os.path.isfile(os.path.join(d, "mesh.pt")):
            raise ValueError(f"CharacterLoad: '{character}' not found in {LIB_DIR}")
        f = torch.load(os.path.join(d, "mesh.pt"), map_location="cpu", weights_only=False)
        m = MESH(f.pop("vertices"), f.pop("faces"))
        for k, v in f.items():
            setattr(m, k, v)
        pose = torch.load(os.path.join(d, "character.pt"), map_location="cpu", weights_only=False)
        blank = torch.zeros(1, 64, 64, 3)
        src = _img_load(os.path.join(d, "source.png")) if os.path.isfile(os.path.join(d, "source.png")) else blank
        th = _img_load(os.path.join(d, "thumb.png")) if os.path.isfile(os.path.join(d, "thumb.png")) else src
        return (m, pose, th, src, character)


NODE_CLASS_MAPPINGS = {
    "MeshOrbitCameras": MeshOrbitCameras,
    "MeshProjectTexture": MeshProjectTexture,
    "CharacterSave": CharacterSave,
    "CharacterLoad": CharacterLoad,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MeshOrbitCameras": "Mesh Orbit Cameras (depth per view)",
    "MeshProjectTexture": "Mesh Project Texture (multi-view)",
    "CharacterSave": "Character Library: Save",
    "CharacterLoad": "Character Library: Load",
}
