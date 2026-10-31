"""Clothed-figure depth and body fitting helpers.

SapiensRegions   splits a Sapiens2 body-part segmentation into person / skin (trusted) / clothing / hair+glasses / face masks.
Sapiens2Depth    metric depth (metres) of the clothed figure from a Sapiens2 pointmap model.
MHRSliderFit     refits a SAM 3D Body (MHR) mesh with bounded body sliders (trunk width/depth at 5 heights, limb girths) so it
                 matches the visible skin and stays inside the clothed silhouette; natural shapes only, nothing can tear.
DepthToControl   the clothed figure's depth as a control image for Qwen."""
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
from comfy_extras.sam3d_body import face_expression as fx


def _sapiens_mod():
    for m in list(sys.modules.values()):
        if isinstance(m, types.ModuleType) and "_OpNamespace" not in type(m).__name__ and "torch" not in getattr(m, "__name__", "") and hasattr(m, "_iter_chunks") and hasattr(m, "Sapiens2Pointmap"):
            return m
    raise RuntimeError("ComfyUI-Sapiens2 is not loaded")


class Sapiens2Depth:
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("depth_m", "preview")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "sapiens2_model": ("SAPIENS2_MODEL",),
                             "frames_per_batch": ("INT", {"default": 1, "min": 1, "max": 16})}}

    def run(self, image, sapiens2_model, frames_per_batch):
        mod = _sapiens_mod()
        to_model_input = getattr(mod, "to_model_input")
        crop_pad_and_resize = getattr(mod, "crop_pad_and_resize")
        patcher, task = sapiens2_model
        if task != "pointmap":
            raise ValueError(f"loaded checkpoint is task={task!r}, expected 'pointmap'")
        comfy.model_management.load_model_gpu(patcher)
        x, orig_hw, padding = to_model_input(image, device=patcher.load_device, dtype=torch.float16)
        chunks = []
        for _, out in mod._iter_chunks(patcher, x, frames_per_batch, desc="Sapiens2 depth"):
            if isinstance(out, tuple):
                pm = out[0].to(torch.float32)
                pm = pm / out[1].to(torch.float32).clamp(min=1e-6).view(-1, 1, 1, 1)
            else:
                pm = out.to(torch.float32)
            pm = crop_pad_and_resize(pm, padding, orig_hw[0], orig_hw[1])
            chunks.append(pm.cpu())
        z = torch.cat(chunks, 0)[:, 2]  # (B,H,W) metres
        lo, hi = torch.quantile(z.flatten(), 0.02), torch.quantile(z.flatten(), 0.98)
        pv = ((z - lo) / (hi - lo + 1e-6)).clamp(0, 1)
        return (z, (1 - pv)[..., None].repeat(1, 1, 1, 3))


def _vertex_normals(v, faces):
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    fn = torch.cross(b - a, c - a, dim=-1)
    vn = torch.zeros_like(v)
    for i in range(3):
        vn.index_add_(0, faces[:, i], fn)
    return F.normalize(vn, dim=-1)


def _prep_mask(m, H, W, dev):
    m = m.float()
    if m.dim() == 2:
        m = m[None]
    m = m[:1]
    if m.shape[-2:] != (H, W):
        m = F.interpolate(m[None], size=(H, W), mode="bilinear", align_corners=False)[0]
    return m[None].to(dev)  # 1,1,H,W


class MHRSliderFit:
    """Character-creator style fit: the base body is resized by a small set of body sliders (trunk width/depth at
    hips / waist / bust / shoulders, girth of upper arms, forearms, thighs, calves), each limited to a range, until it
    fills the clothed shell without clipping through it. Neighbouring trunk heights may only differ by `max_step`."""
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    RETURN_TYPES = ("MHR_POSE_DATA", "STRING")
    RETURN_NAMES = ("mhr_pose_data", "sliders")
    TKNOTS = [-0.2, 0.0, 0.35, 0.7, 1.0]
    TNAMES = ["upper thigh", "hips", "waist", "bust", "shoulders"]
    LIMBS = ["upper arm", "forearm", "thigh", "calf"]

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sam3d_body_model": ("SAM3D_BODY_MODEL",),
            "mhr_pose_data": ("MHR_POSE_DATA",),
            "shell_depth": ("MASK",),
            "person": ("MASK",),
            "image_width": ("INT", {"default": 832, "min": 64, "max": 8192}),
            "image_height": ("INT", {"default": 1248, "min": 64, "max": 8192}),
            "trunk_min": ("FLOAT", {"default": 0.9, "min": 0.5, "max": 1.0, "step": 0.01, "tooltip": "lowest trunk width/depth slider"}),
            "trunk_max": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 2.5, "step": 0.01, "tooltip": "highest trunk width/depth slider"}),
            "limb_min": ("FLOAT", {"default": 0.9, "min": 0.5, "max": 1.0, "step": 0.01}),
            "limb_max": ("FLOAT", {"default": 1.4, "min": 1.0, "max": 2.5, "step": 0.01}),
            "max_step": ("FLOAT", {"default": 0.25, "min": 0.02, "max": 1.0, "step": 0.01,
                                   "tooltip": "largest allowed ratio change (log) between neighbouring trunk heights: keeps shapes natural"}),
            "thickness": ("FLOAT", {"default": 0.05, "min": 0.005, "max": 0.3, "step": 0.005}),
            "silhouette_pull": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
            "iterations": ("INT", {"default": 300, "min": 20, "max": 1500}),
        }, "optional": {"trusted": ("MASK",)}}

    def run(self, *a, **k):
        with torch.inference_mode(False), torch.enable_grad():
            return self._run(*a, **k)

    def _run(self, sam3d_body_model, mhr_pose_data, shell_depth, person, image_width, image_height, trunk_min, trunk_max,
             limb_min, limb_max, max_step, thickness, silhouette_pull, iterations, trusted=None):
        comfy.model_management.load_model_gpu(sam3d_body_model)
        inner = sam3d_body_model.model
        dev = comfy.model_management.get_torch_device()
        H, W = image_height, image_width
        Zs0 = F.avg_pool2d(_prep_mask(shell_depth, H, W, dev), 9, 1, 4)
        M = _prep_mask(person, H, W, dev)
        Mb = F.avg_pool2d(M, 15, 1, 7)
        T = F.avg_pool2d(_prep_mask(trusted, H, W, dev), 15, 1, 7) if trusted is not None else torch.zeros_like(M)
        faces = inner.head_pose.faces.to(dev).long()
        tm = ((T > 0.5) & (M > 0.5)).float()
        bd = (tm - (-F.max_pool2d(-tm, 5, 1, 2))).clamp(0, 1)[0, 0]
        by, bx = torch.nonzero(bd > 0.5, as_tuple=True)
        if len(bx) > 700:
            idx = torch.randperm(len(bx), device=dev)[:700]
            by, bx = by[idx], bx[idx]
        bpts = torch.stack([bx.float(), by.float()], -1) if len(bx) else None
        tk = torch.tensor(self.TKNOTS, device=dev)
        frames = [[dict(p) for p in f] for f in mhr_pose_data["frames"]]
        logs = []

        def sig_to_s(u, lo, hi):
            return lo + (hi - lo) * torch.sigmoid(u)

        def u_init(lo, hi):
            x = (1.0 - lo) / (hi - lo)
            return float(np.log(x / (1 - x)))

        def interp(vals, t):
            tt = t.clamp(self.TKNOTS[0], self.TKNOTS[-1])
            i = torch.clamp(torch.bucketize(tt, tk) - 1, 0, len(self.TKNOTS) - 2)
            t0, t1 = tk[i], tk[i + 1]
            f_ = ((tt - t0) / (t1 - t0)).clamp(0, 1)
            f_ = f_ * f_ * (3 - 2 * f_)
            return vals[i] * (1 - f_) + vals[i + 1] * f_

        for fi, persons in enumerate(frames):
            for pid, p in enumerate(persons):
                if p.get("pred_vertices") is None:
                    continue
                V = torch.from_numpy(np.asarray(p["pred_vertices"], np.float32)).to(dev)
                kp = torch.from_numpy(np.asarray(p["pred_keypoints_3d"], np.float32)).to(dev)
                cam_t = torch.from_numpy(np.asarray(p["pred_cam_t"], np.float32)).to(dev).reshape(3)
                f = float(np.asarray(p["focal_length"]).reshape(-1)[0])
                kp2 = np.asarray(p["pred_keypoints_2d"], np.float32)
                k3 = np.asarray(p["pred_keypoints_3d"], np.float32)
                x0, y0, z0 = k3[0] + np.asarray(p["pred_cam_t"], np.float32)
                cx, cy = float(kp2[0, 0] - f * x0 / max(z0, 1e-6)), float(kp2[0, 1] - f * y0 / max(z0, 1e-6))
                LS, RS, LE, RE, LH, RH, LK, RK, LA, RA = 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
                LW, RW = 62, 41
                hipmid, neck = (kp[LH] + kp[RH]) / 2, (kp[LS] + kp[RS]) / 2
                up = F.normalize(neck - hipmid, dim=0)
                rr = kp[LS] - kp[RS]
                ex = F.normalize(rr - (rr @ up) * up, dim=0)
                ez = torch.cross(ex, up, dim=0)
                Ltr = (neck - hipmid).norm()
                # segments: (p0, p1, radius, kind)
                segs = [("trunk", hipmid, neck, 0.42 * Ltr), ("uarmL", kp[LS], kp[LE], 0.14 * Ltr), ("uarmR", kp[RS], kp[RE], 0.14 * Ltr),
                        ("farmL", kp[LE], kp[LW], 0.12 * Ltr), ("farmR", kp[RE], kp[RW], 0.12 * Ltr),
                        ("thighL", kp[LH], kp[LK], 0.2 * Ltr), ("thighR", kp[RH], kp[RK], 0.2 * Ltr),
                        ("calfL", kp[LK], kp[LA], 0.16 * Ltr), ("calfR", kp[RK], kp[RA], 0.16 * Ltr),
                        ("head", neck, kp[0], 0.45 * Ltr), ("handL", kp[LW], kp[LW] + 1e-3, 0.25 * Ltr),
                        ("handR", kp[RW], kp[RW] + 1e-3, 0.25 * Ltr), ("footL", kp[LA], kp[LA] + 1e-3, 0.22 * Ltr),
                        ("footR", kp[RA], kp[RA] + 1e-3, 0.22 * Ltr)]

                def seg_dist(V_, p0, p1):
                    d = p1 - p0
                    ln = d.norm().clamp(min=1e-6)
                    dn = d / ln
                    tt = ((V_ - p0) @ dn).clamp(0, ln)
                    c = p0 + tt[:, None] * dn
                    return (V_ - c).norm(dim=-1), c

                nd = torch.stack([seg_dist(V, s[1], s[2])[0] / s[3] for s in segs], 1)
                wts = torch.softmax(-nd * 6.0, dim=1)  # (N, S)

                K = len(self.TKNOTS)
                uw = torch.full((K,), u_init(trunk_min, trunk_max), device=dev, requires_grad=True)
                ud = torch.full((K,), u_init(trunk_min, trunk_max), device=dev, requires_grad=True)
                ul = torch.full((len(self.LIMBS),), u_init(limb_min, limb_max), device=dev, requires_grad=True)

                def deform():
                    wk, dk = sig_to_s(uw, trunk_min, trunk_max), sig_to_s(ud, trunk_min, trunk_max)
                    ls = sig_to_s(ul, limb_min, limb_max)
                    q = V - hipmid
                    a_, b_, t_ = q @ ex, q @ ez, (q @ up) / Ltr
                    wf, df = interp(wk, t_), interp(dk, t_)
                    Vt = hipmid + (a_ * wf)[:, None] * ex + (b_ * df)[:, None] * ez + ((q @ up))[:, None] * up
                    out = V.clone()
                    delta = torch.zeros_like(V)
                    delta = delta + wts[:, 0:1] * (Vt - V)
                    lim_map = {"uarmL": 0, "uarmR": 0, "farmL": 1, "farmR": 1, "thighL": 2, "thighR": 2, "calfL": 3, "calfR": 3}
                    for si, sg in enumerate(segs):
                        if sg[0] in lim_map:
                            _, c = seg_dist(V, sg[1], sg[2])
                            delta = delta + wts[:, si:si + 1] * ((c + (V - c) * ls[lim_map[sg[0]]]) - V)
                    return V + delta, wk, dk, ls

                def sample(img, vc):
                    u = f * vc[:, 0] / vc[:, 2].clamp(min=1e-6) + cx
                    w = f * vc[:, 1] / vc[:, 2].clamp(min=1e-6) + cy
                    g = torch.stack([u / W * 2 - 1, w / H * 2 - 1], -1)[None, None]
                    return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=False)[0, 0, 0]

                with torch.no_grad():
                    vc = V + cam_t
                    nrm = _vertex_normals(V, faces)
                    if ((V - V.mean(0)) * nrm).sum(-1).mean() < 0:
                        nrm = -nrm
                    front0 = ((nrm * (-F.normalize(vc, dim=-1))).sum(-1) > 0.2)
                    sel = front0 & (sample(M, vc) > 0.5)
                    tsel = sel & (sample(T, vc) > 0.5)
                    sel = tsel if tsel.sum() > 60 else sel
                    zs = sample(Zs0, vc)
                    A = torch.stack([zs[sel], torch.ones_like(zs[sel])], -1)
                    sol = torch.linalg.lstsq(A, vc[sel, 2:3]).solution.reshape(2)
                    a_al, b_al = sol[0], sol[1]
                    sign = 1.0 if ((V - V.mean(0)) * _vertex_normals(V, faces)).sum(-1).mean() >= 0 else -1.0

                opt = torch.optim.Adam([uw, ud, ul], lr=0.05)
                for it in range(iterations):
                    opt.zero_grad()
                    Vd, wk, dk, ls = deform()
                    vc = Vd + cam_t
                    nr = _vertex_normals(Vd, faces) * sign
                    front = ((nr * (-F.normalize(vc, dim=-1))).sum(-1) > 0.2).detach()
                    zsv = a_al * sample(Zs0, vc) + b_al
                    inm = (sample(M, vc) > 0.5).detach()
                    trw = sample(T, vc).detach().clamp(0, 1)
                    d = vc[:, 2] - zsv
                    sel = (front & inm).float()
                    clip = (F.relu(-d - 0.004) ** 2 * sel).sum() / sel.sum().clamp(min=1)
                    near = (d.detach().abs() < thickness).float()
                    infl = ((d.clamp(min=0) ** 2) * (trw + (1 - trw) * 0.3) * sel * near).sum() / sel.sum().clamp(min=1)
                    sil = (((1 - sample(Mb, vc)) ** 2)).mean()
                    cham = torch.zeros((), device=dev)
                    if bpts is not None and silhouette_pull > 0:
                        pp = torch.stack([f * vc[:, 0] / vc[:, 2].clamp(min=1e-6) + cx, f * vc[:, 1] / vc[:, 2].clamp(min=1e-6) + cy], -1)
                        dm = torch.cdist(bpts, pp).min(1).values / 40.0
                        cham = torch.where(dm < 1, 0.5 * dm ** 2, dm - 0.5).mean()
                    sm_ = F.relu((torch.log(wk[1:] / wk[:-1])).abs() - max_step) ** 2 + F.relu((torch.log(dk[1:] / dk[:-1])).abs() - max_step) ** 2
                    loss = 200 * clip + 80 * infl + 6 * sil + 10 * silhouette_pull * cham + 20 * sm_.sum()
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    Vd, wk, dk, ls = deform()
                    p["pred_vertices"] = Vd.cpu().numpy().astype(np.float32)
                    p["slider_state"] = {"wk": [float(x) for x in wk], "dk": [float(x) for x in dk], "ls": [float(x) for x in ls]}
                    lines = [f"{n:>11}: width x{float(w_):.2f}, depth x{float(d_):.2f}" for n, w_, d_ in zip(self.TNAMES, wk, dk)]
                    lines += [f"{n:>11}: girth x{float(v):.2f}" for n, v in zip(self.LIMBS, ls)]
                    logs.append(f"person {pid}\n" + "\n".join(lines))
                frames[fi][pid] = p
        out = dict(mhr_pose_data)
        out["frames"] = frames
        out["mhr_faces"] = faces.cpu()
        return (out, "\n".join(logs) or "no person")


def _nconv(img, mask, sigma):
    """masked gaussian blur (normalized convolution); img,mask: 1,1,H,W"""
    k = int(sigma * 3) * 2 + 1
    x = torch.arange(k, device=img.device, dtype=img.dtype) - k // 2
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    def blur(t):
        t = F.conv2d(F.pad(t, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
        return F.conv2d(F.pad(t, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1))
    return blur(img * mask) / blur(mask).clamp(min=1e-4)


def _edge_dist(mask, maxd=140):
    """distance (px) to the mask edge, by repeated erosion. mask: 1,1,H,W float"""
    d = torch.zeros_like(mask)
    cur = mask
    for _ in range(maxd):
        d = d + cur
        cur = -F.max_pool2d(-cur, 3, 1, 1)
        if cur.sum() == 0:
            break
    return d


def _raster_depth(u, v, z, faces, H, W):
    """z-buffer triangle rasterization (barycentric, vectorised over triangles, bucketed by screen size); (H*W,) depth, inf = empty."""
    dev = u.device
    U, V_, Z = u[faces], v[faces], z[faces]
    ok = torch.isfinite(U).all(1) & torch.isfinite(V_).all(1) & (Z > 0.05).all(1)
    x0 = torch.floor(U.min(1).values).long()
    y0 = torch.floor(V_.min(1).values).long()
    x1 = torch.ceil(U.max(1).values).long()
    y1 = torch.ceil(V_.max(1).values).long()
    ok &= (x1 >= 0) & (x0 < W) & (y1 >= 0) & (y0 < H)
    x0, y0, x1, y1 = x0.clamp(0, W - 1), y0.clamp(0, H - 1), x1.clamp(0, W - 1), y1.clamp(0, H - 1)  # clip to the image
    den = (V_[:, 1] - V_[:, 2]) * (U[:, 0] - U[:, 2]) + (U[:, 2] - U[:, 1]) * (V_[:, 0] - V_[:, 2])
    ok &= den.abs() > 1e-8
    size = torch.maximum(x1 - x0, y1 - y0)
    dep = torch.full((H * W,), float("inf"), device=dev)
    lo = -1
    for hi in (12, 32, 80, 200, 600, 2000):
        sel = (ok & (size > lo) & (size <= hi)).nonzero()[:, 0]
        lo = hi
        if len(sel) == 0:
            continue
        u_, v_, z_, X0, Y0, X1, Y1, d_ = U[sel], V_[sel], Z[sel], x0[sel], y0[sel], x1[sel], y1[sel], den[sel]
        for dy in range(hi + 1):
            for dx in range(hi + 1):
                px, py = X0 + dx, Y0 + dy
                inb = (px <= X1) & (py <= Y1) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
                if not inb.any():
                    continue
                qx, qy = px.float() + 0.5, py.float() + 0.5
                l0 = ((v_[:, 1] - v_[:, 2]) * (qx - u_[:, 2]) + (u_[:, 2] - u_[:, 1]) * (qy - v_[:, 2])) / d_
                l1 = ((v_[:, 2] - v_[:, 0]) * (qx - u_[:, 2]) + (u_[:, 0] - u_[:, 2]) * (qy - v_[:, 2])) / d_
                l2 = 1 - l0 - l1
                hit = inb & (l0 >= -0.02) & (l1 >= -0.02) & (l2 >= -0.02)
                if not hit.any():
                    continue
                zz = l0 * z_[:, 0] + l1 * z_[:, 1] + l2 * z_[:, 2]
                dep.scatter_reduce_(0, (py[hit] * W + px[hit]), zz[hit], "amin")
    return dep


class DepthToControl:
    """Clothed-figure depth (Sapiens2Depth) as a control image: near = bright, background black, same scaling as the body fituin depth."""
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_control",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"shell_depth": ("MASK",), "person": ("MASK",)}}

    def run(self, shell_depth, person):
        sd = shell_depth.float()
        if sd.dim() == 2:
            sd = sd[None]
        H, W = sd.shape[-2:]
        dev = sd.device
        D = sd[:1][None]
        P = (_prep_mask(person, H, W, dev) > 0.5).float()
        zb = D[P > 0.5]
        lo, hi = (zb.min(), zb.max()) if zb.numel() else (D.min(), D.max())
        dp = ((1 - (D - lo) / (hi - lo + 1e-6)).clamp(0, 1) * P).repeat(1, 3, 1, 1)
        return (dp.permute(0, 2, 3, 1).cpu(),)


_SEG_CLASSES = ["Background", "Apparel", "Eyeglass", "Face_Neck", "Hair", "Left_Foot", "Left_Hand", "Left_Lower_Arm", "Left_Lower_Leg",
                "Left_Shoe", "Left_Sock", "Left_Upper_Arm", "Left_Upper_Leg", "Lower_Clothing", "Right_Foot", "Right_Hand",
                "Right_Lower_Arm", "Right_Lower_Leg", "Right_Shoe", "Right_Sock", "Right_Upper_Arm", "Right_Upper_Leg", "Torso",
                "Upper_Clothing", "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue"]
_SKIN = ["Face_Neck", "Left_Upper_Arm", "Right_Upper_Arm", "Left_Lower_Arm", "Right_Lower_Arm", "Left_Hand", "Right_Hand",
         "Left_Upper_Leg", "Right_Upper_Leg", "Left_Lower_Leg", "Right_Lower_Leg", "Left_Foot", "Right_Foot", "Torso"]
_CLOTH = ["Apparel", "Upper_Clothing", "Lower_Clothing", "Left_Shoe", "Right_Shoe", "Left_Sock", "Right_Sock"]
_FACE = ["Face_Neck", "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue"]
_DROP = ["Hair", "Eyeglass"]


class SapiensRegions:
    """Splits a Sapiens2 body-part segmentation into the regions the body fit needs, in one node."""
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK", "IMAGE", "MASK")
    RETURN_NAMES = ("person", "trusted", "clothing", "remove", "face", "preview", "fit_mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"class_id_mask": ("MASK", {"tooltip": "class_id_mask from Sapiens2 Seg"}),
                             "num_classes": ("INT", {"default": 29, "min": 2, "max": 256})},
                "optional": {"drop_classes": ("STRING", {"default": "Hair,Eyeglass", "tooltip":
                    "Comma-separated Sapiens classes removed from the silhouette (the Edit Planner decides these from the instruction; empty = none)"}),
                             "tight_classes": ("STRING", {"default": "", "tooltip":
                    "Comma-separated clothing classes that are form-fitting (the Edit Planner decides): they count as body evidence in fit_mask"}),
                             "covered_classes": ("STRING", {"default": "", "tooltip":
                    "Comma-separated skin classes the NEW outfit covers: they stop being trusted skin (so the old neckline / sleeves do not leak into the depth) and follow the fitted body instead"})}}

    def run(self, class_id_mask, num_classes, drop_classes="Hair,Eyeglass", covered_classes="", tight_classes=""):
        ids = (class_id_mask.float() * max(num_classes - 1, 1)).round().long()
        sel = lambda names: torch.isin(ids, torch.tensor([_SEG_CLASSES.index(n) for n in names], device=ids.device)).float()
        person = (ids != 0).float()
        drop_names = [c.strip() for c in (drop_classes or '').split(',') if c.strip() in _SEG_CLASSES]
        trusted, cloth, face = sel(_SKIN), sel(_CLOTH), sel(_FACE)
        tight_names = [c.strip() for c in (tight_classes or '').split(',') if c.strip() in _CLOTH]
        fit_mask = torch.clamp(trusted + (sel(tight_names) if tight_names else 0), 0, 1)  # skin + form-fitting clothing: what the body-slider fit measures
        cov_names = [c.strip() for c in (covered_classes or '').split(',') if c.strip() in _SKIN and c.strip() not in _FACE]
        if cov_names:
            cov = sel(cov_names)
            trusted = trusted * (1 - cov)
            cloth = torch.clamp(cloth + cov, 0, 1)
        drop = sel(drop_names) if drop_names else torch.zeros_like(person)
        pv = torch.zeros(ids.shape + (3,))
        pv[..., :] = (person * 0.25)[..., None]
        for m, c in ((trusted, (0.25, 0.8, 0.35)), (cloth, (0.9, 0.3, 0.3)), (drop, (0.3, 0.45, 0.95)), (face, (1.0, 0.85, 0.3))):
            m = m.cpu()[..., None] > 0.5
            pv = torch.where(m, torch.tensor(c).expand_as(pv), pv)
        return (person, trusted, cloth, drop, face, pv, fit_mask)


NODE_CLASS_MAPPINGS = {"SapiensRegions": SapiensRegions, "Sapiens2Depth": Sapiens2Depth, "MHRSliderFit": MHRSliderFit,
                       "DepthToControl": DepthToControl}
NODE_DISPLAY_NAME_MAPPINGS = {"SapiensRegions": "Sapiens Regions (person / skin / clothing / hair)", "Sapiens2Depth": "Sapiens2 Metric Depth",
                              "MHRSliderFit": "MHR Body Sliders Fit",
                              "DepthToControl": "Clothed Depth to Control Image"}
