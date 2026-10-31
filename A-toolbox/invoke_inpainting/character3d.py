"""Re-pose and re-camera the fitted MHR character, with a projected texture from the source image.

`MHRPoseTransfer`  copies the pose (body + hands, optionally the global orientation) from a reference image's SAM 3D Body
                   prediction onto the character: shape, proportions and the body-slider fit are kept.
`MHRViewRender`    renders the posed character from an orbited camera: control depth, shaded mannequin, and a colour image
                   projected from the source photo (pixels the source never saw come back as `unseen`)."""
import numpy as np
import torch
import torch.nn.functional as F

import comfy.model_management
from comfy_extras.sam3d_body import face_expression as fx

from .shell_fit import _raster_depth, _nconv, MHRSliderFit

_TK = MHRSliderFit.TKNOTS
(_LS, _RS, _LE, _RE, _LH, _RH, _LK, _RK, _LA, _RA) = 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
_LW, _RW = 62, 41


def _apply_sliders(V, kp, wk, dk, ls):
    """The same body-slider deformation MHRSliderFit optimises, applied to a mesh in a new pose."""
    dev = V.device
    tk = torch.tensor(_TK, device=dev)
    hipmid, neck = (kp[_LH] + kp[_RH]) / 2, (kp[_LS] + kp[_RS]) / 2
    up = F.normalize(neck - hipmid, dim=0)
    rr = kp[_LS] - kp[_RS]
    ex = F.normalize(rr - (rr @ up) * up, dim=0)
    ez = torch.cross(ex, up, dim=0)
    Ltr = (neck - hipmid).norm()
    segs = [("trunk", hipmid, neck, 0.42 * Ltr), ("uarmL", kp[_LS], kp[_LE], 0.14 * Ltr), ("uarmR", kp[_RS], kp[_RE], 0.14 * Ltr),
            ("farmL", kp[_LE], kp[_LW], 0.12 * Ltr), ("farmR", kp[_RE], kp[_RW], 0.12 * Ltr),
            ("thighL", kp[_LH], kp[_LK], 0.2 * Ltr), ("thighR", kp[_RH], kp[_RK], 0.2 * Ltr),
            ("calfL", kp[_LK], kp[_LA], 0.16 * Ltr), ("calfR", kp[_RK], kp[_RA], 0.16 * Ltr),
            ("head", neck, kp[0], 0.45 * Ltr), ("handL", kp[_LW], kp[_LW] + 1e-3, 0.25 * Ltr),
            ("handR", kp[_RW], kp[_RW] + 1e-3, 0.25 * Ltr), ("footL", kp[_LA], kp[_LA] + 1e-3, 0.22 * Ltr),
            ("footR", kp[_RA], kp[_RA] + 1e-3, 0.22 * Ltr)]

    def seg_dist(p0, p1):
        d = p1 - p0
        ln = d.norm().clamp(min=1e-6)
        dn = d / ln
        tt = ((V - p0) @ dn).clamp(0, ln)
        c = p0 + tt[:, None] * dn
        return (V - c).norm(dim=-1), c

    nd = torch.stack([seg_dist(s[1], s[2])[0] / s[3] for s in segs], 1)
    wts = torch.softmax(-nd * 6.0, dim=1)

    def interp(vals, t):
        tt = t.clamp(_TK[0], _TK[-1])
        i = torch.clamp(torch.bucketize(tt, tk) - 1, 0, len(_TK) - 2)
        t0, t1 = tk[i], tk[i + 1]
        f_ = ((tt - t0) / (t1 - t0)).clamp(0, 1)
        f_ = f_ * f_ * (3 - 2 * f_)
        return vals[i] * (1 - f_) + vals[i + 1] * f_

    q = V - hipmid
    a_, b_, t_ = q @ ex, q @ ez, (q @ up) / Ltr
    Vt = hipmid + (a_ * interp(wk, t_))[:, None] * ex + (b_ * interp(dk, t_))[:, None] * ez + (q @ up)[:, None] * up
    delta = wts[:, 0:1] * (Vt - V)
    lim_map = {"uarmL": 0, "uarmR": 0, "farmL": 1, "farmR": 1, "thighL": 2, "thighR": 2, "calfL": 3, "calfR": 3}
    for si, sg in enumerate(segs):
        if sg[0] in lim_map:
            _, c = seg_dist(sg[1], sg[2])
            delta = delta + wts[:, si:si + 1] * ((c + (V - c) * ls[lim_map[sg[0]]]) - V)
    return V + delta


def _cam(p):
    f = float(np.asarray(p["focal_length"]).reshape(-1)[0])
    kp2, kp3 = np.asarray(p["pred_keypoints_2d"], np.float32), np.asarray(p["pred_keypoints_3d"], np.float32)
    ct = np.asarray(p["pred_cam_t"], np.float32).reshape(3)
    x0, y0, z0 = kp3[0] + ct
    return f, float(kp2[0, 0] - f * x0 / max(z0, 1e-6)), float(kp2[0, 1] - f * y0 / max(z0, 1e-6)), ct


def _raster_ids(u, v, z, faces, H, W, dep):
    """Per pixel: the id of the front-most triangle (-1 = none) and its barycentric weights."""
    dev = u.device
    U, V_, Z = u[faces], v[faces], z[faces]
    ok = torch.isfinite(U).all(1) & torch.isfinite(V_).all(1) & (Z > 0.05).all(1)
    x0, y0 = torch.floor(U.min(1).values).long(), torch.floor(V_.min(1).values).long()
    x1, y1 = torch.ceil(U.max(1).values).long(), torch.ceil(V_.max(1).values).long()
    ok &= (x1 >= 0) & (x0 < W) & (y1 >= 0) & (y0 < H)
    x0, y0, x1, y1 = x0.clamp(0, W - 1), y0.clamp(0, H - 1), x1.clamp(0, W - 1), y1.clamp(0, H - 1)
    den = (V_[:, 1] - V_[:, 2]) * (U[:, 0] - U[:, 2]) + (U[:, 2] - U[:, 1]) * (V_[:, 0] - V_[:, 2])
    ok &= den.abs() > 1e-8
    size = torch.maximum(x1 - x0, y1 - y0)
    ids = torch.full((H * W,), -1, dtype=torch.long, device=dev)
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
                inb = (px <= X1) & (py <= Y1)
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
                idx = py.clamp(0, H - 1) * W + px.clamp(0, W - 1)
                m = hit & (zz <= dep[idx] + 1e-5)
                if m.any():
                    ids[idx[m]] = sel[m]
    have = ids >= 0
    bary = torch.zeros(H * W, 3, device=dev)
    if have.any():
        fid = ids[have]
        pu, pv = U[fid], V_[fid]
        pix = have.nonzero()[:, 0]
        qx, qy = (pix % W).float() + 0.5, (pix // W).float() + 0.5
        d_ = den[fid]
        l0 = ((pv[:, 1] - pv[:, 2]) * (qx - pu[:, 2]) + (pu[:, 2] - pu[:, 1]) * (qy - pv[:, 2])) / d_
        l1 = ((pv[:, 2] - pv[:, 0]) * (qx - pu[:, 2]) + (pu[:, 0] - pu[:, 2]) * (qy - pv[:, 2])) / d_
        bary[have] = torch.stack([l0, l1, 1 - l0 - l1], 1)
    return ids, bary


class MHRPoseTransfer:
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    RETURN_TYPES = ("MHR_POSE_DATA",)
    RETURN_NAMES = ("posed",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sam3d_body_model": ("SAM3D_BODY_MODEL",),
            "character": ("MHR_POSE_DATA", {"tooltip": "the fitted character (MHR Body Sliders Fit)"}),
            "copy_hands": ("BOOLEAN", {"default": True}),
            "copy_global_rotation": ("BOOLEAN", {"default": True, "tooltip": "also copy how the body is turned relative to the camera"}),
            "apply_sliders": ("BOOLEAN", {"default": True, "tooltip": "re-apply the character's fitted body sliders to the new pose"}),
        }, "optional": {"reference": ("MHR_POSE_DATA", {"tooltip": "SAM 3D Body prediction of the pose reference image; leave empty (bypass the reference loader) to keep the character's own pose"})}}

    def run(self, sam3d_body_model, character, copy_hands, copy_global_rotation, apply_sliders, reference=None):
        if reference is None:
            return (character,)
        comfy.model_management.load_model_gpu(sam3d_body_model)
        inner = sam3d_body_model.model
        dev = comfy.model_management.get_torch_device()
        frames = [[dict(p) for p in f] for f in character["frames"]]
        pc, pr = frames[0][0], reference["frames"][0][0]
        pc["body_pose_params"] = pr["body_pose_params"]
        if copy_hands:
            pc["hand_pose_params"] = pr["hand_pose_params"]
        if copy_global_rotation:
            pc["global_rot"] = pr["global_rot"]
        fx.regenerate_mesh_from_params(inner, frames)
        pc = frames[0][0]
        st = pc.get("slider_state")
        if apply_sliders and st:
            V = torch.from_numpy(np.asarray(pc["pred_vertices"], np.float32)).to(dev)
            kp = torch.from_numpy(np.asarray(pc["pred_keypoints_3d"], np.float32)).to(dev)
            T = lambda k: torch.tensor(st[k], device=dev, dtype=torch.float32)
            pc["pred_vertices"] = _apply_sliders(V, kp, T("wk"), T("dk"), T("ls")).cpu().numpy().astype(np.float32)
        out = dict(character)
        out["frames"] = frames
        if out.get("mhr_faces") is None:
            out["mhr_faces"] = inner.head_pose.faces.cpu()
        return (out,)


class MHRViewRender:
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("control_depth", "shaded", "color", "body_mask", "unseen")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "posed": ("MHR_POSE_DATA", {"tooltip": "the posed character (MHR Pose Transfer, or the fitted one for a pure camera move)"}),
            "source_pose_data": ("MHR_POSE_DATA", {"tooltip": "the fitted character in the SOURCE photo's pose (texture comes from there)"}),
            "source_image": ("IMAGE",),
            "yaw_degrees": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0, "tooltip": "orbit around the character: + turns the camera to the character's left"}),
            "pitch_degrees": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 1.0, "tooltip": "+ looks from above"}),
            "zoom": ("FLOAT", {"default": 1.0, "min": 0.3, "max": 3.0, "step": 0.05}),
            "fill_unseen": ("BOOLEAN", {"default": True, "tooltip": "smear nearby seen colours into never-seen surface so Qwen gets a plausible start"}),
        }}

    def run(self, posed, source_pose_data, source_image, yaw_degrees, pitch_degrees, zoom, fill_unseen):
        dev = comfy.model_management.get_torch_device()
        H, W = source_image.shape[1], source_image.shape[2]
        faces = posed.get("mhr_faces")
        if faces is None:
            faces = source_pose_data.get("mhr_faces")
        if faces is None:
            raise RuntimeError("MHRViewRender: no mesh faces; run MHR Body Sliders Fit / MHR Pose Transfer first")
        faces = faces.to(dev).long()
        pn, ps = posed["frames"][0][0], source_pose_data["frames"][0][0]
        f, cx, cy, ct = _cam(pn)
        Vp = torch.from_numpy(np.asarray(pn["pred_vertices"], np.float32)).to(dev) + torch.from_numpy(ct).to(dev)
        c = Vp.mean(0)
        a, b = np.radians(yaw_degrees), np.radians(pitch_degrees)
        Ry = torch.tensor([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]], dtype=torch.float32, device=dev)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]], dtype=torch.float32, device=dev)
        Vn = (Vp - c) @ (Rx @ Ry).T + c
        u = f * Vn[:, 0] / Vn[:, 2] + cx
        v = f * Vn[:, 1] / Vn[:, 2] + cy
        if zoom != 1.0:  # zoom about the character, not the frame corner
            uc, vc = u.mean(), v.mean()
            u, v = (u - uc) * zoom + uc, (v - vc) * zoom + vc
        z = Vn[:, 2]
        dep = _raster_depth(u, v, z, faces, H, W)
        ids, bary = _raster_ids(u, v, z, faces, H, W, dep)
        body = (ids >= 0)
        # depth control (near = bright)
        D = torch.where(body, dep, torch.zeros_like(dep))
        zb = D[body]
        lo, hi = (zb.min(), zb.max()) if zb.numel() else (D.min(), D.max())
        ctrl = ((1 - (D - lo) / (hi - lo + 1e-6)).clamp(0, 1) * body.float())
        # shading from face normals
        fv = Vn[faces]
        fn = F.normalize(torch.cross(fv[:, 1] - fv[:, 0], fv[:, 2] - fv[:, 0], dim=1), dim=1)
        out_sign = 1.0 if ((fv.mean(1) - Vn.mean(0)) * fn).sum(1).mean() >= 0 else -1.0
        fn = fn * out_sign  # outward
        L = F.normalize(torch.tensor([-0.4, -0.5, -0.8], device=dev), dim=0)
        facing = (fn @ L).clamp(0, 1)
        shade = torch.full((H * W,), 0.82, device=dev)
        shade[body] = facing[ids[body]] * 0.75 + 0.2
        # projective texture: where did each surface point appear in the source photo?
        fs, cxs, cys, cts = _cam(ps)
        Vs = torch.from_numpy(np.asarray(ps["pred_vertices"], np.float32)).to(dev) + torch.from_numpy(cts).to(dev)
        us = fs * Vs[:, 0] / Vs[:, 2].clamp(min=1e-6) + cxs
        vs = fs * Vs[:, 1] / Vs[:, 2].clamp(min=1e-6) + cys
        depth_s = _raster_depth(us, vs, Vs[:, 2], faces, H, W)
        color = torch.full((H * W, 3), 0.5, device=dev)
        unseen = torch.zeros(H * W, device=dev)
        if body.any():
            fid = ids[body]
            tri = faces[fid]
            bw = bary[body]
            su = (us[tri] * bw).sum(1)
            sv = (vs[tri] * bw).sum(1)
            sz = (Vs[:, 2][tri] * bw).sum(1)
            ix = su.floor().long().clamp(0, W - 1)
            iy = sv.floor().long().clamp(0, H - 1)
            inside = (su >= 0) & (su < W) & (sv >= 0) & (sv < H)
            seen = inside & (sz <= depth_s[iy * W + ix] + 0.03)
            # surface facing the source camera
            fsn = F.normalize(torch.cross(Vs[faces[:, 1]] - Vs[faces[:, 0]], Vs[faces[:, 2]] - Vs[faces[:, 0]], dim=1), dim=1)
            so = 1.0 if ((Vs[faces].mean(1) - Vs.mean(0)) * fsn).sum(1).mean() >= 0 else -1.0
            vdir = F.normalize(-Vs[faces].mean(1), dim=1)
            seen &= ((fsn * so)[fid] * vdir[fid]).sum(1) > 0.15
            img = source_image[0:1].permute(0, 3, 1, 2)[:, :3].to(dev)
            grid = torch.stack([su / W * 2 - 1, sv / H * 2 - 1], -1)[None, None]
            samp = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, 0, :].T
            col = color[body]
            col[seen] = samp[seen]
            color[body] = col
            un = unseen[body]
            un[~seen] = 1.0
            unseen[body] = un
        color_img = color.view(1, H, W, 3).permute(0, 3, 1, 2)
        unseen_m = unseen.view(1, 1, H, W)
        body_m = body.float().view(1, 1, H, W)
        if fill_unseen and (unseen_m > 0.5).any() and ((body_m > 0.5) & (unseen_m < 0.5)).any():
            seen_m = body_m * (1 - unseen_m)
            for sg in (12.0, 40.0):
                sm = torch.cat([_nconv(color_img[:, k:k + 1], seen_m, sg) for k in range(3)], 1)
                color_img = torch.where((unseen_m > 0.5) & (body_m > 0.5), sm, color_img)
                seen_m = torch.maximum(seen_m, unseen_m * body_m)
        to_img = lambda t: t.permute(0, 2, 3, 1).cpu().clamp(0, 1)
        shade_img = shade.view(1, 1, H, W).repeat(1, 3, 1, 1)
        ctrl_img = ctrl.view(1, 1, H, W).repeat(1, 3, 1, 1)
        return (to_img(ctrl_img), to_img(shade_img), to_img(color_img), body_m[:, 0].cpu(), unseen_m[:, 0].cpu())


def _umeyama(src, dst):
    """Similarity transform (s, R, t) with dst ~ s * R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    U, S, Vh = torch.linalg.svd(cov)
    d = torch.ones(3, device=src.device)
    if torch.det(U) * torch.det(Vh) < 0:
        d[-1] = -1
    R = U @ torch.diag(d) @ Vh
    var = (a ** 2).sum() / len(src)
    s = (S * d).sum() / var.clamp(min=1e-12)
    t = mu_d - s * R @ mu_s
    return s, R, t


def _nearest(a, b, chunk=6000):
    """index of the nearest row of b for every row of a, and the distance"""
    idx, dist = [], []
    for i in range(0, len(a), chunk):
        d = torch.cdist(a[i:i + chunk], b)
        m, j = d.min(1)
        idx.append(j)
        dist.append(m)
    return torch.cat(idx), torch.cat(dist)


class MeshRigPose:
    """Rig a generated (Pixal3D / TRELLIS) character mesh with the MHR skeleton and re-pose it.

    The mesh is aligned to the SAM 3D Body body of the same photo (similarity ICP), every mesh vertex copies the skinning
    weights of its nearest body vertices, and the joint motion between the photo's pose and the reference pose is applied
    with linear blend skinning. The output mesh is back in the input mesh's own frame, texture and UVs untouched."""
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    RETURN_TYPES = ("MESH", "STRING")
    RETURN_NAMES = ("posed_mesh", "report")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sam3d_body_model": ("SAM3D_BODY_MODEL",),
            "mesh": ("MESH", {"tooltip": "the generated, textured character"}),
            "character": ("MHR_POSE_DATA", {"tooltip": "SAM 3D Body prediction of the SAME photo the mesh was generated from"}),
            "copy_hands": ("BOOLEAN", {"default": True}),
            "copy_global_rotation": ("BOOLEAN", {"default": True}),
            "icp_iterations": ("INT", {"default": 40, "min": 0, "max": 200}),
            "weight_neighbours": ("INT", {"default": 4, "min": 1, "max": 16}),
        }, "optional": {"reference": ("MHR_POSE_DATA", {"tooltip": "pose to copy; empty keeps the photo's pose"})}}

    def run(self, sam3d_body_model, mesh, character, copy_hands, copy_global_rotation, icp_iterations, weight_neighbours, reference=None):
        import copy as _copy
        comfy.model_management.load_model_gpu(sam3d_body_model)
        inner = sam3d_body_model.model
        head = inner.head_pose
        rig = head.mhr
        dev = comfy.model_management.get_torch_device()
        pc = character["frames"][0][0]
        pt = dict(pc)
        if reference is not None:
            pr = reference["frames"][0][0]
            pt["body_pose_params"] = pr["body_pose_params"]
            if copy_hands:
                pt["hand_pose_params"] = pr["hand_pose_params"]
            if copy_global_rotation:
                pt["global_rot"] = pr["global_rot"]

        def fwd(p):
            g = lambda k: torch.from_numpy(np.asarray(p[k], np.float32))[None].to(dev)
            with torch.no_grad():
                verts, _kp, jc, _mp, jr = head.mhr_forward(
                    global_trans=torch.zeros_like(g("global_rot")), global_rot=g("global_rot"), body_pose_params=g("body_pose_params"),
                    hand_pose_params=g("hand_pose_params"), scale_params=g("scale_params"), shape_params=g("shape_params"),
                    expr_params=g("expr_params"), return_keypoints=True, return_joint_coords=True, return_model_params=True,
                    return_joint_rotations=True)
            return verts[0].float(), jc[0].float(), jr[0].float()

        Vs, Js, Rs = fwd(pc)
        _, Jt, Rt = fwd(pt)
        n = int(mesh.vertices.shape[1] if mesh.vertex_counts is None else mesh.vertex_counts[0])
        G = mesh.vertices[0, :n].to(dev).float()
        # --- align the generated mesh (any scale/offset) to the SAM 3D Body body in the photo's pose
        hs, hg = (Vs[:, 1].max() - Vs[:, 1].min()), (G[:, 1].max() - G[:, 1].min())
        s = (hs / hg.clamp(min=1e-6))
        R = torch.eye(3, device=dev)
        t = (Vs.max(0).values + Vs.min(0).values) / 2 - s * ((G.max(0).values + G.min(0).values) / 2)
        sub = G[torch.randperm(n, device=dev)[:4000]]
        for _ in range(icp_iterations):
            A = s * (sub @ R.T) + t
            j, d = _nearest(A, Vs)
            keep = d <= torch.quantile(d, 0.6)
            j2, _ = _nearest(Vs, A)
            src = torch.cat([sub[keep], sub[j2]])
            dst = torch.cat([Vs[j[keep]], Vs])
            s, R, t = _umeyama(src, dst)
        A = s * (sub @ R.T) + t
        _, dd = _nearest(A, Vs)
        Ga = s * (G @ R.T) + t
        # --- skin weights: dense per-joint weights of the body, copied to the mesh from the nearest body vertices
        NJ = Js.shape[0]
        Wm = torch.zeros(Vs.shape[0], NJ, device=dev)
        vi = rig.lbs_vert_indices.to(dev).long()
        si = rig.lbs_skin_indices.to(dev).long()
        Wm.index_put_((vi, si), rig.lbs_skin_weights.to(dev).float(), accumulate=True)
        Rrel = Rt @ Rs.transpose(1, 2)  # (NJ,3,3): rotates a point about joint j from the photo pose to the reference pose
        out = torch.empty_like(Ga)
        k = weight_neighbours
        for i in range(0, n, 6000):
            c = Ga[i:i + 6000]
            d = torch.cdist(c, Vs)
            dv, jx = d.topk(k, dim=1, largest=False)
            a = 1.0 / (dv + 1e-3)
            a = a / a.sum(1, keepdim=True)
            w = (Wm[jx] * a[..., None]).sum(1)  # (m, NJ)
            w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
            x = c[:, None, :] - Js[None]  # (m, NJ, 3)
            y = torch.einsum("jab,mjb->mja", Rrel, x) + Jt[None]
            out[i:i + 6000] = (w[..., None] * y).sum(1)
        Gp = ((out - t) / s) @ R  # back to the mesh's own frame
        new = _copy.copy(mesh)
        verts = mesh.vertices.clone()
        verts[0, :n] = Gp.to(verts.device, verts.dtype)
        new.vertices = verts
        new.normals = None
        new.tangents = None
        rep = (f"alignment: scale {float(s):.4f}, rms {float(dd.pow(2).mean().sqrt()):.4f} m, median {float(dd.median()):.4f} m; "
               f"vertices {n}; joints {NJ}; reference {'yes' if reference is not None else 'no (same pose)'}")
        return (new, rep)


class MHRNeutralPose:
    """Her fitted body in the neutral rest pose (MHR zero pose, facing the camera), framed full-body on the source image's canvas.
    Render its openpose skeleton / depth to have Qwen redraw her as a clean full-body A-pose reference."""
    CATEGORY = "invoke_inpainting"
    FUNCTION = "run"
    RETURN_TYPES = ("MHR_POSE_DATA",)
    RETURN_NAMES = ("neutral",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sam3d_body_model": ("SAM3D_BODY_MODEL",),
            "character": ("MHR_POSE_DATA", {"tooltip": "the fitted character (MHR Body Sliders Fit)"}),
            "image_width": ("INT", {"default": 832, "min": 64, "max": 8192}),
            "image_height": ("INT", {"default": 1248, "min": 64, "max": 8192}),
            "body_fraction": ("FLOAT", {"default": 0.88, "min": 0.3, "max": 1.0, "step": 0.01, "tooltip": "how much of the frame the whole body (height and arm span) fills at most"}),
            "apply_sliders": ("BOOLEAN", {"default": True}),
        }}

    def run(self, sam3d_body_model, character, image_width, image_height, body_fraction, apply_sliders):
        pc = character["frames"][0][0]
        f, cx, cy, _ = _cam(pc)
        z0 = lambda k: np.zeros_like(np.asarray(pc[k], np.float32))
        ref = {"frames": [[{"body_pose_params": z0("body_pose_params"), "hand_pose_params": z0("hand_pose_params"), "global_rot": z0("global_rot")}]]}
        (out,) = MHRPoseTransfer().run(sam3d_body_model, character, True, True, apply_sliders, reference=ref)
        p = out["frames"][0][0]
        V = np.asarray(p["pred_vertices"], np.float32)
        ymin, ymax = float(V[:, 1].min()), float(V[:, 1].max())
        xc, yc = float((V[:, 0].min() + V[:, 0].max()) / 2), (ymin + ymax) / 2
        xmin, xmax = float(V[:, 0].min()), float(V[:, 0].max())
        z = max(f * (ymax - ymin) / (body_fraction * image_height), f * (xmax - xmin) / (body_fraction * image_width))  # arms must fit too
        ct = np.array([(image_width / 2 - cx) * z / f - xc, (image_height / 2 - cy) * z / f - yc, z], np.float32)
        p["pred_cam_t"] = ct
        for kp3, kp2 in (("pred_keypoints_3d", "pred_keypoints_2d"), ("pred_face_keypoints_3d", "pred_face_keypoints_2d")):
            if p.get(kp3) is not None:
                k3 = np.asarray(p[kp3], np.float32) + ct
                p[kp2] = np.stack([f * k3[:, 0] / np.maximum(k3[:, 2], 1e-6) + cx, f * k3[:, 1] / np.maximum(k3[:, 2], 1e-6) + cy], -1)
        out["frames"][0][0] = p
        return (out,)


NODE_CLASS_MAPPINGS = {"MHRPoseTransfer": MHRPoseTransfer, "MHRViewRender": MHRViewRender, "MeshRigPose": MeshRigPose, "MHRNeutralPose": MHRNeutralPose}
NODE_DISPLAY_NAME_MAPPINGS = {"MHRPoseTransfer": "MHR Pose Transfer (copy a reference pose)",
                              "MHRViewRender": "MHR View Render (orbit camera + projected texture)",
                              "MeshRigPose": "Mesh Rig + Pose (generated mesh, MHR skeleton)",
                              "MHRNeutralPose": "MHR Neutral Pose (full-body A-pose reference)"}
