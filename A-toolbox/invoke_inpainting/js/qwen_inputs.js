import { app } from "../../scripts/app.js";

// "Qwen Inputs": one node for the reference image / 3D model, saved pose and typed reference images.
const REF_TYPES = {
    "Outfit": "Wear this clothing",
    "Background": "Use this scene behind the character",
    "Same Character": "Another angle of the same character (more accurate 3D model)",
    "Other Character": "A second character to include in the image",
    "Face": "Use this face / identity",
    "Hairstyle": "Use this hairstyle and colour",
    "Object / Prop": "Add this object to the image",
    "Style": "Match this art style and colours",
    "Lighting": "Match this lighting",
};
const MAX_REFS = 3;
const C = { bg: "#2b1d10", panel: "#3a2816", line: "#6b4a26", acc: "#ffb84d", txt: "#f1e6d8", dim: "#b9a58c" };

const el = (tag, css, html) => { const e = document.createElement(tag); if (css) e.style.cssText = css; if (html !== undefined) e.innerHTML = html; return e; };
const btn = (label, fn, css = "") => { const b = el("button", `background:${C.panel};color:${C.txt};border:1px solid ${C.line};border-radius:4px;padding:3px 8px;cursor:pointer;font:11px sans-serif;${css}`, label); b.onclick = fn; return b; };
const sel = (opts, val, fn, css = "") => {
    const s = el("select", `background:#1c130b;color:${C.txt};border:1px solid ${C.line};border-radius:4px;padding:2px;font:11px sans-serif;${css}`);
    for (const o of opts) { const op = el("option"); op.value = o; op.textContent = o; if (o === val) op.selected = true; s.appendChild(op); }
    s.onchange = () => fn(s.value); return s;
};
const imgUrl = (n) => `/view?filename=${encodeURIComponent(n)}&type=input&t=${Date.now()}`;

async function upload(file) {
    const fd = new FormData(); fd.append("image", file); fd.append("type", "input"); fd.append("overwrite", "true");
    const r = await fetch("/upload/image", { method: "POST", body: fd }); const j = await r.json(); return j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
}

app.registerExtension({
    name: "midfr.qwenInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "QwenInputs") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;
            // the frontend auto-previews anything it thinks is an input image under the node; this node draws its own thumbnails
            Object.defineProperty(node, "imgs", { get: () => null, set: () => {}, configurable: true });
            const cw = node.widgets.find((w) => w.name === "config");
            cw.type = "hidden"; cw.hidden = true; cw.computeSize = () => [0, -4];
            if (cw.element) cw.element.style.display = "none";
            const cfg = Object.assign({ image: "", use_model: false, model: "", use_pose: false, pose_from: "image", pose_image: "", pose: "", save_pose: true, use_refs: false, refs: [] }, (() => { try { return JSON.parse(cw.value); } catch (e) { return {}; } })());
            const st = { inputs: [], lib: [], poses: [], note: "" };
            const root = el("div", `background:${C.bg};color:${C.txt};font:12px sans-serif;padding:6px;border-radius:6px;width:100%;box-sizing:border-box;`);
            const save = () => { cw.value = JSON.stringify(cfg); };
            const fit = () => requestAnimationFrame(() => { dw.computeSize = (w) => [w, root.scrollHeight + 6]; const s = node.computeSize(); node.setSize([Math.max(node.size[0], 400), s[1]]); app.graph.setDirtyCanvas(true, true); });

            const picker = (value, onChange, label) => {
                const wrap = el("div", `display:flex;gap:8px;align-items:flex-start;margin:4px 0;`);
                const th = el("div", `width:84px;height:104px;background:#120c06;border:1px solid ${C.line};border-radius:4px;display:flex;align-items:center;justify-content:center;overflow:hidden;flex:none`);
                if (value) th.innerHTML = `<img src="${imgUrl(value)}" style="max-width:100%;max-height:100%">`; else th.innerHTML = `<span style="color:${C.dim};font-size:10px">no image</span>`;
                const col = el("div", "display:flex;flex-direction:column;gap:4px;flex:1;min-width:0");
                if (label) col.appendChild(el("div", `color:${C.dim};font-size:11px`, label));
                col.appendChild(sel(["(choose)"].concat(st.inputs), value || "(choose)", (v) => { onChange(v === "(choose)" ? "" : v); render(); }, "width:100%"));
                const f = el("input"); f.type = "file"; f.accept = "image/*"; f.style.display = "none";
                f.onchange = async () => { if (f.files[0]) { const n = await upload(f.files[0]); await loadInputs(); onChange(n); render(); } };
                col.appendChild(btn("Upload image…", () => f.click())); col.appendChild(f);
                wrap.appendChild(th); wrap.appendChild(col); return wrap;
            };
            const gallery = (items, selected, onPick, thumb, badges) => {
                const g = el("div", "display:grid;grid-template-columns:repeat(auto-fill,minmax(84px,1fr));gap:5px;max-height:210px;overflow:auto;margin:4px 0");
                if (!items.length) g.appendChild(el("div", `color:${C.dim};font-size:11px;grid-column:1/-1`, "Nothing saved yet."));
                for (const it of items) {
                    const name = it.name ?? it;
                    const c = el("div", `cursor:pointer;border:2px solid ${selected === name ? C.acc : C.line};border-radius:4px;background:#120c06;overflow:hidden;text-align:center;font-size:10px`);
                    c.title = name;
                    c.innerHTML = `<img src="${thumb(name)}" style="width:100%;height:90px;object-fit:contain;display:block"><div style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;padding:1px 2px">${name}</div>${badges ? badges(it) : ""}`;
                    c.onclick = () => { onPick(name); render(); };
                    g.appendChild(c);
                }
                return g;
            };
            const section = (title, hint) => { const d = el("div", `border-top:1px solid ${C.line};margin-top:6px;padding-top:5px`); d.appendChild(el("div", `font-weight:600;color:${C.acc}`, title)); if (hint) d.appendChild(el("div", `color:${C.dim};font-size:11px;margin-bottom:3px`, hint)); return d; };
            const check = (label, hint, key) => {
                const d = el("div", `border-top:1px solid ${C.line};margin-top:6px;padding-top:5px`);
                const row = el("label", "display:flex;gap:6px;align-items:center;cursor:pointer;font-weight:600");
                const cb = el("input"); cb.type = "checkbox"; cb.checked = !!cfg[key]; cb.onchange = () => { cfg[key] = cb.checked; render(); };
                row.appendChild(cb); row.appendChild(el("span", "", label)); d.appendChild(row);
                d.appendChild(el("div", `color:${C.dim};font-size:11px;margin:1px 0 3px 22px`, hint));
                return d;
            };

            const importPanel = () => {
                const d = el("div", `background:${C.panel};border:1px dashed ${C.line};border-radius:4px;padding:5px;margin-top:4px`);
                d.appendChild(el("div", `font-weight:600`, "Import a 3D model"));
                d.appendChild(el("div", `color:${C.dim};font-size:11px`, "Pick the model (.glb .gltf .obj .stl) together with its texture / .mtl files. Untextured models are textured from the reference image; models without a rig are rigged automatically when a pose is applied. FBX / USDZ: convert to GLB first."));
                const f = el("input"); f.type = "file"; f.multiple = true; f.accept = ".glb,.gltf,.obj,.stl,.mtl,.png,.jpg,.jpeg,.bin"; f.style.cssText = "display:none";
                const row = el("div", "display:flex;gap:4px;flex-wrap:wrap;margin-top:4px;align-items:center");
                const nm = el("input", `background:#1c130b;color:${C.txt};border:1px solid ${C.line};border-radius:4px;padding:2px;width:110px;font:11px sans-serif`); nm.placeholder = "name (optional)";
                let up = "auto", fc = "+Z";
                row.appendChild(el("span", `color:${C.dim};font-size:11px`, "Up")); row.appendChild(sel(["auto", "Y", "Z"], up, (v) => (up = v)));
                row.appendChild(el("span", `color:${C.dim};font-size:11px`, "Faces")); row.appendChild(sel(["+Z", "-Z", "+X", "-X"], fc, (v) => (fc = v)));
                row.appendChild(nm);
                const msg = el("div", `color:${C.dim};font-size:11px;margin-top:3px`, st.note);
                f.onchange = async () => {
                    if (!f.files.length) return;
                    msg.textContent = "Importing…";
                    const fd = new FormData(); for (const x of f.files) fd.append("files", x, x.name);
                    fd.append("name", nm.value); fd.append("up", up); fd.append("facing", fc);
                    const r = await fetch("/midfr/import_model", { method: "POST", body: fd }); const j = await r.json();
                    if (j.error) { st.note = "Import failed: " + j.error; render(); return; }
                    await loadLib(); cfg.model = j.name; st.note = `Imported “${j.name}”` + (j.textured ? " with its texture." : " (no texture found: it will be textured from the reference image)."); render();
                };
                row.appendChild(btn("Choose files & import…", () => f.click())); row.appendChild(f);
                d.appendChild(row); d.appendChild(msg); return d;
            };

            const render = () => {
                save(); root.innerHTML = "";
                // 1 · input
                if (!cfg.use_model) {
                    const s = section("Reference image", "The image the edit starts from.");
                    s.appendChild(picker(cfg.image, (v) => (cfg.image = v))); root.appendChild(s);
                } else {
                    const s = section("3D model library", "Pick a saved or imported model; it replaces the reference image.");
                    s.appendChild(gallery(st.lib, cfg.model, (n) => (cfg.model = n), (n) => `/midfr/character_thumb?name=${encodeURIComponent(n)}`,
                        (it) => { const t = []; if (it.imported) t.push("imported"); if (!it.textured) t.push("no texture"); if (!it.rigged) t.push("auto-rig"); return t.length ? `<div style="font-size:9px;color:${C.acc}">${t.join(" · ")}</div>` : ""; }));
                    s.appendChild(btn("Refresh library", async () => { await loadLib(); render(); }));
                    s.appendChild(importPanel());
                    s.appendChild(el("div", `color:${C.dim};font-size:11px;margin-top:4px`, "Texture / identity reference (optional, used when the model has no texture):"));
                    s.appendChild(picker(cfg.image, (v) => (cfg.image = v)));
                    root.appendChild(s);
                }
                // 2 · checkboxes
                const c1 = check("Use 3D Model", "Switch the input to your 3D model library (saved characters and imported models).", "use_model"); root.appendChild(c1);
                const c2 = check("Use Saved Pose", "Apply a pose to the model or character: from a saved pose, or from the pose in an image.", "use_pose");
                if (cfg.use_pose) {
                    const tabs = el("div", "display:flex;gap:4px;margin:3px 0 3px 22px");
                    for (const [k, l] of [["saved", "Saved pose"], ["image", "From image"]]) tabs.appendChild(btn(l, () => { cfg.pose_from = k; render(); }, cfg.pose_from === k ? `border-color:${C.acc};color:${C.acc};` : ""));
                    c2.appendChild(tabs);
                    if (cfg.pose_from === "saved") {
                        c2.appendChild(gallery(st.poses, cfg.pose, (n) => (cfg.pose = n), (n) => `/midfr/pose_thumb?name=${encodeURIComponent(n)}`));
                        c2.appendChild(btn("Refresh poses", async () => { await loadPoses(); render(); }));
                    } else {
                        c2.appendChild(picker(cfg.pose_image, (v) => (cfg.pose_image = v), "Image of the pose to copy"));
                        const row = el("label", `display:flex;gap:6px;align-items:center;color:${C.dim};font-size:11px;cursor:pointer`);
                        const cb = el("input"); cb.type = "checkbox"; cb.checked = !!cfg.save_pose; cb.onchange = () => { cfg.save_pose = cb.checked; save(); };
                        row.appendChild(cb); row.appendChild(el("span", "", "Save this pose to the pose library (named after the image)")); c2.appendChild(row);
                    }
                }
                root.appendChild(c2);
                const c3 = check("Use Other Reference", "Add up to 3 more images and say what each one is for.", "use_refs");
                if (cfg.use_refs) {
                    cfg.refs.forEach((rf, i) => {
                        const box = el("div", `background:${C.panel};border:1px solid ${C.line};border-radius:4px;padding:4px;margin:4px 0`);
                        const head = el("div", "display:flex;gap:6px;align-items:center");
                        head.appendChild(el("span", `color:${C.dim};font-size:11px`, `Reference ${i + 1} is`));
                        head.appendChild(sel(Object.keys(REF_TYPES), rf.type, (v) => { rf.type = v; render(); }, "flex:1"));
                        head.appendChild(btn("✕", () => { cfg.refs.splice(i, 1); render(); }));
                        box.appendChild(head);
                        box.appendChild(el("div", `color:${C.acc};font-size:10px;margin:2px 0`, REF_TYPES[rf.type]));
                        box.appendChild(picker(rf.image, (v) => (rf.image = v)));
                        c3.appendChild(box);
                    });
                    if (cfg.refs.length < MAX_REFS) c3.appendChild(btn(`+ Add reference (${cfg.refs.length}/${MAX_REFS})`, () => { cfg.refs.push({ type: "Outfit", image: "" }); render(); }));
                }
                root.appendChild(c3);
                fit();
            };

            const loadInputs = async () => { try { st.inputs = await (await fetch("/midfr/input_images")).json(); } catch (e) { } };
            const loadLib = async () => { try { st.lib = await (await fetch("/midfr/library")).json(); } catch (e) { } };
            const loadPoses = async () => { try { st.poses = (await (await fetch("/midfr/poses")).json()); } catch (e) { } };
            const dw = node.addDOMWidget("qwen_inputs_ui", "div", root, { serialize: false, hideOnZoom: false });
            node.setSize([420, 360]);
            node.onConfigure = ((orig) => function () {
                const r = orig?.apply(this, arguments);
                try { Object.assign(cfg, JSON.parse(cw.value)); } catch (e) { }
                render(); return r;
            })(node.onConfigure);
            (async () => { await Promise.all([loadInputs(), loadLib(), loadPoses()]); if (!cfg.image && st.inputs.length && !cfg.use_model) cfg.image = st.inputs[0]; render(); })();
            return r;
        };
    },
});
