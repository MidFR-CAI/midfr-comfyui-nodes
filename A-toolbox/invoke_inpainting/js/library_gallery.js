import { app } from "../../scripts/app.js";

// Thumbnail gallery on "Input: image or saved model": click a character to select it (and switch use_library on).
const GALLERIES = {
    QwenInputSelect: { list: "/midfr/characters", thumb: "/midfr/character_thumb", combo: "character", toggle: "use_library", extra: null },
    QwenPoseSelect: { list: "/midfr/poses", thumb: "/midfr/pose_thumb", combo: "pose", toggle: "use_saved", extra: "use_pose" },
};

app.registerExtension({
    name: "midfr.libraryGallery",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const cfg = GALLERIES[nodeData.name];
        if (!cfg) return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const node = this;
            const wrap = document.createElement("div");
            wrap.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(70px,1fr));gap:4px;overflow:auto;max-height:190px;padding:2px;";
            const refresh = async () => {
                let names = [];
                try { names = await (await fetch(cfg.list)).json(); } catch (e) { return; }
                const cw = node.widgets.find((w) => w.name === cfg.combo);
                if (cw) { cw.options.values = names.length ? names : ["(empty)"]; if (!names.includes(cw.value) && names.length) cw.value = names[0]; }
                wrap.innerHTML = "";
                for (const n of names) {
                    const cell = document.createElement("div");
                    cell.title = n;
                    cell.style.cssText = "cursor:pointer;border:2px solid " + (cw?.value === n ? "#ffb84d" : "#444") + ";border-radius:4px;background:#111;text-align:center;font:10px sans-serif;color:#ddd;overflow:hidden";
                    cell.innerHTML = `<img src="${cfg.thumb}?name=${encodeURIComponent(n)}&t=${Date.now()}" style="width:100%;height:80px;object-fit:contain;display:block"><div style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${n}</div>`;
                    cell.onclick = () => {
                        if (cw) cw.value = n;
                        const uw = node.widgets.find((w) => w.name === cfg.toggle);
                        if (uw) uw.value = true;
                        const xw = cfg.extra && node.widgets.find((w) => w.name === cfg.extra);
                        if (xw) xw.value = true;
                        app.graph.setDirtyCanvas(true, true);
                        refresh();
                    };
                    wrap.appendChild(cell);
                }
            };
            this.addWidget("button", "refresh library", null, refresh);
            const dw = this.addDOMWidget("library_gallery", "gallery", wrap, { serialize: false, hideOnZoom: false });
            dw.computeSize = (w) => [w, 200];
            this.setSize([Math.max(this.size[0], 330), this.size[1] + 210]);
            setTimeout(refresh, 300);
            return r;
        };
    },
});
