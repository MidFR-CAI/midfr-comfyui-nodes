import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Single "Live Preview" node: shows an image for every step the workflow reports (StepPreview nodes), newest large,
// earlier steps as a clickable thumbnail strip. Frontend-only node (not part of the API prompt).
const steps = [];   // {label, img}
let sel = -1;
const nodes = new Set();

function redraw() { nodes.forEach(n => n.setDirtyCanvas(true, true)); }

app.registerExtension({
    name: "midfr.livePreview",
    async setup() {
        api.addEventListener("execution_start", () => { steps.length = 0; sel = -1; redraw(); });
        api.addEventListener("midfr.step_preview", (e) => {
            const img = new Image();
            img.onload = () => { steps.push({ label: e.detail.label, img }); sel = steps.length - 1; redraw(); };
            img.src = "data:image/jpeg;base64," + e.detail.image;
        });
    },
    registerCustomNodes() {
        class LivePreview extends LGraphNode {
            constructor() {
                super("Live Preview");
                this.size = [460, 600];
                this.serialize_widgets = false;
                this.isVirtualNode = true;
                this.color = "#7a4a12"; this.bgcolor = "#3a2a14";
                this.thumbs = [];
                nodes.add(this);
            }
            onRemoved() { nodes.delete(this); }
            onMouseDown(e, pos) {
                const [w, h] = this.size;
                if (pos[1] > h - 78) {
                    for (const t of this.thumbs) {
                        if (pos[0] >= t.x && pos[0] <= t.x + t.w) { sel = t.i; redraw(); return true; }
                    }
                }
                return false;
            }
            onDrawForeground(ctx) {
                if (this.flags.collapsed) return;
                const [w, h] = this.size;
                const top = 34, strip = 78;
                ctx.save();
                ctx.fillStyle = "#111"; ctx.fillRect(8, top, w - 16, h - top - strip - 8);
                if (!steps.length) {
                    ctx.fillStyle = "#999"; ctx.font = "14px sans-serif"; ctx.textAlign = "center";
                    ctx.fillText("waiting for the first step...", w / 2, top + (h - top - strip) / 2);
                } else {
                    const s = steps[Math.max(0, sel)];
                    const aw = w - 16, ah = h - top - strip - 8;
                    const k = Math.min(aw / s.img.width, ah / s.img.height);
                    const dw = s.img.width * k, dh = s.img.height * k;
                    ctx.drawImage(s.img, 8 + (aw - dw) / 2, top + (ah - dh) / 2, dw, dh);
                    ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(8, top, aw, 22);
                    ctx.fillStyle = "#fff"; ctx.font = "13px sans-serif"; ctx.textAlign = "left";
                    ctx.fillText(`${Math.max(0, sel) + 1}/${steps.length}  ${s.label}`, 14, top + 16);
                }
                // thumbnail strip (last 8 steps)
                this.thumbs = [];
                const first = Math.max(0, steps.length - 8);
                const tw = Math.min(70, (w - 16) / 8 - 4);
                for (let i = first; i < steps.length; i++) {
                    const x = 8 + (i - first) * (tw + 4), y = h - strip + 2;
                    const s = steps[i];
                    const k = Math.min(tw / s.img.width, (strip - 22) / s.img.height);
                    ctx.drawImage(s.img, x, y, s.img.width * k, s.img.height * k);
                    ctx.strokeStyle = i === sel ? "#ffb84d" : "#444"; ctx.lineWidth = 2;
                    ctx.strokeRect(x, y, tw, strip - 22);
                    ctx.fillStyle = "#ddd"; ctx.font = "9px sans-serif"; ctx.textAlign = "left";
                    ctx.fillText(s.label.slice(0, 13), x, h - 8);
                    this.thumbs.push({ x, w: tw, i });
                }
                ctx.restore();
            }
        }
        LivePreview.title = "Live Preview (workflow steps)";
        LivePreview.category = "invoke_inpainting/pipeline";
        LiteGraph.registerNodeType("MidfrLivePreview", LivePreview);
    },
});
