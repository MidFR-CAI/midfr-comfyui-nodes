import { app } from "../../scripts/app.js";

// Mode Selection drives which subgraph is queued: before a prompt is built, the subgraph nodes whose
// properties.midfr_mode differs from the selected mode are muted (mode 2), so they never reach the backend.
const MODES = ["Simple edit", "Advanced edit", "Model extraction and edit"];

app.registerExtension({
    name: "midfr.modeMute",
    async setup() {
        const orig = app.queuePrompt.bind(app);
        app.queuePrompt = async function (...args) {
            const g = app.graph;
            const sel = g?._nodes?.find((n) => n.type === "QwenModeSelect");
            const saved = [];
            if (sel) {
                const w = sel.widgets?.find((x) => x.name === "mode");
                const idx = MODES.indexOf(w?.value);
                for (const n of g._nodes) {
                    const m = n.properties?.midfr_mode;
                    if (m === undefined || m === null) continue;
                    saved.push([n, n.mode]);
                    n.mode = Number(m) === idx ? 0 : 2;
                }
            }
            try {
                return await orig(...args);
            } finally {
                for (const [n, m] of saved) n.mode = m;
                if (saved.length) app.graph.setDirtyCanvas(true, true);
            }
        };
    },
});
