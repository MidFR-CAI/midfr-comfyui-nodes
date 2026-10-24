import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

function createModal(nodeId, initialText) {
    const overlay = document.createElement("div");
    overlay.style.cssText = `
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0,0,0,0.6);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
    `;

    const box = document.createElement("div");
    box.style.cssText = `
        background: #202020;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 16px;
        width: min(600px, 90vw);
        max-height: 80vh;
        display: flex;
        flex-direction: column;
        gap: 10px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.5);
    `;

    const title = document.createElement("div");
    title.textContent = `Edit text — node #${nodeId} (workflow paused)`;
    title.style.cssText = "color:#eee; font-weight:600; font-family:sans-serif; font-size:14px;";

    const textarea = document.createElement("textarea");
    textarea.value = initialText ?? "";
    textarea.style.cssText = `
        width: 100%;
        min-height: 240px;
        resize: vertical;
        background: #111;
        color: #eee;
        border: 1px solid #444;
        border-radius: 4px;
        padding: 8px;
        font-family: monospace;
        font-size: 13px;
        box-sizing: border-box;
    `;

    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex; justify-content:flex-end; gap:8px;";

    const abortBtn = document.createElement("button");
    abortBtn.textContent = "Abort";
    abortBtn.style.cssText = `
        background: #7a3a3a; color: #eee; border: none;
        padding: 8px 14px; border-radius: 4px; cursor: pointer; font-weight: 600;
    `;

    const resumeBtn = document.createElement("button");
    resumeBtn.textContent = "Resume";
    resumeBtn.style.cssText = `
        background: #3a7a3a; color: white; border: none;
        padding: 8px 18px; border-radius: 4px; cursor: pointer; font-weight: 600;
    `;

    btnRow.appendChild(abortBtn);
    btnRow.appendChild(resumeBtn);
    box.appendChild(title);
    box.appendChild(textarea);
    box.appendChild(btnRow);
    overlay.appendChild(box);
    document.body.appendChild(overlay);

    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);

    // Ctrl/Cmd+Enter also resumes, for convenience.
    textarea.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
            resumeBtn.click();
        }
    });

    async function submit(path, text) {
        try {
            await api.fetchApi(path, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ node_id: nodeId, text }),
            });
        } finally {
            overlay.remove();
        }
    }

    resumeBtn.onclick = () => {
        resumeBtn.disabled = true;
        abortBtn.disabled = true;
        resumeBtn.textContent = "Resuming...";
        submit("/textedit/submit", textarea.value);
    };

    abortBtn.onclick = () => {
        resumeBtn.disabled = true;
        abortBtn.disabled = true;
        abortBtn.textContent = "Aborting...";
        submit("/textedit/abort", textarea.value);
    };
}

app.registerExtension({
    name: "textedit.pause",
    async setup() {
        api.addEventListener("textedit.pause", (event) => {
            const { node_id, text } = event.detail;
            createModal(node_id, text);
        });
    },
});
