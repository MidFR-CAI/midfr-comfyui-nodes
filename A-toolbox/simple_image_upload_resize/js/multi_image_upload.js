import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NUM_SLOTS = 9;
const NONE = "none";

function viewUrl(value) {
  const i = value.lastIndexOf("/");
  const sub = i >= 0 ? value.slice(0, i) : "";
  const name = i >= 0 ? value.slice(i + 1) : value;
  const p = new URLSearchParams({ filename: name, subfolder: sub });
  // Small server-made thumbnail, never the full-resolution file.
  return api.apiURL("/multi_image_upload/thumb?" + p.toString());
}

async function uploadFile(file) {
  const body = new FormData();
  body.append("image", file);
  body.append("type", "input");
  const resp = await api.fetchApi("/upload/image", { method: "POST", body });
  if (resp.status !== 200) throw new Error(`${resp.status} ${resp.statusText}`);
  const data = await resp.json();
  return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

app.registerExtension({
  name: "SimpleImageUploadResize.MultiImageUpload",

  async nodeCreated(node) {
    if (node.comfyClass !== "MultiImageUploadResize") return;

    // Hidden value holders (serialized with the workflow).
    const valueWidgets = [];
    for (let i = 1; i <= NUM_SLOTS; i++) {
      const w = node.widgets.find((x) => x.name === `image_${i}`);
      if (!w) return;
      w.hidden = true;
      w.computeSize = () => [0, -4];
      valueWidgets.push(w);
    }

    // ---- UI ----
    const root = document.createElement("div");
    Object.assign(root.style, {
      display: "flex", flexDirection: "column", gap: "6px",
      width: "100%", height: "100%", boxSizing: "border-box", pointerEvents: "auto",
    });

    const toolbar = document.createElement("div");
    Object.assign(toolbar.style, { display: "flex", alignItems: "center", gap: "8px" });
    const counter = document.createElement("span");
    Object.assign(counter.style, { flex: "1", fontSize: "12px", opacity: "0.7" });
    const clearAll = document.createElement("button");
    clearAll.textContent = "Clear all";
    Object.assign(clearAll.style, {
      padding: "3px 10px", cursor: "pointer", borderRadius: "4px",
      border: "1px solid #666", background: "#333", color: "#eee", fontSize: "12px",
    });
    toolbar.append(counter, clearAll);

    const grid = document.createElement("div");
    Object.assign(grid.style, {
      display: "grid", flex: "1", minHeight: "0", gap: "6px",
      gridTemplateColumns: "repeat(3, 1fr)", gridTemplateRows: "repeat(3, 1fr)",
    });
    root.append(toolbar, grid);

    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = "image/*";
    fileInput.multiple = true;
    fileInput.style.display = "none";
    root.append(fileInput);
    let pickIndex = 0;

    const boxes = [];

    const getVal = (i) => valueWidgets[i].value || NONE;
    const setVal = (i, v) => {
      valueWidgets[i].value = v || NONE;
      node.setDirtyCanvas(true, true);
    };

    function refresh() {
      let filled = 0;
      boxes.forEach((b, i) => {
        const v = getVal(i);
        if (v !== NONE) {
          filled++;
          if (b.img.dataset.v !== v) {  // only reload when the file actually changed
            b.img.dataset.v = v;
            b.img.src = viewUrl(v);
          }
          b.img.style.display = "block";
          b.hint.style.display = "none";
          b.x.style.display = "block";
          b.box.title = v;
          b.box.style.borderStyle = "solid";
        } else {
          delete b.img.dataset.v;
          b.img.removeAttribute("src");
          b.img.style.display = "none";
          b.hint.style.display = "flex";
          b.x.style.display = "none";
          b.box.title = "Drop an image here or click to browse";
          b.box.style.borderStyle = "dashed";
        }
      });
      counter.textContent = `${filled} / ${NUM_SLOTS} loaded`;
    }

    async function loadFiles(files, startIndex) {
      files = [...files].filter((f) => f.type.startsWith("image/"));
      if (!files.length) return;
      // First file -> the target box; the rest -> following empty boxes.
      const targets = [startIndex];
      for (let i = 0; i < NUM_SLOTS && targets.length < files.length; i++) {
        if (i !== startIndex && getVal(i) === NONE) targets.push(i);
      }
      for (let k = 0; k < Math.min(files.length, targets.length); k++) {
        try {
          setVal(targets[k], await uploadFile(files[k]));
        } catch (e) {
          console.error("Upload failed", e);
          alert("Upload failed: " + e.message);
        }
      }
      refresh();
    }

    for (let i = 0; i < NUM_SLOTS; i++) {
      const box = document.createElement("div");
      Object.assign(box.style, {
        position: "relative", border: "2px dashed #666", borderRadius: "6px",
        background: "#1e1e1e", overflow: "hidden", cursor: "pointer", minHeight: "0",
      });
      const img = document.createElement("img");
      Object.assign(img.style, {
        width: "100%", height: "100%", objectFit: "contain", display: "none",
        pointerEvents: "none",
      });
      img.draggable = false;
      img.decoding = "async";
      const hint = document.createElement("div");
      hint.textContent = `${i + 1}\nDrop image`;
      Object.assign(hint.style, {
        position: "absolute", inset: "0", display: "flex", alignItems: "center",
        justifyContent: "center", textAlign: "center", whiteSpace: "pre-line",
        fontSize: "12px", opacity: "0.5", pointerEvents: "none",
      });
      const badge = document.createElement("div");
      badge.textContent = i + 1;
      Object.assign(badge.style, {
        position: "absolute", left: "4px", top: "2px", fontSize: "11px",
        background: "rgba(0,0,0,.6)", color: "#fff", padding: "0 5px",
        borderRadius: "3px", pointerEvents: "none",
      });
      const x = document.createElement("div");
      x.textContent = "✕";
      x.title = "Clear this image";
      Object.assign(x.style, {
        position: "absolute", right: "4px", top: "2px", display: "none",
        background: "rgba(0,0,0,.7)", color: "#fff", width: "18px", height: "18px",
        lineHeight: "18px", textAlign: "center", borderRadius: "50%", fontSize: "11px",
      });
      x.addEventListener("click", (e) => {
        e.stopPropagation();
        setVal(i, NONE);
        refresh();
      });

      box.addEventListener("click", () => { pickIndex = i; fileInput.click(); });
      box.addEventListener("dragover", (e) => {
        e.preventDefault(); e.stopPropagation();
        box.style.borderColor = "#4a9eff";
      });
      box.addEventListener("dragleave", () => { box.style.borderColor = "#666"; });
      box.addEventListener("drop", (e) => {
        e.preventDefault(); e.stopPropagation();
        box.style.borderColor = "#666";
        if (e.dataTransfer?.files?.length) loadFiles(e.dataTransfer.files, i);
      });

      box.append(img, hint, badge, x);
      grid.append(box);
      boxes.push({ box, img, hint, x });
    }

    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) loadFiles(fileInput.files, pickIndex);
      fileInput.value = "";
    });
    clearAll.addEventListener("click", () => {
      for (let i = 0; i < NUM_SLOTS; i++) setVal(i, NONE);
      refresh();
    });

    const widget = node.addDOMWidget("image_slots", "custom", root, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: () => 380,
    });
    widget.computeSize = (w) => [w, 380];

    // Restore previews when a saved workflow is loaded.
    const origConfigure = node.onConfigure;
    node.onConfigure = function () {
      const r = origConfigure ? origConfigure.apply(this, arguments) : undefined;
      refresh();
      return r;
    };

    const sz = node.computeSize();
    node.setSize([Math.max(sz[0], 380), Math.max(sz[1], 480)]);
    refresh();
  },
});
