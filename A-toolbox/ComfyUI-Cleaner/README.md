# Resource Reaper

Frees memory before every ComfyUI generation, and tells you what else on the machine is holding onto the GPU.

## Install

```
cd ComfyUI/custom_nodes
# copy this folder into custom_nodes
pip install psutil nvidia-ml-py
```

Use the same Python that runs ComfyUI. On a portable Windows build that is:

```
python_embeded\python.exe -m pip install psutil nvidia-ml-py
```

Restart ComfyUI. The console should print `[ResourceReaper] loaded — auto-clean on, mode adaptive`, and a **Memory** tab appears in the sidebar.

Neither dependency is required: without `psutil` you lose the suggestions, without `nvidia-ml-py` you lose per-program VRAM figures, and the cleaning still works.

## How "always first" works

A node cannot guarantee it runs first. ComfyUI orders nodes by data dependency, so a node only runs before whatever consumes its output — put a cleanup node anywhere in the middle of a graph and half the models are already loaded by the time it fires.

So the automatic pass does not use a node. It wraps `PromptExecutor.execute`, which is the one function every queued prompt passes through, on the worker thread, after the job leaves the queue and before any node runs. That is genuinely first.

The wrap is installed at startup and is disabled by the **Free memory before every run** setting rather than by unpatching, so toggling it takes effect immediately and never leaves ComfyUI in a half-patched state.

## Cleaning levels

| Level | What it does | Cost |
|---|---|---|
| Light | Python GC, flush the CUDA allocator cache. Models stay resident. | None |
| Standard | Light, plus ComfyUI releases models it considers unneeded, plus a forced allocator flush. | Small |
| Deep | Standard, plus unload every model and collect IPC handles. | Full model reload next run |

**Adaptive** is the default and the only setting that doesn't tax every run: it reads free VRAM first and picks light, standard or deep from there. Below 25% free it goes deep; within 15 points of that it goes standard; otherwise light. Change the threshold in the panel.

**Drop cached node results** is separate and off by default. It clears ComfyUI's per-node result cache and its object storage — which is what actually holds checkpoints, LoRAs and CLIP models in system RAM between runs. It reclaims the most, and it forces your whole graph to re-execute next time, so it's throttled to every Nth run.

## The nodes

Three, under `utils/resource reaper`. All of them report `IS_CHANGED = NaN`, so they never cache and always fire.

- **Free Memory (passthrough)** — cleans, then forwards whatever is wired through it. To make it truly first in a graph, right-click your checkpoint loader, convert `ckpt_name` to an input, and route it through this node. Nothing can load before the loader gets its filename.
- **Free Memory After Run** — cleans once images exist, so VRAM sits free while you're looking at the result rather than while you're waiting for the next one.
- **Memory Status** — prints current VRAM and what the last automatic pass reclaimed. For checking whether any of this is helping.

## The panel

The sidebar tab polls `/reaper/status` every three seconds while it's open.

The VRAM bar is split: blue is ComfyUI, the coloured remainder is everything else. If the remainder is large, cleaning inside ComfyUI won't help you and the suggestions below will say what to close.

Suggestions are ranked by severity and are specific — *Close chrome.exe, holding 2.1 GB of VRAM on GPU 0* rather than "free up memory". They cover other programs holding VRAM, RAM and swap pressure, ComfyUI's own footprint, and low disk space on the volume it writes to.

Closing programs from the panel is off by default. Turn on **Let this panel close other programs** and each actionable suggestion gets a close button with a confirmation. It sends a normal quit request and waits five seconds, never a force kill, and it refuses anything on the protected list — OS services, the desktop session, and ComfyUI itself.

## If it's still slowing down

Cleaning treats the symptom. The usual causes, roughly in order of how often they're the real problem:

- **VRAM spilling into shared system memory.** On Windows the NVIDIA driver silently starts using system RAM when VRAM fills, instead of erroring. Everything keeps working and runs about ten times slower. In the panel this shows as VRAM pegged near 100%. Adaptive mode catches it; disabling *CUDA - Sysmem Fallback Policy* for python.exe in the NVIDIA Control Panel makes it fail loudly instead, which is often what you want while diagnosing.
- **Model thrash.** Two large models that don't fit together, being swapped in and out every run. Deep cleaning makes this worse, not better. Fix the workflow.
- **RAM filling with cached models.** Turn on cache dropping, or start ComfyUI with `--disable-smart-memory`, or add `--cache-lru 1` to limit what's kept.
- **The browser tab.** Every previewed image stays in tab memory. A ComfyUI tab left open for a few hundred generations can hold several gigabytes. Reload it.

The console line after each pass — how much was reclaimed and how long it took — tells you which of these you have. If passes consistently reclaim near zero and things still slow down, the problem isn't memory you can free from inside ComfyUI.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /reaper/status` | Memory, processes, suggestions, settings, last pass |
| `POST /reaper/clean` | `{"level": "standard", "purge_execution_cache": false}` |
| `GET, POST /reaper/config` | Read and write settings, persisted to `config.json` |
| `POST /reaper/terminate` | `{"pid": 1234}`, 403 unless closing is enabled |
