# Lora Vault

A lean ComfyUI LoRA manager built around four ideas:

1. **Download only what you use.** Saving a LoRA stores its metadata and previews (a few hundred KB).
   The weights download the first time a workflow runs, and old unused ones are deleted when the
   cache passes your size cap.
2. **📌 Pin the ones you can't lose.** Pinned LoRAs are kept forever and never evicted.
3. **Know when Civitai removes something.** A background check re-queries every saved model. If one
   disappears you keep its last metadata + previews, a `backups/<id>/snapshot.json`, and (if the
   weights were on disk) they are auto-pinned.
4. **Everything goes through Firefox.** ComfyUI never connects to Civitai. The Firefox extension runs
   every request (API, search, previews, file downloads) with Firefox's own connection and Civitai
   login, then hands the result to ComfyUI.

No subscription, licence or account of any kind. A Civitai API key is optional (the Firefox login is used).

## Install

1. Copy this folder to `ComfyUI/custom_nodes/comfyui-lora-vault` and restart ComfyUI.
   (No pip installs - it only uses what ComfyUI already has.)
2. **Firefox extension** (folder `extension/`):
   - Quick try: `about:debugging` → *This Firefox* → *Load Temporary Add-on…* → pick `extension/manifest.json`.
     Temporary add-ons disappear when Firefox restarts.
   - Permanent: Firefox Developer Edition / Nightly with `xpinstall.signatures.required = false`
     in `about:config`, or sign it for yourself (free, "unlisted") with `npx web-ext sign`.
   - Approve the permissions it asks for (civitai.com, downloads).
   - **Keep it permanently (normal Firefox):** Firefox forgets unsigned add-ons on restart. Get Mozilla to
     sign your private copy (free, nothing is published): make an account at addons.mozilla.org, open
     <https://addons.mozilla.org/developers/addon/api/key/> to create API keys, then run
     `extension\sign_extension.bat <JWT issuer> <JWT secret>`. Install the `.xpi` it produces in `extension\signed\`
     via `about:addons` → gear → *Install Add-on From File*. When you change the extension, raise `version`
     in `manifest.json` and sign again.
   - (Mozilla may put an unlisted submission into manual review for days; the Developer Edition route below needs no review.)
   - **Or** use Firefox Developer Edition / Nightly / ESR: set `xpinstall.signatures.required` to `false`
     in `about:config`, zip the contents of `extension/` (manifest.json at the zip's top level), rename to
     `.xpi`, install it from file.
3. Log in to civitai.com in that Firefox. Keep Firefox open while ComfyUI downloads/searches.
4. Open `http://127.0.0.1:8188/lora_vault` (the vault manager). The header shows
   **● Firefox helper connected** when the extension is talking to ComfyUI.

## Use

- Add a **Lora Vault Loader** node (*loaders → lora vault*). Its panel has:
  - a **base-model dropdown** (Civitai's own list) and a **search box** that searches all of Civitai,
    filtered to that base model; **Add** puts the LoRA in your vault and in the node;
  - **Saved** to browse what you've already added; **In node** to see what's loaded;
  - per-LoRA cards: preview, trigger words (click to copy), strengths, stats, description, tags,
    licence, other versions, example-image prompts, 📌 Pin, "Check still exists", warnings
    (different base model, removed from Civitai).
- **Every Civitai model type is accepted and treated as a LoRA.** If an uploader tagged a LoRA as Checkpoint, Other, etc., it still saves and loads
  (the original tag stays on the card as a chip). The vault search has an **all types** checkbox that lifts the LoRA-only filter.
- The node outputs `model`, `clip`, `trigger_words` and a `loaded` summary. Text format is
  `<lora:v12345:0.8>` or `<lora:v12345:0.8:0.6>` (model:clip); names and file names also work.
- Each card has **one strength** (step 0.05) that applies to both model and clip. The **⚙ cog** holds the
  advanced options: a **layer range** (first/last layer, blank = all) and **split clip / model strength**.
  Text form: `<lora:v12345:0.8:layers=14-41>` (also `14-` or `-20`), `<lora:v12345:0.8:0.6:layers=14-41>`.
- **Fused qkv LoRAs are split automatically.** Some LoRAs store attention as one fused `qkv` weight, which
  ComfyUI silently drops on models with separate q/k/v. The loader splits these exactly (no approximation), so
  they actually apply. The `loaded` output reports e.g. `split 24 fused qkv`.
- On Civitai model pages the extension adds an **Add to Lora Vault** button (and ✔ badges on cards you own).

## Vault page (`/lora_vault`)

- Category chips under the header (Character, Style, Concept, Clothing, Poses, Background, Tool, Vehicle,
  Buildings, Objects, Animal, Action, Assets, Base Model) filter by the tags Civitai put on each model; pick several to combine.
- Library grouped by model with a chip per saved version. **Click one** to open the detail view:
  preview gallery, per-image generation data (prompt, negative prompt, steps, sampler, CFG, seed,
  resources, copy buttons), version + model description, trigger words, stats, licence, tags, file info.
- **Versions / collections:** every version on the Civitai page is listed. Save any of them, or
  *Save all N versions*; use several together in one node (`Add also` in the node panel).
- **Blur mature previews** toggle (header, and in the node bar): one setting shared everywhere.
  Untick it to see mature previews automatically; click a blurred image to reveal just that one.
- Mature previews: civitai.com's API hides them from a version's own image list, so the vault also
  queries the images endpoint (and civitai.red) and remembers images it has already seen.

## What is saved locally

| | Unpinned | 📌 Pinned |
|---|---|---|
| Text metadata (descriptions, trigger words, every example's prompt/steps/sampler…) | yes | yes |
| Preview images on disk | 1 (setting) | up to 40 (setting) |
| Weights | download cache (evicted) | kept forever |
| Self-contained export next to the weights (`<name>.vault.json`, `.txt`, `.previews/`) | no | yes |

Unpinning removes the export and trims previews back. A refresh never deletes previews you already have.

## Where things live

| What | Where |
|---|---|
| Library, settings, previews, removal backups | `ComfyUI/user/lora_vault/` |
| Download cache (evicted oldest-first past the cap) | `<loras folder>/_vault/cache/` |
| Pinned weights (kept forever) | `<loras folder>/_vault/pinned/` |

## Notes & limits

- **Firefox and ComfyUI must run on the same PC** (Firefox saves the file; ComfyUI moves it into place).
- Firefox has to be open for searching, saving, downloading and the daily removal check. If it isn't,
  you get a clear error and nothing is marked removed.
- A removal needs **two** consecutive "not found" checks, except when Civitai says
  Archived/TakenDown. Civitai hides mature models on some hosts, so each lookup tries
  `civitai.com` then `civitai.red` before counting a miss.
- The extension talks to a small local server on `127.0.0.1:8199` (extension origins only).
- Tests: `cd tests && python -m pytest` (offline, against a fake Civitai).


## Downloads: Firefox finds the link, ComfyUI downloads the file

With the default `download_mode: direct` (Settings -> "Downloading the LoRA file"), everything Civitai-related
(API, search, previews) still goes through Firefox, but the LoRA itself is downloaded **by ComfyUI directly from
Civitai's CDN** so a slow VPN/proxy in Firefox doesn't throttle it. Firefox only follows your Civitai download link
(with your login / key) until it sees the signed CDN address, stops, and hands that address back. ComfyUI streams it
with resume and a SHA-256 check; your API key is never sent to the CDN. If that can't work (older extension,
expired link, CDN unreachable) the vault falls back to a normal Firefox download automatically. Choose
"Through Firefox's download manager" to always use Firefox. Needs extension 0.4.0 (adds the `webRequest` permission).

## Batch remove

On the vault page: **Select…** (top bar) turns on select mode. Click tiles to pick them, shift-click for a range, or
drag a box over several (Ctrl/Shift-drag adds to the selection). **Select all shown** picks everything the current
search / base-model / category filter shows, which is how you remove a whole category. **Remove selected…** removes
every saved version of each picked model and asks whether to delete the downloaded files too (pinned items are called
out). In a model's detail view, **Remove all versions** does the same for just that model. Esc leaves select mode.

## Load LoRA (layers, model only)

A small companion node for LoRAs already in `models/loras`: pick the file, a strength, and `start_layer` / `end_layer` (inclusive, `-1` = to the end). Fused qkv LoRAs are split automatically, same as the vault loader.
