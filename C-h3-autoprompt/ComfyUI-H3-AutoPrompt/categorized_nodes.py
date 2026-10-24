"""
categorized_nodes.py -- the new pipeline: MiniMaxH3MediaLoader (the original
pack, untouched) -> H3 Subject Categorizer (this pack's new panel, tags each
item) -> H3 Categorized Composer (this file) -> legal H3 prompt.

Replaces vlm_engine.py's SYSTEM_PROMPT-driven classification entirely for
this path. describe_engine.py's much narrower prompt is the only model call
left that decides content; everything format-critical (grouping, retention,
anchor status, audio role, voice-donor pairing) is explicit user input via
categorize.py, not inferred.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from . import categorize as cat
from . import grammar as g
from . import media_io
from .describe_engine import DescribeEngine
from .vlm_engine import LlamaCppVLM, TransformersVLM, VLMConfig
from .whisper_engine import FasterWhisperTranscriber, DialogueClip, SpeakerTurn, transcribe_clip
from . import composer as c

CATEGORY = "conditioning/video_models/h3_autoprompt"

RETENTION_EXPLANATIONS = {
    "fully_preserved": "this reference is reproduced exactly -- pose, framing, and setting "
                        "are carried into the target video as shown.",
    "partially_preserved": "the referenced content is used, but some defined characteristics "
                            "are changed or only partially retained.",
    "attribute_transfer": "only the subject's defining visual traits (not the specific "
                           "pose/setting shown) are carried into the target video.",
    "weak_reference": "only broad similarity in style, category, or atmosphere is retained.",
}


# --------------------------------------------------------------------------
# Tensor indexing -- mirrors the ORIGINAL pack's own _partition()/computeTags()
# ordering exactly (verified against their real nodes.py / web/medialoader.js),
# so "Picture 2" here always means the same picture their own editor would
# call "Picture 2".
# --------------------------------------------------------------------------

def _index_bundle(references: dict) -> Dict[str, object]:
    items = references.get("items") or []
    pictures = references.get("pictures") or []
    videos = references.get("videos") or []
    video_audios = references.get("video_audios") or []
    audios = references.get("audios") or []

    out: Dict[str, object] = {}
    p = v = a = 0
    vid_pos = 0
    for it in items:
        if it.get("enabled") is False:
            continue
        if it.get("kind") == "picture":
            if p < len(pictures):
                out[f"Picture {p + 1}"] = pictures[p]
            p += 1

    for it in items:
        if it.get("enabled") is False or it.get("kind") != "video":
            continue
        if v < len(videos):
            out[f"Video {v + 1}"] = videos[v]
        has_paired = it.get("has_audio") and (it.get("audio_mode") or "paired") == "paired"
        if has_paired:
            a += 1
            if vid_pos < len(video_audios) and video_audios[vid_pos] is not None:
                out[f"Audio {a}"] = video_audios[vid_pos]
        v += 1
        vid_pos += 1

    j = 0
    for it in items:
        if it.get("enabled") is False:
            continue
        is_std_audio = it.get("kind") == "audio"
        is_std_video_audio = (it.get("kind") == "video" and it.get("has_audio")
                               and it.get("audio_mode") == "standalone")
        if is_std_audio or is_std_video_audio:
            a += 1
            if j < len(audios):
                out[f"Audio {a}"] = audios[j]
            j += 1

    return out


# --------------------------------------------------------------------------
# H3 Subject Categorizer -- UI-only node. Its real job is the panel (web/
# categorizer.js); this class just carries the categorized item JSON through
# alongside the untouched references bundle.
# --------------------------------------------------------------------------

class H3SubjectCategorizer:
    """Wire the ORIGINAL pack's 'Fantastic H3 Media Loader' references output
    straight into this node's `references` input, then use the panel (click
    the node) to tag each item: which subject it belongs to, whether it's an
    identity reference or a concrete frame anchor, and for audio, whether
    it's dialogue or a voice-timbre donor. Output feeds H3 Categorized
    Composer."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("H3_REFS", "STRING")
    RETURN_NAMES = ("references", "categorized_items_json")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "references": ("H3_REFS",),
                # Written by the panel: the SAME items list from `references`,
                # each dict extended with cat_* fields. Kept separate from
                # `references` itself (rather than mutating it) so this node
                # is idempotent and safe to re-run.
                "categorized_items": ("STRING", {"multiline": True, "default": "[]"}),
            },
        }

    def run(self, references: dict, categorized_items: str = "[]"):
        try:
            items = json.loads(categorized_items or "[]")
        except Exception:
            items = []
        # Fall back to the bundle's own (uncategorized) items if the panel
        # hasn't been used yet, so wiring this node in doesn't break anything
        # -- everything just defaults to no subject assignment, which the
        # Composer reports clearly rather than silently guessing one.
        if not items:
            items = list(references.get("items") or [])
        out = dict(references)
        out["items"] = items
        return (out, json.dumps(items))


# --------------------------------------------------------------------------
# H3 Categorized Composer
# --------------------------------------------------------------------------

class H3CategorizedComposer:
    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "validation_report")
    FUNCTION = "compose"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "references": ("H3_REFS",),
            "brief": ("STRING", {"multiline": True, "default": ""}),
            "duration_seconds": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 30.0}),
            "vlm_backend": (["llama_cpp_gguf", "transformers_hf"],),
            "vlm_model_path_or_id": ("STRING", {"default": ""}),
            "vlm_mmproj_path": ("STRING", {"default": ""}),
            "whisper_model_size": ("STRING", {"default": "large-v3-turbo"}),
            "device": (["cuda", "cpu"],),
            "use_diarization": ("BOOLEAN", {"default": True}),
            # Real limits vary by model/mmproj/llama-cpp-python build -- seen in
            # practice: a build that silently drops images past some cutoff instead
            # of erroring cleanly (MTMDChatHandler "not all media inputs were
            # consumed", consumed < media_items). Rather than guess one number that
            # might be wrong for someone else's setup, this is a widget: lower it if
            # you hit that error, raise it if your build/hardware can handle more.
            "max_vlm_images": ("INT", {"default": 6, "min": 1, "max": 24}),
            # Was hardcoded (8192) with no way to change it -- silent input truncation
            # ("cutting off important parts of the prompt") without an error is one of
            # the likely symptoms of running out of headroom here, given how much the
            # system prompt and image count have both grown. Raise it if your model/
            # hardware can support more context; lower it if you're VRAM-constrained.
            "vlm_n_ctx": ("INT", {"default": 16384, "min": 2048, "max": 131072}),
            # Written by this node's own "Open categorizer..." panel button (web/
            # categorizer.js) -- the categorization step now lives on this same
            # node instead of requiring a separate H3SubjectCategorizer upstream.
            # That node still exists for workflows that want categorization as
            # its own step feeding multiple consumers, but isn't required.
            "categorized_items": ("STRING", {"multiline": True, "default": "[]"}),
            # Written by the panel's "Shots" section (one text box per shot, added there
            # specifically to stop bleeding): a JSON list of per-shot brief strings.
            # When non-empty, this REPLACES "brief" as the shot-writing input entirely --
            # shot count and boundaries are then the user's own explicit choice, not a
            # VLM judgment call, which is what was letting one shot's content/dialogue
            # bleed into another. "brief" still applies when this is empty.
            "shot_briefs": ("STRING", {"multiline": True, "default": "[]"}),
        }}

    def compose(self, references: dict, brief: str, duration_seconds: float,
                vlm_backend: str, vlm_model_path_or_id: str, vlm_mmproj_path: str,
                whisper_model_size: str, device: str, use_diarization: bool,
                max_vlm_images: int = 6, vlm_n_ctx: int = 16384, categorized_items: str = "[]",
                shot_briefs: str = "[]"):
        try:
            items = json.loads(categorized_items or "[]")
        except Exception:
            items = []
        # Same fallback H3SubjectCategorizer used: if the panel hasn't been used
        # yet (or this node is fed straight from a loader with nothing saved),
        # fall back to the bundle's own uncategorized items rather than erroring
        # -- everything just defaults to no subject assignment, which shows up
        # clearly in the plan-summary line below instead of failing silently.
        if not items:
            items = list(references.get("items") or [])
        plan = cat.build_plan(items)
        problems_pre: List[str] = []

        if not plan.subjects and not plan.anchors:
            problems_pre.append(
                "No item has a subject/anchor assignment yet -- open the H3 Subject "
                "Categorizer panel and tag at least one picture or video before composing."
            )

        # Raw per-item categorization as actually received by Python -- shows exactly
        # what cat_* fields each item carries, independent of what build_plan() did
        # with them. Cuts through "is the panel not saving, or is build_plan dropping
        # it" without needing another round of guessing: if an item you categorized
        # doesn't show its cat_* fields here at all, the panel/Save is the problem;
        # if it DOES show them but isn't reflected in "Categorized: ..." above, the
        # parsing in build_plan() is the problem.
        item_diag_lines = []
        for it in items:
            if it.get("enabled") is False:
                continue
            cat_fields = {k: v for k, v in it.items() if k.startswith("cat_")}
            if cat_fields:
                item_diag_lines.append(f"  {it.get('kind')}:{it.get('file', '?')} -> {cat_fields}")
        item_diag = ("Raw per-item categorization received:\n" + "\n".join(item_diag_lines)
                     if item_diag_lines else
                     "Raw per-item categorization received: NONE of the loaded items carry "
                     "any cat_* field at all -- the panel's Save likely isn't reaching this "
                     "node's categorized_items widget.")

        tensor_index = _index_bundle(references)

        # -- gather images for the description call, one per subject/anchor -----
        # Deduplicated by tag: a picture that's BOTH an anchor AND a subject source
        # (a combination this conversation has been using a lot) used to get sent
        # to the VLM as two separate image blocks with the same underlying pixel
        # data -- wasteful of context, and very likely the direct cause of a real
        # crash seen in practice: MTMDChatHandler(_process_mtmd_prompt): "not all
        # media inputs were consumed by MTMD chunks (consumed=3, media_items=4)".
        # That mismatch is consistent with the chat template/vision encoder not
        # producing a chunk per listed image block when some of those blocks are
        # byte-identical duplicates. Sending each unique picture at most once,
        # with a combined label when it serves multiple roles, avoids the
        # duplication entirely rather than trying to work around whatever the
        # underlying chunking bug actually is.
        images: List = []
        labels: List[str] = []
        _sent_tag_index: Dict[str, int] = {}

        def _add_image(tag: str, extra_label: str) -> bool:
            if tag in _sent_tag_index:
                idx = _sent_tag_index[tag]
                labels[idx] = f"{labels[idx]} / {extra_label}"
                return True
            if len(images) >= max_vlm_images:
                return False
            t = tensor_index.get(tag)
            if t is None:
                return False
            img = media_io.image_tensor_to_pil(t if t.dim() == 4 and t.shape[0] == 1 else t[:1])
            images.append(img)
            labels.append(extra_label)
            _sent_tag_index[tag] = len(images) - 1
            return True

        # Priority order matters now that there's a hard budget: identity/anchor
        # images are essential (without them the VLM can't describe who a subject
        # is at all), so they're always added first and never displaced. Only
        # whatever budget remains goes to choreography frames.
        for s in plan.subjects:
            for src_tag in s.sources[:2]:  # a couple of references is plenty for a description
                kind_hint = f", a {s.kind}" if s.kind and s.kind != "person" else ""
                _add_image(src_tag, f"Subject {s.slot} ({src_tag}{kind_hint})")
        for anchor in plan.anchors:
            _add_image(anchor.tag, f"{anchor.tag} (anchor)")

        # Character-swap videos get MULTIPLE frames spread across their timeline, not
        # just one -- the worked example this feature is built from describes the
        # source footage's actual observed motion second-by-second ("at the same
        # three-second mark he raises both hands..."), which is only possible if the
        # VLM can actually see how the performance changes over time. A single frame
        # (what every other reference gets) can't support that; this is the one case
        # where the composer trades "narrow VLM call" for "enough visual context to
        # describe a fixed choreography faithfully" -- still not asking it to invent
        # new content, only to transcribe what's actually in the footage.
        #
        # But this competes with the same max_vlm_images budget as everything else --
        # sending it unconditionally is what caused a real crash in practice
        # (MTMDChatHandler "not all media inputs were consumed", consumed=6 of 9,
        # right after this feature started sending up to 6 extra frames per swap
        # video with no cap at all). Whatever budget is left after identity/anchor
        # images is split evenly across the swap video's available frames, sampled
        # spread across the timeline rather than just taking the first N -- fewer,
        # evenly-spaced frames is a better trade than dropping the later half of
        # the performance entirely.
        swap_video_tags = {vs.tag for vs in plan.video_structures if vs.is_character_swap}
        remaining = max(0, max_vlm_images - len(images))
        per_video_budget = remaining // max(1, len(swap_video_tags)) if swap_video_tags else 0
        for tag in swap_video_tags:
            if per_video_budget <= 0:
                break
            t = tensor_index.get(tag)
            if t is None or t.dim() != 4:
                continue
            available = t.shape[0]
            n_frames = min(available, per_video_budget)
            if n_frames <= 0:
                continue
            idxs = [round(i * (available - 1) / max(1, n_frames - 1)) for i in range(n_frames)] \
                if n_frames > 1 else [0]
            for i in idxs:
                approx_t = (i / max(1, available - 1)) * duration_seconds
                img = media_io.image_tensor_to_pil(t[i:i + 1])
                images.append(img)
                labels.append(f"{tag} at ~{approx_t:.1f}s (choreography reference)")

        subject_tags_for_prompt = [f"Subject {s.slot}" for s in plan.subjects] + \
                                   [a.tag for a in plan.anchors]

        # -- narrow VLM call: descriptions + shots only --------------------------
        if vlm_backend == "llama_cpp_gguf":
            backend = LlamaCppVLM(VLMConfig(model_path=vlm_model_path_or_id,
                                             mmproj_path=vlm_mmproj_path,
                                             n_gpu_layers=-1 if device == "cuda" else 0,
                                             n_ctx=vlm_n_ctx))
        else:
            backend = TransformersVLM(model_id=vlm_model_path_or_id, device=device)
        describer = DescribeEngine(backend)
        choreography_notes = [
            (tag, "This video is a character-swap edit: describe its ACTUAL observed "
                  "motion and actions across the timestamped frames above in "
                  "chronological detail (e.g. \"at the X-second mark, ...\"), since this "
                  "exact choreography must be preserved with only the performer's "
                  "identity replaced -- do not invent new narrative content for this "
                  "portion, transcribe what the frames actually show.")
            for tag in swap_video_tags
        ]
        try:
            raw_shot_briefs = json.loads(shot_briefs or "[]")
            if not isinstance(raw_shot_briefs, list):
                raw_shot_briefs = []
        except Exception:
            raw_shot_briefs = []
        # Backward-compatible: earlier saves are bare strings (just shot text). Normalize
        # everything to {"text": str, "dialogue": [...]} so explicit per-shot dialogue
        # attribution (added later) can ride the same shot_briefs channel.
        shot_brief_objs = [
            {"text": s, "dialogue": []} if isinstance(s, str)
            else {"text": s.get("text", ""), "dialogue": s.get("dialogue") or []}
            for s in raw_shot_briefs
        ]
        shot_brief_list = [s["text"] for s in shot_brief_objs]

        result = describer.describe(images, labels, brief, duration_seconds,
                                     notes=plan.notes + choreography_notes,
                                     shot_briefs=shot_brief_list)

        descriptions = result.get("subject_descriptions") or {}
        description_diag = (
            f"Sent {len(images)} image(s) to the VLM {labels}. "
            f"VLM returned subject_descriptions keys: {list(descriptions.keys()) or '(none/empty)'}."
        )
        for s in plan.subjects:
            s.description = _find_description(descriptions, f"Subject {s.slot}") or \
                "a subject defined by the loaded reference media"
        # When a picture serves BOTH as an anchor and as a subject source (the same
        # image cited twice, once each way), the VLM often only writes one
        # description for it -- reasonably, from its perspective the two labels
        # ("Subject 1 (Picture 1)" and "Picture 1 (anchor)") point at the same
        # picture, so a second write-up reads redundant. Reuse the matching
        # subject's description instead of falling back to the generic placeholder
        # in that specific case; only use the placeholder when the anchor genuinely
        # isn't shared with any subject.
        anchor_desc = {}
        for a in plan.anchors:
            found = _find_description(descriptions, a.tag)
            if not found:
                sharing_subject = next((s for s in plan.subjects if a.tag in s.sources), None)
                if sharing_subject and sharing_subject.description:
                    found = sharing_subject.description
            anchor_desc[a.tag] = found

        vlm_shots = result.get("shots") or [{
            "content": brief.strip() or "The scene plays out as described.",
            "subject_mentions": subject_tags_for_prompt, "duration_hint_s": duration_seconds,
        }]

        shot_count_notice = None
        if shot_brief_list:
            # Deterministic backstop: the VLM is told to return exactly one shot per
            # user-provided box, in order, but that's a prompt instruction, not a
            # guarantee -- it can still merge two boxes into one shot or split one box
            # into two. Since the whole point of separate boxes is to fix shot count
            # and boundaries as explicit, non-negotiable user input, a mismatch here
            # is corrected deterministically rather than trusted to have gone right:
            # pad missing shots directly from their own box's raw text (no VLM polish,
            # but guaranteed correct boundary and guaranteed not to contain another
            # shot's content), or truncate extras back to the requested count.
            # original_count is captured BEFORE any mutation -- vlm_shots and
            # result["shots"] are the same list object, so appending to one silently
            # mutates the other; re-reading result["shots"] after padding would always
            # report the already-padded count instead of what the VLM actually returned.
            n = len(shot_brief_list)
            original_count = len(vlm_shots)
            if original_count < n:
                for i in range(original_count, n):
                    vlm_shots.append({
                        "content": shot_brief_list[i], "camera": None,
                        "subject_mentions": subject_tags_for_prompt,
                        "duration_hint_s": duration_seconds / n, "dialogue": [],
                    })
                shot_count_notice = (f"VLM returned {original_count} shot(s) "
                                      f"for {n} user-defined shot box(es) -- padded the "
                                      f"missing one(s) directly from their own box text.")
            elif original_count > n:
                shot_count_notice = (f"VLM returned {original_count} shot(s) for {n} "
                                      f"user-defined shot box(es) -- truncated to the "
                                      f"requested count.")
                vlm_shots = vlm_shots[:n]

            # Explicit per-shot dialogue attribution (from the panel's dialogue-line
            # controls) always wins over whatever the VLM guessed for "dialogue" in that
            # same shot -- this is the whole point of giving it its own UI instead of
            # relying on the VLM to infer who's speaking from prose. subject_N stays in
            # the same "Subject N" string format the VLM itself would use, so downstream
            # handling doesn't need to know which source a line came from; a custom
            # non-subject label rides along as "speaker_label" for the synthetic-clip
            # builder below to key a stable, distinct (Sx) off of.
            for i, sb in enumerate(shot_brief_objs):
                if i >= len(vlm_shots) or not sb["dialogue"]:
                    continue
                vlm_shots[i]["dialogue"] = [
                    {"speaker_subject": f"Subject {d['subject']}" if d.get("subject") else None,
                     "speaker_label": (d.get("label") or "").strip() or None,
                     "text": d.get("text", ""), "language": d.get("language") or "English",
                     "voiceover": bool(d.get("voiceover"))}
                    for d in sb["dialogue"] if (d.get("text") or "").strip()
                ]

        # Deterministic backstop: the VLM is explicitly told never to put dialogue words
        # in "content" (they go only in the structured "dialogue" list), but in practice
        # it sometimes populates BOTH -- the structured field correctly, and also restates
        # the same line as quoted prose in content, producing duplicated dialogue in the
        # final output (once properly <d>-tagged, once as loose narrative text). Rather
        # than rely on prompt compliance alone for something with a clean deterministic
        # fix, strip any of a shot's own dialogue text back out of its content if it
        # shows up there verbatim.
        _scrub_duplicated_dialogue(vlm_shots)

        # -- Whisper: only on items the user tagged "dialogue" -------------------
        transcriber = FasterWhisperTranscriber(model_size=whisper_model_size, device=device,
            compute_type="float16" if device == "cuda" else "int8")
        dialogue_clips: List[DialogueClip] = []
        for ap in plan.audio:
            if ap.role != "dialogue":
                continue
            audio_t = tensor_index.get(ap.tag)
            if audio_t is None:
                continue
            wav_path = _write_temp_wav(audio_t)
            dur = audio_t["waveform"].shape[-1] / audio_t["sample_rate"]
            clip = transcribe_clip(transcriber, wav_path, ap.tag, dur,
                                    use_pyannote=use_diarization)
            dialogue_clips.append(clip)

        speaker_ids = c.assign_speaker_ids(dialogue_clips)
        voiceover_clips = {ap.tag for ap in plan.audio if ap.role == "dialogue" and ap.voiceover}
        sung_clips = {ap.tag for ap in plan.audio if ap.role == "dialogue" and ap.sung}
        subject_of_clip = {ap.tag: f"Subject {ap.subject_slot}" for ap in plan.audio
                            if ap.role == "dialogue" and ap.subject_slot}
        speaker_labels = {}
        for key in speaker_ids:
            clip_tag = key.split("::", 1)[0]
            ap = next((a for a in plan.audio if a.tag == clip_tag), None)
            default = "an unseen narrator" if clip_tag in voiceover_clips else "the speaker"
            speaker_labels[key] = (ap.speaker_description if ap and ap.speaker_description
                                    else default)

        # -- brief-specified dialogue: turn into synthetic clips, reuse the SAME
        # <d>-formatting pipeline as real transcribed audio, rather than trusting the
        # VLM to hand-format <d> tags/language/speaker IDs correctly inside free prose
        # (which is exactly how it ended up unformatted, missing a language tag, AND
        # inventing lines nobody asked for -- there was no structured place for this
        # to go before, so it went into "content" as loose text with no constraints
        # on it at all). One synthetic clip per unique speaker (by subject, or a
        # shared "narrator" clip for unattributed voiceover lines) so repeated lines
        # from the same speaker across shots correctly share one (Sx), exactly like a
        # real multi-line audio clip would.
        hints = [max(0.1, float(s.get("duration_hint_s") or (duration_seconds / max(1, len(vlm_shots)))))
                 for s in vlm_shots]
        total_hint = sum(hints) or 1.0
        cum = 0.0
        shot_bounds = []
        for h in hints:
            start = cum
            cum += h / total_hint * duration_seconds
            shot_bounds.append((start, cum))
        if shot_bounds:
            shot_bounds[-1] = (shot_bounds[-1][0], duration_seconds)

        synthetic_clips: Dict[str, DialogueClip] = {}
        synthetic_custom_labels: Dict[str, str] = {}
        for shot_idx, vs in enumerate(vlm_shots):
            lines = vs.get("dialogue") or []
            if not lines:
                continue
            s_start, s_end = shot_bounds[shot_idx] if shot_idx < len(shot_bounds) else (0.0, duration_seconds)
            cursor = s_start
            for d in lines:
                text = (d.get("text") or "").strip()
                if not text:
                    continue
                subj = d.get("speaker_subject")
                custom_label = (d.get("speaker_label") or "").strip()
                is_vo = bool(d.get("voiceover"))
                if subj:
                    media_tag = f"Brief:{subj}"
                elif custom_label:
                    # Stable per-label tag: the SAME label text (e.g. "a market vendor")
                    # used across multiple shots/lines resolves to the same (Sx), the same
                    # way a real recurring speaker would -- this is the explicit "someone
                    # not a tracked subject" path, distinct from the generic narrator.
                    media_tag = f"Brief:Label:{custom_label}"
                else:
                    media_tag = "Brief:Narrator"
                if media_tag not in synthetic_clips:
                    synthetic_clips[media_tag] = DialogueClip(media_tag=media_tag, turns=[],
                                                                clip_duration_s=duration_seconds)
                    if subj:
                        m = re.match(r"Subject\s*(\d+)", str(subj))
                        if m:
                            subject_of_clip[media_tag] = f"Subject {m.group(1)}"
                    elif custom_label:
                        synthetic_custom_labels[media_tag] = custom_label
                    if is_vo:
                        voiceover_clips.add(media_tag)
                # Natural speech rate (~2.5 words/sec, matching the same estimate used
                # elsewhere in this pack) rather than dividing the whole shot evenly --
                # a short line shouldn't get falsely marked <cutoff> just because it was
                # given the same wide time slice as the shot itself.
                word_count = max(1, len(text.split()))
                nat_dur = max(0.8, word_count / 2.5)
                t0 = min(cursor, s_end)
                t1 = min(t0 + nat_dur, duration_seconds)
                synthetic_clips[media_tag].turns.append(
                    SpeakerTurn(start=t0, end=t1, text=text,
                                lang=(d.get("language") or "English").strip() or "English",
                                cluster=0))
                cursor = t1 + 0.2  # small gap before the next line, if any, in this shot

        if synthetic_clips:
            dialogue_clips = dialogue_clips + list(synthetic_clips.values())
            speaker_ids = c.assign_speaker_ids(dialogue_clips)
            for key in speaker_ids:
                if key in speaker_labels:
                    continue
                clip_tag = key.split("::", 1)[0]
                if clip_tag in synthetic_custom_labels:
                    speaker_labels[key] = synthetic_custom_labels[clip_tag]
                else:
                    speaker_labels[key] = ("an unseen narrator" if clip_tag in voiceover_clips
                                            else "the speaker")

        shots = c.build_shots(vlm_shots, dialogue_clips, duration_seconds, speaker_labels,
                               speaker_ids, subject_of_clip, voiceover_clips=voiceover_clips,
                               sung_clips=sung_clips)

        prompt, _unused = _render_common(
            plan=plan, shots=shots, vlm_shots=vlm_shots, anchor_desc=anchor_desc,
            speaker_ids=speaker_ids,
            overall_soundscape_raw=result.get("overall_soundscape", "N/A"),
            non_diegetic_music_raw=result.get("non_diegetic_music", "N/A"))

        problems = problems_pre + g.validate(prompt, "REF", duration_seconds)
        # Always-visible summary of what was actually parsed from categorization --
        # not conditional on there being a problem. This exists because the failure
        # mode that prompted it (a stale/unswitched role dropdown writing to the
        # wrong fields) produces a prompt that's mechanically VALID -- it just
        # reflects the wrong categorization -- so g.validate() alone can't catch
        # it. Reading this line tells you immediately whether the Composer saw
        # what you think you set, without needing to parse the whole prompt by eye.
        plan_summary = (
            f"Categorized: {len(plan.subjects)} subject(s) "
            f"[{', '.join('Subject ' + str(s.slot) + ' <- ' + '+'.join(s.sources) for s in plan.subjects) or 'none'}], "
            f"{len(plan.anchors)} anchor(s) "
            f"[{', '.join(a.tag for a in plan.anchors) or 'none'}], "
            f"{len(plan.video_structures)} video structure citation(s), "
            f"{len(plan.audio)} audio role(s) "
            f"[{', '.join(a.tag + ':' + a.role for a in plan.audio) or 'none'}]."
        )
        report = plan_summary + "\n" + description_diag + "\n" + item_diag
        if shot_count_notice:
            report += "\n" + shot_count_notice
        report += "\n\n" + (
            "OK - passes the mechanical checklist." if not problems else
            "ISSUES FOUND:\n" + "\n".join(f"- {p}" for p in problems))
        return (prompt, report)


def _render_common(plan, shots, vlm_shots, anchor_desc, speaker_ids,
                    overall_soundscape_raw, non_diegetic_music_raw,
                    subject_sx=None, style_text=None):
    """Shared by every composer that produces a REF-mode prompt from a
    CategorizationPlan, regardless of which loader built that plan (the
    flat-items model or the structured Subject/Shot/Audio loader). Mutates
    `shots` in place (inserting anchor phrasing/replacement text at the
    right shot index) and returns (prompt, "") -- the second value is a
    placeholder for future shared diagnostics; each caller still builds its
    own plan-summary line since the two loaders describe different things."""
    def _subject_text(s) -> str:
        if s.motion_source:
            base = (f"is {s.description}, whose appearance comes from "
                    f"{_bracket(s.sources)} and whose motion comes from <{s.motion_source}>")
        elif getattr(s, "outfit_for_subject", None):
            scope = getattr(s, "outfit_scope", None) or "all shots"
            where = "in every shot" if scope == "all shots" else f"in [{scope}]"
            base = (f"is the complete outfit shown in {_bracket(s.sources)}: "
                    f"{s.description}. These exact garments are worn by "
                    f"<Subject {s.outfit_for_subject}> {where}")
        elif getattr(s, "is_comic_panel", False):
            # With a hand-written panel description the artwork is never sent to
            # the VLM, so there is no generated description to quote here.
            _what = (s.description or "").strip()
            if getattr(s, "keep_bubbles", False):
                # Balloons stay in frame: H3 animates comic pages with their
                # lettering, timing each balloon to its line, and erasing them
                # smears the artwork.
                base = (f"is the comic panel [Shot {s.ref_for_shot}] is built from, "
                        f"shown in {_bracket(s.sources)}"
                        + (f": {_what}" if _what else "") + ". The drawn artwork "
                        f"animates while staying on model, and its speech balloons "
                        f"remain part of the frame -- each balloon appears as its "
                        f"line is spoken and clears when the line ends, with the "
                        f"original lettering unchanged")
            else:
                base = (f"is the scene depicted in {_bracket(s.sources)}, the comic "
                        f"panel [Shot {s.ref_for_shot}] is built from"
                        + (f": {_what}" if _what else "") + ". It is "
                        f"rendered as continuous live motion, with no panel border, "
                        f"speech balloon, caption or lettering visible")
        elif getattr(s, "ref_for_shot", None):
            # A shot's scene/prop reference, not a character.
            base = (f"is {s.description}, the setting shown in {_bracket(s.sources)} "
                    f"for [Shot {s.ref_for_shot}]")
        else:
            base = f"is {s.description} (from {_bracket(s.sources)})"
        # Outfit references are cited explicitly rather than being folded into the
        # identity source list, where they were never actually mentioned.
        # The wearer's own identity image usually shows DIFFERENT clothing. Left
        # unstated, the model has two competing outfit signals and tends to keep the
        # identity image's -- which is why the outfit reference appeared to be
        # ignored. Scope the identity image explicitly to face/hair/build.
        if getattr(s, "outfit_refs", None):
            base += (f". Use {_bracket(s.sources)} for facial identity, hair, and build "
                     f"ONLY -- disregard the clothing visible there entirely")
        return base + "."

    subjects_out = [g.SubjectDef(tag=f"Subject {s.slot}", text=_subject_text(s))
                    for s in plan.subjects]

    def _outfit_clause(s) -> str:
        # Outfits now carry their own <Subject N> retention line (see
        # _subject_retention_explanation), so the wearer's line no longer repeats it.
        return ""

    def _subject_retention_explanation(s) -> str:
        if getattr(s, "outfit_for_subject", None):
            scope = getattr(s, "outfit_scope", None) or "all shots"
            where = "in every shot" if scope == "all shots" else f"in [{scope}]"
            return (f"these garments are transferred onto <Subject {s.outfit_for_subject}> "
                    f"{where}, replacing whatever clothing that subject's own identity "
                    f"reference shows; colors, cut, and accessories match exactly.")
        if s.swap_video and s.retention == "attribute_transfer":
            return (f"replaces the original performer's visual identity only, mapped "
                    f"exactly onto the same body position, pose, and movement at every "
                    f"moment; no new actions, timing, or framing are introduced.")
        return RETENTION_EXPLANATIONS.get(s.retention, RETENTION_EXPLANATIONS["attribute_transfer"])

    def _subject_where(s) -> str:
        computed = _where(vlm_shots, f"Subject {s.slot}")
        if s.swap_video and not computed:
            return "(appears throughout [Shot 1])"
        return computed

    retention = [g.RetentionLine(tag=f"Subject {s.slot}",
                                  where=_subject_where(s),
                                  marker=s.retention,
                                  explanation=_subject_retention_explanation(s)
                                               + _outfit_clause(s))
                 for s in plan.subjects]
    # Outfit references deliberately get NO standalone retention line: guide S2.2.2
    # says a costume image is cited inside the wearer's own <Subject N> definition,
    # not given a picture entry of its own. The wearer's retention line below carries
    # the outfit instead, so the relationship is still tracked.

    ANCHOR_PHRASING = {
        "first_frame": "The shot begins from <{tag}>.",
        "last_frame": "The shot ends on <{tag}>.",
        "keyframe": "The shot's keyframe corresponds to <{tag}>.",
        "storyboard": "<{tag}> serves as the storyboard reference for this shot.",
    }
    ROLE_PHRASE = {"first_frame": "the first frame", "last_frame": "the last frame",
                   "keyframe": "a keyframe", "storyboard": "a storyboard reference"}
    for anchor in plan.anchors:
        desc = anchor_desc.get(anchor.tag) or "the reference image"

        if anchor.role == "first_and_last":
            last_idx = len(shots)
            if last_idx <= 1:
                where_shots = "[Shot 1]"
                subj_text = (f"is both the first and last frame of [Shot 1], creating a "
                             f"seamless loop, showing {desc}.")
                retention_where = "([Shot 1] first and last frame)"
            else:
                where_shots = f"[Shot 1], [Shot {last_idx}]"
                subj_text = (f"is the first frame of [Shot 1] and the last frame of "
                             f"[Shot {last_idx}], creating a seamless loop, showing {desc}.")
                retention_where = f"({where_shots} first and last frame)"
            subjects_out.append(g.SubjectDef(tag=anchor.tag, text=subj_text))
            retention.append(g.RetentionLine(
                tag=anchor.tag, where=retention_where,
                marker="fully_preserved",
                explanation=f"{anchor.tag} is reproduced exactly at both the start and "
                            f"end of the target video, so it loops seamlessly back to "
                            f"itself."))
            shots[0].content = (shots[0].content.rstrip()
                                 + f" The shot begins from <{anchor.tag}>.")
            if 1 <= last_idx <= len(shots):
                shots[last_idx - 1].content = (shots[last_idx - 1].content.rstrip()
                                                 + f" The shot ends on <{anchor.tag}>.")
            continue

        where_shots = ", ".join(f"[Shot {n}]" for n in anchor.shots) or "[Shot 1]"
        if anchor.role == "storyboard":
            # A storyboard plans every shot. Its span is worked out in the loader,
            # before the VLM has decided how many shots there are -- so with a
            # brief driving free shot planning it would otherwise claim to cover
            # only [Shot 1]. Recompute against the shots that actually exist.
            where_shots = ", ".join(f"[Shot {n}]" for n in range(1, len(shots) + 1)) \
                          or "[Shot 1]"
            # Guide S2.2.2's own wording for a shot-planning reference.
            subjects_out.append(g.SubjectDef(
                tag=anchor.tag,
                text=f"is a storyboard reference for {where_shots}, defining their "
                     f"viewpoint, subject placement, and shot order: {desc}."))
            retention.append(g.RetentionLine(
                tag=anchor.tag, where=f"({where_shots} shot planning)",
                marker="weak_reference",
                explanation="only the planned viewpoint, subject placement and shot "
                            "order are followed; it is not a frame that appears in the "
                            "target video."))
            continue
        subjects_out.append(g.SubjectDef(
            tag=anchor.tag,
            text=f"is {ROLE_PHRASE.get(anchor.role, 'a keyframe')} of {where_shots}, showing {desc}."))
        retention.append(g.RetentionLine(
            tag=anchor.tag, where=f"({where_shots} {anchor.role.replace('_', ' ')})",
            marker="fully_preserved",
            explanation=f"{anchor.tag} is reproduced exactly as the anchor for {where_shots}."))
        phrase = ANCHOR_PHRASING.get(anchor.role, ANCHOR_PHRASING["keyframe"]).format(tag=anchor.tag)
        for shot_idx in anchor.shots:
            if 1 <= shot_idx <= len(shots):
                shots[shot_idx - 1].content = shots[shot_idx - 1].content.rstrip() + " " + phrase

    for s in plan.subjects:
        if not s.replaces_original or s.swap_video:
            continue
        tag = f"Subject {s.slot}"
        first_shot_idx = next(
            (i for i, vs in enumerate(vlm_shots, start=1)
             if tag in (vs.get("subject_mentions") or [])), 1)
        if 1 <= first_shot_idx <= len(shots):
            sh = shots[first_shot_idx - 1]
            if "replaces the original occupant" not in sh.content:
                sh.content = (f"<{tag}> replaces the original occupant of the scene. "
                              + sh.content.rstrip())

    VIDEO_STRUCT_TEXT = {
        "editing": "is the source video for the target video edit.",
        "continuation": "is the source video the target video continues from.",
        "camera_reference": "is referenced for its camera movement, cuts, rhythm, "
                             "or temporal structure.",
    }
    VIDEO_STRUCT_EXPL = {
        "continuation": "the target video begins where this clip ends; its closing "
                         "framing, lighting, subject positions and action in progress "
                         "carry forward into [Shot 1] without a cut.",
    }
    VIDEO_STRUCT_WHERE = {
        "editing": "(edit source)", "continuation": "(final state carried forward)",
        "camera_reference": "(cut and pacing structure)",
    }
    for vs in plan.video_structures:
        if vs.is_character_swap:
            swap_subjects = [s for s in plan.subjects if s.swap_video == vs.tag]
            who = " and ".join(f"<Subject {s.slot}>" for s in swap_subjects) or "the new subject"
            subjects_out.append(g.SubjectDef(
                tag=vs.tag,
                text=f"is the source video being edited; its camera framing, motion, cuts, "
                     f"and full performance timing are the fixed structure that must be "
                     f"preserved exactly, with only the performer's visual identity replaced "
                     f"by {who}."))
            retention.append(g.RetentionLine(
                tag=vs.tag, where="(camera framing, motion, cuts, full performance and timing)",
                marker="fully_preserved",
                explanation="every camera position, movement, and the precise sequence and "
                            "rhythm of the original performance are kept exactly as in the "
                            "source video; nothing about the shot itself is altered."))
            continue
        subjects_out.append(g.SubjectDef(
            tag=vs.tag, text=VIDEO_STRUCT_TEXT.get(vs.role, VIDEO_STRUCT_TEXT["camera_reference"])))
        retention.append(g.RetentionLine(
            tag=vs.tag, where=VIDEO_STRUCT_WHERE.get(vs.role, ""), marker=vs.retention,
            explanation=VIDEO_STRUCT_EXPL.get(vs.role)
                        or RETENTION_EXPLANATIONS.get(vs.retention,
                                                       RETENTION_EXPLANATIONS["weak_reference"])))

    AUDIO_RETENTION_EXPLANATIONS = {
        "fully_copy": "is reused 1:1 as the target video's complete final audio track.",
        "partially_copy": "the source audio's spoken words are retained; timing is trimmed "
                           "to fit the target video's duration.",
        "reference": "only timbre, rhythm, or delivery is referenced, without copying the "
                     "original signal.",
        "weak_reference": "only broad similarity in category or atmosphere is retained.",
    }
    donor_tags = {d for _, d in plan.donor_pairs}
    for ap in plan.audio:
        if ap.role != "dialogue" or ap.tag in donor_tags:
            continue
        sx = next((v for k, v in speaker_ids.items() if k.startswith(f"{ap.tag}::")), None)
        who = f"<Subject {ap.subject_slot}> ({sx})" if (sx and ap.subject_slot) else "the target speaker"
        noun = "lyrics" if ap.sung else "dialogue"
        verb = "sung" if ap.sung else "spoken"
        subjects_out.append(g.SubjectDef(
            tag=ap.tag, text=f"is the original {noun} track {verb} by {who} in the target video."))
        marker = ap.retention or "partially_copy"
        retention.append(g.RetentionLine(
            tag=ap.tag, where="", marker=marker,
            explanation=AUDIO_RETENTION_EXPLANATIONS.get(marker, AUDIO_RETENTION_EXPLANATIONS["partially_copy"])))
    # Voice-timbre references that AREN'T part of a donor pair. The structured
    # loader attaches a Vocal Reference straight to a subject and never builds
    # donor_pairs (those only exist when one clip's words are re-voiced with
    # another clip's timbre), so without this branch a voice reference matched
    # neither loop below and vanished from the prompt entirely.
    for ap in plan.audio:
        if ap.role != "voice_timbre" or ap.tag in donor_tags:
            continue
        subj = f"Subject {ap.subject_slot}" if ap.subject_slot else None
        sx = (subject_sx or {}).get(subj)
        if subj and sx:
            who = f"<{subj}> ({sx})"          # guide S2.2.4: reuse the speaker's global ID
        elif subj:
            who = f"<{subj}>"                  # subject never actually speaks
        else:
            who = "the target speaker"
        src = (f" It is the synchronized audio track of <{ap.from_video_tag}>."
               if getattr(ap, "from_video_tag", None) else "")
        subjects_out.append(g.SubjectDef(
            tag=ap.tag, text=f"is the voice-timbre reference for {who}.{src}"))
        retention.append(g.RetentionLine(
            tag=ap.tag, where="", marker=ap.retention or "reference",
            explanation=f"the target speaker follows <{ap.tag}>'s voice timbre and "
                        f"delivery without copying the original signal."))

    for source_tag, donor_tag in plan.donor_pairs:
        donor_ap = next((a for a in plan.audio if a.tag == donor_tag), None)
        sx = next((v for k, v in speaker_ids.items() if k.startswith(f"{source_tag}::")), None)
        subj = donor_ap.subject_slot if donor_ap else None
        who = f"<Subject {subj}> ({sx})" if (sx and subj) else "the target speaker"
        subjects_out.append(g.SubjectDef(
            tag=donor_tag, text=f"is the voice-timbre reference for {who}."))
        marker = (donor_ap.retention if donor_ap else None) or "reference"
        retention.append(g.RetentionLine(
            tag=donor_tag, where="", marker=marker,
            explanation=f"the target speaker follows <{donor_tag}>'s voice timbre and "
                        f"delivery without copying the original signal."))

    # Ambience/music audio is cited in the sound sections, but every reference label
    # must also be DEFINED once and given a retention line (guide S2.2 / S2.4) --
    # without these the tags appeared in overall_soundscape / non_diegetic_music as
    # labels that were never introduced anywhere.
    AUDIO_ROLE_TEXT = {
        "soundscape": "is the ambience and sound-effect reference for the target video.",
        "music": "is the background-music reference for the target video's score.",
    }
    AUDIO_ROLE_EXPL = {
        "fully_copy": "is reused 1:1 in the target video's final audio track.",
        "partially_copy": "part of this audio is copied into the target video's final track.",
        "reference": "only its texture and character are referenced; the signal is not copied.",
        "weak_reference": "only broad similarity in category or atmosphere is retained.",
    }
    _defined = {sd.tag for sd in subjects_out}
    for kind, cites in (("soundscape", plan.soundscape_citations),
                         ("music", plan.music_citations)):
        for tag, marker in cites:
            if tag in _defined:
                continue
            _defined.add(tag)
            _ap = next((a for a in plan.audio if a.tag == tag), None)
            _src = (f" It is the synchronized audio track of <{_ap.from_video_tag}>."
                    if _ap is not None and getattr(_ap, "from_video_tag", None) else "")
            subjects_out.append(g.SubjectDef(tag=tag,
                                              text=AUDIO_ROLE_TEXT[kind] + _src))
            retention.append(g.RetentionLine(
                tag=tag, where="", marker=marker or "reference",
                explanation=AUDIO_ROLE_EXPL.get(marker or "reference",
                                                 AUDIO_ROLE_EXPL["reference"])))

    overall_soundscape = _as_sentence(overall_soundscape_raw)
    non_diegetic_music = _as_sentence(non_diegetic_music_raw)
    SOUNDSCAPE_PHRASING = {
        "fully_copy": "is reused 1:1 as the complete ambience/sound-effect layer",
        "partially_copy": "provides a copied ambience layer that continues",
        "reference": "is referenced for its ambience style, without copying the signal",
        "weak_reference": "provides only a loose atmospheric reference",
    }
    for tag, marker in plan.soundscape_citations:
        phrase = SOUNDSCAPE_PHRASING.get(marker, SOUNDSCAPE_PHRASING["partially_copy"])
        overall_soundscape = _as_sentence(f"<{tag}> {phrase} throughout the target video. "
                                           f"{overall_soundscape}")
    MUSIC_PHRASING = {
        "fully_copy": "is directly reused as the complete audience-only score",
        "partially_copy": "provides part of the audience-only score, reused from the source",
        "reference": "is referenced for its style/instrumentation, without copying the signal",
        "weak_reference": "provides only a loose stylistic reference for the score",
    }
    for tag, marker in plan.music_citations:
        non_diegetic_music = _as_sentence(
            f"<{tag}> {MUSIC_PHRASING.get(marker, MUSIC_PHRASING['fully_copy'])}.")

    task_types = _task_types(plan)
    summary_text = _build_summary(plan)
    # Style comes from the references (or the user's preset). It used to be
    # hardcoded to live-action, which mislabelled an all-anime reference set.
    style_text = (style_text or "").strip() or "Live-action, cinematic"
    low = style_text.lower()
    style_intro = (f"The target video uses a {low} style."
                   if "style" not in low else f"The target video uses {low}.")

    prompt = g.compose_reference(subjects_out, task_types, summary_text, retention,
                                  style_intro, shots, overall_soundscape, non_diegetic_music)
    return prompt, ""


_SPEECH_VERBS_RE = re.compile(
    r"(?:says?|said|exclaims?|exclaimed|asks?|asked|shouts?|shouted|whispers?|"
    r"whispered|replies|replied|calls?\s+out|called\s+out|yells?|yelled|"
    r"mutters?|muttered)", re.IGNORECASE)
_QUOTE_CHARS = "\"'\u201c\u201d\u2018\u2019"


def _scrub_duplicated_dialogue(vlm_shots: List[dict]) -> None:
    """Mutates each shot's "content" in place, stripping any of that shot's
    own dialogue text back out if it shows up there verbatim -- the VLM is
    told never to restate dialogue words in content, but sometimes does it
    anyway alongside the correctly-populated structured field, producing
    duplicated dialogue in the final output. Shared by every composer that
    uses the structured "dialogue" schema, not just one."""
    for vs in vlm_shots:
        lines = vs.get("dialogue") or []
        content = vs.get("content", "")
        for d in lines:
            text = (d.get("text") or "").strip()
            if not text or text not in content:
                continue
            esc = re.escape(text)
            # Prefer removing the whole "<verb>, '<text>'" clause (without consuming
            # any leading subject/pronoun, so "She says, 'X' then stands up" cleanly
            # becomes "She then stands up" rather than losing its subject).
            clause_re = re.compile(
                rf"\s*(?:,\s*)?{_SPEECH_VERBS_RE.pattern}\s*,?\s*"
                rf"[{_QUOTE_CHARS}]{esc}[{_QUOTE_CHARS}]", re.IGNORECASE)
            new_content, n = clause_re.subn("", content)
            if n:
                content = new_content
                continue
            # Fallback: no attribution clause matched around it -- just remove the
            # quoted span itself, leaving the surrounding narrative intact.
            content = re.sub(rf"[{_QUOTE_CHARS}]{esc}[{_QUOTE_CHARS}]", "", content)
        content = re.sub(r"\s{2,}", " ", content)
        content = re.sub(r"\s+([.,!?])", r"\1", content)
        content = re.sub(r"\.\s*\.", ".", content)
        vs["content"] = content.strip()


def _find_description(descriptions: dict, tag: str) -> str:
    """Looks up a VLM-written description for `tag` tolerantly. The schema
    asks for bare keys ("Subject 1", "Picture 1"), but the LABELS actually
    shown to the VLM include extra context ("Subject 1 (Picture 1, a dog)",
    "Picture 1 (anchor)") -- a small local model will often just echo the
    fuller label back as its own JSON key rather than stripping to the bare
    tag the schema asked for. Exact match first, then any key that starts
    with or contains the tag, rather than silently falling back to a
    placeholder on a near-miss. Module-level so every composer (not just
    the flat-items one) can reuse it."""
    if tag in descriptions:
        return (descriptions[tag] or "").strip()
    tl = tag.lower()
    for k, v in descriptions.items():
        kl = str(k).strip().lower()
        if kl.startswith(tl) or tl in kl:
            return (v or "").strip()
    return ""


def _bracket(tags: List[str]) -> str:
    return ", ".join(f"<{t}>" for t in tags) if tags else "<reference media>"


def _where(vlm_shots: List[dict], tag: str) -> str:
    shot_list = [i for i, s in enumerate(vlm_shots, start=1) if tag in (s.get("subject_mentions") or [])]
    return f"(appears in {', '.join(f'[Shot {n}]' for n in shot_list)})" if shot_list else ""


def _as_sentence(text: str) -> str:
    text = (text or "").strip()
    if not text or text == "N/A":
        return text or "N/A"
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


def _task_types(plan: cat.CategorizationPlan) -> List[str]:
    types = ["reference generation"]
    if plan.anchors:
        types.append("keyframe completion")
    for vs in plan.video_structures:
        if vs.role == "editing":
            types.append("video editing")
        elif vs.role == "continuation":
            types.append("video continuation")
        # camera_reference alone stays under reference generation, per guide S2.3:
        # "If a reference video provides only camera movement, cuts, or rhythm, it
        # normally belongs to reference generation."
    # Audio task type follows the actual retention marker, not just role/presence --
    # fully_copy/partially_copy is real signal reuse ("audio reuse"); reference/
    # weak_reference is style-only ("audio reference"). A voice-timbre donor is
    # always "audio reference" since its role means the signal is never copied.
    audio_and_music = [a.retention for a in plan.audio if a.role in ("dialogue",)]
    audio_and_music += [m for _, m in plan.music_citations]
    audio_and_music += [s for _, s in plan.soundscape_citations]
    if any(a.role == "voice_timbre" for a in plan.audio) or \
            any(m in ("reference", "weak_reference") for m in audio_and_music):
        types.append("audio reference")
    if any(m in ("fully_copy", "partially_copy") for m in audio_and_music):
        types.append("audio reuse")
    seen = []
    for t in types:
        if t not in seen:
            seen.append(t)
    return seen


def _build_summary(plan: cat.CategorizationPlan) -> str:
    # Guide S2.3: "For video-editing tasks, begin the summary after the task-type
    # prefix with: 'The target video is an edited version of <Video 1>.'" -- this
    # was never actually implemented; the summary was always the same generic
    # filler regardless of task type, which is wrong specifically for editing
    # tasks where the guide mandates this exact opening.
    editing_videos = [vs.tag for vs in plan.video_structures if vs.role == "editing"]
    if editing_videos:
        opening = f"The target video is an edited version of <{editing_videos[0]}>"
        swap_subjects = [s for s in plan.subjects if s.swap_video == editing_videos[0]]
        if swap_subjects:
            who = " and ".join(f"<Subject {s.slot}>" for s in swap_subjects)
            opening += (f" in which only the performer's visual identity is replaced by "
                        f"{who}; the camera framing, motion, and environment are kept "
                        f"unchanged")
        return opening + "."

    bits = []
    for s in plan.subjects:
        bits.append(f"Subject {s.slot} appears in the target video")
    return (". ".join(bits) + ".") if bits else \
        "The target video follows the user's brief using the subjects defined above."


def _write_temp_wav(audio: dict) -> str:
    import tempfile
    data = media_io.audio_to_wav_bytes(audio)
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.write(data)
    f.close()
    return f.name


# Nothing here is registered as a node anymore. This module is retained purely for
# its shared rendering helpers (_render_common, _find_description,
# _scrub_duplicated_dialogue, _task_types, _build_summary and friends), which the
# structured composer imports.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
