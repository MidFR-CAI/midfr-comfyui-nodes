"""
structured_nodes.py -- ComfyUI node wrappers for the new H3 Structured
Media Loader. The loader itself just validates and passes its JSON state
through (matching how H3SubjectCategorizer already works); the composer
does the real work, reusing categorized_nodes.py's module-level rendering
helpers (RETENTION_EXPLANATIONS, _bracket, _where, _as_sentence,
_task_types, _build_summary, _render_common, _scrub_duplicated_dialogue,
_find_description, _write_temp_wav) rather than duplicating them, since
those only care about a CategorizationPlan -- not which loader built it.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

from . import categorized_nodes as cn
from . import debug_log as dbg
from . import composer as c
from . import grammar as g
from . import structured_loader as sl
from . import media_io
from .describe_engine import DescribeEngine
from .vlm_engine import LlamaCppVLM, TransformersVLM, VLMConfig
from .whisper_engine import (FasterWhisperTranscriber, DialogueClip,
                              SpeakerTurn, transcribe_clip)

CATEGORY = "conditioning/video_models/h3_autoprompt"

# Kept for reference; the dropdown itself now lives in web/allinone_panel.js,
# since the widget is a free STRING so custom styles can be typed.
STYLE_PRESETS = [
    "auto (detect from references)",
    "Live-action, cinematic",
    "2D-animated, anime",
    "2D-animated",
    "3D CG",
    "claymation",
    "watercolor",
    "vintage film",
]


def _replace_body_shots(rep_cfg, result):
    """Shot text for a replacement edit.

    A description the person wrote is used verbatim: nothing in this pack can
    watch the source video, so their account of it is the only reliable one, and
    the guide requires a real scene description rather than a list of reference
    relationships. The VLM's reading is the fallback when they haven't written
    one.
    """
    written = (rep_cfg.get("description") or "").strip()
    if written:
        return [written]
    return [(v.get("content") or "").strip()
            for v in (result.get("shots") or [])
            if (v.get("content") or "").strip()]


class H3StructuredMediaLoader:
    """Loads reference media with categories built in: Subject (Complete /
    Visual Reference / Outfit Reference / Vocal Reference), Shot (Complete /
    First Frame / Last Frame / Motion / Music / SFX / Reference Images), and
    Audio (Dialogue / Music, spanning the whole video). Replaces needing a
    separate categorization step after loading -- click "Open loader..." on
    this node to add media directly into the category it belongs to.

    Shared caps across every category combined: 9 images, 3 videos, 3 audio,
    matching MiniMaxH3ReferenceToVideo's own slot limits."""

    CATEGORY = CATEGORY
    # duration rides along as a FLOAT so it can be wired straight into the
    # prompt builder's duration socket. It's a manual value normally; comic mode
    # computes one from the estimated speech and overrides it.
    RETURN_TYPES = ("H3_STRUCTURED_REFS", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("references", "duration", "width", "height")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            # Written by this node's own "Open loader..." panel. Deliberately
            # required, not optional -- an optional basic-type STRING risks the
            # frontend treating it as a connectable slot instead of a real
            # widget, which silently breaks Save (this exact bug hit an
            # earlier node in this pack and cost several debugging rounds).
            # multiline MUST stay False: a multiline STRING widget is backed by a
            # real DOM textarea, and ComfyUI tears that DOM down when you switch
            # away from the workflow tab -- the value doesn't survive the rebuild,
            # which silently wiped the whole loader state on tab switch. A
            # single-line widget keeps its value on the widget object itself and
            # serializes normally. This is what the original pack's own loader
            # uses for exactly the same reason.
            "loader_state": ("STRING", {"multiline": False, "default": "{}"}),
        }}

    @classmethod
    def IS_CHANGED(cls, loader_state="{}", **kwargs):
        # Without this ComfyUI caches the node's output and won't re-run when only
        # the panel's contents changed, since no visible widget appears to differ.
        return loader_state

    def load(self, loader_state: str = "{}"):
        try:
            state = json.loads(loader_state or "{}")
            if not isinstance(state, dict):
                state = {}
        except Exception:
            state = {}
        err = sl.validate_modes(state) or sl.validate_caps(state)
        if err:
            raise ValueError(f"H3 Structured Media Loader: {err}")
        w, h, why = sl.resolve_dimensions(state)
        print(f"[H3 AutoPrompt] resolution: {why}")
        return (state, sl.resolve_duration(state), w, h)


class H3StructuredComposer:
    """Composes a legal H3 prompt directly from the Structured Media
    Loader's categories -- no separate categorization pass, and no free-text
    brief for shots (shot content comes from each shot's own media: anchors,
    motion reference, reference images). An optional overall brief still adds
    scene-level context the media alone can't carry (tone, pacing intent)."""

    CATEGORY = CATEGORY
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "validation_report")
    FUNCTION = "compose"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "references": ("H3_STRUCTURED_REFS",),
            "brief": ("STRING", {"multiline": True, "default": ""}),
            "duration_seconds": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 30.0}),
            "vlm_backend": (["llama_cpp_gguf", "transformers_hf"],),
            "vlm_model_path_or_id": ("STRING", {"default": ""}),
            "vlm_mmproj_path": ("STRING", {"default": ""}),
            "whisper_model_size": ("STRING", {"default": "large-v3-turbo"}),
            "device": (["cuda", "cpu"],),
            "use_diarization": ("BOOLEAN", {"default": True}),
            "max_vlm_images": ("INT", {"default": 4, "min": 1, "max": 24}),
            "vlm_detail": (["accurate (896px)", "faster (704px)",
                             "fastest (512px)"],),
            "use_vlm": ("BOOLEAN", {"default": True}),
            "vlm_n_ctx": ("INT", {"default": 16384, "min": 2048, "max": 131072}),
            "style": ("STRING", {"multiline": False,
                                  "default": "auto (detect from references)"}),
            # One text box per shot the loader defined, rendered inline on this
            # node by web/structured_composer.js and kept in sync with the
            # connected loader's shot count. Required (not optional) so a real
            # widget is guaranteed to exist -- an optional basic-type STRING can
            # be rendered as a connectable slot instead, which silently breaks
            # the panel's write-back.
            # Single-line for the same reason as loader_state above: a multiline
            # STRING is DOM-backed and loses its value when ComfyUI rebuilds the
            # DOM on a tab switch, which wiped every per-shot prompt. The panel
            # renders its own real textareas and stores their contents here as
            # JSON, so nothing is lost by this widget itself being single-line.
            "shot_prompts": ("STRING", {"multiline": False, "default": "[]"}),
        }}

    def compose(self, references: dict, brief: str, duration_seconds: float,
                vlm_backend: str, vlm_model_path_or_id: str, vlm_mmproj_path: str,
                whisper_model_size: str, device: str, use_diarization: bool,
                max_vlm_images: int = 6, vlm_n_ctx: int = 16384,
                shot_prompts: str = "[]", style: str = "auto (detect from references)",
                vlm_detail: str = "accurate (896px)", use_vlm: bool = True):
        plan, tensor_index, shot_contexts = sl.build_plan_from_structured(references, media_io)
        # Kept so callers (H3 All-in-One) can wire the loaded media straight into
        # MiniMaxH3ReferenceToVideo's ref_* slots without re-decoding every file.
        self._last_tensor_index = tensor_index
        problems_pre: List[str] = []
        if not plan.subjects and not plan.anchors and not shot_contexts:
            problems_pre.append(
                "Nothing loaded yet -- open the H3 Structured Media Loader panel and "
                "add at least one Subject or Shot before composing."
            )

        # -- gather images for the narrow VLM call: subject/anchor sources first
        # (identity is essential -- without it the VLM can't describe who anyone
        # is), then each shot's own reference images, all under one shared budget
        # so a request with many shots can't silently blow past what the chat
        # handler can actually consume in one call (a real crash seen in practice).
        images: List = []
        labels: List[str] = []
        sent_tags: Dict[str, int] = {}

        def add_image(tag: str, label: str):
            if len(images) >= max_vlm_images:
                return
            if tag in sent_tags:
                idx = sent_tags[tag]
                labels[idx] = f"{labels[idx]} / {label}"
                return
            t = tensor_index.get(tag)
            if t is None:
                return
            img = media_io.image_tensor_to_pil(t if t.dim() == 4 and t.shape[0] == 1 else t[:1])
            images.append(img)
            labels.append(label)
            sent_tags[tag] = len(images) - 1

        # A described panel is also registered as a Subject, so skipping only its
        # reference-image entry left the image being sent by this loop anyway --
        # and each image is ~40s of vision encoding.
        described_shots = {c_.index for c_ in shot_contexts
                            if getattr(c_, "user_authored", False)}
        for s in plan.subjects:
            if getattr(s, "is_comic_panel", False) \
                    and getattr(s, "ref_for_shot", None) in described_shots:
                dbg.line(f"panel subject for [Shot {s.ref_for_shot}] not sent "
                          f"-- you described it")
                s.description = ""
                continue
            for src_tag in s.sources[:2]:
                # "an environment", not "a environment" -- this string goes to the VLM
                # as the image's label, so the grammar matters for how it reads it.
                _art = "an" if s.kind and s.kind[0] in "aeiou" else "a"
                kind_hint = f", {_art} {s.kind}" if s.kind and s.kind != "person" else ""
                add_image(src_tag, f"Subject {s.slot} ({src_tag}{kind_hint})")
        for a in plan.anchors:
            add_image(a.tag, f"{a.tag} (anchor)")
        for ctx in shot_contexts:
            if getattr(ctx, "user_authored", False):
                # Described by hand: the model has nothing to add about it, and
                # its text is used verbatim anyway.
                dbg.line(f"[Shot {ctx.index}] described by you -- image not sent "
                          f"to the VLM")
                continue
            for tag in ctx.reference_image_tags:
                add_image(tag, f"{tag} (reference for [Shot {ctx.index}])")

        # -- shots are media-defined, not text-defined: reuse the exact same
        # fixed-shot-count mechanism built for per-shot text boxes (shot_briefs),
        # just with placeholder text per shot instead of user-written text, since
        # this loader has no free-text field for shots at all -- each shot's own
        # anchors/motion/reference images (already in `images`/`labels` and in
        # subjects_out/retention via the plan) carry the content instead.
        # Per-shot prompts come from this node's own inline text boxes (one per
        # shot the loader defined, kept in sync by the JS panel). A shot left
        # blank falls back to the placeholder, so its content is inferred from
        # its own media instead -- mixing written and inferred shots is fine.
        try:
            raw_prompts = json.loads(shot_prompts or "[]")
            if not isinstance(raw_prompts, list):
                raw_prompts = []
        except Exception:
            raw_prompts = []
        # Each entry is {"text": str, "dialogue": [...]}; bare strings are accepted
        # too for anything saved before per-shot dialogue lines existed.
        user_shot_prompts, user_shot_dialogue, user_shot_durations = [], [], []
        for entry in raw_prompts:
            if isinstance(entry, str):
                user_shot_prompts.append(entry)
                user_shot_dialogue.append([])
                user_shot_durations.append(None)
            else:
                entry = entry or {}
                user_shot_prompts.append(str(entry.get("text") or ""))
                user_shot_dialogue.append(entry.get("dialogue") or [])
                try:
                    _d = float(entry.get("duration")) if entry.get("duration") else None
                except (TypeError, ValueError):
                    _d = None
                user_shot_durations.append(_d if (_d and _d > 0) else None)

        # Always compose at least one shot: a loader with no Shot rows at all (just
        # subjects, say) should still produce a valid single-shot prompt rather than an
        # empty detailed_description. A synthetic context stands in for the missing row.
        loader_defined_shots = bool(shot_contexts)
        if not shot_contexts:
            shot_contexts = [sl.ShotMediaContext(index=1)]

        shot_brief_list = []
        for i, ctx in enumerate(shot_contexts):
            if getattr(ctx, "user_authored", False) and (ctx.brief_text or "").strip():
                shot_brief_list.append(
                    "VERBATIM SOURCE -- reproduce this shot's content exactly as "
                    "written, changing only grammar and the insertion of reference "
                    "labels. Do not add, remove, reinterpret or embellish anything: "
                    + ctx.brief_text.strip())
                continue
            written = (ctx.brief_text or "").strip()
            if not written:
                written = ((user_shot_prompts[i] if i < len(user_shot_prompts) else "")
                           or "").strip()
            if written:
                shot_brief_list.append(written)
            else:
                shot_brief_list.append(
                    f"(No text given for this shot -- infer its content from its own "
                    f"reference images and any anchor/motion citations already noted "
                    f"above"
                    f"{', it replaces existing footage entirely (video editing)' if ctx.is_replacement else ''}.)")
        shot_brief_list = shot_brief_list or None
        # With no shots defined in the loader, the brief is the whole-video guide
        # -- hand it over without a fixed shot list so the VLM can plan the shots
        # from it (typically alongside a storyboard). A placeholder list here
        # would pin it to one shot and the brief would never be read.
        if not loader_defined_shots and (brief or "").strip():
            shot_brief_list = None

        dbg.section("VLM INPUT")
        dbg.kv("backend", f"{vlm_backend}  model={vlm_model_path_or_id}")
        dbg.kv("images", f"{len(images)} (cap {max_vlm_images})")
        for _lb in labels:
            dbg.bullet(_lb)
        dbg.kv("shots requested", len(shot_brief_list or []))
        for _i, _sb in enumerate(shot_brief_list or [], 1):
            dbg.bullet(f"[Shot {_i}] {_sb}")

        # If every shot is described by hand and there are no images left needing
        # a description, the model has nothing to contribute -- skip the call
        # rather than spend a model load and several round trips reformatting
        # text that is already final.
        from .describe_engine import CANVAS_PRESETS, set_canvas
        set_canvas(CANVAS_PRESETS.get(str(vlm_detail), 896))
        dbg.kv("vlm detail", f"{vlm_detail} -- {len(images)} image(s) to encode")

        all_authored = bool(shot_contexts) and all(
            getattr(c_, "user_authored", False) for c_ in shot_contexts)
        if not use_vlm or (all_authored and not images):
            dbg.section("VLM SKIPPED")
            dbg.line("turned off" if not use_vlm else
                      "every shot is described by you and no image needs describing")
            result = {
                "subject_descriptions": {},
                "shots": [{"content": t, "subject_mentions": [],
                            "duration_hint_s": duration_seconds / max(1, len(shot_brief_list or [1])),
                            "dialogue": []}
                           for t in (shot_brief_list or [])],
                "overall_soundscape": "Ambient sound consistent with the scene.",
                "non_diegetic_music": "N/A",
                "style": None,
            }
        else:
            # Constructing the backend loads the model, so it only happens on the
            # branch that actually needs it.
            if vlm_backend == "llama_cpp_gguf":
                backend = LlamaCppVLM(VLMConfig(model_path=vlm_model_path_or_id,
                                                 mmproj_path=vlm_mmproj_path,
                                                 n_gpu_layers=-1 if device == "cuda" else 0,
                                                 n_ctx=vlm_n_ctx))
            else:
                backend = TransformersVLM(model_id=vlm_model_path_or_id, device=device)
            describer = DescribeEngine(backend)
            result = describer.describe(images, labels, brief, duration_seconds,
                                         notes=plan.notes, shot_briefs=shot_brief_list)

        dbg.section("VLM OUTPUT")
        dbg.kv("description keys", list((result.get("subject_descriptions") or {}).keys()))
        dbg.kv("shots returned", len(result.get("shots") or []))
        dbg.kv("overall_soundscape", result.get("overall_soundscape"))
        dbg.kv("non_diegetic_music", result.get("non_diegetic_music"))

        # Replace mode takes a dedicated, terse path. The general composer
        # narrates action that doesn't exist here, transcribes dialogue that
        # isn't changing, and cites the clip as a structural edit -- all of which
        # dilutes a job that is really just "swap who is on screen".
        rep_cfg = (references.get("replace") or {}) if isinstance(references, dict) else {}
        if rep_cfg.get("enabled"):
            from . import replace_prompt as rp
            descs = result.get("subject_descriptions") or {}
            for _s in plan.subjects:
                _s.description = cn._find_description(descs, f"Subject {_s.slot}") or ""
            src_tag = next((v.tag for v in plan.video_structures), None) \
                      or next((t for t in tensor_index if t.startswith("Video")), "Video 1")
            src_desc = cn._find_description(descs, src_tag)
            a_tag = next((a.tag for a in plan.audio), None)
            mode = rep_cfg.get("audio_mode") or ("keep" if rep_cfg.get("keep_audio", True)
                                                  else "off")
            # style_text is resolved further down for the general path; here the
            # widget value is used directly, with the VLM's reading as fallback.
            rep_style = (None if str(style).startswith("auto") else str(style)) \
                        or (result.get("style") or "").strip() \
                        or "cinematic, ultra-realistic"
            # Everyone marked in the frame gets a subject slot of their own,
            # numbered after the replacing subjects, so an unreplaced person can
            # be preserved by name instead of being swept away.
            next_slot = max([sp.slot for sp in plan.subjects] or [0]) + 1
            voices = {a.subject_slot: a.tag for a in plan.audio
                      if a.role == "voice_timbre" and a.subject_slot}
            people_out = []
            for _p in (rep_cfg.get("people") or []):
                by = _p.get("subject")
                people_out.append({
                    "slot": next_slot,
                    "where": (_p.get("where") or "").strip(),
                    "replaced_by": int(by) if by else None,
                    "voice_tag": voices.get(int(by)) if by else None,
                    "keep": _p.get("keep") or {},
                })
                next_slot += 1
            prompt = rp.render(plan, src_tag, src_desc, duration_seconds,
                                style=rep_style, audio_tag=a_tag, audio_mode=mode,
                                people=people_out,
                                # The VLM's read of the source footage carries the
                                # body; the guide requires a real scene description,
                                # not a relationship list.
                                shots=_replace_body_shots(rep_cfg, result))
            dbg.section("REPLACE PROMPT")
            dbg.multiline("prompt", prompt)
            problems = g.validate(prompt, "REF", duration_seconds)
            report = (f"Replace: {len(plan.subjects)} subject(s), audio={mode}.\n"
                       + ("OK - passes the mechanical checklist." if not problems
                          else "ISSUES FOUND:\n" + "\n".join(f"- {p}" for p in problems)))
            return (prompt, report)

        descriptions = result.get("subject_descriptions") or {}
        for s in plan.subjects:
            s.description = cn._find_description(descriptions, f"Subject {s.slot}")
            if not s.description:
                # No model ran (or it said nothing). Point at the reference rather
                # than inventing traits that would contradict the picture.
                s.description = ("the character shown in "
                                  + (", ".join(f"<{t}>" for t in s.sources)
                                     or "the loaded reference media")
                                  if s.sources else
                                  "a subject defined by the loaded reference media")
        anchor_desc = {}
        for a in plan.anchors:
            found = cn._find_description(descriptions, a.tag)
            if not found:
                sharing = next((s for s in plan.subjects if a.tag in s.sources), None)
                if sharing and sharing.description:
                    found = sharing.description
            anchor_desc[a.tag] = found

        if dbg.on():
            dbg.section("DESCRIPTION MATCHING")
            for _sp in plan.subjects:
                dbg.bullet(f"Subject {_sp.slot}: {_sp.description}")
            for _t, _d in anchor_desc.items():
                dbg.bullet(f"{_t}: {_d or '(none -- placeholder used)'}")

        vlm_shots = result.get("shots") or [{
            "content": brief.strip() or "The scene plays out as described.",
            "subject_mentions": [f"Subject {s.slot}" for s in plan.subjects],
            "duration_hint_s": duration_seconds,
        }]
        n_shots = len(shot_contexts) or 1
        shot_count_notice = None
        # Only hold the VLM to a fixed shot count when the LOADER defined the
        # shots. With just a brief, the synthetic placeholder context is a
        # stand-in, not a requirement -- reconciling against it would truncate a
        # multi-shot plan back to one and silently discard the rest.
        if shot_contexts and loader_defined_shots:
            original_count = len(vlm_shots)
            if original_count < n_shots:
                for i in range(original_count, n_shots):
                    vlm_shots.append({"content": shot_brief_list[i] if shot_brief_list else "",
                                       "camera": None, "subject_mentions": [], "dialogue": [],
                                       "duration_hint_s": duration_seconds / n_shots})
                shot_count_notice = (f"VLM returned {original_count} shot(s) for {n_shots} "
                                      f"loader-defined shot(s) -- padded the missing one(s).")
            elif original_count > n_shots:
                shot_count_notice = (f"VLM returned {original_count} shot(s) for {n_shots} "
                                      f"loader-defined shot(s) -- truncated to the requested count.")
                vlm_shots = vlm_shots[:n_shots]

        if not loader_defined_shots and len(vlm_shots) != len(shot_contexts):
            shot_contexts = [sl.ShotMediaContext(index=i + 1)
                             for i in range(len(vlm_shots))]

        # Explicit per-shot dialogue from the panel always wins over whatever the VLM
        # guessed for that shot -- that is the whole point of giving it its own UI.
        # speaker_subject keeps the same "Subject N" string form the VLM itself uses;
        # a custom non-subject label rides along as speaker_label so the synthetic
        # clip builder can key a stable, distinct (Sx) off it.
        for i, ctx in enumerate(shot_contexts):
            if ctx.dialogue and i < len(vlm_shots):
                vlm_shots[i]["dialogue"] = ctx.dialogue
            # Comic mode knows exactly who is in each panel, so the retention
            # lines can say where each character appears instead of guessing.
            slots = getattr(ctx, "subject_slots", None)
            if slots and i < len(vlm_shots):
                mentions = vlm_shots[i].setdefault("subject_mentions", [])
                for sl_ in slots:
                    if f"Subject {sl_}" not in mentions:
                        mentions.append(f"Subject {sl_}")

        for i, lines in enumerate(user_shot_dialogue):
            if i >= len(vlm_shots) or not lines:
                continue
            vlm_shots[i]["dialogue"] = [
                {"speaker_subject": f"Subject {d['subject']}" if d.get("subject") else None,
                 "speaker_label": (d.get("label") or "").strip() or None,
                 "text": d.get("text", ""), "language": d.get("language") or "English",
                 "voiceover": bool(d.get("voiceover"))}
                for d in lines if (d.get("text") or "").strip()
            ]

        # Per-shot durations from the panel. The video length is a hard ceiling: any
        # set of durations summing past it is scaled down proportionally rather than
        # silently producing cut times beyond the end of the video (which the guide's
        # checklist rejects outright).
        # Comic panels carry a duration derived from their own dialogue, which
        # takes precedence over anything typed in the shot boxes.
        for i, ctx in enumerate(shot_contexts):
            if ctx.duration and i < len(user_shot_durations):
                user_shot_durations[i] = ctx.duration
            elif ctx.duration:
                user_shot_durations.append(ctx.duration)

        if any(d is not None for d in user_shot_durations):
            n_sh = len(vlm_shots)
            explicit = [user_shot_durations[i] if i < len(user_shot_durations) else None
                        for i in range(n_sh)]
            named = [d for d in explicit if d is not None]
            leftover = max(0.0, duration_seconds - sum(named))
            unset = [i for i, d in enumerate(explicit) if d is None]
            # Unspecified shots share whatever time the named ones leave behind.
            share = (leftover / len(unset)) if unset else 0.0
            resolved = [d if d is not None else share for d in explicit]
            total = sum(resolved)
            if total > duration_seconds and total > 0:
                scale = duration_seconds / total
                resolved = [d * scale for d in resolved]
                dbg.section("SHOT DURATIONS CLAMPED")
                dbg.line(f"requested {total:.2f}s over a {duration_seconds:.2f}s video "
                          f"-- scaled by {scale:.3f}")
            for i, d in enumerate(resolved):
                if i < len(vlm_shots):
                    vlm_shots[i]["duration_hint_s"] = max(0.1, d)
            dbg.section("SHOT DURATIONS")
            for i, d in enumerate(resolved, 1):
                dbg.bullet(f"[Shot {i}] {d:.2f}s")

        if shot_count_notice:
            dbg.section("SHOT COUNT ADJUSTED")
            dbg.line(shot_count_notice)

        # Deterministic dialogue-in-content scrub, same as the categorized
        # composer -- a VLM restating dialogue as loose prose alongside the
        # structured field is a real, observed failure mode, not hypothetical.
        cn._scrub_duplicated_dialogue(vlm_shots)

        # Per-shot durations from the panel override the VLM's guesses. The total is
        # capped at duration_seconds authoritatively here rather than in the UI: the
        # duration input is a link-only socket, so the panel can't know the real
        # ceiling. Shots left blank share whatever time the explicit ones don't use;
        # if the explicit ones alone overrun, every shot is scaled down to fit.
        # Comic panels carry a duration derived from their own dialogue, which
        # takes precedence over anything typed in the shot boxes.
        for i, ctx in enumerate(shot_contexts):
            if ctx.duration and i < len(user_shot_durations):
                user_shot_durations[i] = ctx.duration
            elif ctx.duration:
                user_shot_durations.append(ctx.duration)

        if any(d is not None for d in user_shot_durations):
            n_shots = len(vlm_shots)
            wanted = [user_shot_durations[i] if i < len(user_shot_durations) else None
                      for i in range(n_shots)]
            fixed = sum(d for d in wanted if d)
            blanks = [i for i, d in enumerate(wanted) if not d]
            if fixed > duration_seconds and fixed > 0:
                scale = duration_seconds / fixed
                wanted = [(d * scale if d else None) for d in wanted]
                fixed = duration_seconds
                dbg.line(f"shot durations totalled {fixed / scale:.2f}s > "
                          f"{duration_seconds:.2f}s -- scaled to fit")
            if blanks:
                share = max(0.1, (duration_seconds - fixed) / len(blanks))
                for i in blanks:
                    wanted[i] = share
            for i, d in enumerate(wanted):
                if d:
                    vlm_shots[i]["duration_hint_s"] = d
            dbg.kv("shot durations", [round(d or 0, 2) for d in wanted])

        # -- Whisper: only on items the loader tagged "dialogue" -----------------
        # Built lazily: loading the model costs seconds even when nothing needs
        # transcribing, which was pure waste for a comic page or a replacement
        # whose lines are all typed in.
        transcriber = None
        dialogue_clips: List[DialogueClip] = []
        for ap in plan.audio:
            if ap.role != "dialogue":
                continue
            audio_t = tensor_index.get(ap.tag)
            if audio_t is None:
                continue
            if transcriber is None:
                transcriber = FasterWhisperTranscriber(
                    model_size=whisper_model_size, device=device,
                    compute_type="float16" if device == "cuda" else "int8")
            wav_path = cn._write_temp_wav(audio_t)
            dur = audio_t["waveform"].shape[-1] / audio_t["sample_rate"]
            clip = transcribe_clip(transcriber, wav_path, ap.tag, dur, use_pyannote=use_diarization)
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

        # Establishing statements (clothing, location) are kept separate from the
        # shot's action text and rendered BEFORE any dialogue -- see grammar.Shot's
        # `establish` field. Appending them to content put them after the shot's
        # speech, which H3 read as a mid-shot change rather than a starting state.
        establish_by_shot: Dict[int, List[str]] = {}

        # Guide S2.5.3: reference labels must be inserted in detailed_description
        # "where their roles apply". The outfit tag was only ever defined in
        # subject_definitions and never cited in the body -- which is the section
        # the model actually follows shot by shot, and the likeliest reason a
        # correctly-described outfit still didn't reach the generation.
        for sp in plan.subjects:
            for tag, label in (getattr(sp, "outfit_refs", []) or []):
                if label and label != "all shots":
                    m = re.search(r"(\d+)", str(label))
                    idxs = [int(m.group(1))] if m else [1]
                else:
                    idxs = list(range(1, len(vlm_shots) + 1))
                for si in idxs:
                    if si < 1 or si > len(vlm_shots):
                        continue
                    if any(tag in t for t in establish_by_shot.get(si, [])):
                        continue
                    # PREPEND, and state it as a condition already true at the first
                    # frame. detailed_description is read as a timeline, so appending
                    # the clothing sentence after the shot's action (and after its
                    # dialogue) reads as a change happening LATER in the shot -- which
                    # is exactly what produced an outfit that only swapped in halfway
                    # through.
                    # "remain unchanged" was ambiguous -- unchanged relative to
                    # WHAT? The model read it as "keep the clothing already visible",
                    # i.e. the identity image's outfit, so the reference looked
                    # ignored. Name the swap actively and say what it replaces.
                    outfit_subj = next(
                        (o.slot for o in plan.subjects
                         if getattr(o, "outfit_for_subject", None) == sp.slot
                         and tag in o.sources), None)
                    wearing = (f"<Subject {outfit_subj}>" if outfit_subj else f"<{tag}>")
                    # Register the outfit subject as present in this shot so its
                    # retention line reports where it appears instead of being blank.
                    if outfit_subj:
                        _m = vlm_shots[si - 1].setdefault("subject_mentions", [])
                        if f"Subject {outfit_subj}" not in _m:
                            _m.append(f"Subject {outfit_subj}")
                    establish_by_shot.setdefault(si, []).append(
                        f"<Subject {sp.slot}> is dressed in {wearing} — the exact "
                        f"garments, colors, and accessories from <{tag}> — for the whole "
                        f"shot, starting at the first frame. This outfit replaces the "
                        f"clothing seen in that subject's identity reference; only the "
                        f"face, hair, and build carry over from it.")

        # A character-only edit is a rigid form, not a narrative to invent. The
        # guide's worked example is effectively a template: everything about the
        # source is preserved frame-for-frame and ONLY the performer's identity
        # changes. Letting the VLM write this freely produced invented action
        # ("walks in and sits next to..."), which is the opposite of the intent.
        # So the framing is fixed here and the VLM's text is demoted to a
        # description of what the source footage already does.
        for ctx in shot_contexts:
            if ctx.is_replacement and ctx.index <= len(vlm_shots):
                vlm_shots[ctx.index - 1]["camera"] = None
                vlm_shots[ctx.index - 1]["amplitude"] = None
                vlm_shots[ctx.index - 1]["speed"] = None

        swap_subjects = [s_ for s_ in plan.subjects if getattr(s_, "swap_video", None)]
        for ctx in shot_contexts:
            if not ctx.is_replacement or ctx.index > len(vlm_shots):
                continue
            vs = vlm_shots[ctx.index - 1]
            src = next((v.tag for v in plan.video_structures
                        if v.role == "editing"), None)
            if not src:
                continue
            # Filter from ALL subjects, not just swap_subjects: swap_video is only
            # auto-linked when there's exactly one subject, so a two-person
            # replacement had an empty list and fell through to the vague wording.
            mapped = [s_ for s_ in plan.subjects
                      if getattr(s_, "replaces_person", None)]
            if mapped:
                # replaces_person is the plain phrase the loader stores
                # ("the person on the left"); accept a dict too in case an older
                # saved workflow holds the richer shape.
                parts = []
                for s_ in mapped:
                    rp = s_.replaces_person
                    if isinstance(rp, dict):
                        target = ", ".join(x for x in (
                            (rp.get("description") or "").strip(),
                            (rp.get("position") or "").strip()) if x) or "the performer"
                    else:
                        target = str(rp).strip() or "the performer"
                    parts.append(f"<Subject {s_.slot}> takes the place of {target}")
                who = "; ".join(parts) + " -- each replacing only that one person"
            else:
                who = " and ".join(f"<Subject {s_.slot}>" for s_ in swap_subjects) \
                      or "the replacement subject"
            # Your own text wins; otherwise use the VLM's account of the source
            # footage. The guide's worked example carries BOTH the rigid framing
            # and a detailed description of the action -- framing on its own left
            # H3 with nothing on screen to anchor to, so it animated the stills.
            typed = ((user_shot_prompts[ctx.index - 1]
                      if ctx.index - 1 < len(user_shot_prompts) else "") or "").strip()
            # Deliberately NOT falling back to the VLM's text: for a
            # character-only edit there is no new action, so anything it wrote is
            # invented by definition. Only what you typed survives.
            observed = typed
            establish_by_shot.setdefault(ctx.index, []).insert(0,
                f"This shot is a strict character-only edit of <{src}>: the camera "
                f"framing, camera motion, cuts, lighting, environment and the exact "
                f"timing of every action are identical to the source, and nothing is "
                f"added, removed or re-staged. Only the performer's visual identity is "
                f"replaced: {who}. Each replacement is mapped onto that person's exact "
                f"body position, pose and movement in every frame. Everyone else in the "
                f"footage is left exactly as they are, and no new person is added.")
            # Keep the model's observation of the source, framed as a report of
            # what the footage does rather than as new action.
            vs["content"] = (f"Throughout, the action follows <{src}> exactly: {observed}"
                             if observed else
                             f"Every movement, gesture and beat plays out exactly as it "
                             f"does in <{src}>, at the same moments, with no new action "
                             f"introduced.")

        # Continuation: say plainly that the shot picks up from the END of the
        # source clip. Goes through `establish` so it lands before any dialogue or
        # action -- it describes where the shot starts from, not something that
        # happens during it.
        for ctx in shot_contexts:
            if not ctx.continues_from:
                continue
            establish_by_shot.setdefault(ctx.index, []).insert(0,
                f"The video resumes directly from the final frame of "
                f"<{ctx.continues_from}>, picking up exactly where that clip ends "
                f"and carrying its framing, lighting and action forward without a cut.")

        # A shot's scene/prop reference belongs to that shot specifically -- make sure
        # it's listed among that shot's subject_mentions so its retention line says
        # where it appears, and cite it inline so detailed_description actually
        # references the location instead of dropping it.
        for sp in plan.subjects:
            shot_no = getattr(sp, "ref_for_shot", None)
            if not shot_no or shot_no > len(vlm_shots):
                continue
            vs = vlm_shots[shot_no - 1]
            tag = f"Subject {sp.slot}"
            mentions = vs.setdefault("subject_mentions", [])
            if tag not in mentions:
                mentions.append(tag)
            if not any(tag in t for t in establish_by_shot.get(shot_no, [])):
                establish_by_shot.setdefault(shot_no, []).append(
                    f"The shot takes place in <{tag}>.")

        # Panel-written dialogue becomes synthetic clips so it flows through the SAME
        # <d>-formatting/speaker-ID pipeline as real transcribed audio, rather than the
        # VLM hand-formatting tags in prose (which is exactly how dialogue previously
        # came out unformatted, untagged for language, and duplicated).
        hints = [max(0.1, float(vs.get("duration_hint_s") or (duration_seconds / max(1, len(vlm_shots)))))
                 for vs in vlm_shots]
        total = sum(hints) or 1.0
        bounds, cum = [], 0.0
        for h in hints:
            start = cum
            cum += h / total * duration_seconds
            bounds.append((start, cum))
        if bounds:
            bounds[-1] = (bounds[-1][0], duration_seconds)

        synthetic, custom_labels = {}, {}
        for si, vs in enumerate(vlm_shots):
            lines = vs.get("dialogue") or []
            if not lines:
                continue
            s_start, s_end = bounds[si] if si < len(bounds) else (0.0, duration_seconds)
            cursor = s_start
            for d in lines:
                text = (d.get("text") or "").strip()
                if not text:
                    continue
                subj = d.get("speaker_subject")
                label = (d.get("speaker_label") or "").strip()
                if subj:
                    tag = f"Shot:{subj}"
                elif label:
                    # Stable per-label tag: the same label reused across shots resolves
                    # to the same (Sx), like a real recurring speaker would.
                    tag = f"Shot:Label:{label}"
                else:
                    tag = "Shot:Narrator"
                if tag not in synthetic:
                    synthetic[tag] = DialogueClip(media_tag=tag, turns=[],
                                                   clip_duration_s=duration_seconds)
                    if subj:
                        m = re.match(r"Subject\s*(\d+)", str(subj))
                        if m:
                            subject_of_clip[tag] = f"Subject {m.group(1)}"
                    elif label:
                        custom_labels[tag] = label
                    if d.get("voiceover"):
                        voiceover_clips.add(tag)
                # Natural speech rate rather than dividing the shot evenly, so a short
                # line isn't falsely marked <cutoff> for filling the whole shot.
                dur = max(0.8, max(1, len(text.split())) / 2.5)
                t0 = min(cursor, s_end)
                t1 = min(t0 + dur, duration_seconds)
                synthetic[tag].turns.append(SpeakerTurn(
                    start=t0, end=t1, text=text,
                    lang=(d.get("language") or "English").strip() or "English", cluster=0))
                cursor = t1 + 0.2

        if synthetic:
            dialogue_clips = dialogue_clips + list(synthetic.values())
            speaker_ids = c.assign_speaker_ids(dialogue_clips)
            for key in speaker_ids:
                if key in speaker_labels:
                    continue
                ct = key.split("::", 1)[0]
                speaker_labels[key] = custom_labels.get(
                    ct, "an unseen narrator" if ct in voiceover_clips else "the speaker")

        shots = c.build_shots(vlm_shots, dialogue_clips, duration_seconds, speaker_labels,
                               speaker_ids, subject_of_clip, voiceover_clips=voiceover_clips,
                               sung_clips=sung_clips)

        # subject -> (Sx), so a Vocal Reference can cite the speaker ID of the
        # subject it re-voices (guide S2.2.4 reuses the global ID rather than
        # assigning a new one).
        subject_sx = {}
        for key, sx in speaker_ids.items():
            subj = subject_of_clip.get(key.split("::", 1)[0])
            if subj and subj not in subject_sx:
                subject_sx[subj] = sx

        dbg.section("SHOTS ASSEMBLED")
        for _i, _sh in enumerate(shots, 1):
            dbg.bullet(f"[Shot {_i}] {_sh.content}")
            for _ev in (_sh.dialogue or []):
                dbg.kv("dialogue", f"({_ev.sx}) [{_ev.lang}] {_ev.text}"
                                    + ("  (voiceover)" if _ev.voiceover else ""))
        if speaker_ids:
            dbg.section("SPEAKERS")
            for _k, _sx in speaker_ids.items():
                _clip = _k.split("::", 1)[0]
                dbg.bullet(f"{_sx} <- {_clip}  ({speaker_labels.get(_k, '')})"
                            + (f"  = {subject_of_clip[_clip]}" if _clip in subject_of_clip else ""))

        for _idx, _texts in establish_by_shot.items():
            if 1 <= _idx <= len(shots):
                shots[_idx - 1].establish = " ".join(_texts)

        # "auto" defers to whatever the VLM read off the reference images; any other
        # choice is an explicit override.
        style_text = None if str(style).startswith("auto") else str(style)
        if style_text is None:
            style_text = (result.get("style") or "").strip() or None
            # Cross-check the model's answer against pixel statistics, which don't
            # hallucinate. Only a HIGH-confidence measurement overrides it: flat
            # cel fills carry no sensor noise, photographs always do, but graded
            # or CG footage sits in between where this genuinely can't decide.
            try:
                from . import comic as _cmc
                # Probe a CHARACTER reference specifically -- an outfit shot on a
                # mannequin or a scene plate would describe the wrong thing.
                probe = None
                src_file = None
                for _s2 in (references.get("subjects") or []):
                    for _r in (_s2.get("visual_refs") or []):
                        src_file = _r.get("file"); break
                    if not src_file and (_s2.get("complete") or {}).get("kind") == "picture":
                        src_file = _s2["complete"].get("file")
                    if src_file:
                        break
                if src_file:
                    probe = _cmc.estimate_style(sl.resolve_media_path(src_file))
                if probe and probe.get("confidence") == "high":
                    drawn = probe["style"] == "2D-animated"
                    said_drawn = "anim" in (style_text or "").lower() \
                                 or "anime" in (style_text or "").lower()
                    if drawn != said_drawn:
                        dbg.line(f"style cross-check overrode the model: "
                                 f"{style_text!r} -> {probe['style']!r} "
                                 f"(flat_sd={probe['flat_sd']})")
                        style_text = probe["style"]
            except Exception as _exc:
                dbg.line(f"style cross-check skipped ({_exc})")
        dbg.section("STYLE")
        dbg.kv("setting", style)
        dbg.kv("resolved", style_text or "(fallback: live-action, cinematic)")

        prompt, _unused = cn._render_common(
            plan=plan, shots=shots, vlm_shots=vlm_shots, anchor_desc=anchor_desc,
            speaker_ids=speaker_ids, subject_sx=subject_sx,
            overall_soundscape_raw=result.get("overall_soundscape", "N/A"),
            non_diegetic_music_raw=result.get("non_diegetic_music", "N/A"),
            style_text=style_text)

        dbg.section("FINAL PROMPT")
        dbg.multiline("prompt", prompt)

        problems = problems_pre + g.validate(prompt, "REF", duration_seconds)
        dbg.section("CHECKLIST")
        if problems:
            for _pb in problems:
                dbg.bullet(_pb)
        else:
            dbg.line("passes the guide checklist")
        plan_summary = (
            f"Structured loader: {len(plan.subjects)} subject(s), {len(plan.anchors)} anchor(s), "
            f"{len(shot_contexts)} shot(s) "
            f"({sum(1 for ctx in shot_contexts if ctx.is_replacement)} replacement-type, "
            f"{sum(1 for ctx in shot_contexts if ctx.continues_from)} continuation), "
            f"{len(plan.audio)} audio role(s)."
        )
        report = plan_summary
        # Per-shot durations from the panel. The video length is a hard ceiling: any
        # set of durations summing past it is scaled down proportionally rather than
        # silently producing cut times beyond the end of the video (which the guide's
        # checklist rejects outright).
        # Comic panels carry a duration derived from their own dialogue, which
        # takes precedence over anything typed in the shot boxes.
        for i, ctx in enumerate(shot_contexts):
            if ctx.duration and i < len(user_shot_durations):
                user_shot_durations[i] = ctx.duration
            elif ctx.duration:
                user_shot_durations.append(ctx.duration)

        if any(d is not None for d in user_shot_durations):
            n_sh = len(vlm_shots)
            explicit = [user_shot_durations[i] if i < len(user_shot_durations) else None
                        for i in range(n_sh)]
            named = [d for d in explicit if d is not None]
            leftover = max(0.0, duration_seconds - sum(named))
            unset = [i for i, d in enumerate(explicit) if d is None]
            # Unspecified shots share whatever time the named ones leave behind.
            share = (leftover / len(unset)) if unset else 0.0
            resolved = [d if d is not None else share for d in explicit]
            total = sum(resolved)
            if total > duration_seconds and total > 0:
                scale = duration_seconds / total
                resolved = [d * scale for d in resolved]
                dbg.section("SHOT DURATIONS CLAMPED")
                dbg.line(f"requested {total:.2f}s over a {duration_seconds:.2f}s video "
                          f"-- scaled by {scale:.3f}")
            for i, d in enumerate(resolved):
                if i < len(vlm_shots):
                    vlm_shots[i]["duration_hint_s"] = max(0.1, d)
            dbg.section("SHOT DURATIONS")
            for i, d in enumerate(resolved, 1):
                dbg.bullet(f"[Shot {i}] {d:.2f}s")

        if shot_count_notice:
            report += "\n" + shot_count_notice
        report += "\n\n" + (
            "OK - passes the mechanical checklist." if not problems else
            "ISSUES FOUND:\n" + "\n".join(f"- {p}" for p in problems))
        return (prompt, report)


# H3StructuredComposer is deliberately NOT registered as a node anymore -- the
# All-in-One node calls it directly, so exposing it separately was just a second
# way to do the same thing. It stays a plain class for that internal use.
NODE_CLASS_MAPPINGS = {
    "H3StructuredMediaLoader": H3StructuredMediaLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3StructuredMediaLoader": "H3 AutoPrompt: Structured Media Loader",
}
