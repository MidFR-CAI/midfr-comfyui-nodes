# ComfyUI Text Edit (Pause)

A custom node that pauses the workflow, shows the incoming text in a popup
editor in your browser, and resumes execution with the (optionally) edited
text once you click **Resume**.

## Install

1. Copy the `ComfyUI-TextEditPause` folder into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI (it needs to pick up the new Python route and JS extension).

## Use

1. Add the **Text Edit (Pause)** node (category: `utils/text`).
2. Wire any `STRING` output into its `text` input.
3. Wire its `text` output into whatever should receive the edited text
   (e.g. a CLIP Text Encode prompt box).
4. Queue the prompt. When execution reaches this node, a dialog pops up
   in the browser tab with the incoming text pre-filled.
5. Edit as needed and click **Resume** (or press Ctrl/Cmd+Enter) to
   continue the run with your edited text. Click **Abort** to stop the
   workflow entirely (the node raises an error and the rest of the queued
   prompt is interrupted, just like clicking ComfyUI's own Cancel button).

## Notes / limitations

- Execution is genuinely blocked on the backend while the dialog is open —
  the ComfyUI server thread for that prompt sits waiting, so don't leave it
  paused indefinitely if others share the server.
- The pause is matched to the node via ComfyUI's internal node id for that
  run. If you queue multiple prompts back-to-back, each will pop its own
  dialog in the order they reach the node.
- If the browser tab is closed while paused, the run will hang until you
  reopen ComfyUI in a tab (which will re-fire the listener) or restart the
  server.
