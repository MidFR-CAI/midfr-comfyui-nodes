import threading

from aiohttp import web
from server import PromptServer

try:
    import comfy.model_management as model_management
except ImportError:
    model_management = None


class WorkflowAborted(Exception):
    """Raised when the user clicks Abort in the pause dialog."""
    pass


# Holds one entry per in-flight pause, keyed by the node's unique_id.
# entry = {"event": threading.Event, "result": str, "aborted": bool}
_pending = {}
_lock = threading.Lock()


class TextEditPause:
    """
    Pauses the workflow at this node, sends the incoming text to the
    ComfyUI browser tab, and blocks until the user edits the text and
    clicks "Resume" in the popup dialog. The (possibly edited) text is
    then passed through as the node's output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "pause_and_edit"
    CATEGORY = "utils/text"
    OUTPUT_NODE = False

    def pause_and_edit(self, text, unique_id):
        node_id = str(unique_id)
        event = threading.Event()

        with _lock:
            _pending[node_id] = {"event": event, "result": text, "aborted": False}

        # Tell the frontend to pop open the edit dialog for this node.
        PromptServer.instance.send_sync(
            "textedit.pause",
            {"node_id": node_id, "text": text},
        )

        # Block the prompt-execution thread until /textedit/submit or
        # /textedit/abort is hit.
        event.wait()

        with _lock:
            entry = _pending.pop(node_id, None)

        if entry is not None and entry.get("aborted"):
            if model_management is not None:
                model_management.interrupt_current_processing()
            raise WorkflowAborted("Workflow aborted by user in Text Edit (Pause) node.")

        result_text = entry["result"] if entry is not None else text
        return (result_text,)


routes = PromptServer.instance.routes


@routes.post("/textedit/submit")
async def textedit_submit(request):
    data = await request.json()
    node_id = str(data.get("node_id"))
    edited_text = data.get("text", "")

    with _lock:
        entry = _pending.get(node_id)
        if entry is not None:
            entry["result"] = edited_text
            entry["event"].set()
            return web.json_response({"status": "ok"})

    return web.json_response({"status": "not_found"}, status=404)


@routes.post("/textedit/abort")
async def textedit_abort(request):
    data = await request.json()
    node_id = str(data.get("node_id"))

    with _lock:
        entry = _pending.get(node_id)
        if entry is not None:
            entry["aborted"] = True
            entry["event"].set()
            return web.json_response({"status": "ok"})

    return web.json_response({"status": "not_found"}, status=404)


NODE_CLASS_MAPPINGS = {
    "TextEditPause": TextEditPause,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEditPause": "Text Edit (Pause)",
}
