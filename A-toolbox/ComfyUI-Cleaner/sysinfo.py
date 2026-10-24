"""
Reads the state of the machine ComfyUI is running on and turns it into
suggestions a person can act on.

Two hard rules:
  * nothing here ever terminates anything on its own
  * operating-system and desktop-session processes are never suggested
"""

from __future__ import annotations

import logging
import os
import platform
import sys

log = logging.getLogger("ResourceReaper")

GB = 1024 ** 3

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

_nvml_ready = False
try:
    import pynvml
    try:
        pynvml.nvmlInit()
        _nvml_ready = True
    except Exception as exc:
        log.debug("nvml init failed: %s", exc)
except Exception:
    pynvml = None


# Processes that are load-bearing for the OS or the desktop session. These are
# filtered out of every list, so nothing in the UI can point a person at them.
PROTECTED = {
    "system", "system idle process", "registry", "memory compression",
    "csrss.exe", "wininit.exe", "winlogon.exe", "services.exe", "lsass.exe",
    "smss.exe", "svchost.exe", "dwm.exe", "explorer.exe", "fontdrvhost.exe",
    "ctfmon.exe", "audiodg.exe", "sihost.exe", "runtimebroker.exe",
    "searchindexer.exe", "shellexperiencehost.exe", "startmenuexperiencehost.exe",
    "systemd", "systemd-journald", "systemd-logind", "init", "kthreadd",
    "dbus-daemon", "pipewire", "pulseaudio", "gnome-shell", "kwin_x11",
    "kwin_wayland", "xorg", "wayland", "plasmashell", "windowserver",
    "loginwindow", "kernel_task", "launchd", "coreaudiod",
}

# Known VRAM-hungry neighbours, with the reason they matter.
GPU_NEIGHBOURS = {
    "chrome": "browser GPU process; hardware acceleration reserves VRAM",
    "msedge": "browser GPU process; hardware acceleration reserves VRAM",
    "firefox": "browser GPU process; hardware acceleration reserves VRAM",
    "brave": "browser GPU process; hardware acceleration reserves VRAM",
    "obs64": "OBS holds an encoder and a live GPU surface",
    "obs": "OBS holds an encoder and a live GPU surface",
    "blender": "Blender keeps its scene resident in VRAM",
    "resolve": "DaVinci Resolve reserves a large GPU pool",
    "premiere": "Premiere keeps a GPU playback cache",
    "afterfx": "After Effects keeps a GPU playback cache",
    "photoshop": "Photoshop reserves VRAM for the canvas",
    "unrealeditor": "Unreal keeps its render target in VRAM",
    "unity": "Unity keeps its render target in VRAM",
    "steam": "a running game or shader compile holds VRAM",
    "ollama": "another model server is holding VRAM",
    "lmstudio": "another model server is holding VRAM",
    "koboldcpp": "another model server is holding VRAM",
    "python": "another Python process may hold a CUDA context",
    "pythonw": "another Python process may hold a CUDA context",
}

BROWSER_NAMES = {"chrome", "msedge", "firefox", "brave", "opera", "vivaldi",
                 "chromium", "safari", "arc"}


def _norm(name: str) -> str:
    base = (name or "").lower()
    return base[:-4] if base.endswith(".exe") else base


def _is_protected(name: str) -> bool:
    low = (name or "").lower()
    return low in PROTECTED or _norm(low) in PROTECTED


# --------------------------------------------------------------------------
# raw readings
# --------------------------------------------------------------------------

def memory() -> dict:
    if psutil is None:
        return {"available": False}
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    return {
        "available": True,
        "total": vm.total,
        "used": vm.total - vm.available,
        "free": vm.available,
        "percent": vm.percent,
        "swap_total": sm.total,
        "swap_used": sm.used,
        "swap_percent": sm.percent,
    }


def gpus() -> list[dict]:
    if not _nvml_ready:
        return []
    out = []
    try:
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            entry = {
                "index": i,
                "name": _decode(pynvml.nvmlDeviceGetName(h)),
                "total": int(info.total),
                "used": int(info.used),
                "free": int(info.free),
                "percent": round(info.used / info.total * 100, 1) if info.total else 0,
                "processes": [],
            }
            try:
                entry["utilization"] = int(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:
                entry["utilization"] = None
            try:
                entry["temperature"] = int(
                    pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                entry["temperature"] = None

            procs = []
            for fn in ("nvmlDeviceGetComputeRunningProcesses",
                       "nvmlDeviceGetGraphicsRunningProcesses"):
                try:
                    procs.extend(getattr(pynvml, fn)(h))
                except Exception:
                    continue
            seen = set()
            for p in procs:
                if p.pid in seen:
                    continue
                seen.add(p.pid)
                name = _proc_name(p.pid)
                if _is_protected(name):
                    continue
                entry["processes"].append({
                    "pid": p.pid,
                    "name": name,
                    "vram": int(p.usedGpuMemory) if getattr(p, "usedGpuMemory", None) else None,
                    "is_self": p.pid == os.getpid(),
                })
            entry["processes"].sort(key=lambda p: p["vram"] or 0, reverse=True)
            out.append(entry)
    except Exception as exc:
        log.debug("nvml read failed: %s", exc)
    return out


def _decode(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _proc_name(pid: int) -> str:
    if psutil is None:
        return f"pid {pid}"
    try:
        return psutil.Process(pid).name()
    except Exception:
        return f"pid {pid}"


def processes(limit: int = 12, min_rss: int = 300 * 1024 * 1024) -> list[dict]:
    """Heaviest RAM consumers, grouped by program rather than listed per-PID."""
    if psutil is None:
        return []

    self_pid = os.getpid()
    groups: dict[str, dict] = {}

    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            name = proc.info["name"] or ""
            if _is_protected(name):
                continue
            rss = proc.info["memory_info"].rss if proc.info["memory_info"] else 0
            if rss <= 0:
                continue
            key = _norm(name)
            g = groups.setdefault(key, {
                "name": name, "key": key, "rss": 0, "count": 0,
                "pids": [], "is_self": False,
            })
            g["rss"] += rss
            g["count"] += 1
            g["is_self"] = g["is_self"] or proc.info["pid"] == self_pid
            if len(g["pids"]) < 64:
                g["pids"].append(proc.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue

    rows = [g for g in groups.values() if g["rss"] >= min_rss]
    rows.sort(key=lambda g: g["rss"], reverse=True)
    return rows[:limit]


def disks() -> list[dict]:
    """Free space on the volumes ComfyUI writes to."""
    if psutil is None:
        return []
    paths = {os.path.abspath(os.sep)}
    for env in ("TEMP", "TMPDIR"):
        if os.environ.get(env):
            paths.add(os.environ[env])
    paths.add(os.path.dirname(os.path.abspath(sys.argv[0])) or os.getcwd())

    out = []
    seen_volumes = set()
    for path in sorted(paths):
        try:
            volume = os.stat(path).st_dev
            if volume in seen_volumes:
                continue
            seen_volumes.add(volume)
            usage = psutil.disk_usage(path)
            out.append({
                "path": path,
                "total": usage.total,
                "free": usage.free,
                "percent": usage.percent,
            })
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------
# suggestions
# --------------------------------------------------------------------------

def suggestions(mem: dict, gpu_list: list[dict], proc_list: list[dict],
                disk_list: list[dict]) -> list[dict]:
    """
    Turn readings into things worth doing, highest severity first.
    Severity is one of: critical, warn, info.
    """
    out: list[dict] = []
    self_pid = os.getpid()

    # --- other programs holding VRAM -------------------------------------
    for gpu in gpu_list:
        for p in gpu["processes"]:
            if p["is_self"] or not p["vram"]:
                continue
            if p["vram"] < 200 * 1024 ** 2:
                continue
            key = _norm(p["name"])
            why = GPU_NEIGHBOURS.get(key, "holds a GPU context that ComfyUI cannot reclaim")
            out.append({
                "id": f"gpu-proc-{p['pid']}",
                "severity": "critical" if p["vram"] > 1.5 * GB else "warn",
                "title": f"Close {p['name']}",
                "detail": f"Holding {fmt(p['vram'])} of VRAM on GPU {gpu['index']} — {why}.",
                "pid": p["pid"],
                "process": p["name"],
            })

    # --- VRAM pressure overall -------------------------------------------
    for gpu in gpu_list:
        if gpu["percent"] >= 92:
            out.append({
                "id": f"vram-{gpu['index']}",
                "severity": "critical",
                "title": "VRAM is nearly full",
                "detail": (f"{gpu['name']} is at {gpu['percent']}% "
                           f"({fmt(gpu['free'])} free). On Windows the driver will start "
                           "spilling into shared system memory, which is roughly ten times "
                           "slower than VRAM and is the usual cause of a sudden slowdown. "
                           "Run a deep clean, then close other GPU programs."),
            })
        elif gpu["percent"] >= 80:
            out.append({
                "id": f"vram-{gpu['index']}",
                "severity": "warn",
                "title": "VRAM is getting tight",
                "detail": f"{gpu['name']} is at {gpu['percent']}% with {fmt(gpu['free'])} free.",
            })

    # --- system RAM -------------------------------------------------------
    if mem.get("available"):
        if mem["percent"] >= 90:
            out.append({
                "id": "ram",
                "severity": "critical",
                "title": "System RAM is nearly full",
                "detail": (f"{mem['percent']}% used, {fmt(mem['free'])} free. "
                           "ComfyUI caches models in RAM between runs; when RAM runs out "
                           "the OS pages to disk and every run stalls. Turn on cache "
                           "purging below, or close the largest programs listed."),
            })
        elif mem["percent"] >= 78:
            out.append({
                "id": "ram",
                "severity": "warn",
                "title": "System RAM is filling up",
                "detail": f"{mem['percent']}% used, {fmt(mem['free'])} free.",
            })
        if mem["swap_total"] and mem["swap_percent"] >= 40:
            out.append({
                "id": "swap",
                "severity": "warn",
                "title": "The machine is paging to disk",
                "detail": (f"Swap is {mem['swap_percent']}% used. Anything paged out has to "
                           "be read back from disk before the next run, which shows up as "
                           "a long pause before sampling starts."),
            })

    # --- heavy RAM programs ----------------------------------------------
    for g in proc_list:
        if g["is_self"] or g["rss"] < 2 * GB:
            continue
        if g["key"] in BROWSER_NAMES:
            detail = (f"{g['count']} processes using {fmt(g['rss'])}. Each open ComfyUI tab "
                      "keeps every previewed image in memory — reload the tab to drop them.")
        else:
            detail = f"{g['count']} process(es) using {fmt(g['rss'])}."
        out.append({
            "id": f"ram-proc-{g['key']}",
            "severity": "warn" if g["rss"] > 4 * GB else "info",
            "title": f"Close {g['name']}",
            "detail": detail,
            "pids": g["pids"],
            "process": g["name"],
        })

    # --- ComfyUI's own footprint -----------------------------------------
    if psutil is not None:
        try:
            rss = psutil.Process(self_pid).memory_info().rss
            if rss > 24 * GB:
                out.append({
                    "id": "self-rss",
                    "severity": "warn",
                    "title": "ComfyUI is holding a lot of RAM",
                    "detail": (f"This process is using {fmt(rss)}. That is cached models and "
                               "node outputs from previous runs. Enable cache purging, or "
                               "start ComfyUI with --disable-smart-memory if it keeps growing."),
                })
        except Exception:
            pass

    # --- disk -------------------------------------------------------------
    for d in disk_list:
        if d["free"] < 10 * GB:
            out.append({
                "id": f"disk-{d['path']}",
                "severity": "warn",
                "title": "Low disk space",
                "detail": (f"{fmt(d['free'])} free on {d['path']}. Model loading memory-maps "
                           "files and the OS needs headroom to page; clear space here."),
            })

    if not out:
        out.append({
            "id": "clear",
            "severity": "info",
            "title": "Nothing to close",
            "detail": "Memory has headroom and no other program is holding the GPU.",
        })

    rank = {"critical": 0, "warn": 1, "info": 2}
    out.sort(key=lambda s: rank.get(s["severity"], 3))
    return out


def fmt(num: int | None) -> str:
    if not num:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024 or unit == "TB":
            return f"{num:.1f} {unit}" if unit not in ("B", "KB") else f"{num:.0f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def snapshot() -> dict:
    mem = memory()
    gpu_list = gpus()
    proc_list = processes()
    disk_list = disks()
    return {
        "platform": f"{platform.system()} {platform.release()}",
        "psutil": psutil is not None,
        "nvml": _nvml_ready,
        "memory": mem,
        "gpus": gpu_list,
        "processes": proc_list,
        "disks": disk_list,
        "suggestions": suggestions(mem, gpu_list, proc_list, disk_list),
    }


# --------------------------------------------------------------------------
# termination (opt-in, guarded)
# --------------------------------------------------------------------------

def terminate(pid: int) -> dict:
    """
    Ask a process to exit. Refuses protected processes and refuses to kill
    ComfyUI itself. Sends a terminate signal only — never SIGKILL — so the
    program gets a chance to save.
    """
    if psutil is None:
        return {"ok": False, "error": "psutil is not installed"}
    if pid == os.getpid():
        return {"ok": False, "error": "that is ComfyUI itself"}
    try:
        proc = psutil.Process(pid)
        name = proc.name()
    except psutil.NoSuchProcess:
        return {"ok": False, "error": "process already gone"}
    except psutil.AccessDenied:
        return {"ok": False, "error": "access denied"}

    if _is_protected(name):
        return {"ok": False, "error": f"{name} is a system process"}

    try:
        proc.terminate()
        proc.wait(timeout=5)
        return {"ok": True, "name": name, "pid": pid}
    except psutil.TimeoutExpired:
        return {"ok": False, "name": name,
                "error": f"{name} did not exit within 5s; close it yourself"}
    except psutil.AccessDenied:
        return {"ok": False, "error": f"no permission to close {name}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
