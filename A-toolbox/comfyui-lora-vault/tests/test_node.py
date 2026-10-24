import os, sys, types
import pytest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
from fake_civitai import Fake  # noqa: E402
import importlib.util


def load_pkg():
    """Import the node folder the way ComfyUI does (as a package, so relative imports work)."""
    if "lvpkg" in sys.modules:
        return sys.modules["lvpkg"]
    root = os.path.dirname(HERE)
    spec = importlib.util.spec_from_file_location("lvpkg", os.path.join(root, "__init__.py"), submodule_search_locations=[root])
    mod = importlib.util.module_from_spec(spec); sys.modules["lvpkg"] = mod; spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("LORA_VAULT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LORA_VAULT_LORA_DIR", str(tmp_path / "loras"))
    fake = Fake().start()
    from vault import config
    config.update({"civitai_hosts": [f"http://127.0.0.1:{fake.api_port}"], "civitai_api_key": "KEY123",
                   "transport": "direct", "cache_limit_gb": 0})
    calls = []
    comfy = types.ModuleType("comfy"); sd = types.ModuleType("comfy.sd"); ut = types.ModuleType("comfy.utils")
    ut.load_torch_file = lambda p, safe_load=True: {"path": p}
    sd.load_lora_for_models = lambda m, c, lora, sm, sc: (calls.append((os.path.basename(lora["path"]), sm, sc)) or (m + "+", c + "+"))
    comfy.sd, comfy.utils = sd, ut
    monkeypatch.setitem(sys.modules, "comfy", comfy); monkeypatch.setitem(sys.modules, "comfy.sd", sd); monkeypatch.setitem(sys.modules, "comfy.utils", ut)
    return fake, calls


def test_loader_downloads_on_demand_applies_and_returns_trigger_words(env):
    fake, calls = env
    from vault import snapshot, storage, db
    LoraVaultLoader = load_pkg().nodes.LoraVaultLoader
    snapshot.save_version(version_id=111); snapshot.save_version(version_id=222)
    assert storage.local_state(db.get_version(111)) == "none"
    model, clip, words, loaded = LoraVaultLoader().load("M", "C", "<lora:v111:0.8> <lora:v222:1:0.5>")
    assert (model, clip) == ("M++", "C++")
    assert calls == [("Cool Style - v1.0.safetensors", 0.8, 0.8), ("Pony Thing - v2.safetensors", 1.0, 0.5)]
    assert words == "coolstyle, pt" and "Cool Style / v1.0 @ 0.8" in loaded
    assert storage.local_state(db.get_version(111)) == "cached"


def test_unknown_lora_fails_before_any_download(env):
    fake, calls = env
    from vault import snapshot, storage, db, parse
    LoraVaultLoader = load_pkg().nodes.LoraVaultLoader
    snapshot.save_version(version_id=111)
    with pytest.raises(Exception, match="not in your vault"):
        LoraVaultLoader().load("M", "C", "<lora:v111:1> <lora:v9999:1>")
    assert storage.local_state(db.get_version(111)) == "none" and calls == []


def test_download_missing_off_refuses(env):
    fake, calls = env
    from vault import snapshot
    LoraVaultLoader = load_pkg().nodes.LoraVaultLoader
    snapshot.save_version(version_id=111)
    with pytest.raises(RuntimeError, match="isn't downloaded"):
        LoraVaultLoader().load("M", "C", "<lora:v111:1>", download_missing=False)


def test_fused_qkv_is_split_and_layers_limited(env, monkeypatch):
    import numpy as np
    fake, calls = env
    from vault import snapshot
    LoraVaultLoader = load_pkg().nodes.LoraVaultLoader
    snapshot.save_version(version_id=111)
    P = "diffusion_model.layers.{}.proj"
    key_map = {f"{P.format(i)}.{x}": f"{P.format(i)}.{x}.weight" for i in range(3) for x in ("q", "k", "v")}
    lm = types.ModuleType("comfy.lora"); lm.model_lora_keys_unet = lambda m, km: dict(key_map)
    monkeypatch.setitem(sys.modules, "comfy.lora", lm); sys.modules["comfy"].lora = lm
    lora = {}
    for i in range(3):
        lora[f"{P.format(i)}.qkv.lora_A.weight"] = np.ones((2, 4)); lora[f"{P.format(i)}.qkv.lora_B.weight"] = np.ones((9, 2))
    seen = {}
    sys.modules["comfy.utils"].load_torch_file = lambda p, safe_load=True: lora
    def apply(m, c, sd, sm, sc):
        seen.update(sd); return m, c
    sys.modules["comfy.sd"].load_lora_for_models = apply

    class FakeModel:
        model = object()
        def get_model_object(self, k): return np.zeros((3, 4))
    _, _, _, loaded = LoraVaultLoader().load(FakeModel(), "C", "<lora:v111:0.4:layers=1-2>")
    assert sorted(seen) == sorted(f"{P.format(i)}.{x}.lora_{ab}.weight" for i in (1, 2) for x in "qkv" for ab in "AB")
    assert seen[f"{P.format(1)}.q.lora_B.weight"].shape == (3, 2)
    assert "split 3 fused qkv" in loaded and "layers 1-2" in loaded and "@ 0.4" in loaded
