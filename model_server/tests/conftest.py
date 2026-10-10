"""Shared bootstrap for `model_server` tests.

`safety.py` / `unifolm_wla_action_adapter.py` import `SLICES` from
`unifolm_wla.dataloader.multi_source_dataset.action_mapping`. Importing that
through the normal package path executes `unifolm_wla/dataloader/__init__.py`,
which pulls in accelerate, torch.distributed, lerobot, torchvision, pyarrow and
datasets. `action_mapping` itself needs only numpy + yaml + scipy.

So we register lightweight stubs for the parent packages and load the three real
leaf modules by file path. Tests therefore run on a machine with no GPU stack --
and they still exercise the *real* `SLICES` / `STATE_SLICES` widths, which is the
whole point of not copying the slot tables into the tests.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MSD_DIR = REPO_ROOT / "unifolm_wla" / "dataloader" / "multi_source_dataset"


def _bootstrap_lightweight_action_mapping() -> None:
    full_names = [f"unifolm_wla.dataloader.multi_source_dataset.{m}" for m in
                  ("config", "se3_utils", "action_mapping")]
    if full_names[-1] in sys.modules:
        return
    for name in (
        "unifolm_wla",
        "unifolm_wla.dataloader",
        "unifolm_wla.dataloader.multi_source_dataset",
    ):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = []  # mark as a package so relative imports resolve
            sys.modules[name] = stub
    # Leaf modules: load in dependency order (config, se3_utils, action_mapping).
    for mod in ("config", "se3_utils", "action_mapping"):
        full = f"unifolm_wla.dataloader.multi_source_dataset.{mod}"
        if full in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(full, MSD_DIR / f"{mod}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)


_bootstrap_lightweight_action_mapping()

# `model_server` modules import each other by bare name (the server does
# `sys.path.insert(0, os.path.dirname(__file__))` at runtime).
if str(REPO_ROOT / "model_server") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "model_server"))
