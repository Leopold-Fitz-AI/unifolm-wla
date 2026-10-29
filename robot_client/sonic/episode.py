"""Atomic lifecycle marker shared by a policy client and task evaluator."""
import json
from pathlib import Path
import time


def write_episode(path, episode_id, status, source):
    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps({"episode_id": episode_id, "status": status,
                                        "policy_source": source, "wall_time": time.time()}))
        temporary.replace(path)
