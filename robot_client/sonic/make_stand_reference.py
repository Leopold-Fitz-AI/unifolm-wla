"""Create a static startup reference from SONIC's own default joint angles."""

import argparse
from pathlib import Path
import re

import numpy as np

from .protocol import MUJOCO_TO_ISAAC


def generate(sonic_root, output):
    header = Path(sonic_root) / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"
    match = re.search(r"default_angles\s*=\s*\{([^}]+)\}", header.read_text())
    text = re.sub(r"//[^\n]*", "", match[1])
    q = np.array([float(v) for v in text.split(",") if v.strip()])
    if q.shape != (29,):
        raise ValueError("unknown default joint layout")
    output = Path(output) / "stand"
    output.mkdir(parents=True, exist_ok=True)
    arrays = {"joint_pos": np.tile(q[MUJOCO_TO_ISAAC], (100, 1)),
              "joint_vel": np.zeros((100, 29)),
              "body_pos": np.tile([0, 0, .75], (100, 1)),
              "body_quat": np.tile([1, 0, 0, 0], (100, 1)),
              "body_lin_vel": np.zeros((100, 3)), "body_ang_vel": np.zeros((100, 3))}
    for name, array in arrays.items():
        np.savetxt(output / f"{name}.csv", array, delimiter=",",
                   header=",".join(f"v{i}" for i in range(array.shape[1])), comments="")
    (output / "metadata.txt").write_text("Body part indexes:\n[0]\nTotal timesteps: 100\n")
    return q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sonic-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.sonic_root, args.output)


if __name__ == "__main__":
    main()
