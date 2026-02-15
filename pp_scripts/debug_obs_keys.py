from __future__ import annotations

import numpy as np
import torch
from typing import Any

# 关键：先启动 Isaac Sim / Isaac Lab 的 App
from isaaclab.app import AppLauncher

# 在 import 任何 isaaclab.envs / 你的 env 之前启动 App
app_launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = app_launcher.app

# 现在再导入你的 env
from pick_and_place_project.tasks.pick_place_gr1t2_pi import (
    PickPlaceGR1T2PiEnvCfg,
    PickPlaceGR1T2PiEnv,
)


def make_env():
    cfg = PickPlaceGR1T2PiEnvCfg()
    env = PickPlaceGR1T2PiEnv(cfg)
    return env


def print_tensor_info(x: Any, prefix: str = ""):
    if isinstance(x, torch.Tensor):
        print(f"{prefix}Tensor shape={tuple(x.shape)} dtype={x.dtype} device={x.device}")
    elif isinstance(x, np.ndarray):
        print(f"{prefix}ndarray shape={x.shape} dtype={x.dtype}")
    else:
        print(f"{prefix}{type(x)}")


def recursive_print(d: Any, prefix: str = ""):
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{prefix}{k}:")
            recursive_print(v, prefix + "  ")
    else:
        print_tensor_info(d, prefix)


def guess_image_and_state(obs: dict):
    print("\n===== AUTO GUESS =====")
    for k, v in obs.items():
        if isinstance(v, (torch.Tensor, np.ndarray)):
            shape = tuple(v.shape)
            if len(shape) == 3 and shape[-1] == 3:
                print(f"[IMG?] {k} looks like HWC image: {shape}")
            elif len(shape) == 3 and shape[0] == 3:
                print(f"[IMG?] {k} looks like CHW image: {shape}")
            elif len(shape) == 1 and shape[0] >= 6:
                print(f"[STATE?] {k} looks like state vector: {shape}")


def main():
    env = make_env()

    print("Resetting env...")
    out = env.reset()

    if isinstance(out, tuple):
        obs = out[0]
    else:
        obs = out

    print("\n===== OBS KEYS STRUCTURE =====")
    recursive_print(obs)

    guess_image_and_state(obs)

    print("\nDone.")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
