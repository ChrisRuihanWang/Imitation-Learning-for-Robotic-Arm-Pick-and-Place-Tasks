from __future__ import annotations

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# -------- 1. 启动 Isaac App --------
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(
    headless=args.headless,
    enable_cameras=args.enable_cameras,
)
simulation_app = app_launcher.app

# -------- 2. 导入你的 env --------
from pick_and_place_project.tasks.pick_place_gr1t2_pi import (  # noqa: E402
    PickPlaceGR1T2PiEnvCfg,
    PickPlaceGR1T2PiEnv,
)


def make_env():
    cfg = PickPlaceGR1T2PiEnvCfg()
    env = PickPlaceGR1T2PiEnv(cfg)
    return env


def get_joint_pos_from_obs(obs):
    """
    根据你之前的 obs 结构：
    obs["policy"] shape = (num_envs, 18)
    前 6 个通常是 joint pos 或 joint pos rel
    """
    state = obs["policy"]
    if torch.is_tensor(state):
        state = state[0].detach().cpu().numpy()
    return state[:6].copy()


def main():
    env = make_env()
    obs, info = env.reset()

    print("\n===== ACTION MANAGER INFO =====")
    am = getattr(env, "action_manager", None) or getattr(env, "_action_manager", None)
    print("ActionManager type:", type(am))

    if am is not None:
        for attr in ["cfg", "terms", "_terms", "term_cfgs", "_term_cfgs"]:
            if hasattr(am, attr):
                print(f"\n--- {attr} ---")
                print(getattr(am, attr))

    print("\n===== DELTA TEST START =====")

    q0 = get_joint_pos_from_obs(obs)
    print("Initial joint pos:", q0)

    # 测试动作：只给第一个关节一个 delta
    delta = np.zeros((1, 7), dtype=np.float32)
    delta[0, 0] = 0.5  # 试试大一点 delta

    for i in range(50):
        obs, reward, terminated, truncated, info = env.step(
            torch.from_numpy(delta).to(env.device)
        )

    q1 = get_joint_pos_from_obs(obs)
    print("After 50 steps joint pos:", q1)
    print("Delta joint change:", q1 - q0)

    print("\n===== DONE =====")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()