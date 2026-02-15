# scripts/eval_act_rollout.py
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch

from pick_and_place_project.tasks.pick_place_gr1t2_pi import (
    PickPlaceGR1T2PiEnvCfg,
    PickPlaceGR1T2PiEnv,
)

from pp_scripts.act_infer import ActAgent


@dataclass
class EvalArgs:
    episodes: int = 20
    max_steps: int = 400
    ckpt_rel: str = "checkpoints/act_pickplace/step_0008000"
    render: bool = False  # 预留：如果你有渲染开关，可用


def make_env():
    cfg = PickPlaceGR1T2PiEnvCfg()
    env = PickPlaceGR1T2PiEnv(cfg)
    return env


def get_obs_from_env(env):
    obs = env.get_observations() if hasattr(env, "get_observations") else env._get_observations()

    state = obs["policy"]  # (1,18) torch
    if isinstance(state, torch.Tensor):
        state = state.squeeze(0).detach().cpu().numpy()
    state = state.astype(np.float32)

    img = obs["images"]["rgb"]  # (1,H,W,3) torch float32
    if isinstance(img, torch.Tensor):
        img = img.squeeze(0).detach().cpu().numpy()

    # float [0,1] -> uint8
    if img.max() <= 1.5:
        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img_u8 = np.clip(img, 0.0, 255.0).astype(np.uint8)

    return img_u8, state


def parse_success(info) -> bool | None:
    """
    Try best-effort detect success from info dict.
    Return True/False if detectable, else None.
    """
    if not isinstance(info, dict):
        return None

    # common patterns
    for k in ["is_success", "success", "task_success", "episode_success", "successes"]:
        if k in info:
            v = info[k]
            # v might be bool, numpy, torch, list
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            if torch.is_tensor(v):
                return bool(v.item())
            if isinstance(v, (int, float, np.number)):
                return bool(v)
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return bool(v[0])
    return None


def step_env(env, action):
    """
    Normalize different env.step return signatures.
    Expect either:
      obs, reward, terminated, truncated, info
    or:
      obs, reward, done, info
    """
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        return out
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        terminated = bool(done)
        truncated = False
        return obs, reward, terminated, truncated, info
    raise RuntimeError(f"Unexpected env.step return: {type(out)} len={len(out) if isinstance(out, tuple) else 'NA'}")


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    args = EvalArgs()

    # project root = Pick_and_Place/
    proj_root = Path(__file__).resolve().parents[1]
    ckpt_dir = (proj_root / args.ckpt_rel).resolve()
    assert ckpt_dir.exists(), f"Checkpoint not found: {ckpt_dir}"

    agent = ActAgent(ckpt_dir=ckpt_dir, device="cuda")

    env = make_env()

    success_count = 0
    total_steps = 0
    undetected_success = 0

    for ep in range(args.episodes):
        out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out

        ep_success = False
        ep_success_detected = False
        last_info = None

        for t in range(args.max_steps):
            img_u8, state_18 = get_obs_from_env(env)

            chunk = agent.predict_chunk(img_u8, state_18)   # (T,7)
            action = np.clip(chunk[0], -1.0, 1.0).astype(np.float32)
            action = action[None, :]  # (1,7) 最稳

            obs, reward, terminated, truncated, info = step_env(env, action)
            last_info = info

            s = parse_success(info)
            if s is not None:
                ep_success_detected = True
                ep_success = bool(s)

            if terminated or truncated:
                total_steps += (t + 1)
                break
        else:
            # timeout
            total_steps += args.max_steps

        if not ep_success_detected:
            undetected_success += 1

        if ep_success:
            success_count += 1

        # 打印每个 episode 结果（你可以之后关掉）
        print(f"[EP {ep:03d}] success={ep_success} detected={ep_success_detected} last_info_keys={list(last_info.keys()) if isinstance(last_info, dict) else type(last_info)}")

    env.close()

    sr = success_count / max(1, args.episodes)
    avg_steps = total_steps / max(1, args.episodes)
    print("\n===== EVAL SUMMARY =====")
    print(f"episodes: {args.episodes}")
    print(f"success: {success_count}  success_rate: {sr:.3f}")
    print(f"avg_steps: {avg_steps:.1f}")
    print(f"episodes_without_success_signal_in_info: {undetected_success}")


if __name__ == "__main__":
    main()