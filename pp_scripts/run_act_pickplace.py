from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import cv2
from isaaclab.app import AppLauncher


# ========= 1. 解析 IsaacLab 启动参数并创建唯一的 App =========
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()

app_launcher = AppLauncher(
    headless=args.headless,
    enable_cameras=args.enable_cameras,
)
simulation_app = app_launcher.app

# ========= 2. 现在才可以导入 IsaacLab / 你的 env 和 Agent =========
from pick_and_place_project.tasks.pick_place_gr1t2_pi import (  # noqa: E402
    PickPlaceGR1T2PiEnvCfg,
    PickPlaceGR1T2PiEnv,
)
from act_infer import ActAgent  # noqa: E402


# ========= 3. Env 工厂 =========
def make_env() -> PickPlaceGR1T2PiEnv:
    cfg = PickPlaceGR1T2PiEnvCfg()
    env = PickPlaceGR1T2PiEnv(cfg)
    return env


# ========= 4. 从 env 取出图像 + state，并转成 ACT 需要的格式 =========
def get_obs_from_obsdict(obs) -> Tuple[np.ndarray, np.ndarray]:
    # =====================
    # 1️⃣ state
    # =====================
    state = obs["policy"]  # [num_envs, 18]
    if torch.is_tensor(state):
        state = state[0].detach().cpu().numpy()
    state = state.astype(np.float32)

    # =====================
    # 2️⃣ image
    # =====================
    img = obs["images"]["rgb"]  # [num_envs, H, W, 3] or [num_envs, 3, H, W]
    if torch.is_tensor(img):
        img = img[0].detach().cpu().numpy()

    # CHW -> HWC
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    # float[0,1] -> uint8
    if img.max() <= 1.5:
        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img_u8 = np.clip(img, 0.0, 255.0).astype(np.uint8)

    # =====================
    # 3️⃣ 🔥 关键：resize 到训练分辨率
    # =====================
    img_u8 = cv2.resize(
        img_u8,
        (256, 256),  # (W,H)
        interpolation=cv2.INTER_AREA,
    )

    return img_u8, state



# ========= 5. 主循环：env + ACT policy =========
def main():
    scale_j = 50         # joint delta gain：先从 200/400/800 试
    max_step = 0.065         # 每步 joint delta 的硬限幅（很关键！先 0.05~0.15 试）
    alpha = 0.25           # EMA 平滑系数：0.1 更稳，0.3 更灵敏
    prev = np.zeros(7, np.float32)  # 上一步平滑后的 action
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    proj_root = Path(__file__).resolve().parents[1]
    ckpt_dir = (proj_root / "checkpoints/act_pickplace/step_0008000").resolve()
    agent = ActAgent(ckpt_dir, device="cuda")
    env = make_env()
    obs, info = env.reset()
    done = False
    t = 0
    while simulation_app.is_running() and not done and t < 2000:
        img_u8, state_18 = get_obs_from_obsdict(obs)

        # 1) 不再额外 clip 到 [-1,1]
        chunk = agent.predict_chunk(img_u8, state_18)   # (T,7)
        raw = chunk[0].astype(np.float32)               # (7,)

        # 1) 分离 joint / gripper
        j = raw[:6] * scale_j          # 只放大关节 delta
        g = raw[6]                     # 夹爪不放大（或单独阈值化）

        # 2) joint delta per-step 限幅（防发疯）
        j = np.clip(j, -max_step, max_step)

        # 3) 夹爪建议二值化/滞回（可选，但很推荐）
        #    你现在 g 常接近 0.98，说明它可能一直“开/合”某一边
        g = 1.0 if g > 0.0 else -1.0

        a = np.concatenate([j, [g]]).astype(np.float32)

        # 4) EMA 平滑（关节 + 夹爪都可以平滑，但夹爪通常不用）
        a = (1 - alpha) * prev + alpha * a
        prev = a

        # 5) 最终安全 clip（如果 env 还 rescale_to_limits，这一步也不会坏事）
        a = np.clip(a, -1.0, 1.0)

        action = torch.from_numpy(a)[None, :].to(getattr(env, "device", "cpu"))
        obs, reward, terminated, truncated, info = env.step(action)

        done = bool(terminated.any()) or bool(truncated.any())

        t += 1
        if t % 20 == 0:
            print(f"[t={t}] raw    :", raw)
            print(f"[t={t}] applied:", a)
        print("state range:", state_18.min(), state_18.max())
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
