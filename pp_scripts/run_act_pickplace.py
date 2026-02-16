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
    # 1) state: 23 维
    state = obs["policy"]  # [num_envs, 23]
    if torch.is_tensor(state):
        state = state[0].detach().cpu().numpy()
    state = state.astype(np.float32)

    # 2) image
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

    # 3) resize 到训练分辨率 256x256
    img_u8 = cv2.resize(
        img_u8,
        (256, 256),  # (W,H)
        interpolation=cv2.INTER_AREA,
    )

    return img_u8, state  # img: (256,256,3), state: (23,)


# ========= 5. 主循环：env + ACT policy =========
def main():
    scale_j = 150.0          # 关节缩放
    max_step = 0.1
    alpha = 0.2
    prev = np.zeros(7, np.float32)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    proj_root = Path(__file__).resolve().parents[1]

    ckpt_dir = (proj_root /"checkpoints/act_pickplace/step_00010000/pretrained_model").resolve()
    agent = ActAgent(ckpt_dir, device="cuda")

    env = make_env()
    obs, info = env.reset()

    done = False
    t = 0

    while simulation_app.is_running() and not done and t < 2000:
        img_u8, state_23 = get_obs_from_obsdict(obs)

        # 1) 策略预测
        chunk = agent.predict_chunk(img_u8, state_23)   # (T,7)
        raw = chunk[0].astype(np.float32)               # (7,)

        raw_j = raw[:6]
        raw_g = raw[6]

        # 2) 关节 / 夹爪连续控制
        # 关节：先缩放，再限幅
        j = raw_j * scale_j
        j = np.clip(j, -max_step, max_step)

        # 夹爪：保留连续输出，轻微放大后 clip 到 [-1,1]
        g = raw_g * 1.5
        g = float(np.clip(g, -1.0, 1.0))

        a = np.concatenate([j, [g]]).astype(np.float32)

        # 3) EMA 平滑
        a = (1.0 - alpha) * prev + alpha * a
        prev = a

        # 4) 最终安全 clip
        a = np.clip(a, -1.0, 1.0)

        action = torch.from_numpy(a)[None, :].to(getattr(env, "device", "cpu"))

        # 环境一步 + 渲染
        obs, reward, terminated, truncated, info = env.step(action)
        simulation_app.update()

        done = bool(terminated.any()) or bool(truncated.any())
        t += 1

        # ====== Debug 打印 ======
        # 每步都看 raw_g，方便观察它是否在接近物体时有明显变化
        print(
            f"[t={t}] raw_j={np.round(raw_j,3)}, "
            f"raw_g={raw_g:.3f}, scaled_j={np.round(j,3)}, "
            f"scaled_g={g:.3f}"
        )

        if t % 20 == 0:
            print(f"[t={t}] applied a = {np.round(a,3)}")
        print("state range:", state_23.min(), state_23.max())

    env.close()
    simulation_app.close()



if __name__ == "__main__":
    main()
