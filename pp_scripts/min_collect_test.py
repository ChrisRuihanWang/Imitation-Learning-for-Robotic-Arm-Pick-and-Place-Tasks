from isaaclab.app import AppLauncher
import numpy as np
import torch
from pathlib import Path
import imageio.v2 as imageio


def main():
    app = AppLauncher(headless=False, enable_cameras=True).app

    from pick_and_place_project.tasks.pick_place_gr1t2_pi import make_env
    env = make_env()
    obs, info = env.reset()
    device = env.device

    from isaaclab.sensors.camera import Camera
    camera: Camera = env.scene.sensors["camera"]

    eye = torch.tensor([[0.7, 0.0, 0.9]], device=device)
    target = torch.tensor([[0.7, 0.0, 0.0]], device=device)
    camera.set_world_poses_from_view(eye, target)
    print("CameraData fields:", camera.data.__dict__.keys())

    pos = camera.data.pos_w
    quat = camera.data.quat_w_world
    print("Camera pos:", pos)
    print("Camera quat (xyzw):", quat)
    

    # 保存到 scripts/../data/teleop_npz
    SCRIPT_DIR = Path(__file__).resolve().parent
    save_dir = (SCRIPT_DIR / "../data/teleop_npz").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "min_test.npz"
    save_dir = (SCRIPT_DIR / "../data/teleop_npz").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "min_test.npz"
    png_dir = save_dir / "debug_png"
    png_dir.mkdir(parents=True, exist_ok=True)

    T = 120  # 约 2 秒（取决于你的仿真步频）
    states, rgbs, actions = [], [], []

    for t in range(T):
        a = torch.zeros((1, 7), device=device)

        # 给一点小运动 + 夹爪开合，确保 action 不是全 0
        a[0, 0] = 0.02 if (t // 30) % 2 == 0 else -0.02   # dx
        a[0, 6] = +1.0 if (t // 60) % 2 == 0 else -1.0    # gripper toggle

        obs2, rew, done, trunc, info = env.step(a)

        # state: (1,18)->(18,)
        state_np = obs["policy"].detach().cpu().numpy()[0].copy()

        # rgb: (1,H,W,3) float32 in [0,1] -> uint8
        rgb = obs["images"]["rgb"].detach().cpu().numpy()[0]
        rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

        states.append(state_np)
        rgbs.append(rgb_u8)
        actions.append(a.detach().cpu().numpy()[0].copy())
        # ---- save debug png (every 10 frames) ----
        if t % 10 == 0:
            png_path = png_dir / f"frame_{t:04d}.png"
            imageio.imwrite(png_path, rgb_u8)

        obs = obs2

        if done.any() or trunc.any():
            obs, info = env.reset()

    np.savez_compressed(
        out_path,
        state=np.stack(states, axis=0).astype(np.float32),   # (T,18)
        rgb=np.stack(rgbs, axis=0).astype(np.uint8),         # (T,H,W,3)
        action=np.stack(actions, axis=0).astype(np.float32), # (T,7)
    )
    print("[OK] saved:", out_path)
    
    env.close()
    app.close()

    

if __name__ == "__main__":
    main()
