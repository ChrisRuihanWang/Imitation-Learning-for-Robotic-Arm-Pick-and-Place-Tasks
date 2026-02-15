from isaaclab.app import AppLauncher
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


def main():
    app = AppLauncher(headless=False, enable_cameras=True).app

    from pick_and_place_project.tasks.pick_place_gr1t2_pi import make_env
    env = make_env()
    obs, info = env.reset()

    device = env.device

    # ---- Grab camera handle (Camera sensor) ----
    # 你 teleop 脚本里用过 env.scene.sensors["camera"]，这里沿用
    cam = env.scene.sensors.get("camera", None)
    print("[DEBUG] scene.sensors keys:", list(env.scene.sensors.keys()))
    if cam is None:
        raise KeyError("Cannot find camera in env.scene.sensors['camera']. Check SceneCfg naming.")

    # ---- Force set camera view (关键：不然默认很可能拍不到) ----
    # eye: 相机位置（世界坐标），target: 看向位置（世界坐标）
    eye = torch.tensor([[0.65, 0.00, 0.90]], device=device, dtype=torch.float32)
    target = torch.tensor([[0.65, 0.00, 0.00]], device=device, dtype=torch.float32)
    cam.set_world_poses_from_view(eye, target)

    # ---- Warmup: step a few frames so sensors update ----
    zero_action = torch.zeros((1, 7), device=device, dtype=torch.float32)

    # 先让 Kit 跑几帧
    for _ in range(10):
        app.update()

    # 再 step 多帧，确保 camera buffer 更新到 obs
    for _ in range(30):
        obs, _, _, _, _ = env.step(zero_action)
        app.update()

    # ---- Grab one RGB frame from observations ----
    rgb = obs["images"]["rgb"]  # (1,H,W,3) float32 [0,1]  (理论上)
    rgb_np = rgb.detach().cpu().numpy()[0]

    print("[DEBUG] rgb shape:", rgb_np.shape, "dtype:", rgb_np.dtype,
          "min:", float(rgb_np.min()), "max:", float(rgb_np.max()),
          "mean:", float(rgb_np.mean()))

    # 转 uint8
    rgb_u8 = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)

    # ---- Save ----
    SCRIPT_DIR = Path(__file__).resolve().parent
    data_dir = (SCRIPT_DIR / "../data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = data_dir / f"camera_check_{ts}.png"

    try:
        import imageio.v2 as imageio
        imageio.imwrite(out_path.as_posix(), rgb_u8)
    except Exception:
        from PIL import Image
        Image.fromarray(rgb_u8).save(out_path.as_posix())

    print(f"[OK] Saved camera snapshot: {out_path}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()

