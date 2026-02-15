from isaaclab.app import AppLauncher
import numpy as np
import torch
from pathlib import Path
from pynput import keyboard


def main():
    # ---- App ----
    app = AppLauncher(headless=False, enable_cameras=True).app

    # ---- Env ----
    from pick_and_place_project.tasks.pick_place_gr1t2_pi import make_env
    env = make_env()

    print("CFG CLASS:", type(env.cfg))
    print("CFG MODULE:", type(env.cfg).__module__)
    import inspect
    print("CFG FILE:", inspect.getfile(type(env.cfg)))

    print("cfg episode_length_s:", getattr(env.cfg, "episode_length_s", None))
    print("env max_episode_length:", getattr(env, "max_episode_length", None))
    print("env dt:", getattr(env, "dt", None))

    obs, info = env.reset()
    for _ in range(10):
        app.update()
    a0 = torch.zeros((env.num_envs, 7), device=env.device)
    obs, *_ = env.step(a0)

    from isaaclab.sensors.camera import Camera
    device = env.device

    # ---- Cameras from scene ----
    overhead_cam: Camera = env.scene.sensors["camera"]
    wrist_cam: Camera = env.scene.sensors.get("wrist_camera", None)
    if wrist_cam is None:
        print("[WARN] wrist_camera not found in env.scene.sensors. "
              "Did you add it to SceneCfg as self.scene.wrist_camera?")
    else:
        print("[OK] wrist_camera found.")

    # Set overhead camera view once
    eye = torch.tensor([[0.7, 0.0, 0.9]], device=device)
    target = torch.tensor([[0.7, 0.0, 0.0]], device=device)
    overhead_cam.set_world_poses_from_view(eye, target)

    # ---- Print shapes ----
    print("Action space:", env.action_space)
    print("State shape:", obs["policy"].shape)
    print("RGB shape:", obs["images"]["rgb"].shape, obs["images"]["rgb"].dtype)
    if wrist_cam is not None and "wrist_rgb" in obs["images"]:
        print("Wrist RGB shape:", obs["images"]["wrist_rgb"].shape, obs["images"]["wrist_rgb"].dtype)
    else:
        print("[INFO] obs['images'] has keys:", list(obs["images"].keys()))

    # ---- Save ----
    import re
    from datetime import datetime

    SCRIPT_DIR = Path(__file__).resolve().parent
    save_dir = (SCRIPT_DIR / "../data/teleop_npz").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 单独存 wrist debug 图像的目录
    check_dir = (SCRIPT_DIR / "../data/check_wrist_simple").resolve()
    check_dir.mkdir(parents=True, exist_ok=True)

    def next_episode_id(folder: Path, prefix: str = "ep_", suffix: str = ".npz") -> int:
        pat = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(suffix)}$")
        max_id = -1
        for p in folder.iterdir():
            if not p.is_file():
                continue
            m = pat.match(p.name)
            if m:
                max_id = max(max_id, int(m.group(1)))
        return max_id + 1

    def safe_episode_path(folder: Path, ep_id: int) -> Path:
        while True:
            p = folder / f"ep_{ep_id:05d}.npz"
            if not p.exists():
                return p
            ep_id += 1

    ep_id = next_episode_id(save_dir)
    print(f"[SAVE] directory: {save_dir}")
    print(f"[SAVE] starting ep_id: {ep_id:05d}")
    print(f"[CHECK] wrist debug dir: {check_dir}")

    # ---- Teleop state ----
    pressed = set()
    recording = False
    episode_state = []
    episode_rgb = []
    episode_wrist_rgb = []
    episode_action = []

    # scales
    dpos = 0.03
    drot = 0.20

    # Continuous gripper command
    grip_cmd = -1.0
    grip_rate = 2.5

    def key_to_str(k):
        try:
            return k.char.lower()
        except Exception:
            return str(k)

    pending_reset = False

    def save_episode(current_ep_id: int) -> int:
        nonlocal episode_state, episode_rgb, episode_wrist_rgb, episode_action

        if len(episode_action) <= 5:
            print("[SAVE] skipped (too short)")
            episode_state.clear()
            episode_rgb.clear()
            episode_wrist_rgb.clear()
            episode_action.clear()
            return current_ep_id

        out = {
            "state": np.stack(episode_state, axis=0).astype(np.float32),
            "rgb": np.stack(episode_rgb, axis=0).astype(np.uint8),
            "action": np.stack(episode_action, axis=0).astype(np.float32),
            "meta": np.array([{
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "T": int(len(episode_action)),
                "dt": float(getattr(env, "dt", 0.0) or 0.0),
                "cfg_class": type(env.cfg).__name__,
                "cfg_module": type(env.cfg).__module__,
                "has_wrist": bool(len(episode_wrist_rgb) > 0),
            }], dtype=object),
        }

        if len(episode_wrist_rgb) == len(episode_rgb) and len(episode_wrist_rgb) > 0:
            out["wrist_rgb"] = np.stack(episode_wrist_rgb, axis=0).astype(np.uint8)

        path = safe_episode_path(save_dir, current_ep_id)
        np.savez_compressed(path, **out)
        print(f"[SAVE] {path.name}  T={len(episode_action)}")

        episode_state.clear()
        episode_rgb.clear()
        episode_wrist_rgb.clear()
        episode_action.clear()

        try:
            saved_id = int(path.stem.split("_")[1])
            return saved_id + 1
        except Exception:
            return current_ep_id + 1

    # wrist debug 保存函数：每次按 n 存一张
    def save_wrist_debug(obs_any):
        if "images" not in obs_any or "wrist_rgb" not in obs_any["images"]:
            print("[DEBUG] no wrist_rgb in obs.")
            return

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        raw_path = check_dir / f"wrist_{ts}_raw.png"
        view_path = check_dir / f"wrist_{ts}_view.png"

        w = obs_any["images"]["wrist_rgb"][0].detach().cpu().numpy()
        print("wrist min/max:", w.min(), w.max())

        # 映射到 [0,1] 再存 raw（假设值大致在 [-0.5,0.5]）
        w_norm = (w - (-0.5)) / (0.5 - (-0.5))
        w_norm = np.clip(w_norm, 0.0, 1.0)
        w_raw = (w_norm * 255).astype(np.uint8)

        lo = np.percentile(w_norm, 1.0)
        hi = np.percentile(w_norm, 99.0)
        if hi <= lo + 1e-6:
            lo, hi = float(w_norm.min()), float(w_norm.max())
        if hi <= lo + 1e-6:
            w_view = np.zeros_like(w_raw)
        else:
            w_view = np.clip((w_norm - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)

        from imageio import imwrite
        imwrite(raw_path, w_raw)
        imwrite(view_path, w_view)

        print(f"[DEBUG] saved {raw_path.name} / {view_path.name}  "
              f"raw_range=[{w_norm.min():.3f},{w_norm.max():.3f}] lo/hi=[{lo:.3f},{hi:.3f}]")

    def on_press(k):
        nonlocal recording, ep_id, pending_reset
        s = key_to_str(k)
        pressed.add(s)

        if s == "r":
            recording = not recording
            print(f"[REC] {'ON' if recording else 'OFF'}")
            if not recording:
                ep_id = save_episode(ep_id)

        if s == "m":
            pending_reset = True
            print("[RESET REQUESTED]")

        if s == "v":
            print("gripper: HOLD V -> CLOSE (continuous)")
        if s == "b":
            print("gripper: HOLD B -> OPEN  (continuous)")

        if s == "n":
            save_wrist_debug(obs)

        if s == "Key.esc":
            print("[QUIT]")
            return False

    def on_release(k):
        s = key_to_str(k)
        pressed.discard(s)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("listener started")

    try:
        while app.is_running():
            if pending_reset:
                pending_reset = False
                print("[RESET] main thread")
                obs, info = env.reset()
                pressed.clear()
                for _ in range(2):
                    app.update()
                continue

            dt = float(getattr(env, "dt", 1 / 60) or (1 / 60))
            if ("v" in pressed) and ("b" not in pressed):
                grip_cmd = min(1.0, grip_cmd + grip_rate * dt)
            elif ("b" in pressed) and ("v" not in pressed):
                grip_cmd = max(-1.0, grip_cmd - grip_rate * dt)

            # 7D action tensor
            a = torch.zeros((1, 7), device=device)

            if "w" in pressed: a[0, 0] += dpos
            if "s" in pressed: a[0, 0] -= dpos
            if "a" in pressed: a[0, 1] += dpos
            if "d" in pressed: a[0, 1] -= dpos
            if "q" in pressed: a[0, 2] += dpos
            if "e" in pressed: a[0, 2] -= dpos

            if "i" in pressed: a[0, 3] += drot
            if "k" in pressed: a[0, 3] -= drot
            if "j" in pressed: a[0, 4] += drot
            if "l" in pressed: a[0, 4] -= drot
            if "u" in pressed: a[0, 5] += drot
            if "o" in pressed: a[0, 5] -= drot

            a[0, 6] = float(grip_cmd)

            obs2, rew, done, trunc, info = env.step(a)

            if recording:
                state_np = obs["policy"].detach().cpu().numpy()[0].copy()

                rgb = obs["images"]["rgb"].detach().cpu().numpy()[0]
                rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

                episode_state.append(state_np)
                episode_rgb.append(rgb_u8)
                episode_action.append(a.detach().cpu().numpy()[0].copy())

                if "wrist_rgb" in obs["images"]:
                    w = obs["images"]["wrist_rgb"].detach().cpu().numpy()[0]
                    # 注意：这里如果你以后也想用归一化后的值训练，可以照 debug 里的同样归一化方式
                    w_u8 = np.clip((w + 0.5) / 1.0 * 255.0, 0, 255).astype(np.uint8)
                    episode_wrist_rgb.append(w_u8)

            obs = obs2

    finally:
        listener.stop()
        env.close()
        app.close()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
