from isaaclab.app import AppLauncher
import numpy as np
import torch
from pathlib import Path
from pynput import keyboard


def main():
    # ---- App ----
    app = AppLauncher(headless=False, enable_cameras=True).app

    # ---- Env ----
    try:
        from pick_and_place_project.tasks.pick_place_gr1t2_pi import make_env
        env = make_env()
    except Exception as e:
        print("[WARN] import make_env failed:", repr(e))
        from pick_and_place_project.tasks.pick_place_gr1t2_pi import PickPlaceGR1T2PiEnv, PickPlaceGR1T2PiEnvCfg
        env = PickPlaceGR1T2PiEnv(PickPlaceGR1T2PiEnvCfg())

    import inspect
    print("CFG CLASS:", type(env.cfg))
    print("CFG MODULE:", type(env.cfg).__module__)
    print("CFG FILE:", inspect.getfile(type(env.cfg)))
    print("cfg episode_length_s:", getattr(env.cfg, "episode_length_s", None))
    print("env max_episode_length:", getattr(env, "max_episode_length", None))
    print("env dt:", getattr(env, "dt", None))

    # ---- Reset + warm-up ----
    obs, info = env.reset()
    for _ in range(10):
        app.update()
    a0 = torch.zeros((env.num_envs, 7), device=env.device, dtype=torch.float32)
    obs, *_ = env.step(a0)

    device = env.device

    # ---- Print shapes ----
    print("Action space:", env.action_space)
    print("State shape:", obs["policy"].shape, obs["policy"].dtype)
    print("Image keys:", list(obs["images"].keys()))
    print("RGB shape:", obs["images"]["rgb"].shape, obs["images"]["rgb"].dtype)
    if "wrist_rgb" in obs["images"]:
        print("Wrist RGB shape:", obs["images"]["wrist_rgb"].shape, obs["images"]["wrist_rgb"].dtype)
    else:
        print("[WARN] no wrist_rgb in obs['images']!")

    # ---- Save dirs ----
    import re
    from datetime import datetime

    SCRIPT_DIR = Path(__file__).resolve().parent
    save_dir = (SCRIPT_DIR / "../data/teleop2").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = (save_dir / "debug_images").resolve()
    debug_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"[DEBUG] image debug dir: {debug_dir}")

    # ---- Teleop state ----
    pressed = set()
    recording = False
    pending_reset = False

    episode_state = []
    episode_rgb_raw = []
    episode_wrist_raw = []
    episode_action = []

    # scales
    dpos = 0.03
    drot = 0.20

    # continuous gripper command in [-1, +1]
    grip_cmd = -1.0
    grip_rate = 2.5  # per second

    def key_to_str(k):
        try:
            return k.char.lower()
        except Exception:
            return str(k)

    # --------- raw -> view (ONLY for debug PNG) ----------
    def _raw_to_view_u8(x_raw: np.ndarray) -> np.ndarray:
        """Robustly make raw float image viewable as uint8 for debugging."""
        x = x_raw.astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mn, mx = float(x.min()), float(x.max())

        # If it already looks like [0,1], keep it
        if mx <= 1.5 and mn >= -0.1:
            y = np.clip(x, 0.0, 1.0)
            return (y * 255.0).astype(np.uint8)

        # Otherwise, percentile stretch in raw space
        lo = np.percentile(x, 1.0)
        hi = np.percentile(x, 99.0)
        if hi <= lo + 1e-8:
            return np.zeros_like(x, dtype=np.uint8)
        y = (x - lo) / (hi - lo)
        y = np.clip(y, 0.0, 1.0)
        return (y * 255.0).astype(np.uint8)

    def save_debug_images(obs_any):
        """Press N -> save current overhead+wrist PNG for human check."""
        if "images" not in obs_any:
            print("[DEBUG] no images in obs.")
            return
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # overhead
        if "rgb" in obs_any["images"]:
            oh = obs_any["images"]["rgb"][0].detach().cpu().numpy()
            oh_u8 = _raw_to_view_u8(oh)
            oh_path = debug_dir / f"overhead_{ts}.png"
            try:
                import imageio.v2 as imageio
                imageio.imwrite(str(oh_path), oh_u8)
            except Exception:
                from PIL import Image
                Image.fromarray(oh_u8).save(str(oh_path))
            print(f"[DEBUG] saved {oh_path.name}  raw[min,max,std]=({oh.min():.4f},{oh.max():.4f},{oh.std():.4f})")

        # wrist
        if "wrist_rgb" in obs_any["images"]:
            w = obs_any["images"]["wrist_rgb"][0].detach().cpu().numpy()
            w_u8 = _raw_to_view_u8(w)
            w_path = debug_dir / f"wrist_{ts}.png"
            try:
                import imageio.v2 as imageio
                imageio.imwrite(str(w_path), w_u8)
            except Exception:
                from PIL import Image
                Image.fromarray(w_u8).save(str(w_path))
            print(f"[DEBUG] saved {w_path.name}  raw[min,max,std]=({w.min():.4f},{w.max():.4f},{w.std():.4f})")
        else:
            print("[DEBUG] no wrist_rgb in obs['images'].")

    # ---------- episode save (RAW ONLY) ----------
    def save_episode(current_ep_id: int) -> int:
        nonlocal episode_state, episode_rgb_raw, episode_wrist_raw, episode_action

        lens = {
            "state": len(episode_state),
            "rgb_raw": len(episode_rgb_raw),
            "wrist_raw": len(episode_wrist_raw),
            "action": len(episode_action),
        }
        T = min(lens.values())
        if T <= 5:
            print("[SAVE] skipped (too short / not aligned)", lens)
            episode_state.clear()
            episode_rgb_raw.clear()
            episode_wrist_raw.clear()
            episode_action.clear()
            return current_ep_id

        if len(set(lens.values())) != 1:
            print("[WARN] length mismatch:", lens, "-> trunc to T=", T)

        path = safe_episode_path(save_dir, current_ep_id)

        out = {
            "state": np.stack(episode_state[:T], axis=0).astype(np.float32),
            "rgb_raw": np.stack(episode_rgb_raw[:T], axis=0).astype(np.float32),
            "wrist_rgb_raw": np.stack(episode_wrist_raw[:T], axis=0).astype(np.float32),
            "action": np.stack(episode_action[:T], axis=0).astype(np.float32),
            "meta": np.array([{
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "T": int(T),
                "dt": float(getattr(env, "dt", 0.0) or 0.0),
                "cfg_class": type(env.cfg).__name__,
                "cfg_module": type(env.cfg).__module__,
                "note": "teleop2_raw_only",
            }], dtype=object),
        }

        np.savez_compressed(path, **out)
        print(f"[SAVE] {path.name}  T={T}  keys={list(out.keys())}")

        # quick sanity stats
        rr = out["rgb_raw"]
        print(f"       rgb_raw: shape={rr.shape} min/max/std=({rr.min():.4f},{rr.max():.4f},{rr.std():.4f})")
        wr = out["wrist_rgb_raw"]
        print(f"       wrist_raw: shape={wr.shape} min/max/std=({wr.min():.4f},{wr.max():.4f},{wr.std():.4f})")

        episode_state.clear()
        episode_rgb_raw.clear()
        episode_wrist_raw.clear()
        episode_action.clear()

        try:
            saved_id = int(path.stem.split("_")[1])
            return saved_id + 1
        except Exception:
            return current_ep_id + 1

    # ---------- keyboard callbacks ----------
    def on_press(k):
        nonlocal recording, ep_id, pending_reset, obs
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

        if s == "n":
            save_debug_images(obs)

        if s == "v":
            print("gripper: HOLD V -> CLOSE (continuous)")
        if s == "b":
            print("gripper: HOLD B -> OPEN  (continuous)")

        if s == "Key.esc":
            print("[QUIT]")
            return False

    def on_release(k):
        s = key_to_str(k)
        pressed.discard(s)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("listener started")
    print("Controls: R=record toggle | N=save debug PNGs | M=reset | V/B=gripper close/open | ESC=quit")

    # ---------- main loop ----------
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

            # continuous gripper update
            dt = float(getattr(env, "dt", 1 / 60) or (1 / 60))
            if ("v" in pressed) and ("b" not in pressed):
                grip_cmd = min(1.0, grip_cmd + grip_rate * dt)
            elif ("b" in pressed) and ("v" not in pressed):
                grip_cmd = max(-1.0, grip_cmd - grip_rate * dt)

            # action (7D: dx dy dz dRx dRy dRz grip)
            a = torch.zeros((1, 7), device=device, dtype=torch.float32)

            # translation
            if "w" in pressed: a[0, 0] += dpos
            if "s" in pressed: a[0, 0] -= dpos
            if "a" in pressed: a[0, 1] += dpos
            if "d" in pressed: a[0, 1] -= dpos
            if "q" in pressed: a[0, 2] += dpos
            if "e" in pressed: a[0, 2] -= dpos

            # rotation
            if "i" in pressed: a[0, 3] += drot
            if "k" in pressed: a[0, 3] -= drot
            if "j" in pressed: a[0, 4] += drot
            if "l" in pressed: a[0, 4] -= drot
            if "u" in pressed: a[0, 5] += drot
            if "o" in pressed: a[0, 5] -= drot

            a[0, 6] = float(grip_cmd)

            # step
            obs2, rew, done, trunc, info = env.step(a)

            # record (store obs BEFORE step)
            if recording:
                # state
                episode_state.append(obs["policy"].detach().cpu().numpy()[0].astype(np.float32, copy=True))

                # overhead raw float
                oh = obs["images"]["rgb"].detach().cpu().numpy()[0].astype(np.float32, copy=True)
                episode_rgb_raw.append(oh)

                # wrist raw float (must exist in your config)
                if "wrist_rgb" in obs["images"]:
                    w = obs["images"]["wrist_rgb"].detach().cpu().numpy()[0].astype(np.float32, copy=True)
                else:
                    # fallback: keep alignment
                    if len(episode_wrist_raw) > 0:
                        w = episode_wrist_raw[-1]
                    else:
                        # last-resort blank raw
                        w = np.zeros_like(oh, dtype=np.float32)
                episode_wrist_raw.append(w)

                # action
                episode_action.append(a.detach().cpu().numpy()[0].astype(np.float32, copy=True))

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