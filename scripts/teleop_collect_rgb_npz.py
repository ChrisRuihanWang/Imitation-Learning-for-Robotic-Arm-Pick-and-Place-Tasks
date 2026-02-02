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
    
    from isaaclab.sensors.camera import Camera
   

    device = env.device
    # 从 scene 里拿你在 cfg 里命名的 Camera 实例，例如 "camera"
    camera: Camera = env.scene.sensors["camera"]  # 名字按你 SceneCfg 里用的 key 改

    eye = torch.tensor([[0.7, 0.0, 0.9]], device=device)     # 相机位置
    target = torch.tensor([[0.7, 0.0, 0.0]], device=device)  # 看向桌面中心
    camera.set_world_poses_from_view(eye, target)

    device = env.device
    print("Action space:", env.action_space)
    print("State shape:", obs["policy"].shape)
    print("RGB shape:", obs["images"]["rgb"].shape, obs["images"]["rgb"].dtype)

    # ---- Save ----

    SCRIPT_DIR = Path(__file__).resolve().parent
    save_dir = (SCRIPT_DIR / "../data/teleop_npz").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Teleop state ----
    pressed = set()
    recording = False
    episode_state = []
    episode_rgb = []
    episode_action = []
    ep_id = 0

    # scales (随时调)
    dpos = 0.03   # meters/step
    drot = 0.20   # rad/step
    grip_cmd = -1.0  # default open

    def key_to_str(k):
        try:
            return k.char.lower()
        except Exception:
            return str(k)

    def on_press(k):
        nonlocal recording, grip_cmd, ep_id
        s = key_to_str(k)
        pressed.add(s)

        # record toggle
        if s == "r":
            recording = not recording
            print(f"[REC] {'ON' if recording else 'OFF'}")
            # stop -> save
            if (not recording) and len(episode_action) > 5:
                out = {
                    "state": np.stack(episode_state, axis=0).astype(np.float32),    # (T, 18)
                    "rgb": np.stack(episode_rgb, axis=0).astype(np.uint8),         # (T, 480, 640, 3)
                    "action": np.stack(episode_action, axis=0).astype(np.float32), # (T, 7)
                }
                np.savez_compressed(save_dir / f"ep_{ep_id:05d}.npz", **out)
                print(f"[SAVE] ep_{ep_id:05d}.npz  T={len(episode_action)}")
                ep_id += 1
                episode_state.clear()
                episode_rgb.clear()
                episode_action.clear()

        # reset
        if s == "p":
            print("[RESET]")
            nonlocal obs, info
            obs, info = env.reset()

        # gripper
        if s == "v":
            grip_cmd = +1.0   # close
            print("gripper -> CLOSE")

        if s == "b":
            grip_cmd = -1.0   # open
            print("gripper -> OPEN")

        # quit
        if s == "Key.esc":
            print("[QUIT]")
            return False

    def on_release(k):
        
        s = key_to_str(k)
        print("KEY:", s) 
        pressed.discard(s)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while app.is_running():
            # 7D action tensor
            a = torch.zeros((1, 7), device=device)

            # translation: W/S (x), A/D (y), Q/E (z)
            if "w" in pressed: a[0, 0] += dpos
            if "s" in pressed: a[0, 0] -= dpos
            if "a" in pressed: a[0, 1] += dpos
            if "d" in pressed: a[0, 1] -= dpos
            if "q" in pressed: a[0, 2] += dpos
            if "e" in pressed: a[0, 2] -= dpos

            # rotation: I/K (rx), J/L (ry), U/O (rz)
            if "i" in pressed: a[0, 3] += drot
            if "k" in pressed: a[0, 3] -= drot
            if "j" in pressed: a[0, 4] += drot
            if "l" in pressed: a[0, 4] -= drot
            if "u" in pressed: a[0, 5] += drot
            if "o" in pressed: a[0, 5] -= drot

            # gripper scalar
            a[0, 6] = grip_cmd

            # step
            obs2, rew, done, trunc, info = env.step(a)

            # record
            if recording:
                # state: (1,18) -> (18,)
                state_np = obs["policy"].detach().cpu().numpy()[0].copy()

                # rgb: (1,H,W,3) float32 -> uint8 [0,255]
                rgb = obs["images"]["rgb"].detach().cpu().numpy()[0]
                rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

                episode_state.append(state_np)
                episode_rgb.append(rgb_u8)
                episode_action.append(a.detach().cpu().numpy()[0].copy())

            obs = obs2

            if done.any() or trunc.any():
                print("[EPISODE END] (timeout) Press P to reset. Recording will keep going.")
            if done.any() or trunc.any():
                print("done:", done.cpu().numpy(), "trunc:", trunc.cpu().numpy())
    finally:
        listener.stop()
        env.close()
        app.close()


if __name__ == "__main__":
    main()
