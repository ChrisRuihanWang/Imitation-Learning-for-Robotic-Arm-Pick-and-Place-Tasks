# scripts/sanity_check_env.py
import argparse
import torch
from isaaclab.app import AppLauncher

def main():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()

    # ✅ 先启动 App（相机开关由 CLI 控制）
    app_launcher = AppLauncher(headless=args.headless, enable_cameras=args.enable_cameras)
    simulation_app = app_launcher.app

    # ✅ 再 import isaaclab/mdp/你的env
    import isaaclab.envs.mdp as mdp
    print("MDP camera-related symbols:")
    print([x for x in dir(mdp) if "camera" in x.lower() or "rgb" in x.lower() or "image" in x.lower()][:80])

    from pick_and_place_project.tasks.pick_place_gr1t2_pi import make_env
    env = make_env()
    obs, info = env.reset()

    print("Action space:", env.action_space)
    print("Obs keys:", obs.keys())
    if "policy" in obs:
        print("Policy obs shape:", obs["policy"].shape)

    device = env.device

    for step in range(200):
        action = torch.zeros((1, 7), device=device)

        # 每 50 步切换一次开合
        action[0, 6] = -1.0 if (step // 50) % 2 == 0 else +1.0

        obs, rew, done, trunc, info = env.step(action)

        if done.any() or trunc.any():
            obs, info = env.reset()
        print("Obs keys:", obs.keys())
        if "images" in obs:
            print("Images keys:", obs["images"].keys())
            if "rgb" in obs["images"]:
               print("RGB shape:", obs["images"]["rgb"].shape, obs["images"]["rgb"].dtype)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()



