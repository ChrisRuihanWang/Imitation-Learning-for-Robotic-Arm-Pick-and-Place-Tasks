from isaaclab.app import AppLauncher

def main():
    app_launcher = AppLauncher(headless=False, enable_cameras=True)
    simulation_app = app_launcher.app

    from pick_and_place_project.tasks.pick_place_gr1t2_pi import make_env
    from pick_and_place_project.pi_agent.policy_pi05 import Pi05Policy

    env = make_env()
    policy = Pi05Policy()

    obs, _ = env.reset()

    while simulation_app.is_running():
        
        simulation_app.update()

        
        action = policy.act(obs)
        obs, rew, terminated, truncated, info = env.step(action)

        if terminated.any() or truncated.any():
            obs, _ = env.reset()
        print(env.action_space)

    simulation_app.close()

if __name__ == "__main__":
    main()


