from isaaclab.app import AppLauncher


def main():
    # 1) 先启动 IsaacSim 应用（必须最先做），这里打开相机
    app_launcher = AppLauncher(headless=False, enable_cameras=True)
    simulation_app = app_launcher.app

    # 2) 再导入依赖 isaacsim / isaaclab 的模块
    from pick_and_place_project.tasks.pick_place_gr1t2_pi import make_env
    from pick_and_place_project.pi_agent.policy_pi05 import Pi05Policy

    # 3) 创建环境
    env = make_env()

    # 4) 创建 pi0.5 策略（当前是占位 zero-action 策略）
    policy = Pi05Policy()

    # 5) 重置环境
    obs, _ = env.reset()

    # 简单跑一段时间，确认整条链路畅通
    num_steps = 500
    for _ in range(num_steps):
        # 策略根据观测生成动作
        action = policy.act(obs)

        # 与环境交互
        obs, rew, terminated, truncated, info = env.step(action)

        # 如果 episode 结束，重置环境
        if terminated.any() or truncated.any():
            obs, _ = env.reset()

    # 6) 关闭仿真
    simulation_app.close()


if __name__ == "__main__":
    main()

