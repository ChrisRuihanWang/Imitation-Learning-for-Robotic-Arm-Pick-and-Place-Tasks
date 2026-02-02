from isaaclab.utils.configclass import configclass
from isaaclab.envs import ManagerBasedRLEnv
from pick_and_place_project.tasks.pick_place_cfg import (
    PickPlaceEnvCfg_PLAY,
)


@configclass
class PickPlaceGR1T2PiEnvCfg(PickPlaceEnvCfg_PLAY):
    """基于官方 GR1T2 Pick&Place 环境的配置，用于 pi0.5 微调."""

    def __post_init__(self):
        # 先调用父类的配置
        super().__post_init__()
        # 这里可以按需要微调配置（先给一个合理的默认）
        self.scene.num_envs = 1
        self.episode_length_s = 120.0

        # 如果你只想要 policy 观测里的图像 + 状态，可以在这里添加/修改
        # 例如：self.image_obs_list = ["robot_pov_cam"]  # 具体名字看官方 env 定义
        # 目前先保持官方默认


def make_env() -> ManagerBasedRLEnv:
    """创建一个 ManagerBasedRLEnv 实例，供脚本或训练代码使用."""
    cfg = PickPlaceGR1T2PiEnvCfg()
    env = ManagerBasedRLEnv(cfg)
    # 参考 run_usd_camera.py 的用法
    

    return env
