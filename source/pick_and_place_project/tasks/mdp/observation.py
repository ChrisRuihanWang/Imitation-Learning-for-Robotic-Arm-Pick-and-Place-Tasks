from __future__ import annotations
import torch
from isaaclab.managers import SceneEntityCfg
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera

def franka_gripper_width(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return gripper width as q1 + q2. Shape: (num_envs, 1)."""
    robot = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]   # (N,2)
    return q.sum(dim=-1, keepdim=True)

def image_raw(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    data_type: str = "rgb",
) -> torch.Tensor:
    """不做 normalize，直接返回摄像头原始输出."""
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output[data_type]  # 通常是 uint8 [0,255]
    return images.clone()