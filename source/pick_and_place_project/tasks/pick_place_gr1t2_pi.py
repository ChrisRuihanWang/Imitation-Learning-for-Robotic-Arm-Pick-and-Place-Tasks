from __future__ import annotations

import math
import torch


from isaaclab.utils.configclass import configclass
from isaaclab.envs import ManagerBasedRLEnv

from pick_and_place_project.tasks.pick_place_cfg import PickPlaceEnvCfg_PLAY


@configclass
class PickPlaceGR1T2PiEnvCfg(PickPlaceEnvCfg_PLAY):
    """用于 teleop / 数据采集：延长 episode + reset 后注入噪声（随机化）。"""

    # ===== Objects（物体）随机范围：限定在 robot 与 basket 之间的区域 =====
    # 你现在 basket 初始 pos=(0.7, 0.0, z)，robot 在 (0,0) 附近；这里给一个“夹在中间”的区域
    cube_xy_range: tuple[tuple[float, float], tuple[float, float]] = ((0.50, 0.68), (-0.12, 0.12))
    cube_yaw_deg: float = 45.0  # cube yaw（偏航角）随机范围：[-yaw_deg, +yaw_deg]

    # 避免 cube 与 basket 初始重叠：最小中心距离（米）
    # basket 半边长 0.125m，cube 半边长 0.03m，留点余量建议 >= 0.18~0.22
    cube_basket_min_dist: float = 0.20
    cube_sample_max_tries: int = 50

    # ===== Basket（篮子）只随机角度，不随机位置 =====
    basket_yaw_deg: float = 15.0

    # ===== Robot（机器人）初始关节噪声 =====
    robot_joint_noise_std: float = 0.03  # rad，0.03rad≈1.7°

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 120.0


class PickPlaceGR1T2PiEnv(ManagerBasedRLEnv):
    """reset() 后注入噪声（Noise｜随机扰动）：objects / robot / camera。"""

    # ----------------------------
    # Utils（工具）
    # ----------------------------
    def _default_root_state_w(self, obj):
        if hasattr(obj.data, "default_root_state_w"):
            return obj.data.default_root_state_w
        if hasattr(obj.data, "default_root_state"):
            return obj.data.default_root_state
        if hasattr(obj.data, "root_state_w"):
            return obj.data.root_state_w
        raise AttributeError(f"{obj} has no default/root state fields on obj.data.")

    def _rand_yaw_quat(self, n: int, device, yaw_deg: float) -> torch.Tensor:
        """quat(x,y,z,w), yaw ∈ [-yaw_deg, +yaw_deg]."""
        yaw = (torch.rand((n,), device=device) * 2.0 - 1.0) * (yaw_deg * math.pi / 180.0)
        half = 0.5 * yaw
        quat = torch.zeros((n, 4), device=device)
        quat[:, 2] = torch.sin(half)  # z
        quat[:, 3] = torch.cos(half)  # w
        return quat

    # ----------------------------
    # Noise 1：Objects（物体）随机化
    # ----------------------------
    def _sample_cube_xy_non_overlapping(
        self,
        n: int,
        device,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        basket_xy: torch.Tensor,   # (n,2)
        min_dist: float,
        max_tries: int,
    ) -> torch.Tensor:
        """
        采样 cube 的 (x,y) 并满足与 basket 的最小距离约束。
        返回 shape (n,2).
        """
        xy = torch.empty((n, 2), device=device)
        # 初始全未满足
        valid = torch.zeros((n,), dtype=torch.bool, device=device)

        for _ in range(max_tries):
            # 仅为未满足的 env 采样
            idx = torch.nonzero(~valid, as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                break

            u = torch.rand((idx.numel(), 2), device=device)
            xy_new = torch.empty((idx.numel(), 2), device=device)
            xy_new[:, 0] = x_min + (x_max - x_min) * u[:, 0]
            xy_new[:, 1] = y_min + (y_max - y_min) * u[:, 1]

            # 距离约束：||cube_xy - basket_xy|| >= min_dist
            d = torch.linalg.norm(xy_new - basket_xy[idx], dim=-1)
            ok = d >= min_dist

            # 写入通过的部分
            if ok.any():
                ok_idx = idx[ok]
                xy[ok_idx] = xy_new[ok]
                valid[ok_idx] = True

        # 仍不满足的 env：兜底放到区域角落，尽量远离 basket
        if (~valid).any():
            idx = torch.nonzero(~valid, as_tuple=False).squeeze(-1)
            # 选择一个远离 basket 的角点（粗略策略）
            corners = torch.tensor(
                [[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]],
                device=device,
                dtype=xy.dtype,
            )  # (4,2)
            # 对每个 idx 找最远角点
            b = basket_xy[idx].unsqueeze(1)  # (m,1,2)
            d_corner = torch.linalg.norm(corners.unsqueeze(0) - b, dim=-1)  # (m,4)
            far = torch.argmax(d_corner, dim=-1)  # (m,)
            xy[idx] = corners[far]

        return xy

    def _randomize_objects(self):
        device = self.device
        n = self.num_envs

        cube = self.scene["cube_0"]
        basket = self.scene["basket"]

        # --- basket: 只随机 yaw，不改位置 ---
        basket_state = self._default_root_state_w(basket).clone()
        basket_xy = basket_state[:, 0:2].clone()  # 固定位置
        basket_state[:, 3:7] = self._rand_yaw_quat(n, device, self.cfg.basket_yaw_deg)
        basket.write_root_pose_to_sim(basket_state[:, 0:7])

        # --- cube: 在 robot 与 basket 之间区域采样，且避免与 basket 重叠 ---
        cube_state = self._default_root_state_w(cube).clone()

        (x_min, x_max), (y_min, y_max) = self.cfg.cube_xy_range
        cube_xy = self._sample_cube_xy_non_overlapping(
            n=n,
            device=device,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            basket_xy=basket_xy,
            min_dist=float(self.cfg.cube_basket_min_dist),
            max_tries=int(self.cfg.cube_sample_max_tries),
        )

        cube_state[:, 0:2] = cube_xy
        # z 保持默认（避免穿模）
        #cube_state[:, 3:7] = self._rand_yaw_quat(n, device, self.cfg.cube_yaw_deg)
        cube.write_root_pose_to_sim(cube_state[:, 0:7])

    # ----------------------------
    # Noise 2：Robot（机器人）初始关节噪声
    # ----------------------------
    def _randomize_robot(self):
        device = self.device
        robot = self.scene["robot"]

        if not hasattr(robot.data, "default_joint_pos"):
            print("[warn] robot.data has no default_joint_pos; skip robot noise.")
            return

        q0 = robot.data.default_joint_pos.clone()
        if hasattr(robot.data, "default_joint_vel"):
            qd0 = robot.data.default_joint_vel.clone()
        else:
            qd0 = torch.zeros_like(q0, device=device)

        noise = torch.randn_like(q0) * float(self.cfg.robot_joint_noise_std)
        q = q0 + noise
        robot.write_joint_state_to_sim(q, qd0)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)

        self._randomize_objects()
        self._randomize_robot()

        return obs, info


def make_env() -> ManagerBasedRLEnv:
    cfg = PickPlaceGR1T2PiEnvCfg()
    env = PickPlaceGR1T2PiEnv(cfg)
    return env
