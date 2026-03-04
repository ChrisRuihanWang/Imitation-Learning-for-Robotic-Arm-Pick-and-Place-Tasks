from __future__ import annotations

import torch
from isaaclab.utils import configclass
from isaaclab.managers import ActionTerm, ActionTermCfg


class FrankaGripperAction(ActionTerm):
    """1D gripper command mapped to both finger joints."""

    cfg: "FrankaGripperActionCfg"

    def __init__(self, cfg: "FrankaGripperActionCfg", env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]

        
        joint_ids, joint_names = self.robot.find_joints(list(cfg.joint_names))
        
        self.joint_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=env.device)

        self._raw_actions: torch.Tensor | None = None
        self._processed_actions: torch.Tensor | None = None

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions
        a = actions.squeeze(-1).clamp(-1.0, 1.0)
        target = (a + 1.0) * 0.5 * (self.cfg.close_pos - self.cfg.open_pos) + self.cfg.open_pos
        self._processed_actions = torch.stack([target, target], dim=-1)

    def apply_actions(self):
        self.robot.set_joint_position_target(self._processed_actions, joint_ids=self.joint_ids)


@configclass
class FrankaGripperActionCfg(ActionTermCfg):
    """Config for 1D gripper command mapped to both finger joints."""

    
    class_type: type[ActionTerm] = FrankaGripperAction

    asset_name: str = "robot"
    joint_names: tuple[str, str] = ("panda_finger_joint1", "panda_finger_joint2")
    open_pos: float = 0.04
    close_pos: float = 0.0
