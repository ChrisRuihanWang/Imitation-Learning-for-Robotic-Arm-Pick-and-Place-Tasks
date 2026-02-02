from typing import Any

import torch


class Pi05Policy:
    """封装 pi0.5 的策略接口（当前占位：返回全零动作）"""

    def __init__(self, action_dim: int = 6):
        self.action_dim = action_dim

    def _extract_tensor(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        if isinstance(obs, dict):
            first_key = next(iter(obs.keys()))
            return self._extract_tensor(obs[first_key])
        return torch.as_tensor(obs)

    def act(self, obs: Any) -> torch.Tensor:
        obs_tensor = self._extract_tensor(obs)
        batch_size = obs_tensor.shape[0]
        device = obs_tensor.device
        return torch.zeros((batch_size, self.action_dim), device=device)


