from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

# ✅ 最稳：尽早声明离线（避免任何 HF hub 请求）
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ==== fake accelerate（仅在缺失时注入）====
try:
    import accelerate  # noqa: F401
except Exception:
    fake_accelerate = types.ModuleType("accelerate")

    class DummyAccelerator:
        def __init__(self, *args, **kwargs):
            pass

        def prepare(self, *args, **kwargs):
            if len(args) == 1:
                return args[0]
            return args

    fake_accelerate.Accelerator = DummyAccelerator
    sys.modules["accelerate"] = fake_accelerate
# ========================================

# ✅ 关键：直接导入 ACT，避免 lerobot.policies.__init__ 引入 Groot/transformers
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.processor.pipeline import PolicyProcessorPipeline


def hwc_u8_to_chw_f01(img_hwc_uint8: np.ndarray) -> torch.Tensor:
    """HWC uint8 -> CHW float32 in [0,1]."""
    assert img_hwc_uint8.ndim == 3 and img_hwc_uint8.shape[2] == 3
    return (
        torch.from_numpy(img_hwc_uint8)
        .permute(2, 0, 1)
        .contiguous()
        .float()
        / 255.0
    )


class ActAgent:
    def __init__(self, ckpt_dir: Path, device: str = "cuda"):
        self.device = torch.device(
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )

        self.policy = ACTPolicy.from_pretrained(ckpt_dir).to(self.device)
        self.policy.eval()

        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            ckpt_dir, config_filename="policy_preprocessor.json"
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            ckpt_dir, config_filename="policy_postprocessor.json"
        )

    @torch.no_grad()
    def predict_chunk(self, obs_img_hwc_uint8: np.ndarray, obs_state_23: np.ndarray) -> np.ndarray:
        """
        Args:
          obs_img_hwc_uint8: (H,W,3) uint8
          obs_state_23: (23,) float32/float64

        Returns:
          chunk: (T,7) float32
        """
        img = hwc_u8_to_chw_f01(obs_img_hwc_uint8).unsqueeze(0).to(self.device)  # [1,3,H,W]
        state = (
            torch.from_numpy(obs_state_23.astype(np.float32))
            .unsqueeze(0)
            .to(self.device)
        )  # [1,23]

        batch: Dict[str, Any] = {
            "observation.images.overhead": img,
            "observation.state": state,
        }

        batch = self.preprocessor(batch)

        # 2) preprocessor 之后，补齐 ACT 需要的 key（以防它没做）
        if "observation.environment_state" not in batch and "observation.state" in batch:
            batch["observation.environment_state"] = batch["observation.state"]

        # 3) 同理，补齐 ACT 可能需要的 observation.images 结构（list）
        if "observation.images" not in batch and "observation.images.overhead" in batch:
            batch["observation.images"] = [batch["observation.images.overhead"]]

        out = self.policy.model(batch)
        if isinstance(out, tuple):
            actions_hat = out[0]
        elif isinstance(out, dict):
            actions_hat = out.get("actions", None) or out.get("action", None)
            if actions_hat is None:
                raise KeyError(
                    f"policy.model(batch) dict keys={list(out.keys())}, missing 'actions'/'action'"
                )
        else:
            actions_hat = out

        # ✅ postprocess：兼容 action/actions 两种键
        pp_out = self.postprocessor({"action": actions_hat})
        if "action" in pp_out:
            actions_hat = pp_out["action"]
        elif "actions" in pp_out:
            actions_hat = pp_out["actions"]
        else:
            # 兜底：直接用 pp_out
            actions_hat = next(iter(pp_out.values()))

        return actions_hat.squeeze(0).detach().cpu().numpy().astype(np.float32)
