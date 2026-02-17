from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List

import numpy as np
from PIL import Image
import tyro

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _parse_resize_hw(s: str) -> Optional[Tuple[int, int]]:
    s = s.strip()
    if not s:
        return None
    if "," not in s:
        raise ValueError(f"--args.resize-hw expects 'H,W' like '256,256', got: {s}")
    h, w = s.split(",")
    return int(h), int(w)


@dataclass
class Args:
    # Input
    npz_dir: Path
    repo_id: str  # e.g. "local/pickplace_teleop2"

    # Output parent directory (dataset will be created at root / repo_id)
    root: Path = Path("data/lerobot")

    # Data meaning
    fps: int = 30
    task: str = "Pick the red cube and place it into the basket."
    robot_type: str = "sim_franka"

    # Images
    resize_hw: str = ""   # e.g. "256,256" or empty for no resize
    use_videos: bool = True

    # Overwrite existing dataset_dir
    overwrite: bool = False

    # Optional: store npz["meta"] into frame["info"]["meta"]
    write_info_meta: bool = False


def _require_keys(d: Any, keys: List[str], fname: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise KeyError(f"Missing keys in {fname}: {missing}")


def _build_state_names(state_dim: int) -> List[str]:
    joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
        "panda_finger_joint1", "panda_finger_joint2",
    ]
    state_names = (
        [f"qrel_{n}" for n in joint_names] +
        [f"qdrel_{n}" for n in joint_names] +
        ["grip_qrel_finger1", "grip_qrel_finger2"] +
        ["grip_qdrel_finger1", "grip_qdrel_finger2"] +
        ["grip_width"]
    )
    if state_dim != len(state_names):
        raise ValueError(
            f"state_dim={state_dim} but expected {len(state_names)}.\n"
            f"Obs concat order/dims changed. Please update _build_state_names()."
        )
    return state_names


def _raw_float_to_u8(img_raw: np.ndarray) -> np.ndarray:
    """
    Convert your saved float raw (typically centered/normalized) -> uint8 RGB for LeRobot.
    Uses robust percentile stretch per-frame to avoid black/noise.
    """
    x = img_raw.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # If it already looks like [0,1], use directly
    mn, mx = float(x.min()), float(x.max())
    if mx <= 1.5 and mn >= -0.1:
        y = np.clip(x, 0.0, 1.0)
        return (y * 255.0).astype(np.uint8)

    # Percentile stretch in raw space
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))
    if hi <= lo + 1e-8:
        # totally flat -> blank
        return np.zeros_like(x, dtype=np.uint8)

    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def main(args: Args) -> None:
    npz_dir = args.npz_dir.expanduser().resolve()
    if not npz_dir.exists():
        raise FileNotFoundError(f"npz_dir not found: {npz_dir}")

    resize_hw = _parse_resize_hw(args.resize_hw)

    root_parent = args.root.expanduser().resolve()
    dataset_dir = (root_parent / args.repo_id).resolve()

    if args.overwrite and dataset_dir.exists():
        print(f"[INFO] overwrite=True -> removing existing dataset: {dataset_dir}")
        shutil.rmtree(dataset_dir)

    dataset_dir.parent.mkdir(parents=True, exist_ok=True)

    if dataset_dir.exists():
        raise FileExistsError(
            f"Target dataset already exists: {dataset_dir}\n"
            f"Delete it or pass --args.overwrite."
        )

    # ---------- Scan episodes ----------
    npz_files = sorted(npz_dir.glob("ep_*.npz"))
    if len(npz_files) == 0:
        raise RuntimeError(f"No ep_*.npz found in {npz_dir}")

    # ---------- Infer shapes ----------
    sample = np.load(npz_files[0], allow_pickle=True)
    _require_keys(sample, ["state", "action", "rgb_raw", "wrist_rgb_raw",
                           "oblique_rgb_raw"], 
                  npz_files[0].name)

    state0 = sample["state"]              # (T,S) float32
    action0 = sample["action"]            # (T,A) float32
    rgb0 = sample["rgb_raw"]              # (T,H,W,3) float32
    wrist0 = sample["wrist_rgb_raw"]      # (T,h,w,3) float32
    oblique0 = sample["oblique_rgb_raw"]
    
    if state0.ndim != 2 or action0.ndim != 2 or rgb0.ndim != 4 or wrist0.ndim != 4:
        raise ValueError(
            f"Unexpected shapes:\n"
            f"  state {state0.shape}\n"
            f"  action {action0.shape}\n"
            f"  rgb_raw {rgb0.shape}\n"
            f"  wrist_rgb_raw {wrist0.shape}"
        )

    _, state_dim = state0.shape
    _, action_dim = action0.shape
    _, H_oh, W_oh, C_oh = rgb0.shape
    _, H_wr, W_wr, C_wr = wrist0.shape
    _, H_ob, W_ob, C_ob = oblique0.shape
    if C_oh != 3 or C_wr != 3 or C_ob != 3:
        raise ValueError(f"Images must have 3 channels, got overhead={C_oh}, wrist={C_wr}, oblique={C_ob}")
    
    state_names = _build_state_names(state_dim)

    # declared image shapes
    if resize_hw is not None:
        H_out, W_out = resize_hw
        overhead_shape = (H_out, W_out, 3)
        wrist_shape = (H_out, W_out, 3)
        oblique_shape = (H_out, W_out, 3)
    else:
        overhead_shape = (H_oh, W_oh, 3)
        wrist_shape = (H_wr, W_wr, 3)
        oblique_shape = (H_ob, W_ob, 3)

    features = {
        "observation.images.overhead": {
            "dtype": "video" if args.use_videos else "image",
            "shape": overhead_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "video" if args.use_videos else "image",
            "shape": wrist_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": state_names,
        },
        "observation.images.oblique": {
            "dtype": "video" if args.use_videos else "image",
            "shape": oblique_shape,   
            "names": ["height", "width", "channel"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"a{i}" for i in range(action_dim)],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=dataset_dir,
        fps=args.fps,
        robot_type=args.robot_type,
        features=features,
        use_videos=args.use_videos,
    )

    print(f"[OK] repo_id={args.repo_id}")
    print(f"[OK] dataset_dir={dataset_dir}")
    print(f"[OK] npz_dir={npz_dir} episodes={len(npz_files)}")
    print(f"[OK] state_dim={state_dim} action_dim={action_dim}")
    print(f"[OK] overhead_raw={H_oh}x{W_oh} wrist_raw={H_wr}x{W_wr} resize={resize_hw}")
    print(f"[OK] overhead_raw={H_oh}x{W_oh} wrist_raw={H_wr}x{W_wr} oblique_raw={H_ob}x{W_ob} resize={resize_hw}")

    for ep_idx, f in enumerate(npz_files):
        data = np.load(f, allow_pickle=True)
        _require_keys(data, ["state", "action", "rgb_raw", "wrist_rgb_raw",
                             "oblique_rgb_raw"], f.name)

        state = data["state"].astype(np.float32)
        action = data["action"].astype(np.float32)
        rgb_raw = data["rgb_raw"]
        wrist_raw = data["wrist_rgb_raw"]
        oblique_raw = data["oblique_rgb_raw"]
        T = min(len(state), len(action), len(rgb_raw), len(wrist_raw),len(oblique_raw))
        if T < 5:
            print(f"[WARN] Skip short episode {f.name}: T={T}")
            continue

        meta_payload: Optional[Dict[str, Any]] = None
        if args.write_info_meta and "meta" in data:
            try:
                meta_payload = data["meta"][0] if isinstance(data["meta"], np.ndarray) else data["meta"]
            except Exception:
                meta_payload = None

        for t in range(T):
            oh_u8 = _raw_float_to_u8(rgb_raw[t])
            wr_u8 = _raw_float_to_u8(wrist_raw[t])
            ob_u8 = _raw_float_to_u8(oblique_raw[t])

            img_oh = Image.fromarray(oh_u8)
            img_wr = Image.fromarray(wr_u8)
            img_ob = Image.fromarray(ob_u8)

            if resize_hw is not None:
                img_oh = img_oh.resize((W_out, H_out), resample=Image.BILINEAR)
                img_wr = img_wr.resize((W_out, H_out), resample=Image.BILINEAR)
                img_ob = img_ob.resize((W_out, H_out), resample=Image.BILINEAR)

            frame = {
                "observation.images.overhead": img_oh,
                "observation.images.wrist": img_wr,
                "observation.images.oblique": img_ob, 
                "observation.state": state[t],
                "action": action[t],
                "task": args.task,
            }
            if meta_payload is not None:
                frame["info"] = {"meta": meta_payload}

            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"[EP] {ep_idx:03d} saved: {f.name} (T={T})")

    print("\n[DONE] Conversion finished.")
    print("Load with:")
    print("  from pathlib import Path")
    print("  from lerobot.datasets.lerobot_dataset import LeRobotDataset")
    print(f'  ds = LeRobotDataset("{args.repo_id}", root=Path("{dataset_dir}"))')
    print("  print(ds[0].keys())")


if __name__ == "__main__":
    tyro.cli(main)