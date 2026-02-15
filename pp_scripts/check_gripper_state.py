from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path("data/lerobot/local/pickplace_teleop").resolve()
ds = LeRobotDataset("local/pickplace_teleop", root=root)

g = []
for i in range(len(ds)):
    a = ds[i]["action"].numpy()
    g.append(a[-1])
g = np.array(g)
print("gripper min/max/mean/std:", g.min(), g.max(), g.mean(), g.std())
print("percentiles:", np.percentile(g, [1,5,25,50,75,95,99]))