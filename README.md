# Imitation Learning for Robotic Arm Pick-and-Place Tasks
Robot Learning | Robotic Manipulation | Isaac Lab

This project implements an **imitation learning pipeline for robotic manipulation** using the **Franka Panda robot** in **Isaac Lab simulation**.

The goal is to learn a **pick-and-place policy from teleoperation demonstrations**, train a policy using collected datasets, and deploy the trained policy back into the simulation environment.

---

## Pipeline

Isaac Lab Environment  
↓  
Teleoperation Demonstrations  
↓  
LeRobot Dataset Construction  
↓  
π0.5 VLA Policy Training  
↓  
Policy Deployment in Isaac Lab


---

## Demo

![Pick and Place Task](assets/demo.gif)


---

## Features

- Isaac Lab robotic manipulation environment
- Teleoperation-based demonstration collection
- Dataset conversion to **LeRobot format**
- Imitation learning for **robotic pick-and-place**
- Policy deployment in Isaac Lab simulation

---

## Tech Stack

**Simulation**

- Isaac Sim / Isaac Lab

**Machine Learning**

- PyTorch  
- Vision-Language-Action Model (π0.5)

**Robotics**

- Franka Panda Manipulator
- Teleoperation Demonstrations
- Robot Manipulation Learning

---

## Repository Structure

```text
Imitation-Learning-for-Robotic-Arm-Pick-and-Place-Tasks
│
├── pp_scripts/                                   # End-to-end pipeline scripts
│   ├── teleop_collect_rgb_npz.py                 # Collect teleop demonstrations (RGB + state/action) -> .npz
│   ├── convert_npz_to_lerobot_teleop2.py         # Convert demonstrations to LeRobot dataset format
│   ├── act_infer.py                              # Policy inference / rollout evaluation
│   └── run_pi_pickplace.py                       # Deploy trained π0.5 policy in Isaac Lab
│
├── source/pick_and_place_project/                # Custom Isaac Lab project package
│   ├── __init__.py
│   ├── pi_agent/
│   │   └── policy_pi05.py                        # π0.5 policy wrapper
│   │
│   └── tasks/
│       ├── pick_place_cfg.py                     # Task configuration
│       ├── pick_place_gr1t2_pi.py                # Task environment entry
│       │
│       └── mdp/
│           ├── actions.py                        # Action definitions
│           ├── observation.py                    # Observation construction
│           └── terminations.py                   # Episode termination logic
│
├── README.md                                     # Project documentation
├── LICENSE                                       # MIT license
├── pyproject.toml                                # Python project configuration
└── .gitignore                                    # Ignore datasets, checkpoints, env files
```

---

## Dataset

Demonstrations are collected through teleoperation and converted to **LeRobot dataset format** for training.

Datasets and checkpoints are **not included in this repository**.

---

## Future Work

Planned improvements include:

- Improving teleoperation data quality
- Training larger Vision-Language-Action models
- Improving manipulation robustness
- Exploring sim-to-real transfer

---

## Author

Ruihan Wang  
MSc Student — KTH Royal Institute of Technology  




