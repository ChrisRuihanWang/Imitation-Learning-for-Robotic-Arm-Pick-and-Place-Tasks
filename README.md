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

*(Add a GIF or short video here later)*


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
Imitation-Learning-for-Robotic-Arm-Pick-and-Place-Tasks
│
├── pp_scripts/                                   # End-to-end pipeline scripts
│   ├── teleop_collect_rgb_npz.py                 # Collect teleop demonstrations (RGB + state/action) -> .npz
│   ├── convert_npz_to_lerobot_teleop2.py         # Convert .npz demonstrations to LeRobot dataset format
│   ├── act_infer.py                              # Policy inference / rollout evaluation (to be migrated to π0.5 naming)
│   └── run_pi_pickplace.py                       # Deploy trained π0.5 policy back into Isaac Lab simulation
│
├── source/pick_and_place_project/                # Custom Isaac Lab project package
│   ├── __init__.py
│   ├── pi_agent/                                 # π0.5 policy wrapper & integration
│   │   └── policy_pi05.py                         # π0.5 interface (load model, run inference, action post-processing)
│   │
│   └── tasks/                                    # Pick-and-place task implementation
│       ├── pick_place_cfg.py                      # Task config (scene, robot, sensors, controllers, etc.)
│       ├── pick_place_gr1t2_pi.py                 # Task environment entry (registration + wiring task, robot, cameras)
│       │
│       └── mdp/                                  # MDP components (Isaac Lab manager-based style)
│           ├── actions.py                          # Action space definition (e.g., arm/gripper commands)
│           ├── observation.py                      # Observation construction (state + multi-view RGB, normalization)
│           └── terminations.py                     # Episode termination conditions (success/failure/timeouts)
│
├── README.md                                      # Project documentation
├── LICENSE                                        # MIT license
├── pyproject.toml                                 # Python package / build configuration
└── .gitignore                                     # Ignore datasets, checkpoints, local env files, etc.


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




