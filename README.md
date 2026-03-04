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
scripts/
teleop_collect.py # teleoperation data collection
train_policy.py # model training / fine-tuning
deploy_policy.py # policy inference in simulation

source/
tasks/ # Isaac Lab task environment

data/ # demonstration datasets (not included)
checkpoints/ # trained models (not included)


---

## Dataset

Demonstrations are collected through teleoperation and converted to **LeRobot dataset format** for training.

Datasets and checkpoints are **not included in this repository**.

---

## Future Work

- Training larger **Vision-Language-Action models**
- Improving manipulation generalization
- Sim-to-real transfer for robotic manipulation

---

## Author

Ruihan Wang  
MSc Student — KTH Royal Institute of Technology  




