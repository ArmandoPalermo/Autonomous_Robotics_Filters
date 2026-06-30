# Autonomous Robotics Filters

This repository contains a collection of exercises and implementations focused on the main
probabilistic filters used in autonomous robotics.  
The goal is to provide clear, modular, and extensible examples for studying and experimenting
with localization and sensor fusion techniques.

---

## 📚 Full Documentation for EKF fusion

👉 **SensorFusion_EKF/Docummentation_EKF_Fusion.pdf** 

---

## 📦 Repo Content

### 🔹 1. Particle Filter and EKF - 2D
Implementation of a particle filter for robot state estimation in environments with
non‑Gaussian noise.  
Includes:
- particle generation
- motion update using kinematic models
- sensor update with likelihood field
- resampling 
- visualization of particle evolution

Directory: `particle_and_EKF_2DFilter/`

---

### 🔹 3. EKF Sensor Fusion (Real ROSbag)
Implementation of an **Extended Kalman Filter** for fusing data from a real ROSbag
(DLIO + Odometry + GPS).  

Directory: `SensorFusion_EKF/`

