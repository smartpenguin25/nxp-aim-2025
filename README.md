# 🤖 NXP AIM 2025 — Autonomous Rover

> Autonomous warehouse rover for the **NXP AIM India 2025** competition.
> Built on ROS2 Humble, the rover performs full map exploration, sequential
> shelf identification via QR codes, and object detection using YOLO.

---

## 🏆 Competition Overview

The **NXP AIM (Autonomous Intelligent Machine) 2025** challenge requires a
simulated rover to autonomously navigate a warehouse environment, identify
shelves using QR codes, and detect & publish objects found on each shelf —
in strict sequential order.

---

##  Mission Flow
  Autonomous Exploration  →  Full map coverage using SLAM
  Shelf 1                 →  Navigate → Decode QR → Detect & Publish Objects
  Shelf 2 (unlocked)      →  Navigate → Decode QR → Detect & Publish Objects
  Shelf 3 (unlocked)      →  Navigate → Decode QR → Detect & Publish Objects
  Shelf 4 (unlocked)      →  Navigate → Decode QR → Detect & Publish Objects
>  Each shelf only unlocks **after** the previous shelf's QR is decoded
> and object data is successfully published.

---

## Features

- **Autonomous Exploration** — Full map coverage before shelf scanning begins
- **QR Code Scanning** — Decodes shelf QR codes to unlock the next shelf
- **Object Detection (YOLO)** — Detects and classifies objects on each shelf
- **Shelf Sequencing** — Strict sequential logic; rover never skips a shelf
- **ROS2 Topics** — Object data published on ROS2 topics after each shelf scan

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | ROS2 Humble |
| Simulation | Gazebo |
| Object Detection | YOLOv5 / YOLO11 |
| QR Scanning | OpenCV + pyzbar |
| Navigation | Nav2 |
| Mapping | SLAM Toolbox |
| Language | Python 3 |

---


## ⚙️ Setup & Installation

### Prerequisites
- Ubuntu 22.04
- ROS2 Humble
- Gazebo
- Python 3.10+

### Install Dependencies
```bash
sudo apt update
sudo apt install ros-humble-nav2-bringup ros-humble-slam-toolbox
pip install opencv-python pyzbar ultralytics
```

### Clone the Repository
```bash
cd ~/ros2_ws/src
git clone https://github.com/smartpenguin25/nxp-aim-2025.git
```

### Build
```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

---

## 🚀 Launch

### Full Mission (Exploration + Shelf Scanning)
```bash
ros2 launch nxp_aim_india_2025 full_mission.launch.py
```

### Exploration Only
```bash
ros2 launch nxp_aim_india_2025 exploration.launch.py
```

### Shelf Scanning Only
```bash
ros2 launch nxp_aim_india_2025 shelf_scan.launch.py
```

---

## ROS2 Topics

| Topic | Type | Description |
|---|---|---|
| `/shelf/qr_data` | `std_msgs/String` | Decoded QR data from current shelf |
| `/shelf/objects` | `std_msgs/String` | Detected objects published after scan |
| `/shelf/status` | `std_msgs/Int32` | Current shelf number being processed |
| `/exploration/status` | `std_msgs/Bool` | Exploration complete flag |

---

##Team

- **Team Name:** smartpenguin25
- **Competition:** NXP AIM India 2025
- **Platform:** Simulation (Gazebo + ROS2 Humble)

---

## 📄 License

This project is open source
