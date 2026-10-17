# 🦾 BiomechAI 5.0

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20%26%20FaceMesh-00C49F?logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![macOS](https://img.shields.io/badge/Platform-macOS%20(Metal%20Accelerated)-black?logo=apple&logoColor=white)](https://apple.com)

> **BiomechAI** is an intelligent, real-time computer vision system for **360° biomechanical posture analysis**, **V-Taper & body fat estimation**, and **facial adiposity / jawline sharpness evaluation** featuring an asynchronous **English Voice Coach** and a high-tech **glassmorphism HUD dashboard**.

---

## 📋 Table of Contents
- [Key Features](#-key-features)
  - [1. Voice-Guided Sequential 360° Scan](#1--voice-guided-sequential-360-scan)
  - [2. Facial Adiposity & Jawline Scanner](#2--facial-adiposity--jawline-scanner)
  - [3. 360° Posture & Spinal Biomechanics](#3--360-posture--spinal-biomechanics)
  - [4. Glassmorphism Cyberpunk HUD](#4--glassmorphism-cyberpunk-hud)
- [Architecture & Mechanics](#-architecture--mechanics)
- [Quick Start](#-quick-start)
- [Keyboard Shortcuts](#-keyboard-shortcuts)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Privacy & Offline Execution](#-privacy--offline-execution)
- [License](#-license)

---

## 🌟 Key Features

### 1. 🔊 Voice-Guided Sequential 360° Scan
Operate hands-free from 2 meters away. The integrated voice coach (Apple macOS Samantha Siri engine) guides you through a multi-angle diagnostic:
* **Stage 1 (Front View):** *"Voice-guided body scan initiated. Please stand upright facing the camera."* -> Automatic 3-2-1 countdown -> Front profile locked.
* **Stage 2 (Back View):** *"Front scan complete! Now, please turn around to show your back."* -> Automatic back detection -> Shoulder drop delta and spinal alignment captured.
* **Stage 3 (Side View):** *"Back scan complete! Now, please turn ninety degrees to the side."* -> Thoracic kyphosis and forward head posture (Text Neck) evaluated.
* **Summary Synthesis:** Speaks immediate findings aloud (e.g. V-Taper score and shoulder asymmetry in cm) and exports a complete 3D biometric report.

---

### 2. 👤 Facial Adiposity & Jawline Scanner
Analyzes 468 3D facial landmarks (MediaPipe FaceMesh) to estimate subcutaneous body fat and mandibular definition without requiring full-body framing:
* **Jawline V-Angle (Gonial Taper):** Mandibular sharpness angle (< 76° indicates low body fat and prominent bone definition).
* **Cheek-to-Jaw Ratio (CJR):** Buccal fat fullness compared against zygomatic cheekbone width.
* **Jawline Sharpness Score (%0 - %100):** Real-time bone prominence rating.
* **Facial Body Fat Estimate (%):** Adiposity categorization (`9% - 13% Lean/Chiseled` to `21% - 26% Rounded`).
* **fWHR & Facial Symmetry:** Facial width-to-height ratio and bilateral balance scoring.

---

### 3. ⚖️ 360° Posture & Spinal Biomechanics
* **Left / Right Shoulder Drop:** Converts pixel delta into real centimeters calibrated to user height (`Mild Left Shoulder Drop: 2.1 cm / 4.2°`).
* **Lateral Spinal Deviation (Scoliosis Indicator):** Measures spinal column axis offset from the neutral vertical line.
* **Thoracic Kyphosis:** Angle between ear, acromion, and hip to identify hunchback tendencies.
* **Forward Head Posture (Text Neck):** Horizontal offset between ear and shoulder line caused by smartphone/desk ergonomics.
* **Corrective Physical Therapy Plan:** Specific prescriptions (Chin-Tucks, Face-Pulls, Wall Angels, Unilateral Rows).

---

### 4. 🖼️ Glassmorphism Cyberpunk HUD
* **Frosted Glass Cards:** Semi-transparent dark panels (`alpha=0.86`) with rounded corners (`radius=16`) and cybernetic targeting brackets (`⌜ ⌝ ⌞ ⌟`).
* **Anti-Aliased Typography:** Crystal-clear TrueType rendering powered by Pillow and Apple Helvetica / San Francisco.
* **Apple Watch Style Radial Gauges:** Circular glowing ring arcs for posture, symmetry, and jawline definition.
* **Silhouette Segmentation Modes (`[B]` Key):** Normal Camera, Dark Studio Mode, or Holographic Cyber Laser scanlines.

---

## 🧠 Architecture & Mechanics

```
                  ┌────────────────────────┐
                  │   Webcam Feed (M2)     │
                  └───────────┬────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
     [Mode: Body]                         [Mode: Face]
            │                                   │
 ┌──────────▼───────────┐            ┌──────────▼───────────┐
 │   MediaPipe Pose     │            │  MediaPipe FaceMesh  │
 │  (33 3D Landmarks)   │            │ (468 3D Landmarks)   │
 └──────────┬───────────┘            └──────────┬───────────┘
            │                                   │
 ┌──────────▼───────────┐            ┌──────────▼───────────┐
 │ Anthropometric Math  │            │ Craniofacial Math    │
 │ - V-Taper Ratio      │            │ - Jawline V-Angle    │
 │ - Shoulder Drop (cm) │            │ - Cheek/Jaw Ratio    │
 │ - Kyphosis / Neck    │            │ - Facial Fat Est. %  │
 └──────────┬───────────┘            └──────────┬───────────┘
            │                                   │
            └─────────────────┬─────────────────┘
                              │
               ┌──────────────▼─────────────┐
               │    Glassmorphism HUD &     │
               │  Apple Watch Radial Gauges │
               └──────────────┬─────────────┘
                              │
               ┌──────────────▼─────────────┐
               │   Native English TTS Coach │
               │   (macOS Siri / Samantha)  │
               └────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/biomech-ai.git
cd biomech-ai
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| **`[T]`** or **`[SPACE]`** | **Start Voice-Guided 360° Scan:** Sequential Front -> Back -> Side scan with voice instructions |
| **`[V]`** | **Toggle Voice Coach:** Mute / Unmute the spoken English voice guidance |
| **`[F]`** | **Switch Mode:** 360° Body Scan <-> Facial Fat & Jawline Scanner |
| **`[B]`** | **Silhouette Mode:** Cycle between Normal, Dark Studio, and Cyber Laser effects |
| **`[S]`** | **Manual Save:** Save instantaneous HUD screenshot (`.png`) and telemetry report (`.txt`) |
| **`[H]`** | **Toggle Overlays:** Show or hide skeletal lines and facial mesh contours |
| **`[Q]`** or **`[ESC]`** | Cleanly exit application |

---

## ⚙️ Configuration

Customize your anthropometric profile in [config.py](file:///Users/ekremcakir/.gemini/antigravity-ide/scratch/ai-body-scanner/config.py):

```python
USER_CONFIG = {
    "gender": "male",      # 'male' or 'female'
    "height_cm": 170,      # Height in cm (used for cm drop calibration)
    "weight_kg": 63,       # Weight in kg
    "age": 23,             # Age
}
```

*(Optional)* Enable Google Gemini Multimodal Vision AI for secondary assessment:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

---

## 🛠️ Troubleshooting

### macOS Camera Selection (FaceTime HD vs. iPhone Continuity Camera)
On macOS Ventura/Sonoma, if an iPhone is nearby, macOS may route camera feed to iPhone Continuity Camera. To use the built-in MacBook camera:
- In the macOS menu bar, click the green camera icon and select **FaceTime HD Camera**, or
- On your iPhone: Go to **Settings > General > AirPlay & Handoff > Continuity Camera** and toggle it OFF.

---

## 🔒 Privacy & Offline Execution

All computer vision inference (MediaPipe Pose, FaceMesh, and OpenCV graphics) executes **100% locally on your machine** utilizing Apple Silicon GPU/Metal acceleration. Camera snapshots and biometric telemetry reports are saved locally to your disk and are strictly excluded via `.gitignore`. No personal media is ever transmitted over the network.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the [LICENSE](LICENSE) file for details.
