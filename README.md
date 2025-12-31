# Vampire Survivors Reinforced Auto Learning

This project implements a reinforcement learning–style automation system that
plays **Vampire Survivors** by reading pixels from the game screen and sending
keyboard inputs in real time. It uses Python, OpenCV, and Gymnasium-style
environments to experiment with reward shaping and control logic.

The goal of this repo is to demonstrate how to:
- Capture and preprocess game frames from the desktop.
- Track key UI elements (player, HUD, health/XP bars) using template matching.
- Wrap the game as a Gym-like environment.
- Train and debug a PPO agent using Stable-Baselines3.

> Note: This project was built for educational / portfolio purposes (specifically as my final project in my EE-595 graduate class) and may
> require small tweaks (e.g., coordinates, templates) for different
> resolutions and setups.

---

## Project Structure

```text
vs_rl/
├── capture_fixed.py        # Full-monitor capture + preprocessing
├── vs_env_fixed.py         # Gymnasium Env wrapping Vampire Survivors
├── train_fixed.py          # PPO training script with hotkey controls
├── reward.py               # Reward utilities and bar/template helpers
├── vision.py               # Player tracking and enemy density estimation
├── controls.py             # Keyboard controller using pydirectinput
├── curriculum.py           # Simple curriculum schedule (reward scaling)
├── config_fixed.yaml       # Main config: ROI, templates, rewards, hyperparams
├── config.yaml             # Alternate/legacy config (similar structure)
├── debug_roi.py            # Visualize configured ROIs on a live capture
├── test_capture.py         # Quick test: grab a monitor screenshot
├── templates/
│   ├── player.png          # Template for player sprite matching
│   ├── hud.png             # Template for in-game HUD detection
│   └── game_over.png       # Template for game-over detection
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

The training artifacts (TensorBoard logs, trained model weights, my personal attempts.) are not
included in the repository to keep it lightweight and focused on the code for potentially future downloaders.

## Tech Stack
- Python
- OpenCV for image processing and template matching.
- mss for screen capture.
- pydirectinput for sending keyboard inputs to the game.
- Stable-Baselines3 (PPO) for training (used by train_fixed.py).
- A legal copy of the game Vampire Survivors (it's a good game even regardless of this project)

## Setup
1. Clone this repository:
  ```
git clone https://github.com/AnotherAnotherAustin/VampireSurvivorsReinforcedAutoLearning.git
cd VampireSurvivorsReinforcedAutoLearning/vs_rl
```
2. Create and activate a virtual environment (it was easier to get it to run this way):
```
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```
3. Install the dependencies:
```
pip install -r requirements.txt
```
