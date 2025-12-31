# Vampire Survivors Reinforced Auto Learning

This project implements a reinforcement learning–style automation system that
plays the top-down game **Vampire Survivors** by reading pixels from the game screen and sending
keyboard inputs in real time. It uses Python, OpenCV, and Gymnasium-style
environments to experiment with reward shaping and control logic.

This is a repo that houses all the required files for a reinforcement learning agent that views a working game of Vampire Survivors and gradually improves its play style.

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
.
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

## Testing Screen Capture and ROIs
But before we do anything fancy here, it would be good to verify that the capture and ROIs are set correctly.
1. Ingame settings:

***Resolution:*** 1920x1080

***Window Mode:*** Fullscreen Window
> These are the only required ingame settings for the code to function correctly, you can have the rest of the settings be whatever you want.

2. Test full-monitor capture *(optional)*
  ```
  python test_capture.py
  ```
This should save **test_capture.png** in the currently directory. Open it and check that it properly took an image of your ingame screen.

3. Debug XP/HP bar ROIs *(optional)*
```
python debug_roi.py
```
This overlays rectangles for the XP ane HP bars ingame defined in **config.yaml**.
Use this to fine-tune your specific coordinates for each under the ***roi:*** section if you need to.

## Environment and Training
### Gym-style environment
The main environment class is defined in **vs_env_fixed.py**:
- Observations: stacked grayscale frames (shapes based on *obs_width*, *obs_height*, and *frame_stack* in **config_fixed.yaml**).
- Actions: movement directions via *KeyController* in **controls.py**.
- Reward: combines time alive, HP changes, and enemy density around the player (see **reward.py** and the *reward:*/*enemy_penalty:* sections in the config).

### Running the whole thing (Finally...)
> Note: This is resource heavy and time-intensive on your computer and won't work without the proper resolution/game settings.
```
python train_fixed.py
```
The script:
- Creates ***VampireSurvivorsEnv("config_fixed.yaml")***.
- Wraps it in Stable-Baselines3 utilities (*DummyVecEnv*, *VecTransposeImage*).
- Trains a PPO agent while allowing basic hotkeys in the console (pause/quit via *ConsoleHotkeyCallback*).

Trained models and logs are written to local directories (e.g. *models/*,*tb/*) which are **intentionally not committed** to this repository. 
> (sorry... you gotta work for your personal Vampire Survivors auto-player)

## Future Improvements
As implied before (as well with the sorry state this repository was in before) this was cobbled together for a graduate level class, so there are future ideas I may improve in the future (or some ideas for you if you're feeling *crazy enough*), since I see you silly gooses already cloning this class project :)

- Refactor the environment into a reusable Python package/module.
- Add richer logging and metrics dashboards.
- Allow movement option of *click* for the agent
- Integrate automatic menu traversal (to save the amount of times, I've had to navigate the menu for the next run)
- Add configuration presets for different resolutions.
- Integrate a clean training/evaluation CLI.
