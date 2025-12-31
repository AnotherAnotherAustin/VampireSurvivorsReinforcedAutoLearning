# Vampire Survivors Reinforced Auto Learning

This project implements a reinforcement learning agent that plays the game
**Vampire Survivors** by reading pixels from the game screen and issuing
actions in real time. It combines Python, OpenCV, and Gymnasium-like
interfaces with custom reward shaping and automation tooling.

## Features

- Screen-capture based environment (no game source modifications required)
- Real-time control loop for capturing frames and sending actions
- Configurable rewards via `config_fixed.yaml` and `reward.py`
- Logging and basic evaluation tools to measure performance over time

## Tech Stack

- Python
- OpenCV
- MSS (for screen capture)
- Gymnasium-style environment wrappers

## Project Structure

```text
VampireSurvivorsReinforcedAutoLearning/
├── capture_fixed.py       # Main script for screen capture and interaction loop
├── reward.py              # Reward function definitions and shaping logic
├── config_fixed.yaml      # Configuration for capture regions, keys, and parameters
├── requirements.txt       # Python dependencies
├── templates/             # (Optional) helper configs/templates
└── README.md              # Project overview and usage
