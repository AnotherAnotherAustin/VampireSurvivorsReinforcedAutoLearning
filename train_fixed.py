import yaml
import time
import os
import msvcrt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from vs_env_fixed import VampireSurvivorsEnv
from curriculum import set_curriculum


from stable_baselines3.common.callbacks import BaseCallback

class ConsoleHotkeyCallback(BaseCallback):
    """Hotkeys in the training console:
    - Press 'p' to pause/resume (no actions sent while paused)
    - Press 'q' to save and quit safely
    """
    def __init__(self, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self._paused = False

    def _get_base_env(self):
        # VecEnv -> first env -> unwrap
        env = self.training_env.envs[0]
        return env.unwrapped if hasattr(env, "unwrapped") else env

    def _toggle_pause(self):
        self._paused = not self._paused
        env = self._get_base_env()
        if hasattr(env, "set_paused"):
            env.set_paused(self._paused)
        if self.verbose:
            print(f"[hotkeys] {'PAUSED' if self._paused else 'RESUMED'} (press 'p' to toggle, 'q' to quit)")

    def _safe_quit(self):
        # unpause + release keys + save model
        env = self._get_base_env()
        if hasattr(env, "set_paused"):
            env.set_paused(False)
        # Save
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.model.save(self.save_path)
        if self.verbose:
            print(f"[hotkeys] Saved model to: {self.save_path}")
            print("[hotkeys] Stopping training now.")
        return False  # returning False stops training

    def _on_step(self) -> bool:
        # Non-blocking key check
        while msvcrt.kbhit():
            ch = msvcrt.getwch().lower()
            if ch == 'p':
                self._toggle_pause()
            elif ch == 'q':
                return self._safe_quit()

        # If paused, block training loop here until unpaused or quit.
        if self._paused:
            if self.verbose:
                print("[hotkeys] Training paused. Focus the game; press 'p' to resume or 'q' to quit.")
            while True:
                time.sleep(0.1)
                if msvcrt.kbhit():
                    ch = msvcrt.getwch().lower()
                    if ch == 'p':
                        self._toggle_pause()
                        break
                    if ch == 'q':
                        return self._safe_quit()
            return True

        return True


def make_env():
    return VampireSurvivorsEnv("config_fixed.yaml")

def build_env():
    env = DummyVecEnv([make_env])
    env = VecTransposeImage(env)
    return env

if __name__ == "__main__":
    print("Starting training in 5 seconds. Click the game window so it has focus...")
    time.sleep(5)

    phases = [
        (1, 200_000),
        (2, 300_000),
        (3, 500_000),
    ]

    model = None
    for phase, steps in phases:
        cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
        set_curriculum(cfg, phase)
        with open("config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        env = build_env()

        if model is None:
            model = PPO(
                policy="CnnPolicy",
                env=env,
                verbose=1,
                n_steps=2048,
                batch_size=64,
                learning_rate=2.5e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                tensorboard_log="./tb/",
                seed=0,
            )
            hotkeys = ConsoleHotkeyCallback(
            save_path="models/ppo_vs_rl",
            verbose=1
            )

        else:
            model.set_env(env)

        print(f"\n=== PHASE {phase} | {steps} timesteps ===\n")
        model.learn(total_timesteps=steps, callback=hotkeys)
        model.save(f"vs_ppo_phase{phase}.zip")
        env.close()

    model.save("vs_ppo_final.zip")
