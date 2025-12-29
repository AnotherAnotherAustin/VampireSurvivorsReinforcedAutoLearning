import time
import yaml
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque

from capture_fixed import MonitorCapture, preprocess
from controls import KeyController
from reward import crop, bar_fill_ratio, TemplateMatcher
from vision import PlayerTracker, enemy_density_ring

class VampireSurvivorsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config_path="config.yaml"):
        super().__init__()
        cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
        self.cfg = cfg

        self.obs_w = int(cfg["obs_width"])
        self.obs_h = int(cfg["obs_height"])
        self.stack_n = int(cfg["frame_stack"])
        self.fps = float(cfg["fps"])
        self.dt = 1.0 / self.fps
        self.action_repeat = int(cfg["action_repeat"])

        keys = cfg["keys"]
        self.controller = KeyController(keys["up"], keys["down"], keys["left"], keys["right"])
        self.cap = MonitorCapture(int(cfg.get("monitor_index", 1)))

        self.roi_xp = cfg["roi"]["xp_bar"]
        self.roi_hp = cfg["roi"]["hp_bar"]

        self.game_over_matcher = TemplateMatcher(cfg["templates"]["game_over"], threshold=0.75)

        # NEW: gameplay HUD matcher for reset waiting
        self.hud_matcher = TemplateMatcher(
            cfg["templates"]["hud"],
            threshold=float(cfg["reset_wait"]["hud_threshold"])
        )
        self.reset_max_seconds = float(cfg["reset_wait"]["max_seconds"])
        self.reset_check_fps = float(cfg["reset_wait"]["check_fps"])

        vcfg = cfg["vision"]
        self.player = PlayerTracker(
            cfg["templates"]["player"],
            threshold=float(vcfg["player_match_threshold"]),
            search_radius=int(vcfg["search_radius"]),
        )
        self.prev_player_xy = None

        ep = cfg["enemy_penalty"]
        self.r_in = int(ep["ring_inner"])
        self.r_out = int(ep["ring_outer"])
        self.enemy_w = float(ep["density_weight"])

        ip = cfg["idle_penalty"]
        self.idle_speed_thr = float(ip["speed_px_threshold"])
        self.idle_w = float(ip["weight"])

        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.obs_h, self.obs_w, self.stack_n), dtype=np.uint8
        )

        self.frames = deque(maxlen=self.stack_n)
        self.steps = 0
        self.paused = False  # toggled by hotkeys
        self.prev_xp = None
        self.prev_hp = None

        # Reward config (safe defaults)
        rcfg = cfg.get('reward', {})
        self.time_reward = float(rcfg.get('time_reward', 0.01))
        self.hp_loss_scale = float(rcfg.get('hp_loss_scale', 5.0))
        self.max_neg = float(rcfg.get('max_negative_per_step', 1.0))
        self.max_pos = float(rcfg.get('max_positive_per_step', 1.0))
        self.idle_when_unknown_penalty = float(rcfg.get('idle_when_unknown_penalty', 0.0))

    def _action_to_keys(self, a: int):
        if a == 0: return []
        if a == 1: return [self.controller.up]
        if a == 2: return [self.controller.down]
        if a == 3: return [self.controller.left]
        if a == 4: return [self.controller.right]
        if a == 5: return [self.controller.up, self.controller.left]
        if a == 6: return [self.controller.up, self.controller.right]
        if a == 7: return [self.controller.down, self.controller.left]
        if a == 8: return [self.controller.down, self.controller.right]
        return []

    def _grab_frame(self):
        return self.cap.grab()

    def _get_obs(self, frame_bgr):
        proc = preprocess(frame_bgr, self.obs_w, self.obs_h)
        if len(self.frames) == 0:
            for _ in range(self.stack_n):
                self.frames.append(proc)
        else:
            self.frames.append(proc)
        return np.stack(list(self.frames), axis=-1)

    def _compute_signals(self, frame_bgr):
        xp = bar_fill_ratio(crop(frame_bgr, self.roi_xp))
        hp = bar_fill_ratio(crop(frame_bgr, self.roi_hp))
        return xp, hp

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.controller.release_all()
        self.frames.clear()
        self.steps = 0
        self.paused = False  # toggled by hotkeys
        self.prev_xp = None
        self.prev_hp = None

        # Reward config (safe defaults)
        rcfg = self.cfg.get('reward', {})
        self.time_reward = float(rcfg.get('time_reward', 0.01))
        self.hp_loss_scale = float(rcfg.get('hp_loss_scale', 5.0))
        self.max_neg = float(rcfg.get('max_negative_per_step', 1.0))
        self.max_pos = float(rcfg.get('max_positive_per_step', 1.0))
        self.idle_when_unknown_penalty = float(rcfg.get('idle_when_unknown_penalty', 0.0))
        self.prev_player_xy = None
        self.player.last_xy = None

        # Wait until you're actually in gameplay (HUD visible)
        deadline = time.time() + self.reset_max_seconds
        last_frame = None
        while True:
            frame = self._grab_frame()
            last_frame = frame

            if self.hud_matcher.matches(frame):
                obs = self._get_obs(frame)
                xp, hp = self._compute_signals(frame)
                self.prev_xp, self.prev_hp = xp, hp

                cx, cy, _ = self.player.locate(frame)
                if cx is not None:
                    self.prev_player_xy = (cx, cy)

                return obs, {}

            # If we timed out, optionally continue anyway with the last captured frame.
            if time.time() > deadline:
                allow = bool(self.cfg.get("reset_wait", {}).get("allow_timeout_start", False))
                if allow and last_frame is not None:
                    obs = self._get_obs(last_frame)
                    xp, hp = self._compute_signals(last_frame)
                    self.prev_xp, self.prev_hp = xp, hp
                    return obs, {"reset_timeout": True}
                # Otherwise, keep waiting (likely in menu). Print a hint occasionally.
                if int(time.time()) % 5 == 0:
                    print('[vs_env] Waiting for gameplay HUD... start a run in-game (Alt-Tab back and click).')
                deadline = time.time() + self.reset_max_seconds

            time.sleep(1.0 / self.reset_check_fps)

    
    def set_paused(self, paused: bool):
        """Pause/unpause environment actions (used by training hotkeys)."""
        self.paused = bool(paused)
        if self.paused:
            # ensure no stuck keys while paused
            self.controller.release_all()

    def step(self, action):
        self.steps += 1

        # If paused, do not send actions; just return the latest observation.
        if self.paused:
            self.controller.release_all()
            frame = self._grab_frame()
            obs = self._get_obs(frame)
            time.sleep(self.dt)
            return obs, 0.0, False, False, {"paused": True, "steps": self.steps}

        self.controller.hold(self._action_to_keys(int(action)))

        total_reward = 0.0
        terminated = False
        truncated = False
        obs = None

        for _ in range(self.action_repeat):
            time.sleep(self.dt)
            frame = self._grab_frame()
            obs = self._get_obs(frame)

            if self.game_over_matcher.matches(frame):
                terminated = True
                total_reward -= 25.0
                break

            xp, hp = self._compute_signals(frame)
            total_reward += self.time_reward

            if self.prev_xp is not None:
                xp_scale = float(self.cfg.get("_xp_scale", 2.0))
                total_reward += xp_scale * (xp - self.prev_xp)

            if self.prev_hp is not None:
                dhp = hp - self.prev_hp
                if dhp < 0:
                    total_reward += self.hp_loss_scale * dhp

            self.prev_xp, self.prev_hp = xp, hp

            cx, cy, _ = self.player.locate(frame)
            if cx is not None:
                dens = enemy_density_ring(frame, cx, cy, self.r_in, self.r_out)
                total_reward -= self.enemy_w * dens

                if self.prev_player_xy is not None:
                    px, py = self.prev_player_xy
                    speed = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    if speed < self.idle_speed_thr:
                        total_reward -= self.idle_w
                self.prev_player_xy = (cx, cy)
            else:
                # Vision can fail due to effects/animations. Don't punish too hard for missing detection.
                total_reward -= self.idle_when_unknown_penalty
                self.prev_player_xy = None

        # Clip reward for training stability
        if total_reward > self.max_pos:
            total_reward = self.max_pos
        if total_reward < -self.max_neg:
            total_reward = -self.max_neg

        return obs, float(total_reward), terminated, truncated, {"steps": self.steps, "terminated": terminated}

    def close(self):
        self.controller.release_all()
