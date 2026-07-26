"""
env_wrapper.py
==============
Phase 5 — Gymnasium environment.

Wraps the Phase 2 surrogate environment model and the Phase 4 personalized
reward function into a standard `gymnasium.Env` so that Stable-Baselines3
(PPO/SAC/...) can train against it.

Env model: GRU only (dual_gru_env_model.DualGRUEnvModel).
    dual_lgbm_env_model_v2.py (LightGBM) is an alternate/experimental
    surrogate from an earlier iteration and is intentionally NOT wired in
    here — GRU is the canonical Phase 2 model used across Phase 5/6/7
    (see training_report.json / meta.json for the trained artifact this
    expects: gru_T.pt, gru_RH.pt, scaler_T.pkl, scaler_RH.pkl, meta.json).

Reward:  hvac_reward_env.reward_from_env_model()  (Phase 4)
Comfort: hvac_reward_env.ComfortModel              (Phase 3, personalized T*/RH*)

Action space: Discrete(8)
    0     -> AC off
    1..7  -> setpoint 24..30 degC (step 1 degC)

Observation space: Box(9,), normalized to [-1, 1]
    [T_indoor, RH_indoor, T_outdoor, RH_outdoor,
     hour_sin, hour_cos, AC_state, GHI, occupancy]

Dynamics: indoor T/RH are simulated by the GRU model in response to the
agent's action. Exogenous variables the agent cannot control (T_outdoor,
RH_outdoor, GHI, occupancy, hour) are replayed from real recorded episodes
in `data_path`, sampled per-apartment so a trajectory never crosses two
different apartments. If no data_path is given (or file missing), a
synthetic diurnal fallback trajectory is used instead — useful for smoke
tests, but NOT a substitute for the real CAMaRSEC-driven environment.

One caveat inherited from Phase 2: the GRU only forecasts at t+30 and
t+60 minutes (no t+15 head). Since the dataset resolution and episode
step are both 15 min, this env advances physical state using the t+30
forecast as the next-step approximation — the closest horizon available.
This is a known simplification of the surrogate, not a bug; keep it in
mind when interpreting short-timescale agent behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from hvac_reward_env import ComfortModel, load_comfort_model, reward_from_env_model

# ── Action space ────────────────────────────────────────────────────────────
N_ACTIONS = 8
SETPOINT_BASE = 24.0     # action=1 -> 24 degC
SETPOINT_STEP = 1.0      # action=k -> SETPOINT_BASE + (k-1)*STEP

STEP_MINUTES = 15
DEFAULT_EPISODE_LENGTH = 96   # 96 * 15min = 24h

# ── Observation normalization ranges ────────────────────────────────────────
# Must stay in sync with phase7_evaluation_update.py's decode_obs()/denorm ranges.
T_MIN, T_MAX = 18.0, 35.0
RH_MIN, RH_MAX = 20.0, 95.0
T_OUT_MIN, T_OUT_MAX = 20.0, 40.0
RH_OUT_MIN, RH_OUT_MAX = 30.0, 95.0
GHI_MIN, GHI_MAX = 0.0, 1200.0

EXOGENOUS_COLS = ["T_Outdoor", "RH_Outdoor", "GHI", "Occupancy_State", "hour_sin", "hour_cos"]
REQUIRED_COLS = EXOGENOUS_COLS + ["T_Indoor", "RH_Indoor", "location_id"]


def _normalize(v: float, lo: float, hi: float) -> float:
    return float(np.clip(2 * (v - lo) / (hi - lo) - 1, -1, 1))


def action_to_setpoint(action: int) -> float:
    """Convert discrete action -> setpoint degC. Action 0 (off) -> 27.0 nominal, for logging only."""
    if action == 0:
        return 27.0
    return SETPOINT_BASE + (action - 1) * SETPOINT_STEP


class HVACRLEnv(gym.Env):
    """Gymnasium environment for personalized HVAC control (GRU surrogate + Bayesian reward)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_model,                                     # DualGRUEnvModel, already loaded (Phase 2)
        comfort_model_path: Optional[Path] = None,      # Phase 3 posterior JSON; None -> hard-coded Run C
        data_path: Optional[Path] = None,                # CSV with real exogenous trajectories
        episode_length: int = DEFAULT_EPISODE_LENGTH,
        apartment_id: Optional[str] = None,               # restrict sampling to one apartment; None = any
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.env_model = env_model
        self.comfort_model: ComfortModel = load_comfort_model(comfort_model_path)
        self.episode_length = episode_length
        self.apartment_id = apartment_id

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._groups: Optional[dict[str, pd.DataFrame]] = self._load_exogenous(data_path)

        # Populated by reset()
        self._traj: Optional[pd.DataFrame] = None
        self._step_idx = 0
        self._T = self._RH = self._T_out = self._RH_out = 0.0
        self._GHI = self._occupancy = self._hour_sin = self._hour_cos = 0.0
        self._ac_state = 0.0
        self._prev_action_norm = 0.0

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_exogenous(self, data_path: Optional[Path]) -> Optional[dict[str, pd.DataFrame]]:
        """
        Load real recorded trajectories, grouped per apartment (location_id),
        using the SAME feature engineering as Phase 2 training
        (dual_gru_env_model.load_and_prepare: hour_sin/cos, GHI fill).
        Returns None if no usable data is found -> caller falls back to the
        synthetic diurnal trajectory.
        """
        if data_path is None or not Path(data_path).exists():
            return None

        from Env_model.dual_gru_env_model import load_and_prepare

        df = load_and_prepare(Path(data_path))
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"data_path missing required columns: {missing}")

        df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
        if df.empty:
            return None

        groups = {
            loc: g.reset_index(drop=True)
            for loc, g in df.groupby("location_id")
            if len(g) >= self.episode_length + 5
        }
        return groups or None

    def _sample_trajectory(self) -> Optional[pd.DataFrame]:
        """Pick one apartment + a random contiguous window of length episode_length+1."""
        if not self._groups:
            return None

        if self.apartment_id is not None:
            candidates = [self.apartment_id] if self.apartment_id in self._groups else []
        else:
            candidates = list(self._groups.keys())
        if not candidates:
            return None

        loc = candidates[int(self._rng.integers(0, len(candidates)))]
        g = self._groups[loc]
        max_start = len(g) - self.episode_length - 1
        start = int(self._rng.integers(0, max_start)) if max_start > 0 else 0
        return g.iloc[start:start + self.episode_length + 1].reset_index(drop=True)

    def _exogenous_row(self, idx: int) -> dict:
        if self._traj is not None and idx < len(self._traj):
            row = self._traj.iloc[idx]
            return {
                "T_Outdoor": float(row["T_Outdoor"]),
                "RH_Outdoor": float(row["RH_Outdoor"]),
                "GHI": float(row["GHI"]),
                "Occupancy_State": float(row["Occupancy_State"]),
                "hour_sin": float(row["hour_sin"]),
                "hour_cos": float(row["hour_cos"]),
            }
        # Synthetic diurnal fallback (used only when no real data is available)
        hour = (idx * STEP_MINUTES / 60.0) % 24
        ghi = 800.0 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0.0
        return {
            "T_Outdoor": 30.0 + 4.0 * np.sin(2 * np.pi * (hour - 15) / 24),
            "RH_Outdoor": 75.0,
            "GHI": float(max(0.0, ghi)),
            "Occupancy_State": 1.0,
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
        }

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        # Required by Gymnasium's env_checker: seeds self.np_random and marks
        # the RNG as initialized. We reuse it as self._rng for sampling.
        super().reset(seed=seed)
        if seed is not None:
            self._rng = self.np_random

        self._step_idx = 0
        self._ac_state = 0.0
        self._prev_action_norm = 0.0

        self._traj = self._sample_trajectory()
        if self._traj is not None:
            row0 = self._traj.iloc[0]
            self._T = float(row0["T_Indoor"])
            self._RH = float(row0["RH_Indoor"])
        else:
            self._T = 30.0 + float(self._rng.normal(0, 1))
            self._RH = 72.0 + float(self._rng.normal(0, 3))

        exo0 = self._exogenous_row(0)
        self._T_out = exo0["T_Outdoor"]
        self._RH_out = exo0["RH_Outdoor"]
        self._GHI = exo0["GHI"]
        self._occupancy = exo0["Occupancy_State"]
        self._hour_sin = exo0["hour_sin"]
        self._hour_cos = exo0["hour_cos"]

        initial_state = {
            "T_Indoor": self._T, "RH_Indoor": self._RH,
            "T_Outdoor": self._T_out, "RH_Outdoor": self._RH_out,
            "AC_State": 0.0, "setpoint_proxy": self._T,
            "GHI": self._GHI, "Occupancy_State": self._occupancy,
            "hour_sin": self._hour_sin, "hour_cos": self._hour_cos,
        }
        self.env_model.reset(initial_state)

        return self._obs(), {}

    def step(self, action: int):
        action = int(action)
        ac_on = 1.0 if action > 0 else 0.0
        setpoint = action_to_setpoint(action)
        action_norm = action / (N_ACTIONS - 1)   # 0..1, same scale as other reward terms

        state = {
            "T_Indoor": self._T, "RH_Indoor": self._RH,
            "T_Outdoor": self._T_out, "RH_Outdoor": self._RH_out,
            "AC_State": ac_on, "setpoint_proxy": setpoint,
            "GHI": self._GHI, "Occupancy_State": self._occupancy,
            "hour_sin": self._hour_sin, "hour_cos": self._hour_cos,
        }

        rc = reward_from_env_model(
            env_model=self.env_model,
            state=state,
            action=action_norm,
            previous_action=self._prev_action_norm,
            occupied=bool(self._occupancy > 0.5),
            comfort_model=self.comfort_model,
        )

        # Advance indoor state using the t+30min GRU forecast (closest horizon
        # to the 15-min env step available from Phase 2 — see module docstring).
        self._T = rc.T_t30
        self._RH = rc.RH_t30
        self._ac_state = ac_on
        self._prev_action_norm = action_norm

        self._step_idx += 1
        exo = self._exogenous_row(self._step_idx)
        self._T_out = exo["T_Outdoor"]
        self._RH_out = exo["RH_Outdoor"]
        self._GHI = exo["GHI"]
        self._occupancy = exo["Occupancy_State"]
        self._hour_sin = exo["hour_sin"]
        self._hour_cos = exo["hour_cos"]

        terminated = self._step_idx >= self.episode_length
        truncated = False
        info = {
            "comfort": rc.comfort_mean,
            "energy": rc.energy_mean,
            "safety": rc.safety_penalty,
            "reward": rc.total_reward,
        }
        return self._obs(), rc.total_reward, terminated, truncated, info

    def _obs(self) -> np.ndarray:
        return np.array([
            _normalize(self._T, T_MIN, T_MAX),
            _normalize(self._RH, RH_MIN, RH_MAX),
            _normalize(self._T_out, T_OUT_MIN, T_OUT_MAX),
            _normalize(self._RH_out, RH_OUT_MIN, RH_OUT_MAX),
            self._hour_sin,
            self._hour_cos,
            self._ac_state,
            _normalize(self._GHI, GHI_MIN, GHI_MAX),
            self._occupancy,
        ], dtype=np.float32)
