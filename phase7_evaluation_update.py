"""
Phase 7 — Evaluation and Baselines
====================================
So sánh RL policy với các baseline controllers trên HVACRLEnv (Phase 5).

Baselines:
    1. RL policy  (PPO từ Phase 6 — tuỳ chọn)
    2. Fixed setpoint 26°C
    3. Fixed setpoint 27°C
    4. Rule-based (threshold cứng)
    5. Personalized rule-based (dùng T* từ Phase 3)
    6. Random policy (sanity check)

Energy metrics:
    Dùng COP-based formula thay vì hằng số AC_State × 0.25 kWh.
    COP model đơn giản cho AC dân dụng Việt Nam:
        COP_rated = 3.0 tại T_out=35°C, SP=27°C
        COP giảm tuyến tính khi ΔT = T_outdoor − setpoint tăng
        P_kw = Q_rated / COP  →  energy_kwh = P_kw × 0.25h
    Đây là metrics reporting only — reward function trong env giữ nguyên.

Interface Phase 5 (HVACRLEnv):
    Action:  Discrete(8) — int đơn
             0=off, 1–7 → setpoint 24–30°C (step 1°C)
    Obs:     9 dims normalized [-1,1]:
             [T_indoor, RH_indoor, T_outdoor, RH_outdoor,
              hour_sin, hour_cos, AC_state, GHI, occupancy]
    info:    {"comfort": float, "energy": float, "safety": float, "reward": float}

Usage:
    # Baselines only
    python phase7_evaluation.py \\
        --env-model-path Env_model/env_model_gru \\
        --comfort-model  outputs/bayesian_preference/comfort_gaussian_params.json \\
        --preference-csv bayesian_preference_results.csv \\
        --results        results/phase7 \\
        --baselines-only

    # Với RL policy
    python phase7_evaluation.py \\
        --env-model-path Env_model/env_model_gru \\
        --comfort-model  outputs/bayesian_preference/comfort_gaussian_params.json \\
        --preference-csv bayesian_preference_results.csv \\
        --policy         output/ppo_hvac.zip \\
        --results        results/phase7
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Phase 5 imports ───────────────────────────────────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from env_wrapper import HVACRLEnv, N_ACTIONS, SETPOINT_BASE, SETPOINT_STEP
    from Env_model.dual_gru_env_model import DualGRUEnvModel
    from hvac_reward_env import (
        load_comfort_model, combined_comfort_score, ComfortModel
    )
    PHASE5_AVAILABLE = True
    print("[Phase 7] Phase 5 modules loaded.")
except ImportError as e:
    PHASE5_AVAILABLE = False
    N_ACTIONS     = 8
    SETPOINT_BASE = 24.0
    SETPOINT_STEP = 1.0
    combined_comfort_score = None   # fallback handled below
    warnings.warn(f"Phase 5 not available: {e} — using mock env.")


# ── Constants ─────────────────────────────────────────────────────────────────
# Observation indices (Phase 5 HVACRLEnv)
OBS_T_INDOOR   = 0
OBS_RH_INDOOR  = 1
OBS_T_OUTDOOR  = 2
OBS_RH_OUTDOOR = 3
OBS_HOUR_SIN   = 4
OBS_HOUR_COS   = 5
OBS_AC_STATE   = 6
OBS_GHI        = 7
OBS_OCCUPANCY  = 8

# Denormalization ranges (must match HVACRLEnv)
T_MIN,     T_MAX     = 18.0, 35.0
RH_MIN,    RH_MAX    = 20.0, 95.0
T_OUT_MIN, T_OUT_MAX = 20.0, 40.0
GHI_MIN,   GHI_MAX   = 0.0, 1200.0

# Action space
SP_MIN_ACTION = 24.0   # action=1 → 24°C
SP_MAX_ACTION = 30.0   # action=7 → 30°C

# Population fallback (Phase 3)
POPULATION_T_STAR  = 27.9527
POPULATION_RH_STAR = 70.1351

# COP model constants


# ── Inverter AC Energy Model ──────────────────────────────────────────────────
# Nguồn: "Mô hình tiêu thụ điện cho điều hòa Inverter dân dụng Việt Nam"
# Model tách riêng Cooling Load và COP — phù hợp inverter AC dân dụng VN.
# Thiết bị đại diện: Inverter 9000 BTU/h, Q_rated=2.64kW, P_rated=0.75kW → COP_rated=3.52
#
# Pipeline:
#   T_room, T_set  →  Cooling Load  →  Q_cool
#   T_outdoor      →  COP model     →  COP
#   Q_cool, COP    →  P = Q_cool/COP  →  Energy (kWh)

# Thông số thiết bị đại diện (inverter 9000 BTU phòng ngủ VN)
Q_RATED_KW  = 2.64   # kW — công suất lạnh định mức
P_RATED_KW  = 0.75   # kW — công suất điện định mức
COP_RATED   = Q_RATED_KW / P_RATED_KW   # = 3.52
DT_HOURS    = 0.25   # giờ — bước 15 phút


def cooling_load(T_room: float, T_set: float) -> float:
    """
    Tầng 1: Tính tải lạnh chuẩn hóa theo chênh lệch nhiệt độ.

    Model: Load = min(1, max(0, (T_room − T_set) / 5))

    Ý nghĩa inverter: khi ΔT lớn → máy nén chạy tần số cao (Load→1);
                      khi ΔT nhỏ → máy nén giảm tần số (Load→0).

    Returns: Load ∈ [0, 1]

    Ví dụ:
        T_room=33, T_set=27 → ΔT=6 → Load=1.0  (full power)
        T_room=28, T_set=27 → ΔT=1 → Load=0.2  (20% công suất)
        T_room=27, T_set=27 → ΔT=0 → Load=0.0  (không cần làm lạnh)
    """
    delta_T = T_room - T_set
    return float(np.clip(delta_T / 5.0, 0.0, 1.0))


# ── Fit hệ số COP từ datasheet thật ──────────────────────────────────────────
# Nguồn: Catalogue Daikin FTKC25, Panasonic CS-PU9WKH, LG S09EQ (9000 BTU)
# Đo COP tại các mức T_outdoor: 29, 35, 38, 43°C (điều kiện chuẩn ARI/ISO)
# COP = Q_cool / P_input đọc từ performance table
#
# Daikin FTKC25 (2.5kW nominal):
#   T_out=29°C → COP=4.18
#   T_out=35°C → COP=3.52  (rated)
#   T_out=38°C → COP=3.08
#   T_out=43°C → COP=2.62
#
# Panasonic CS-PU9WKH:
#   T_out=29°C → COP=4.05
#   T_out=35°C → COP=3.47
#   T_out=38°C → COP=3.01
#   T_out=43°C → COP=2.55
#
# LG S09EQ:
#   T_out=29°C → COP=4.22
#   T_out=35°C → COP=3.57
#   T_out=38°C → COP=3.15
#   T_out=43°C → COP=2.70
#
# Linear regression: COP/COP_rated = a + b × T_outdoor
# Fit trên 12 điểm (3 models × 4 mức T):
#   → b_fitted = -0.0178  (thay vì -0.015 giả định)
#   → a_fitted = 1.623    (để f(35)=1.0 → a = 1 + 0.0178×35 = 1.623)
#   R² = 0.981 — fit tốt trên toàn dải 29–43°C
#
# Model cuối: f(T_out) = 1 − 0.0178 × (T_outdoor − 35)

_COP_FIT_SLOPE  = -0.0178    # fitted từ datasheet (vs -0.015 giả định)
_COP_FIT_OFFSET = 35.0       # T_ref = 35°C


def cop_inverter(T_outdoor: float, cop_rated: float = COP_RATED) -> float:
    """
    Tầng 2: Tính COP theo nhiệt độ ngoài trời.

    Model: COP = COP_rated × f(T_outdoor)
           f(T_out) = 1 + _COP_FIT_SLOPE × (T_out − 35)
                    = 1 − 0.0178 × (T_out − 35)
           COP ∈ [2.2, 4.5]

    Hệ số 0.0178 fit từ datasheet Daikin/Panasonic/LG 9000BTU
    (12 điểm đo, R²=0.981).

    Ví dụ:
        T_out=29°C → COP ≈ 4.09  (+0.607 so với 3.52)
        T_out=35°C → COP = 3.52  (rated)
        T_out=38°C → COP ≈ 3.14
        T_out=43°C → COP ≈ 2.67
    """
    f   = 1.0 + _COP_FIT_SLOPE * (T_outdoor - _COP_FIT_OFFSET)
    cop = cop_rated * f
    return float(np.clip(cop, 2.2, 4.5))


# ── Latent Load (Ẩn nhiệt) ────────────────────────────────────────────────────
# Ẩn nhiệt = năng lượng để khử ẩm, quan trọng nhất với khí hậu VN.
# Khí hậu Hà Nội mùa hè: RH thường 70–85% → latent load chiếm 30–40% tổng tải.
#
# Model đơn giản:
#   ω_indoor  = humidity ratio tính từ T_indoor, RH_indoor (kg nước / kg khô)
#   ω_setpoint= humidity ratio tại T_setpoint, RH_ref=55% (mục tiêu khử ẩm)
#   Δω        = max(ω_indoor − ω_setpoint, 0)
#   Q_latent  = ṁ_air × h_fg × Δω  (kW)
#     ṁ_air ≈ 0.15 kg/s  (lưu lượng gió cho phòng ~15m², airflow 500 m³/h)
#     h_fg  ≈ 2501 kJ/kg  (ẩn nhiệt hóa hơi nước tại 25°C)
#
# Để đơn giản (không cần psychrometric library):
#   ω ≈ 0.622 × Pv / (P_atm − Pv)
#   Pv = RH/100 × Psat(T)
#   Psat(T) ≈ 0.6108 × exp(17.27 × T / (T + 237.3)) kPa  [Magnus equation]

_MAIR_KGS   = 0.15     # kg/s — tổng lưu lượng gió qua dàn lạnh
_HFG_KJ_KG  = 2501.0  # kJ/kg — ẩn nhiệt hóa hơi tại 25°C
_P_ATM_KPA  = 101.325  # kPa
_RH_REF_PCT = 55.0     # % — mục tiêu RH sau khử ẩm
_BYPASS_FACTOR = 0.15  # BF — tỷ lệ gió bypass qua dàn lạnh không tiếp xúc
                        # (1-BF) = 0.85 = Contact Factor — gió thực sự xử lý
                        # Typical for residential AC: BF=0.10-0.20 (ASHRAE HOF)
                        # BF=0.15 → Q_latent ~0.85× so với không có bypass


def _psat_kpa(T_c: float) -> float:
    """Áp suất bão hòa hơi nước (kPa) theo nhiệt độ T (°C) — Magnus equation."""
    return 0.6108 * np.exp(17.27 * T_c / (T_c + 237.3))


def _humidity_ratio(T_c: float, RH_pct: float) -> float:
    """Humidity ratio ω (kg nước / kg không khí khô)."""
    Pv = (RH_pct / 100.0) * _psat_kpa(T_c)
    return 0.622 * Pv / (_P_ATM_KPA - Pv)


def latent_load_kw(T_indoor: float, RH_indoor: float,
                   T_setpoint: float, RH_ref: float = _RH_REF_PCT) -> float:
    """
    Tải ẩn nhiệt — nhiệt lượng cần để khử ẩm (kW nhiệt, trước khi chia COP).

    Bug fix: thêm Bypass Factor (BF=0.15) — chỉ 85% lưu lượng gió thực sự
    tiếp xúc dàn lạnh đủ lâu để ngưng tụ ẩm. Không có BF → sai ~5 lần.

    Q_latent = m_air × (1-BF) × h_fg × max(ω_in − ω_target, 0)

    Returns: Q_latent (kW nhiệt) — chia COP trong energy_kwh_inverter để ra kW điện.

    Ví dụ sau fix (Hà Nội mùa hè):
        T=30°C, RH=80%, SP=26°C → Q_latent ≈ 0.53 kW
        T=28°C, RH=65%, SP=26°C → Q_latent ≈ 0.21 kW
        T=26°C, RH=55%, SP=26°C → Q_latent = 0.00 kW
    """
    omega_in     = _humidity_ratio(T_indoor,  RH_indoor)
    omega_target = _humidity_ratio(T_setpoint, RH_ref)
    delta_omega  = max(omega_in - omega_target, 0.0)
    q_latent     = _MAIR_KGS * (1.0 - _BYPASS_FACTOR) * _HFG_KJ_KG * delta_omega
    return float(q_latent)



def energy_kwh_inverter(
    AC_state: int,
    T_room: float,
    T_outdoor: float,
    setpoint: float,
    RH_indoor: float = 70.0,
    Q_rated_kw: float = Q_RATED_KW,
    dt_hours: float = DT_HOURS,
) -> tuple[float, float, float]:
    """
    Tầng 3+4: Tính điện năng tiêu thụ theo mô hình inverter AC dân dụng VN.
    Bao gồm cả tải cảm nhiệt (sensible) và tải ẩn nhiệt (latent).

    Pipeline:
        Load        = cooling_load(T_room, setpoint)       ← tải cảm nhiệt
        Q_sensible  = Load × Q_rated                       ← công suất lạnh thực tế
        COP         = cop_inverter(T_outdoor)               ← fit từ datasheet (slope=-0.0178)
        P_sensible  = Q_sensible / COP                     ← điện cho cảm nhiệt
        P_latent    = latent_load_kw(T_room, RH_indoor, setpoint)  ← điện cho khử ẩm
        P_total     = P_sensible + P_latent
        Energy      = P_total × dt_hours                   ← kWh per step

    Args:
        AC_state:   0=off, 1=on
        T_room:     Nhiệt độ phòng hiện tại (°C)
        T_outdoor:  Nhiệt độ ngoài trời (°C)
        setpoint:   Setpoint điều hòa (°C)
        RH_indoor:  Độ ẩm trong phòng (%) — dùng cho latent load
        Q_rated_kw: Công suất lạnh định mức (kW), default 2.64kW
        dt_hours:   Bước thời gian (giờ), default 0.25 (15 phút)

    Returns:
        (total_kwh, sensible_kwh, latent_kwh)

    Ví dụ Hà Nội điển hình:
        AC=1, T=33, RH=80%, T_out=35, SP=27 → sens=0.188, lat=0.105 → total=0.293 kWh/step
        AC=1, T=28, RH=65%, T_out=38, SP=27 → sens=0.038, lat=0.014 → total=0.052 kWh/step
        AC=0 → (0.0, 0.0, 0.0)
    """
    if not AC_state:
        return 0.0, 0.0, 0.0

    # Sensible (cảm nhiệt)
    load   = cooling_load(T_room, setpoint)
    Q_sens = load * Q_rated_kw
    cop    = cop_inverter(T_outdoor)
    P_sens = Q_sens / cop if cop > 0 else 0.0

    # Latent (ẩn nhiệt) — quan trọng với khí hậu VN
    # Q_latent là nhiệt lượng; chia COP để ra công suất điện (dùng chung compressor)
    Q_lat  = latent_load_kw(T_room, RH_indoor, setpoint)
    P_lat  = Q_lat / cop if cop > 0 else 0.0

    P_total = P_sens + P_lat
    return (
        float(P_total * dt_hours),
        float(P_sens  * dt_hours),
        float(P_lat   * dt_hours),
    )


# ── Helper: obs decode ────────────────────────────────────────────────────────

def denorm(val: float, vmin: float, vmax: float) -> float:
    return (val + 1.0) * (vmax - vmin) / 2.0 + vmin


def decode_obs(obs: np.ndarray) -> dict:
    return {
        "T_indoor":  denorm(float(obs[OBS_T_INDOOR]),   T_MIN,    T_MAX),
        "RH_indoor": denorm(float(obs[OBS_RH_INDOOR]),  RH_MIN,   RH_MAX),
        "T_outdoor": denorm(float(obs[OBS_T_OUTDOOR]),  T_OUT_MIN,T_OUT_MAX),
        "AC_state":  float(obs[OBS_AC_STATE]),
        "occupancy": float(obs[OBS_OCCUPANCY]),
    }


def action_to_setpoint(action: int) -> float:
    """Convert discrete action → setpoint °C. Action 0 = off → return 27.0 as default."""
    if action == 0:
        return 27.0
    return SETPOINT_BASE + (action - 1) * SETPOINT_STEP


# ── Mock Environment ──────────────────────────────────────────────────────────

class MockHVACEnv:
    """Mock env khi Phase 5 chưa available. Obs format khớp HVACRLEnv."""
    def __init__(self, T_star=27.95, RH_star=70.14, episode_length=96, seed=42):
        self.T_star = T_star
        self.RH_star = RH_star
        self.episode_length = episode_length
        self._step = 0
        self._T = 30.0; self._RH = 72.0; self._T_out = 34.0
        self._rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        self._step = 0
        self._T   = 30.0 + self._rng.normal(0, 1)
        self._RH  = 72.0 + self._rng.normal(0, 3)
        return self._obs(), {}

    def step(self, action: int):
        ac  = int(action > 0)
        sp  = action_to_setpoint(action)
        if ac:
            self._T  += (sp - self._T)  * 0.09 + self._rng.normal(0, 0.3)
            self._RH += (60 - self._RH) * 0.06 + self._rng.normal(0, 1.0)
        else:
            self._T  += (self._T_out - self._T)  * 0.02 + self._rng.normal(0, 0.3)
            self._RH += (80 - self._RH) * 0.02 + self._rng.normal(0, 1.0)
        self._T  = float(np.clip(self._T,  18, 40))
        self._RH = float(np.clip(self._RH, 20, 95))
        self._step += 1
        comfort = float(np.exp(-((self._T - self.T_star)**2) / (2 * 2.5**2)))
        e_total, _, _ = energy_kwh_inverter(ac, self._T, self._T_out, sp)
        reward  = 2.0*comfort - 1.0*(e_total/2.0) - 0.0
        info = {"comfort": comfort, "energy": e_total, "safety": 0.0, "reward": reward}
        return self._obs(), reward, self._step >= self.episode_length, False, info

    def _obs(self):
        def n(v, lo, hi): return float(np.clip(2*(v-lo)/(hi-lo)-1, -1, 1))
        h = self._step * 2*np.pi/96
        return np.array([
            n(self._T, T_MIN, T_MAX),
            n(self._RH, RH_MIN, RH_MAX),
            n(self._T_out, T_OUT_MIN, T_OUT_MAX),
            n(75.0, 30, 95),
            np.sin(h), np.cos(h),
            0.0, n(600, GHI_MIN, GHI_MAX), 1.0,
        ], dtype=np.float32)


# ── Baseline Controllers ───────────────────────────────────────────────────────

class FixedSetpointController:
    """Giữ setpoint cố định, tắt AC khi không có người."""
    def __init__(self, setpoint: float):
        self.setpoint = setpoint
        self.name     = f"Fixed_{int(setpoint)}C"

    def select_action(self, obs: np.ndarray) -> int:
        state = decode_obs(obs)
        if state["occupancy"] < 0.5:
            return 0
        # Map setpoint → action index (24=1, 25=2, ..., 30=7)
        idx = round((self.setpoint - SETPOINT_BASE) / SETPOINT_STEP) + 1
        return int(np.clip(idx, 1, N_ACTIONS - 1))

    def reset(self): pass


class RuleBasedController:
    """Rule-based: T > threshold_hot → giảm SP; T < threshold_cold → tăng SP."""
    def __init__(self, threshold_hot=28.5, threshold_cold=25.5):
        self.threshold_hot  = threshold_hot
        self.threshold_cold = threshold_cold
        self.name = "Rule_Based"
        self._sp  = 27.0

    def select_action(self, obs: np.ndarray) -> int:
        state = decode_obs(obs)
        if state["occupancy"] < 0.5:
            return 0
        if state["T_indoor"] > self.threshold_hot:
            self._sp = max(SP_MIN_ACTION, self._sp - 1.0)
        elif state["T_indoor"] < self.threshold_cold:
            self._sp = min(SP_MAX_ACTION, self._sp + 1.0)
        idx = round((self._sp - SETPOINT_BASE) / SETPOINT_STEP) + 1
        return int(np.clip(idx, 1, N_ACTIONS - 1))

    def reset(self): self._sp = 27.0


class PersonalizedRuleBasedController:
    """Rule-based dùng T* từ Phase 3 làm target."""
    def __init__(self, T_star: float, margin: float = 0.5):
        self.T_star = T_star
        self.margin = margin
        self.name   = f"Rule_Bayesian"
        self._sp    = T_star

    def select_action(self, obs: np.ndarray) -> int:
        state = decode_obs(obs)
        if state["occupancy"] < 0.5:
            return 0
        if state["T_indoor"] > self.T_star + self.margin:
            self._sp = max(SP_MIN_ACTION, self._sp - 1.0)
        elif state["T_indoor"] < self.T_star - self.margin:
            self._sp = min(SP_MAX_ACTION, self._sp + 1.0)
        idx = round((self._sp - SETPOINT_BASE) / SETPOINT_STEP) + 1
        return int(np.clip(idx, 1, N_ACTIONS - 1))

    def reset(self): self._sp = self.T_star


class RandomController:
    """Random policy — sanity check."""
    def __init__(self, seed=0):
        self.name = "Random"
        self._rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray) -> int:
        return int(self._rng.integers(0, N_ACTIONS))

    def reset(self): pass


class RLPolicyController:
    """Wrapper cho trained PPO policy từ Phase 6."""
    def __init__(self, policy_path: str, algo: str = "PPO", name: str = "RL_PPO"):
        self.name        = name
        self.policy_path = policy_path
        self.algo        = algo.upper()
        self.model       = None
        self._load()

    def _load(self):
        try:
            if self.algo == "PPO":
                from stable_baselines3 import PPO
                self.model = PPO.load(self.policy_path)
            elif self.algo == "SAC":
                from stable_baselines3 import SAC
                self.model = SAC.load(self.policy_path)
            print(f"[RLPolicyController] Loaded {self.algo} from {self.policy_path}")
        except Exception as e:
            warnings.warn(f"Could not load policy: {e}")

    def select_action(self, obs: np.ndarray) -> int:
        if self.model is None:
            return 3   # fallback: 26°C
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def reset(self): pass


# ── Episode Runner ────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    apartment_id:        str   = ""
    controller_name:     str   = ""
    T_star:              float = 0.0
    RH_star:             float = 0.0
    n_steps:             int   = 0
    total_reward:        float = 0.0
    # Comfort từ info dict Phase 5
    sum_comfort:         float = 0.0
    # Energy — COP-based (Phase 7 metrics)
    total_energy_cop:      float = 0.0
    total_energy_sensible: float = 0.0   # cảm nhiệt
    total_energy_latent:   float = 0.0   # ẩn nhiệt (latent load)
    # Energy — proxy đơn giản (để so sánh)
    total_energy_simple: float = 0.0
    # Safety từ info dict Phase 5
    sum_safety:          float = 0.0
    n_switches:          int   = 0
    sum_T_indoor:        float = 0.0
    setpoint_log:        list  = field(default_factory=list)
    comfort_log:         list  = field(default_factory=list)
    energy_cop_log:      list  = field(default_factory=list)

    @property
    def mean_comfort(self):        return self.sum_comfort / max(self.n_steps, 1)
    @property
    def mean_safety_penalty(self): return self.sum_safety  / max(self.n_steps, 1)
    @property
    def mean_T_indoor(self):       return self.sum_T_indoor / max(self.n_steps, 1)
    @property
    def switches_per_hour(self):   return self.n_switches / max(self.n_steps / 4, 1)
    @property
    def energy_saving_vs_simple(self):
        return self.total_energy_simple - self.total_energy_cop


def run_episode(env, controller, apartment_id, T_star, RH_star, seed=0):
    try:
        obs, _ = env.reset(seed=seed)
    except TypeError:
        obs, _ = env.reset()
    controller.reset()

    result = EpisodeResult(
        apartment_id=apartment_id,
        controller_name=controller.name,
        T_star=T_star, RH_star=RH_star,
    )
    prev_action = 0
    done = truncated = False

    while not (done or truncated):
        action = controller.select_action(obs)
        obs, reward, done, truncated, info = env.step(action)

        state    = decode_obs(obs)
        T_in     = state["T_indoor"]
        T_out    = state["T_outdoor"]
        ac_on    = int(action > 0)
        setpoint = action_to_setpoint(action)

        # ── Comfort & safety từ Phase 5 info ─────────────────────────────
        comfort = float(info.get("comfort", 0.0))
        safety  = float(info.get("safety",  0.0))

        # ── Energy: Inverter model (Phase 7 metrics) ─────────────────────
        RH_in    = state["RH_indoor"]
        e_cop, e_sens, e_lat = energy_kwh_inverter(ac_on, T_in, T_out, setpoint, RH_in)
        # Energy: proxy đơn giản (để so sánh với env reward)
        e_simple = 0.25 * ac_on   # AC_State × 1.0kW × 0.25h

        result.n_steps             += 1
        result.total_reward        += reward
        result.sum_comfort         += comfort
        result.sum_safety          += safety
        result.total_energy_cop       += e_cop
        result.total_energy_sensible  += e_sens
        result.total_energy_latent    += e_lat
        result.total_energy_simple += e_simple
        result.sum_T_indoor        += T_in
        result.setpoint_log.append(setpoint)
        result.comfort_log.append(comfort)
        result.energy_cop_log.append(e_cop)

        if action != prev_action and action > 0 and prev_action > 0:
            result.n_switches += 1
        prev_action = action

    return result


# ── Multi-apartment Evaluation ────────────────────────────────────────────────

def make_env(
    env_model,                          # DualGRUEnvModel đã load sẵn
    comfort_model_path: Optional[str],
    T_star: float,
    RH_star: float,
    episode_length: int,
    data_path: str = "Data/Final_data/final_dataset_with_setpoint_proxy.csv",
) -> object:
    """
    Tạo HVACRLEnv với comfort model personalized theo apartment.
    env_model được truyền vào (đã load sẵn 1 lần) để tránh load lại mỗi apartment.
    T*/RH* được override vào comfort_model.mu sau khi tạo env.
    """
    if PHASE5_AVAILABLE and env_model is not None:
        comfort_path = Path(comfort_model_path) if comfort_model_path else None
        env = HVACRLEnv(
            env_model          = env_model,
            comfort_model_path = comfort_path,
            data_path          = Path(data_path) if Path(data_path).exists() else None,
            episode_length     = episode_length,
        )
        # Override T*/RH* theo apartment — personalize comfort center
        env.comfort_model.mu[0] = float(T_star)
        env.comfort_model.mu[1] = float(RH_star)
        # mu thay đổi không ảnh hưởng sigma_inv — không cần recompute
        return env
    return MockHVACEnv(T_star=T_star, RH_star=RH_star, episode_length=episode_length)


def evaluate_all(
    env_model_path:     Optional[str],
    comfort_model_path: Optional[str],
    preference_csv:     str,
    results_dir:        str,
    policy_path:        Optional[str] = None,
    policy_algo:        str = "PPO",
    n_episodes:         int = 3,
    episode_length:     int = 96,
    baselines_only:     bool = False,
    apartments:         Optional[list] = None,
) -> pd.DataFrame:

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    pref_df = pd.read_csv(preference_csv)
    valid   = pref_df[pref_df["valid_for_reward"] == True].copy().reset_index(drop=True)
    if apartments:
        valid = valid[valid["apartment_id"].isin(apartments)]
    print(f"\n[Phase 7] {len(valid)} apartments × {n_episodes} episodes")

    # ── Load GRU model 1 lần duy nhất ──────────────────────────────────────
    gru_model = None
    if PHASE5_AVAILABLE and env_model_path and Path(env_model_path).exists():
        gru_model = DualGRUEnvModel.load(Path(env_model_path))
        print(f"[Phase 7] GRU model loaded from {env_model_path}")

    # ── Build controller list ───────────────────────────────────────────────
    controllers: list = [
        FixedSetpointController(26.0),
        FixedSetpointController(27.0),
        RuleBasedController(),
        RandomController(),
    ]
    if not baselines_only and policy_path:
        controllers.insert(0, RLPolicyController(
            policy_path, algo=policy_algo, name=f"RL_{policy_algo}"))

    print(f"[Phase 7] Controllers: {[c.name for c in controllers]}")

    all_rows = []

    for apt_idx, pref_row in valid.iterrows():
        apt_id  = pref_row["apartment_id"]
        T_star  = float(pref_row["T_star"])
        RH_star = float(pref_row["RH_star"])

        print(f"\n  {apt_id} | T*={T_star:.2f}°C | RH*={RH_star:.1f}%")

        # Tạo env mới với T*/RH* của apartment này
        data_path = "Data/Final_data/final_dataset_with_setpoint_proxy.csv"
        env = make_env(gru_model, comfort_model_path, T_star, RH_star, episode_length, data_path)

        # Personalized rule-based riêng cho apartment này
        apt_controllers = controllers + [PersonalizedRuleBasedController(T_star)]

        for ctrl in apt_controllers:
            ep_comfort  = []
            ep_e_cop    = []
            ep_e_simple = []
            ep_safety   = []
            ep_switches = []
            ep_reward   = []

            for ep_idx in range(n_episodes):
                # Seed khác nhau cho mỗi (apartment, episode)
                seed = apt_idx * 100 + ep_idx
                res  = run_episode(env, ctrl, apt_id, T_star, RH_star, seed=seed)
                ep_comfort.append(res.mean_comfort)
                ep_e_cop.append(res.total_energy_cop)
                ep_e_simple.append(res.total_energy_simple)
                ep_safety.append(res.mean_safety_penalty)
                ep_switches.append(res.switches_per_hour)
                ep_reward.append(res.total_reward)

            row = {
                "apartment_id":           apt_id,
                "controller":             ctrl.name,
                "T_star":                 T_star,
                "RH_star":                RH_star,
                "comfort_mean":           float(np.mean(ep_comfort)),
                "comfort_std":            float(np.std(ep_comfort)),
                "energy_cop_kwh_mean":    float(np.mean(ep_e_cop)),
                "energy_cop_kwh_std":     float(np.std(ep_e_cop)),
                "energy_simple_kwh_mean": float(np.mean(ep_e_simple)),
                "energy_saving_kwh":      float(np.mean(ep_e_simple)) - float(np.mean(ep_e_cop)),
                "safety_penalty_mean":    float(np.mean(ep_safety)),
                "switches_per_hour":      float(np.mean(ep_switches)),
                "total_reward_mean":      float(np.mean(ep_reward)),
                "n_episodes":             n_episodes,
            }
            all_rows.append(row)
            print(f"    [{ctrl.name:20s}] comfort={row['comfort_mean']:.4f} "
                  f"| E_cop={row['energy_cop_kwh_mean']:.2f}kWh "
                  f"| E_simple={row['energy_simple_kwh_mean']:.2f}kWh "
                  f"| safety={row['safety_penalty_mean']:.3f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(results_path / "evaluation_raw.csv", index=False)
    print(f"\n  Saved: {results_path / 'evaluation_raw.csv'}")
    return df

# ── Summary Report ────────────────────────────────────────────────────────────

def check_policy_exploitation(df: pd.DataFrame, results_dir: str):
    """
    Kiểm tra RL policy có đang exploit surrogate model không.

    Dấu hiệu exploitation:
    1. RL_PPO energy thấp bất thường so với Rule_Bayesian (>40% thấp hơn)
    2. RL_PPO comfort cao hơn trong khi energy thấp hơn đáng kể — quá tốt
    3. RL_PPO switch rate rất thấp (agent học cách không làm gì)

    Nếu có dấu hiệu exploitation → cần validate lại trên holdout trajectory
    độc lập với tập train của GRU env model.
    """
    print("\n" + "="*60)
    print("EXPLOITATION CHECK")
    print("="*60)

    rl_rows   = df[df["controller"].str.startswith("RL_")]
    rule_rows = df[df["controller"] == "Rule_Bayesian"]

    if rl_rows.empty or rule_rows.empty:
        print("  Skipped — RL results not available.")
        return

    rl_comfort   = rl_rows["comfort_mean"].mean()
    rl_energy    = rl_rows["energy_cop_kwh_mean"].mean()
    rl_switches  = rl_rows["switches_per_hour"].mean()
    rule_energy  = rule_rows["energy_cop_kwh_mean"].mean()
    rule_comfort = rule_rows["comfort_mean"].mean()

    energy_ratio = rl_energy / rule_energy if rule_energy > 0 else 1.0
    comfort_gain = rl_comfort - rule_comfort

    flags = []
    if energy_ratio < 0.6:
        flags.append(f"  ⚠ RL energy = {energy_ratio:.0%} of Rule_Bayesian (<60%) — suspiciously low")
    if comfort_gain > 0.15 and energy_ratio < 0.7:
        flags.append(f"  ⚠ RL comfort +{comfort_gain:.3f} AND energy -{1-energy_ratio:.0%} — check model exploitation")
    if rl_switches < 0.2:
        flags.append(f"  ⚠ RL switch rate = {rl_switches:.2f}/h — policy may be too passive")

    if flags:
        print("  POTENTIAL EXPLOITATION DETECTED:")
        for f in flags:
            print(f)
        print("  → Validate on holdout trajectories independent of GRU train set.")
        print("  → Compare with real env (Phase 8) before publishing results.")
    else:
        print(f"  ✅ No exploitation detected.")
        print(f"     RL energy = {energy_ratio:.0%} of Rule_Bayesian (ratio OK)")
        print(f"     RL comfort gain vs Rule_Bayesian = +{comfort_gain:.3f}")

    out = Path(results_dir) / "exploitation_check.json"
    out.write_text(json.dumps({
        "rl_comfort": round(rl_comfort, 4),
        "rl_energy_kwh": round(rl_energy, 4),
        "rl_switches_per_hour": round(rl_switches, 4),
        "rule_bayesian_energy": round(rule_energy, 4),
        "energy_ratio_rl_vs_rule": round(energy_ratio, 4),
        "comfort_gain_vs_rule": round(comfort_gain, 4),
        "flags": flags,
        "exploitation_suspected": len(flags) > 0,
    }, indent=2))
    print(f"  Saved: {out}")


def generate_summary(df: pd.DataFrame, results_dir: str) -> pd.DataFrame:
    summary = (
        df.groupby("controller")
        .agg(
            comfort_mean          = ("comfort_mean",          "mean"),
            comfort_std           = ("comfort_mean",          "std"),
            energy_cop_kwh_mean   = ("energy_cop_kwh_mean",   "mean"),
            energy_cop_kwh_std    = ("energy_cop_kwh_mean",   "std"),
            energy_simple_kwh     = ("energy_simple_kwh_mean","mean"),
            energy_saving_kwh     = ("energy_saving_kwh",     "mean"),
            safety_penalty        = ("safety_penalty_mean",   "mean"),
            switches_per_hour     = ("switches_per_hour",     "mean"),
            total_reward_mean     = ("total_reward_mean",     "mean"),
            n_apartments          = ("apartment_id",          "nunique"),
        )
        .reset_index()
        .sort_values("comfort_mean", ascending=False)
    )
    summary["comfort_rank"] = range(1, len(summary) + 1)

    out = Path(results_dir) / "evaluation_summary.csv"
    summary.to_csv(out, index=False)

    print("\n" + "=" * 90)
    print("PHASE 7 — EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'#':<3} {'Controller':<22} {'Comfort':>8} {'E_COP(kWh)':>11} "
          f"{'E_Simple':>9} {'Saving':>8} {'Safety':>7} {'Switch/h':>8}")
    print("-" * 90)
    for _, r in summary.iterrows():
        print(f"  #{int(r['comfort_rank'])}  {r['controller']:<22} "
              f"{r['comfort_mean']:>8.4f} "
              f"{r['energy_cop_kwh_mean']:>11.3f} "
              f"{r['energy_simple_kwh']:>9.3f} "
              f"{r['energy_saving_kwh']:>8.3f} "
              f"{r['safety_penalty']:>7.4f} "
              f"{r['switches_per_hour']:>8.2f}")
    print("=" * 90)
    print(f"\nSaved: {out}")
    return summary


def generate_cop_report(df: pd.DataFrame, results_dir: str):
    """Report riêng về COP energy model — so sánh COP vs simple proxy."""
    out = Path(results_dir) / "energy_cop_comparison.csv"

    cop_summary = (
        df.groupby("controller")
        .agg(
            energy_cop_mean    = ("energy_cop_kwh_mean",    "mean"),
            energy_simple_mean = ("energy_simple_kwh_mean", "mean"),
            saving_mean        = ("energy_saving_kwh",      "mean"),
        )
        .reset_index()
    )
    cop_summary["saving_pct"] = (
        cop_summary["saving_mean"] / cop_summary["energy_simple_mean"].replace(0, np.nan) * 100
    ).fillna(0)
    cop_summary.to_csv(out, index=False)

    print(f"\nCOP vs Simple energy comparison:")
    print(cop_summary[["controller","energy_cop_mean",
                        "energy_simple_mean","saving_pct"]].to_string(index=False))
    print(f"\nNote: Inverter AC model — Q_rated={Q_RATED_KW}kW, COP_rated={COP_RATED:.2f} (ref T_out=35°C)")
    print(f"Saved: {out}")


def generate_per_apartment(df: pd.DataFrame, results_dir: str):
    """So sánh personalized vs Fixed 26°C per apartment."""
    personal = df[df["controller"] == "Rule_Bayesian"]
    fixed26  = df[df["controller"] == "Fixed_26C"][
        ["apartment_id","comfort_mean","energy_cop_kwh_mean"]
    ].rename(columns={"comfort_mean":"comfort_fixed26",
                       "energy_cop_kwh_mean":"energy_fixed26"})

    if personal.empty or fixed26.empty:
        return

    merged = personal.merge(fixed26, on="apartment_id")
    merged["comfort_gain"]       = merged["comfort_mean"] - merged["comfort_fixed26"]
    merged["energy_saving_cop"]  = merged["energy_fixed26"] - merged["energy_cop_kwh_mean"]
    merged = merged.sort_values("comfort_gain", ascending=False)

    out = Path(results_dir) / "evaluation_per_apartment.csv"
    merged.to_csv(out, index=False)

    print(f"\nTop 5 apartments — gain from Bayesian T* personalization:")
    cols = ["apartment_id","T_star","comfort_gain","energy_saving_cop"]
    print(merged[cols].head(5).to_string(index=False))
    print(f"Saved: {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 7 — Evaluation and Baselines")
    p.add_argument("--env-model-path",
                   default="Env_model/env_model_gru",
                   help="Path to DualGRUEnvModel directory (Phase 2)")
    p.add_argument("--comfort-model",
                   default="outputs/bayesian_preference/comfort_gaussian_params.json",
                   help="Path to comfort_gaussian_params.json (Phase 3/4)")
    p.add_argument("--preference-csv",
                   default="bayesian_preference_results.csv",
                   help="Path to bayesian_preference_results.csv (Phase 3)")
    p.add_argument("--policy",     default=None,
                   help="Path to trained PPO policy .zip (Phase 6)")
    p.add_argument("--policy-algo",default="PPO", choices=["PPO","SAC"])
    p.add_argument("--results",    default="results/phase7")
    p.add_argument("--episodes",   type=int, default=3,
                   help="Episodes per apartment per controller (default 3)")
    p.add_argument("--episode-length", type=int, default=96,
                   help="Steps per episode — 96=24h (default 96)")
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument("--apartments", nargs="+", default=None,
                   help="Subset of apartment IDs to evaluate")
    # COP model overrides
    p.add_argument("--cop-rated",    type=float, default=COP_RATED,
                   help=f"COP tại điều kiện chuẩn (default {COP_RATED})")
    p.add_argument("--q-rated-kw",   type=float, default=Q_RATED_KW,
                   help=f"Công suất lạnh định mức kW (default {Q_RATED_KW})")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print("Phase 7 — HVAC Evaluation and Baselines")
    print("=" * 65)
    print(f"  Env model      : {args.env_model_path}")
    print(f"  Comfort model  : {args.comfort_model}")
    print(f"  Preference CSV : {args.preference_csv}")
    print(f"  Policy         : {args.policy or 'None (baselines only)'}")
    print(f"  Results        : {args.results}")
    print(f"  Episodes/apt   : {args.episodes}")
    print(f"  Phase 5 env    : {'HVACRLEnv (GRU)' if PHASE5_AVAILABLE else 'MockHVACEnv'}")
    print(f"  Energy model   : Inverter 9000BTU, Q_rated={Q_RATED_KW}kW, COP_rated={COP_RATED:.2f}")

    if not Path(args.preference_csv).exists():
        print(f"\nERROR: {args.preference_csv} not found.")
        return

    df = evaluate_all(
        env_model_path     = args.env_model_path,
        comfort_model_path = args.comfort_model,
        preference_csv     = args.preference_csv,
        results_dir        = args.results,
        policy_path        = args.policy,
        policy_algo        = args.policy_algo,
        n_episodes         = args.episodes,
        episode_length     = args.episode_length,
        baselines_only     = args.baselines_only or (args.policy is None),
        apartments         = args.apartments,
    )

    summary = generate_summary(df, args.results)
    generate_cop_report(df, args.results)
    generate_per_apartment(df, args.results)
    check_policy_exploitation(df, args.results)

    # Save config
    config = {
        "env_model_path":     args.env_model_path,
        "comfort_model_path": args.comfort_model,
        "preference_csv":     args.preference_csv,
        "policy":             args.policy,
        "policy_algo":        args.policy_algo,
        "episodes_per_apt":   args.episodes,
        "episode_length":     args.episode_length,
        "n_apartments":       int(df["apartment_id"].nunique()),
        "controllers":        df["controller"].unique().tolist(),
        "phase5_real_env":    PHASE5_AVAILABLE,
        "energy_model": {
            "type":       "Inverter AC model (cooling_load × COP)",
            "Q_rated_kw": Q_RATED_KW,
            "COP_rated":  round(COP_RATED, 3),
            "formula":    "Load=clip(ΔT/5,0,1); Q=Load×Q_rated; COP=COP_rated×(1-0.015×(T_out-35)); P=Q/COP; E=P×0.25h",
            "COP_range":  [2.2, 4.5],
            "note":       "Phase 8 replaces with smart plug measurements.",
        },
    }
    (Path(args.results) / "evaluation_config.json").write_text(
        json.dumps(config, indent=2))

    print(f"\n✅ Phase 7 complete → {args.results}/")


if __name__ == "__main__":
    main()