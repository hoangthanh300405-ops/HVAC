"""
hvac_reward.py
==============
Predictive reward function for the HVAC RL agent.

Horizons: t+30 and t+60 minutes — aligned with add_set_point.py which
uses window +2 (30 min) and +4 (60 min) at 15-min resolution.
t+45 is dropped for consistency and simplicity.

Comfort model: Bayesian 2D Gaussian (μ, Σ) from Bayesian Run C,
supplemented by a Heat Index Gaussian term.

Energy: explicit proxy — AC_State × P_RATED_KW × Δt.
No learned energy model; no power meter in CAMaRSEC.
Documented limitation; Phase 8 will replace with smart plug data.

Occupant preferences (T*, RH*): inferred from behavioral signals.
Because explicit comfort ratings are unavailable in the CAMaRSEC dataset,
the Bayesian model infers preferred thermal conditions from AC stable-operation
plateaus — defined as the 30–60 minute window after AC activation during which
indoor temperature stabilises. These plateau observations serve as weak
pseudo-labels of occupant-preferred conditions.

Posterior (Bayesian Run, Hanoi dataset):
  T*   = 28.86 °C   RH*  = 57.52 %
  σT   = 2.55 °C    σRH  = 12.78 %    ρ = 0.095
  N    = 4,012 plateau observations across 41 Hanoi apartments
  Prior: southern-China comfort dataset (informative prior)

Interface
---------
The reward function expects the environment model to implement:

    env_model.predict(state: dict) -> dict
        Keys in output: T_Indoor_t30, T_Indoor_t60,
                        RH_Indoor_t30, RH_Indoor_t60,
                        energy_proxy_t30_kWh, energy_proxy_t60_kWh

This contract is satisfied by DualLGBMEnvModel in dual_lgbm_env_model.py.
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZONS_MINUTES: tuple[int, ...] = (30, 60)
HORIZON_LABELS: tuple[str, ...] = ("t30", "t60")
ALPHA_WEIGHTS: tuple[float, ...] = (0.5, 0.5)  # equal weight per horizon

P_RATED_KW: float = 1.0   # nominal rated power; used for energy normalisation
P_MAX_KW: float = 1.5     # clip ceiling for normalisation

# Comfort component mixture
W_2D: float = 0.8    # Gaussian 2D (learned covariance from Bayesian)
W_HI: float = 0.2    # Heat Index Gaussian (engineering heuristic)

# Reward weights
W_COMFORT: float = 3.0   # tăng comfort
W_ENERGY:  float = 0.8   # không để quá thấp
W_STABILITY: float = 0.3   # small; stability does not distinguish good vs bad change
W_SWITCH: float = 0.5

# Stability tolerances (per 15-min step transition)
STABILITY_TOL_T: float = 0.5    # °C
STABILITY_TOL_RH: float = 3.0   # %

# Safety hard penalty
W_SAFETY: float = 50.0
SAFE_T_MIN: float = 20.0
SAFE_T_MAX: float = 32.0
SAFE_RH_MAX: float = 85.0
MAX_SWITCHES_PER_HOUR: int = 6


# ---------------------------------------------------------------------------
# Comfort model parameters
# ---------------------------------------------------------------------------

@dataclass
class ComfortModel:
    """
    Bayesian 2D Gaussian comfort parameters from Bayesian Run C.

    μ  = [T*, RH*]  posterior mean of preferred temperature / humidity.
    Σ  = 2×2 covariance matrix encoding uncertainty and T-RH correlation.
    Σ_inv = precomputed inverse for fast Mahalanobis distance.
    HI_star = Heat Index at μ (°C).
    sigma_HI = tolerance for HI Gaussian term (hand-tuned, ablate in paper).
    """
    mu: np.ndarray         # shape (2,)  [T*, RH*]
    sigma_mat: np.ndarray  # shape (2,2)
    sigma_inv: np.ndarray  # shape (2,2)
    HI_star: float = 26.50
    sigma_HI: float = 3.0

    @classmethod
    def from_json(cls, path: Path) -> "ComfortModel":
        """Load parameters from the Bayesian output JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        mu = np.array([data["T_star"], data["RH_star"]], dtype=float)
        sigma_mat = np.array(data["sigma_matrix"], dtype=float)
        sigma_inv = np.linalg.inv(sigma_mat)
        obj = cls(mu=mu, sigma_mat=sigma_mat, sigma_inv=sigma_inv)
        if "HI_star" in data:
            obj.HI_star = float(data["HI_star"])
        if "sigma_HI" in data:
            obj.sigma_HI = float(data["sigma_HI"])
        return obj

    @classmethod
    def from_bayesian_run_c(cls) -> "ComfortModel":
        """
        Posterior from Bayesian Run — Hanoi CAMaRSEC dataset.

        μ derived from 4,012 AC stable-operation plateau observations
        (30–60 min window after AC activation) across 41 Hanoi apartments
        (47 BR files total; 6 low-AC APs excluded, AP41/42 absent).
        Plateau indoor temperature / humidity serve as weak pseudo-labels
        of occupant-preferred conditions (behavioral inference; no explicit
        comfort ratings available in CAMaRSEC).

        Southern-China comfort dataset used as informative prior.

        Parameters
        ----------
        T*   = 28.86 °C  (mean, plateau obs); median = 29.00 °C
        RH*  = 57.52 %   (mean, plateau obs); median = 57.30 %
        σT   = 2.55 °C     σRH  = 12.78 %    ρ = 0.095

        Σ reconstructed from (σT, σRH, ρ) as:
            Σ = [[σT²,        ρ·σT·σRH ],
                 [ρ·σT·σRH,  σRH²      ]]

        Σ⁻¹ computed analytically; verified S_2D(μ) = 1.0.

        HI* = NWS Heat Index evaluated at μ = (28.86 °C, 57.52 %).
        Note: σT, σRH, ρ from report CI95. Update full Σ when available.
        """
        mu = np.array([28.86, 57.52])

        # Σ from report posterior marginals and correlation
        sigma_T, sigma_RH, rho = 2.55, 12.78, 0.095
        sigma_mat = np.array([
            [sigma_T ** 2,               rho * sigma_T * sigma_RH],
            [rho * sigma_T * sigma_RH,   sigma_RH ** 2            ],
        ])
        sigma_inv = np.linalg.inv(sigma_mat)

        # HI* at real posterior mean from CAMaRSEC plateau data
        HI_star = nws_heat_index_celsius(28.86, 57.52)  # ≈ 30.44 °C

        return cls(mu=mu, sigma_mat=sigma_mat, sigma_inv=sigma_inv, HI_star=HI_star)


def load_comfort_model(json_path: Path | None = None) -> ComfortModel:
    """Load from JSON if available, else fall back to hard-coded Run C values."""
    if json_path is not None and Path(json_path).exists():
        print(f"  [comfort] Loaded from {json_path}")
        return ComfortModel.from_json(Path(json_path))
    print("  [comfort] Using hard-coded Bayesian Run C posterior.")
    return ComfortModel.from_bayesian_run_c()


# ---------------------------------------------------------------------------
# Heat Index (NWS Rothfusz)
# ---------------------------------------------------------------------------

def nws_heat_index_celsius(T_c: float, RH: float) -> float:
    """
    Compute NWS Heat Index.

    Uses simple Steadman formula first; upgrades to Rothfusz regression
    above ~80°F. Returns result in °C.

    Note: Heat Index is designed for outdoor apparent heat. Using it
    indoors is an engineering heuristic, not an indoor comfort standard.
    Ablate weight W_HI in sensitivity analysis.
    """
    T_f = T_c * 9 / 5 + 32
    # Simple Steadman
    hi_simple = 0.5 * (T_f + 61.0 + (T_f - 68.0) * 1.2 + RH * 0.094)
    if (hi_simple + T_f) / 2 < 80:
        return (hi_simple - 32) * 5 / 9

    # Rothfusz polynomial
    hi_f = (
        -42.379
        + 2.04901523 * T_f
        + 10.14333127 * RH
        - 0.22475541 * T_f * RH
        - 0.00683783 * T_f ** 2
        - 0.05481717 * RH ** 2
        + 0.00122874 * T_f ** 2 * RH
        + 0.00085282 * T_f * RH ** 2
        - 0.00000199 * T_f ** 2 * RH ** 2
    )
    # Humidity adjustments
    if RH < 13 and 80 < T_f < 112:
        hi_f -= ((13 - RH) / 4) * math.sqrt((17 - abs(T_f - 95)) / 17)
    elif RH > 85 and 80 < T_f < 87:
        hi_f += ((RH - 85) / 10) * ((87 - T_f) / 5)

    return (hi_f - 32) * 5 / 9


# ---------------------------------------------------------------------------
# Comfort score per horizon
# ---------------------------------------------------------------------------

def gaussian_2d_comfort_score(T: float, RH: float, model: ComfortModel) -> float:
    """
    Unnormalised Mahalanobis kernel from Bayesian posterior.

      S_2D(x) = exp[-½ (x−μ)ᵀ Σ⁻¹ (x−μ)]

    Preserves learned covariance and T-RH correlation ρ.
    Two independent 1D Gaussians would assume ρ=0 and miss the comfort ellipse.
    """
    x = np.array([T, RH]) - model.mu
    d2 = float(x @ model.sigma_inv @ x)
    return float(np.exp(-0.5 * d2))


def heat_index_comfort_score(T: float, RH: float, model: ComfortModel) -> float:
    """
    Gaussian around HI* (Heat Index at preferred state μ).

      S_HI = exp[-½ ((HI - HI*) / σ_HI)²]
    """
    hi = nws_heat_index_celsius(T, RH)
    return float(np.exp(-0.5 * ((hi - model.HI_star) / model.sigma_HI) ** 2))


def combined_comfort_score(T: float, RH: float, model: ComfortModel) -> float:
    """
    Combined comfort: 0.8 × S_2D  +  0.2 × S_HI  ∈ (0, 1].

    Both components equal 1 at μ, so combined score = 1 at μ.
    Mixture weights (0.8/0.2) are hand-tuned; ablation recommended.
    """
    s2d = gaussian_2d_comfort_score(T, RH, model)
    shi = heat_index_comfort_score(T, RH, model)
    return W_2D * s2d + W_HI * shi


# ---------------------------------------------------------------------------
# Environment model protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EnvModel(Protocol):
    """
    Contract that any surrogate environment model must satisfy.

    predict(state) must return a dict with at minimum:
        T_Indoor_t30, T_Indoor_t60
        RH_Indoor_t30, RH_Indoor_t60
        energy_proxy_t30_kWh, energy_proxy_t60_kWh
    """
    def predict(self, state: dict) -> dict: ...


# ---------------------------------------------------------------------------
# Predictive reward components
# ---------------------------------------------------------------------------

@dataclass
class RewardComponents:
    """All intermediate terms for logging and ablation studies."""
    # Per-horizon comfort
    comfort_t30: float = 0.0
    comfort_t60: float = 0.0
    comfort_mean: float = 0.0

    # Per-horizon energy (normalised)
    energy_norm_t30: float = 0.0
    energy_norm_t60: float = 0.0
    energy_mean: float = 0.0

    # Stability across transitions
    stability_t30: float = 0.0
    stability_t60: float = 0.0
    stability_mean: float = 0.0

    # Switching penalty
    switch_penalty: float = 0.0

    # Safety
    safety_penalty: float = 0.0

    # Occupancy gate
    occupied: bool = True

    # Final scalar
    total_reward: float = 0.0

    # Forecasts (for logging)
    T_t30: float = float("nan")
    T_t60: float = float("nan")
    RH_t30: float = float("nan")
    RH_t60: float = float("nan")

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _stability_score(dT: float, dRH: float) -> float:
    """
    Penalise large step-to-step changes in the forecast trajectory.
    Does NOT distinguish beneficial from harmful changes — only magnitude.
    Therefore given low weight and occupancy-gated.
    """
    return float(np.exp(-0.5 * ((dT / STABILITY_TOL_T) ** 2 + (dRH / STABILITY_TOL_RH) ** 2)))


def _safety_violations(forecasts: dict, current_T: float, current_RH: float) -> float:
    """Return 1.0 if any hard safety bound is violated in current or forecast state."""
    temps = [current_T, forecasts["T_Indoor_t30"], forecasts["T_Indoor_t60"]]
    rhs = [current_RH, forecasts["RH_Indoor_t30"], forecasts["RH_Indoor_t60"]]
    violations = (
        any(t < SAFE_T_MIN or t > SAFE_T_MAX for t in temps)
        or any(rh > SAFE_RH_MAX for rh in rhs)
    )
    return 1.0 if violations else 0.0


def reward_components(
    forecasts: dict,
    current_T: float,
    current_RH: float,
    action: float,
    previous_action: float,
    occupied: bool,
    comfort_model: ComfortModel,
    p_max_kw: float = P_MAX_KW,
) -> RewardComponents:
    """
    Compute all reward terms from pre-computed forecasts.

    Parameters
    ----------
    forecasts : output of env_model.predict(state)
    current_T, current_RH : state at time t (used as trajectory start)
    action : current HVAC setpoint or on/off in [0,1]
    previous_action : action at t−1
    occupied : whether the room is currently occupied
    comfort_model : ComfortModel instance
    p_max_kw : ceiling for energy normalisation

    Returns
    -------
    RewardComponents dataclass with all terms filled.
    """
    rc = RewardComponents()
    rc.occupied = occupied

    # ── Extract forecasts ───────────────────────────────────────────────────
    T30 = float(forecasts["T_Indoor_t30"])
    T60 = float(forecasts["T_Indoor_t60"])
    RH30 = float(forecasts["RH_Indoor_t30"])
    RH60 = float(forecasts["RH_Indoor_t60"])
    E30 = float(forecasts.get("energy_proxy_t30_kWh", action * P_RATED_KW * 0.5))
    E60 = float(forecasts.get("energy_proxy_t60_kWh", action * P_RATED_KW * 1.0))

    rc.T_t30, rc.T_t60 = T30, T60
    rc.RH_t30, rc.RH_t60 = RH30, RH60

    # ── Comfort at each horizon ─────────────────────────────────────────────
    rc.comfort_t30 = combined_comfort_score(T30, RH30, comfort_model)
    rc.comfort_t60 = combined_comfort_score(T60, RH60, comfort_model)
    rc.comfort_mean = sum(
        a * c for a, c in zip(ALPHA_WEIGHTS, [rc.comfort_t30, rc.comfort_t60])
    )
    # Gate comfort by occupancy (no comfort penalty for empty room)
    comfort_occupied = rc.comfort_mean * (1.0 if occupied else 0.0)

    # ── Energy at each horizon ──────────────────────────────────────────────
    # Normalise interval energy against (p_max × Δt per step × n_steps_to_horizon)
    rc.energy_norm_t30 = min(E30 / (p_max_kw * 0.5), 1.0)   # 0.5 h = 30 min
    rc.energy_norm_t60 = min(E60 / (p_max_kw * 1.0), 1.0)   # 1.0 h = 60 min
    rc.energy_mean = sum(
        a * e for a, e in zip(ALPHA_WEIGHTS, [rc.energy_norm_t30, rc.energy_norm_t60])
    )

    # ── Stability of forecast trajectory ───────────────────────────────────
    # Transitions: current→t30→t60
    rc.stability_t30 = _stability_score(T30 - current_T, RH30 - current_RH)
    rc.stability_t60 = _stability_score(T60 - T30, RH60 - RH30)
    rc.stability_mean = sum(
        a * s for a, s in zip(ALPHA_WEIGHTS, [rc.stability_t30, rc.stability_t60])
    )
    # Stability only bonuses when occupied (no point stabilising an empty room)
    stability_occupied = rc.stability_mean * (1.0 if occupied else 0.0)

    # ── Switching penalty ──────────────────────────────────────────────────
    # Penalises chattering; different from stability (action vs state).
    rc.switch_penalty = abs(action - previous_action)

    # ── Safety ─────────────────────────────────────────────────────────────
    rc.safety_penalty = _safety_violations(forecasts, current_T, current_RH)

    # ── Aggregate ─────────────────────────────────────────────────────────
    raw = (
        W_COMFORT * comfort_occupied
        + W_STABILITY * stability_occupied
        - W_ENERGY * rc.energy_mean
        - W_SWITCH * rc.switch_penalty
        - W_SAFETY * rc.safety_penalty
    )

    # Normalise to [−1, 1] for RL stability
    # Denominator = sum of all positive weight terms
    pos_ceiling = W_COMFORT + W_STABILITY
    neg_floor = W_ENERGY + W_SWITCH + W_SAFETY
    rc.total_reward = float(
        2 * np.clip((raw + neg_floor) / (pos_ceiling + neg_floor), 0, 1) - 1
    )
    return rc


def reward_from_env_model(
    env_model: EnvModel,
    state: dict,
    action: float,
    previous_action: float,
    occupied: bool,
    comfort_model: ComfortModel,
    p_max_kw: float = P_MAX_KW,
) -> RewardComponents:
    """
    Full reward pipeline: call env model → compute reward components.

    Parameters
    ----------
    env_model : DualLGBMEnvModel or any object satisfying EnvModel protocol
    state : dict with current sensor readings (passed to env_model.predict)
    action, previous_action, occupied, comfort_model : see reward_components()
    """
    if not isinstance(env_model, EnvModel):
        raise TypeError("env_model must implement predict(state) -> dict")

    forecasts = env_model.predict(state)

    # Validate forecast contract
    required_keys = {
        "T_Indoor_t30", "T_Indoor_t60",
        "RH_Indoor_t30", "RH_Indoor_t60",
    }
    missing = required_keys - forecasts.keys()
    if missing:
        raise ValueError(f"env_model.predict() missing keys: {missing}")

    rh30 = forecasts["RH_Indoor_t30"]
    rh60 = forecasts["RH_Indoor_t60"]
    if not (0 <= rh30 <= 100 and 0 <= rh60 <= 100):
        raise ValueError(
            f"RH forecast out of [0,100]: RH_t30={rh30:.1f}, RH_t60={rh60:.1f}. "
            "Both Gaussian 2D and Heat Index require valid RH. No silent fallback."
        )

    return reward_components(
        forecasts=forecasts,
        current_T=float(state["T_Indoor"]),
        current_RH=float(state["RH_Indoor"]),
        action=action,
        previous_action=previous_action,
        occupied=occupied,
        comfort_model=comfort_model,
        p_max_kw=p_max_kw,
    )


def total_reward(
    env_model: EnvModel,
    state: dict,
    action: float,
    previous_action: float,
    occupied: bool,
    comfort_model: ComfortModel,
) -> float:
    """Convenience wrapper returning only the scalar reward for RL step."""
    rc = reward_from_env_model(
        env_model, state, action, previous_action, occupied, comfort_model
    )
    return rc.total_reward


# ---------------------------------------------------------------------------
# Legacy diagnostic — not used in main training pipeline
# ---------------------------------------------------------------------------

def instantaneous_reward_components(
    T: float,
    RH: float,
    action: float,
    previous_action: float,
    occupied: bool,
    comfort_model: ComfortModel,
    p_max_kw: float = P_MAX_KW,
) -> dict:
    """
    One-step (instantaneous) reward for ablation comparison only.

    NOT used in the main pipeline. Retained to allow:
      - Ablation: predictive reward vs instantaneous reward
      - Diagnostic of comfort surface without env model dependency

    Note: does not double-count future outcomes (no overlap risk),
    but also does not account for control lag.
    """
    comfort = combined_comfort_score(T, RH, comfort_model) * (1.0 if occupied else 0.0)
    energy = min(action * P_RATED_KW / p_max_kw, 1.0)
    switch = abs(action - previous_action)
    safety = 1.0 if (T < SAFE_T_MIN or T > SAFE_T_MAX or RH > SAFE_RH_MAX) else 0.0
    raw = W_COMFORT * comfort - W_ENERGY * energy - W_SWITCH * switch - W_SAFETY * safety
    pos_ceil = W_COMFORT
    neg_floor = W_ENERGY + W_SWITCH + W_SAFETY
    total = float(2 * np.clip((raw + neg_floor) / (pos_ceil + neg_floor), 0, 1) - 1)
    return {
        "comfort": round(comfort, 4),
        "energy_norm": round(energy, 4),
        "switch_penalty": round(switch, 4),
        "safety_penalty": round(safety, 4),
        "total_reward": round(total, 4),
        "mode": "instantaneous (ablation only)",
    }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  hvac_reward.py — smoke test (no trained env model)")
    print("  Horizons: t+30, t+60 minutes")
    print("=" * 60)

    comfort_model = load_comfort_model()

    # Fake forecast matching report Section 8 style
    fake_forecasts = {
        "T_Indoor_t30": 28.0,
        "T_Indoor_t60": 27.2,
        "RH_Indoor_t30": 63.0,
        "RH_Indoor_t60": 60.0,
        "energy_proxy_t30_kWh": 0.225,   # 0.9 kW × 0.25 h
        "energy_proxy_t60_kWh": 0.350,
    }

    rc = reward_components(
        forecasts=fake_forecasts,
        current_T=30.0,
        current_RH=70.0,
        action=1.0,
        previous_action=0.2,
        occupied=True,
        comfort_model=comfort_model,
    )

    print(f"\n  Comfort t30 : {rc.comfort_t30:.4f}")
    print(f"  Comfort t60 : {rc.comfort_t60:.4f}")
    print(f"  Comfort mean: {rc.comfort_mean:.4f}")
    print(f"  Energy  mean: {rc.energy_mean:.4f}")
    print(f"  Stability   : {rc.stability_mean:.4f}")
    print(f"  Switch pen. : {rc.switch_penalty:.4f}")
    print(f"  Safety pen. : {rc.safety_penalty:.4f}")
    print(f"\n  Total reward: {rc.total_reward:.4f}  ∈ [−1, 1]")

    print("\n  [Ablation] Instantaneous reward at current state (30°C, 70% RH):")
    inst = instantaneous_reward_components(30.0, 70.0, 1.0, 0.2, True, comfort_model)
    for k, v in inst.items():
        print(f"    {k}: {v}")

    print("\n  Smoke test passed.")