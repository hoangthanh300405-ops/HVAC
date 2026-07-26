from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np


FORECAST_HORIZONS_MINUTES = (30, 60)


@dataclass(frozen=True)
class ForecastPoint:
    """One environment-model prediction used by the long-horizon reward.

    ``hvac_power`` is the predicted mean HVAC power over the interval ending at
    this horizon (e.g., 0-30, 30-60 minutes), not an instantaneous meter
    sample. Temperature alone is insufficient to infer electrical consumption.
    """

    horizon_minutes: int
    temperature_c: float
    relative_humidity: float
    hvac_power: float

    def __post_init__(self) -> None:
        values = (
            self.temperature_c,
            self.relative_humidity,
            self.hvac_power,
        )
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive.")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Forecast values must be finite.")
        if not 0.0 <= self.relative_humidity <= 100.0:
            raise ValueError("Forecast relative_humidity must be in [0, 100].")
        if self.hvac_power < 0:
            raise ValueError("Forecast hvac_power cannot be negative.")


class EnvironmentForecastModel(Protocol):
    """Minimal interface expected from an HVAC environment model."""

    def predict_horizons(
        self,
        state: Mapping[str, float],
        action: float,
        horizons_minutes: Sequence[int],
    ) -> Sequence[ForecastPoint]:
        """Predict T, RH, and interval-mean HVAC power at each horizon."""


@dataclass(frozen=True)
class ComfortModel:
    """Posterior comfort parameters used by the HVAC reward."""

    mu: tuple[float, float]
    covariance: tuple[tuple[float, float], tuple[float, float]]
    heat_index_target_c: float | None = None
    heat_index_sigma_c: float = 3.0
    gaussian_2d_weight: float = 0.8
    heat_index_weight: float = 0.2

    def __post_init__(self) -> None:
        covariance = np.asarray(self.covariance, dtype=float)
        if covariance.shape != (2, 2):
            raise ValueError("covariance must be a 2x2 matrix.")
        if np.linalg.eigvalsh(covariance).min() <= 0:
            raise ValueError("covariance must be positive definite.")
        if self.heat_index_sigma_c <= 0:
            raise ValueError("heat_index_sigma_c must be positive.")
        if self.gaussian_2d_weight < 0 or self.heat_index_weight < 0:
            raise ValueError("Comfort mixture weights must be non-negative.")
        if not math.isclose(
            self.gaussian_2d_weight + self.heat_index_weight,
            1.0,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("Comfort mixture weights must sum to one.")

    @property
    def resolved_heat_index_target_c(self) -> float:
        if self.heat_index_target_c is not None:
            return float(self.heat_index_target_c)
        return float(nws_heat_index_celsius(self.mu[0], self.mu[1]))


@dataclass(frozen=True)
class RewardWeights:
    """Weights for the formalized total reward."""

    comfort: float = 2.0
    stability: float = 0.20
    energy: float = 0.10
    switching: float = 0.02


@dataclass(frozen=True)
class StabilityScales:
    temperature_delta_c: float = 0.5
    humidity_delta_percent: float = 3.0

    def __post_init__(self) -> None:
        if self.temperature_delta_c <= 0 or self.humidity_delta_percent <= 0:
            raise ValueError("Stability scales must be positive.")


def load_comfort_model(
    path: Path | str = Path(
        "outputs/bayesian_preference/comfort_gaussian_params.json"
    ),
    heat_index_sigma_c: float = 3.0,
) -> ComfortModel:
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    # JSON uses T_star and RH_star instead of mu
    mu = (float(data["T_star"]), float(data["RH_star"]))
    sigma_mat = data["sigma_matrix"]

    return ComfortModel(
        mu=mu,
        covariance=(
            (float(sigma_mat[0][0]), float(sigma_mat[0][1])),
            (float(sigma_mat[1][0]), float(sigma_mat[1][1])),
        ),
        heat_index_target_c=data.get("HI_star"),
        heat_index_sigma_c=data.get("sigma_HI", heat_index_sigma_c),
    )


def nws_heat_index_celsius(
    temperature_c: float | np.ndarray,
    relative_humidity: float | np.ndarray,
) -> float | np.ndarray:
    temperature_c_array = np.asarray(temperature_c, dtype=float)
    rh = np.asarray(relative_humidity, dtype=float)
    temperature_c_array, rh = np.broadcast_arrays(temperature_c_array, rh)
    rh = np.clip(rh, 0.0, 100.0)
    temperature_f = temperature_c_array * 9.0 / 5.0 + 32.0

    simple = 0.5 * (
        temperature_f
        + 61.0
        + (temperature_f - 68.0) * 1.2
        + rh * 0.094
    )
    simple = 0.5 * (simple + temperature_f)

    regression = (
        -42.379
        + 2.04901523 * temperature_f
        + 10.14333127 * rh
        - 0.22475541 * temperature_f * rh
        - 0.00683783 * temperature_f**2
        - 0.05481717 * rh**2
        + 0.00122874 * temperature_f**2 * rh
        + 0.00085282 * temperature_f * rh**2
        - 0.00000199 * temperature_f**2 * rh**2
    )

    low_humidity = (
        (rh < 13.0)
        & (temperature_f >= 80.0)
        & (temperature_f <= 112.0)
    )
    low_adjustment = ((13.0 - rh) / 4.0) * np.sqrt(
        np.maximum(0.0, (17.0 - np.abs(temperature_f - 95.0)) / 17.0)
    )
    regression = np.where(low_humidity, regression - low_adjustment, regression)

    high_humidity = (
        (rh > 85.0)
        & (temperature_f >= 80.0)
        & (temperature_f <= 87.0)
    )
    high_adjustment = ((rh - 85.0) / 10.0) * (
        (87.0 - temperature_f) / 5.0
    )
    regression = np.where(
        high_humidity,
        regression + high_adjustment,
        regression,
    )
    heat_index_f = np.where(simple >= 80.0, regression, simple)
    heat_index_c = (heat_index_f - 32.0) * 5.0 / 9.0
    if heat_index_c.ndim == 0:
        return float(heat_index_c)
    return heat_index_c


def gaussian_2d_comfort_score(
    temperature_c: float | np.ndarray,
    relative_humidity: float | np.ndarray,
    model: ComfortModel,
) -> float | np.ndarray:
    temperature = np.asarray(temperature_c, dtype=float)
    humidity = np.asarray(relative_humidity, dtype=float)
    temperature, humidity = np.broadcast_arrays(temperature, humidity)
    points = np.stack([temperature, humidity], axis=-1)
    delta = points - np.asarray(model.mu, dtype=float)
    inverse = np.linalg.inv(np.asarray(model.covariance, dtype=float))
    mahalanobis = np.einsum("...i,ij,...j->...", delta, inverse, delta)
    score = np.exp(-0.5 * mahalanobis)
    if score.ndim == 0:
        return float(score)
    return score


def heat_index_comfort_score(
    temperature_c: float | np.ndarray,
    relative_humidity: float | np.ndarray,
    model: ComfortModel,
) -> float | np.ndarray:
    heat_index = np.asarray(
        nws_heat_index_celsius(temperature_c, relative_humidity),
        dtype=float,
    )
    standardized = (
        heat_index - model.resolved_heat_index_target_c
    ) / model.heat_index_sigma_c
    score = np.exp(-0.5 * standardized**2)
    if score.ndim == 0:
        return float(score)
    return score


def combined_comfort_score(
    temperature_c: float | np.ndarray,
    relative_humidity: float | np.ndarray,
    model: ComfortModel,
) -> float | np.ndarray:
    gaussian_score = gaussian_2d_comfort_score(
        temperature_c,
        relative_humidity,
        model,
    )
    heat_index_score = heat_index_comfort_score(
        temperature_c,
        relative_humidity,
        model,
    )
    return (
        model.gaussian_2d_weight * gaussian_score
        + model.heat_index_weight * heat_index_score
    )


def stability_score(
    delta_temperature_c: float,
    delta_relative_humidity: float,
    scales: StabilityScales = StabilityScales(),
) -> float:
    standardized_distance = (
        (delta_temperature_c / scales.temperature_delta_c) ** 2
        + (
            delta_relative_humidity
            / scales.humidity_delta_percent
        )
        ** 2
    )
    return float(math.exp(-0.5 * standardized_distance))


def normalized_energy_cost(
    hvac_power: float,
    maximum_hvac_power: float,
) -> float:
    if maximum_hvac_power <= 0:
        raise ValueError("maximum_hvac_power must be positive.")
    return float(np.clip(hvac_power / maximum_hvac_power, 0.0, 1.0))


def switching_cost(
    action: float,
    previous_action: float,
    maximum_action_change: float = 1.0,
) -> float:
    if maximum_action_change <= 0:
        raise ValueError("maximum_action_change must be positive.")
    return float(
        np.clip(
            abs(action - previous_action) / maximum_action_change,
            0.0,
            1.0,
        )
    )


def _validated_forecasts(
    forecasts: Sequence[ForecastPoint],
) -> tuple[ForecastPoint, ...]:
    ordered = tuple(sorted(forecasts, key=lambda item: item.horizon_minutes))
    actual_horizons = tuple(item.horizon_minutes for item in ordered)
    if actual_horizons != FORECAST_HORIZONS_MINUTES:
        raise ValueError(
            f"Forecasts must contain exactly one point at {FORECAST_HORIZONS_MINUTES}; "
            f"received {actual_horizons}."
        )
    return ordered


def _normalized_horizon_weights(
    horizon_weights: Sequence[float] | None,
) -> np.ndarray:
    if horizon_weights is None:
        return np.full(len(FORECAST_HORIZONS_MINUTES), 0.5)
    weights = np.asarray(horizon_weights, dtype=float)
    if weights.shape != (len(FORECAST_HORIZONS_MINUTES),):
        raise ValueError(f"horizon_weights must contain {len(FORECAST_HORIZONS_MINUTES)} values.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("horizon_weights must be finite and non-negative.")
    if float(weights.sum()) <= 0:
        raise ValueError("At least one horizon weight must be positive.")
    return weights / weights.sum()


def _normalized_occupancy_forecast(
    occupied: bool,
    occupancy_forecast: Sequence[float] | None,
) -> np.ndarray:
    if occupancy_forecast is None:
        return np.full(len(FORECAST_HORIZONS_MINUTES), 1.0 if occupied else 0.0)
    occupancy = np.asarray(occupancy_forecast, dtype=float)
    if occupancy.shape != (len(FORECAST_HORIZONS_MINUTES),):
        raise ValueError(f"occupancy_forecast must contain {len(FORECAST_HORIZONS_MINUTES)} values.")
    if not np.all(np.isfinite(occupancy)):
        raise ValueError("occupancy_forecast must be finite.")
    if np.any((occupancy < 0) | (occupancy > 1)):
        raise ValueError("occupancy_forecast values must be in [0, 1].")
    return occupancy


def future_comfort_scores(
    forecasts: Sequence[ForecastPoint],
    model: ComfortModel,
) -> np.ndarray:
    ordered = _validated_forecasts(forecasts)
    return np.asarray(
        [
            combined_comfort_score(
                point.temperature_c,
                point.relative_humidity,
                model,
            )
            for point in ordered
        ],
        dtype=float,
    )


def future_energy_costs(
    forecasts: Sequence[ForecastPoint],
    maximum_hvac_power: float,
) -> np.ndarray:
    ordered = _validated_forecasts(forecasts)
    return np.asarray(
        [
            normalized_energy_cost(point.hvac_power, maximum_hvac_power)
            for point in ordered
        ],
        dtype=float,
    )


def future_stability_scores(
    current_temperature_c: float,
    current_relative_humidity: float,
    forecasts: Sequence[ForecastPoint],
    scales: StabilityScales = StabilityScales(),
) -> np.ndarray:
    ordered = _validated_forecasts(forecasts)
    previous_temperature = current_temperature_c
    previous_humidity = current_relative_humidity
    scores: list[float] = []
    for point in ordered:
        scores.append(
            stability_score(
                point.temperature_c - previous_temperature,
                point.relative_humidity - previous_humidity,
                scales,
            )
        )
        previous_temperature = point.temperature_c
        previous_humidity = point.relative_humidity
    return np.asarray(scores, dtype=float)


def reward_components(
    *,
    forecasts: Sequence[ForecastPoint],
    current_temperature_c: float,
    current_relative_humidity: float,
    maximum_hvac_power: float,
    action: float,
    previous_action: float,
    occupied: bool,
    comfort_model: ComfortModel,
    horizon_weights: Sequence[float] | None = None,
    occupancy_forecast: Sequence[float] | None = None,
    weights: RewardWeights = RewardWeights(),
    stability_scales: StabilityScales = StabilityScales(),
) -> dict[str, float]:
    ordered = _validated_forecasts(forecasts)
    aggregation_weights = _normalized_horizon_weights(horizon_weights)
    occupancy = _normalized_occupancy_forecast(
        occupied,
        occupancy_forecast,
    )
    comfort_values = future_comfort_scores(ordered, comfort_model)
    stability_values = future_stability_scores(
        current_temperature_c,
        current_relative_humidity,
        ordered,
        stability_scales,
    )
    energy_values = future_energy_costs(ordered, maximum_hvac_power)

    comfort = float(np.dot(aggregation_weights, comfort_values))
    occupied_comfort = float(
        np.dot(aggregation_weights, occupancy * comfort_values)
    )
    stability = float(np.dot(aggregation_weights, stability_values))
    occupied_stability = float(
        np.dot(aggregation_weights, occupancy * stability_values)
    )
    energy = float(np.dot(aggregation_weights, energy_values))
    switching = switching_cost(action, previous_action)

    comfort_term = weights.comfort * occupied_comfort
    stability_term = weights.stability * occupied_stability
    energy_term = weights.energy * energy
    switching_term = weights.switching * switching

    raw_reward = comfort_term + stability_term - energy_term - switching_term

    # ponytail: remove aggressive clipping and normalization
    # reward is now natural: [best_possible, worst_possible]
    total_reward = raw_reward

    # Provide baseline incentive when not occupied to avoid stuck gradients
    if not occupied:
        total_reward += 0.1

    result = {
        "comfort_score": comfort,
        "occupied_comfort_score": occupied_comfort,
        "stability_score": stability,
        "occupied_stability_score": occupied_stability,
        "energy_cost": energy,
        "switching_cost": switching,
        "comfort_term": comfort_term,
        "stability_term": stability_term,
        "energy_term": energy_term,
        "switching_term": switching_term,
        "total_reward": total_reward,
    }
    for index, point in enumerate(ordered):
        suffix = f"h{point.horizon_minutes}"
        result[f"comfort_{suffix}"] = float(comfort_values[index])
        result[f"stability_{suffix}"] = float(stability_values[index])
        result[f"energy_{suffix}"] = float(energy_values[index])
        result[f"occupancy_{suffix}"] = float(occupancy[index])
        result[f"weight_{suffix}"] = float(aggregation_weights[index])
    return result


def reward_from_env_model(
    *,
    env_model: EnvironmentForecastModel,
    state: Mapping[str, float],
    action: float,
    previous_action: float,
    occupied: bool,
    maximum_hvac_power: float,
    comfort_model: ComfortModel,
    horizon_weights: Sequence[float] | None = None,
    occupancy_forecast: Sequence[float] | None = None,
    weights: RewardWeights = RewardWeights(),
    stability_scales: StabilityScales = StabilityScales(),
) -> dict[str, float]:
    try:
        current_temperature = float(state["temperature_c"])
        current_humidity = float(state["relative_humidity"])
    except KeyError as error:
        raise KeyError(
            "state must contain temperature_c and relative_humidity."
        ) from error
    forecasts = env_model.predict_horizons(
        state,
        action,
        FORECAST_HORIZONS_MINUTES,
    )
    return reward_components(
        forecasts=forecasts,
        current_temperature_c=current_temperature,
        current_relative_humidity=current_humidity,
        maximum_hvac_power=maximum_hvac_power,
        action=action,
        previous_action=previous_action,
        occupied=occupied,
        comfort_model=comfort_model,
        horizon_weights=horizon_weights,
        occupancy_forecast=occupancy_forecast,
        weights=weights,
        stability_scales=stability_scales,
    )


def total_reward(
    env_model: EnvironmentForecastModel,
    state: Mapping[str, float],
    action: float,
    previous_action: float,
    occupied: bool,
    comfort_model: ComfortModel,
) -> float:
    return float(reward_components(
        forecasts=env_model.predict_horizons(state, action, FORECAST_HORIZONS_MINUTES),
        current_temperature_c=state["temperature_c"],
        current_relative_humidity=state["relative_humidity"],
        maximum_hvac_power=1.0,
        action=action,
        previous_action=previous_action,
        occupied=occupied,
        comfort_model=comfort_model,
    )["total_reward"])
