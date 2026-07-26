"""
dual_lgbm_env_model.py
======================
Phase 2 — Dual LightGBM Surrogate Environment

Two independent LightGBM models trained on separate physical targets:
  - Model A (LightGBM_T) : T_Indoor_{t+30}, T_Indoor_{t+60}
  - Model B (LightGBM_RH): RH_Indoor_{t+30}, RH_Indoor_{t+60}

Energy is not modelled by a learned model. It is an explicit proxy:
  energy_proxy = AC_State_t × P_RATED_KW  (kWh over 15-min step)

Horizons align with add_set_point.py window logic:
  +2 steps (30 min), +4 steps (60 min) at 15-min resolution.

Train/test split:
  AP01–AP40  → surrogate training
  AP41–AP49  → holdout evaluation (no leakage)

Usage
-----
  python dual_lgbm_env_model.py --data path/to/final_dataset_with_setpoint_proxy.csv
  python dual_lgbm_env_model.py --data ... --eval-only --model-dir outputs/env_model
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STEP_MINUTES: int = 15          # dataset resolution
HORIZONS: tuple[int, ...] = (2, 4)    # steps ahead → 30 min, 60 min
HORIZON_LABELS: tuple[str, ...] = ("t30", "t60")

P_RATED_KW: float = 1.0        # nominal AC rated power (kW); 1 step = 0.25 h
                                # → energy_proxy = AC_State × 1.0 × 0.25 kWh

TRAIN_AP_MAX: int = 40          # AP01–AP40 → train; AP41–AP49 → holdout

# ---------------------------------------------------------------------------
# Feature sets — designed around physical response characteristics
# ---------------------------------------------------------------------------
# Both models share the same 12-feature input vector.
# Lag at t-1 captures temporal dependency without needing a recurrent arch.
SHARED_FEATURES: list[str] = [
    "T_Indoor",          # current indoor temp
    "T_Indoor_lag1",     # indoor temp one step ago  (temporal inertia)
    "RH_Indoor",         # current indoor humidity
    "RH_Indoor_lag1",    # indoor humidity one step ago
    "T_Outdoor",         # outdoor temp  (heat load driver)
    "RH_Outdoor",        # outdoor humidity  (moisture source for RH model)
    "AC_State",          # binary on/off  (direct action)
    "setpoint_proxy",    # estimated cooling target from add_set_point.py
    "GHI",               # solar irradiance  (heat gain through glass)
    "Occupancy_State",   # occupancy flag
    "hour_sin",          # cyclic hour encoding
    "hour_cos",
]

# Targets: direct multi-output per model, one column per horizon
T_TARGET_COLS: list[str] = ["T_Indoor_t30", "T_Indoor_t60"]
RH_TARGET_COLS: list[str] = ["RH_Indoor_t30", "RH_Indoor_t60"]

# ---------------------------------------------------------------------------
# LightGBM hyper-parameters
# ---------------------------------------------------------------------------
LGBM_PARAMS_T: dict = {
    # Temperature responds quickly and directly to AC controls.
    # Moderate num_leaves; strong L1/L2 to avoid overfitting on few APs.
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "n_estimators": 600,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.2,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

LGBM_PARAMS_RH: dict = {
    # Humidity has higher inertia and depends on outdoor moisture.
    # More leaves for non-linear interactions; stronger regularisation
    # because RH is more stochastic and easier to overfit.
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 127,
    "learning_rate": 0.04,
    "n_estimators": 700,
    "min_child_samples": 30,
    "reg_alpha": 0.2,
    "reg_lambda": 0.4,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.75,
    "bagging_freq": 5,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _extract_ap_number(location_id: str) -> int:
    """Parse AP number from location_id like 'AP3_BR' → 3."""
    try:
        return int("".join(filter(str.isdigit, location_id.split("_")[0])))
    except (ValueError, IndexError):
        return 0


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """
    Load dataset, engineer features and target columns.

    Expects columns produced by add_set_point.py:
      timestamp, location_id, T_Indoor, RH_Indoor, T_Outdoor, RH_Outdoor,
      AC_State, Occupancy_State, setpoint_proxy  (+ optionally GHI).

    Returns a DataFrame with lag features, cyclic time encoding,
    energy_proxy, and shifted target columns appended.
    """
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    # ── GHI: fill with zeros when absent (solar data not in CAMaRSEC) ──────
    if "GHI" not in df.columns:
        print("  [info] GHI column not found — filling with 0 (no solar sensor).")
        df["GHI"] = 0.0

    # ── Lag features per location (avoids bleed across apartments) ──────────
    for col in ["T_Indoor", "RH_Indoor"]:
        df[f"{col}_lag1"] = df.groupby("location_id")[col].shift(1)

    # ── Cyclic hour encoding ────────────────────────────────────────────────
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # ── Energy proxy (explicit, not learned) ────────────────────────────────
    # energy_proxy_kWh: AC_State × P_rated × Δt
    # Limitation documented in paper: no power meter in CAMaRSEC.
    # Phase 8 will replace with smart plug measurements.
    df["energy_proxy_kWh"] = df["AC_State"].fillna(0) * P_RATED_KW * (STEP_MINUTES / 60)

    # ── Shifted target columns per location ─────────────────────────────────
    for steps, label in zip(HORIZONS, HORIZON_LABELS):
        df[f"T_Indoor_{label}"] = df.groupby("location_id")["T_Indoor"].shift(-steps)
        df[f"RH_Indoor_{label}"] = df.groupby("location_id")["RH_Indoor"].shift(-steps)

    # ── AP number for train/holdout split ───────────────────────────────────
    df["ap_number"] = df["location_id"].apply(_extract_ap_number)

    print(f"  Rows loaded: {len(df):,}")
    print(f"  Locations : {df['location_id'].nunique()}")
    print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def split_train_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministic split by AP number.
      AP01–AP40  → train surrogate
      AP41–AP49  → holdout evaluation  (no overlap, no leakage)
    """
    train = df[df["ap_number"] <= TRAIN_AP_MAX].copy()
    holdout = df[df["ap_number"] > TRAIN_AP_MAX].copy()
    print(f"  Train   : {len(train):,} rows  (AP01–AP40)")
    print(f"  Holdout : {len(holdout):,} rows (AP41–AP49)")
    return train, holdout


def resolve_data_path(path_arg: Path | None) -> Path:
    """Resolve the dataset path from common workspace locations."""
    if path_arg is not None and path_arg.exists():
        return path_arg

    candidates: list[Path] = []
    if path_arg is not None:
        candidates.append(path_arg)

    workspace_root = Path(__file__).resolve().parent.parent
    candidates.extend(
        [
            Path("Data/Final_data/final_dataset_with_setpoint_proxy.csv"),
            Path("data set /Data/Final_data/final_dataset_with_setpoint_proxy.csv"),
            workspace_root / "Data/Final_data/final_dataset_with_setpoint_proxy.csv",
            workspace_root / "data set " / "Data/Final_data/final_dataset_with_setpoint_proxy.csv",
            workspace_root / "data set " / "final_dataset_with_setpoint_proxy.csv",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(workspace_root.rglob("final_dataset_with_setpoint_proxy.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        "Could not find final_dataset_with_setpoint_proxy.csv. "
        "Pass --data with the correct file path."
    )


def get_xy(
    df: pd.DataFrame,
    target_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows with NaN in features or targets; return X, y."""
    required = SHARED_FEATURES + target_cols
    mask = df[required].notna().all(axis=1)
    clean = df[mask]
    return clean[SHARED_FEATURES], clean[target_cols]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    lgbm_params: dict,
    model_name: str,
) -> list[lgb.LGBMRegressor]:
    """
    Train one LGBMRegressor per target column (direct multi-output).
    Returns a list of fitted models aligned with y_train.columns.
    """
    models: list[lgb.LGBMRegressor] = []
    for col in y_train.columns:
        print(f"  Training {model_name} → {col} ...")
        m = lgb.LGBMRegressor(**lgbm_params)
        m.fit(X_train, y_train[col])
        models.append(m)
    return models


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    models: list[lgb.LGBMRegressor],
    X: pd.DataFrame,
    y: pd.DataFrame,
    model_name: str,
    split_label: str,
) -> dict:
    """Compute MAE, RMSE, R² for each target; return dict of metrics."""
    results: dict[str, dict] = {}
    print(f"\n{'─'*60}")
    print(f"  {model_name}  [{split_label}]")
    print(f"{'─'*60}")

    for model, col in zip(models, y.columns):
        pred = model.predict(X)
        mae = mean_absolute_error(y[col], pred)
        rmse = mean_squared_error(y[col], pred) ** 0.5
        r2 = r2_score(y[col], pred)
        results[col] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}
        unit = "°C" if "T_" in col else "%"
        print(f"  {col:<22}  MAE={mae:.3f}{unit}  RMSE={rmse:.3f}{unit}  R²={r2:.4f}")

    return results


def feature_importance_report(
    models: list[lgb.LGBMRegressor],
    target_cols: list[str],
    model_name: str,
) -> dict:
    """Aggregate feature importances across horizon models."""
    print(f"\n  Feature importance — {model_name}")
    importance: dict[str, float] = {f: 0.0 for f in SHARED_FEATURES}
    for model in models:
        imp = dict(zip(SHARED_FEATURES, model.feature_importances_))
        for f, v in imp.items():
            importance[f] += v
    # Average across horizons
    n = len(models)
    importance = {f: round(v / n, 1) for f, v in importance.items()}
    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, score in ranked:
        bar = "█" * int(score / max(v for _, v in ranked) * 20)
        print(f"  {feat:<24} {bar:<20} {score:.0f}")
    return dict(ranked)


# ---------------------------------------------------------------------------
# Inference interface (used by reward function / RL env)
# ---------------------------------------------------------------------------

class DualLGBMEnvModel:
    """
    Thin wrapper around the two trained model lists.

    predict(state_dict) → dict with keys:
        T_Indoor_t30, T_Indoor_t60,
        RH_Indoor_t30, RH_Indoor_t60,
        energy_proxy_t30, energy_proxy_t60  (from explicit proxy formula)

    The predict() interface is intentionally simple so the reward function
    can call it without knowing internal implementation details.
    """

    def __init__(
        self,
        models_T: list[lgb.LGBMRegressor],
        models_RH: list[lgb.LGBMRegressor],
    ) -> None:
        self.models_T = models_T
        self.models_RH = models_RH

    def predict(self, state: dict) -> dict:
        """
        Parameters
        ----------
        state : dict with keys matching SHARED_FEATURES.
                Missing GHI defaults to 0; missing setpoint_proxy defaults
                to T_Indoor (AC probably off).

        Returns
        -------
        dict with predicted T and RH at each horizon, plus energy proxy.
        """
        # Build feature vector (1 row)
        row: dict[str, float] = {
            "T_Indoor": float(state["T_Indoor"]),
            "T_Indoor_lag1": float(state.get("T_Indoor_lag1", state["T_Indoor"])),
            "RH_Indoor": float(state["RH_Indoor"]),
            "RH_Indoor_lag1": float(state.get("RH_Indoor_lag1", state["RH_Indoor"])),
            "T_Outdoor": float(state["T_Outdoor"]),
            "RH_Outdoor": float(state["RH_Outdoor"]),
            "AC_State": float(state["AC_State"]),
            "setpoint_proxy": float(state.get("setpoint_proxy", state["T_Indoor"])),
            "GHI": float(state.get("GHI", 0.0)),
            "Occupancy_State": float(state.get("Occupancy_State", 1.0)),
            "hour_sin": float(state["hour_sin"]),
            "hour_cos": float(state["hour_cos"]),
        }
        X = pd.DataFrame([row])[SHARED_FEATURES]

        output: dict[str, float] = {}

        # Temperature predictions
        for model, label in zip(self.models_T, HORIZON_LABELS):
            output[f"T_Indoor_{label}"] = float(model.predict(X)[0])

        # Humidity predictions
        for model, label in zip(self.models_RH, HORIZON_LABELS):
            output[f"RH_Indoor_{label}"] = float(model.predict(X)[0])

        # Energy proxy — explicit formula, NOT a learned model
        # energy_proxy_kWh = AC_State × P_rated × Δt
        # Holds AC_State constant over the horizon (held-action assumption).
        # Future work: replace with smart plug data (Phase 8).
        ac = row["AC_State"]
        delta_t = STEP_MINUTES / 60  # hours per step
        for steps, label in zip(HORIZONS, HORIZON_LABELS):
            output[f"energy_proxy_{label}_kWh"] = ac * P_RATED_KW * delta_t * steps

        return output

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        for i, (model, label) in enumerate(zip(self.models_T, HORIZON_LABELS)):
            model.booster_.save_model(str(model_dir / f"lgbm_T_{label}.txt"))
        for i, (model, label) in enumerate(zip(self.models_RH, HORIZON_LABELS)):
            model.booster_.save_model(str(model_dir / f"lgbm_RH_{label}.txt"))
        print(f"  Models saved to {model_dir}")

    @classmethod
    def load(cls, model_dir: Path) -> "DualLGBMEnvModel":
        models_T, models_RH = [], []
        for label in HORIZON_LABELS:
            t_path = model_dir / f"lgbm_T_{label}.txt"
            rh_path = model_dir / f"lgbm_RH_{label}.txt"
            if not t_path.exists() or not rh_path.exists():
                raise FileNotFoundError(
                    f"Model files not found in {model_dir}. Run training first."
                )
            m_t = lgb.LGBMRegressor()
            m_t._Booster = lgb.Booster(model_file=str(t_path))
            m_t.fitted_ = True
            models_T.append(m_t)

            m_rh = lgb.LGBMRegressor()
            m_rh._Booster = lgb.Booster(model_file=str(rh_path))
            m_rh.fitted_ = True
            models_RH.append(m_rh)

        print(f"  Models loaded from {model_dir}")
        return cls(models_T, models_RH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Dual LightGBM surrogate env model.")
    p.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data set " / "Data" / "Final_data" / "final_dataset_with_setpoint_proxy.csv",
        help=(
            "Path to final_dataset_with_setpoint_proxy.csv (output of add_set_point.py). "
            "Defaults to the workspace copy under the data set folder."
        ),
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("outputs/env_model"),
        help="Directory to save/load trained models. Default: outputs/env_model",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/env_model/training_report.json"),
        help="Path to save JSON training report.",
    )
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load saved models and run holdout evaluation only.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Dual LightGBM Surrogate Environment — Phase 2")
    print(f"  Horizons : {[f'+{h*STEP_MINUTES}min' for h in HORIZONS]}")
    print(f"  P_rated  : {P_RATED_KW} kW  (energy proxy, no power meter)")
    print(f"  Train AP : AP01–AP{TRAIN_AP_MAX}  |  Holdout: AP{TRAIN_AP_MAX+1}–AP49")
    print("=" * 60)

    data_path = resolve_data_path(args.data)
    print(f"Using data file: {data_path}")
    df = load_and_prepare(data_path)
    train_df, holdout_df = split_train_holdout(df)

    report: dict = {
        "horizons_minutes": [h * STEP_MINUTES for h in HORIZONS],
        "p_rated_kw": P_RATED_KW,
        "energy_note": (
            "Energy is a proxy: AC_State × P_rated × Δt. "
            "No power meter exists in CAMaRSEC. "
            "Phase 8 will replace with smart plug data."
        ),
        "train_ap_range": f"AP01–AP{TRAIN_AP_MAX}",
        "holdout_ap_range": f"AP{TRAIN_AP_MAX+1}–AP49",
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "shared_features": SHARED_FEATURES,
        "metrics": {},
        "feature_importance": {},
    }

    if args.eval_only:
        print("\nEval-only mode: loading saved models ...")
        env_model = DualLGBMEnvModel.load(args.model_dir)
    else:
        # ── Model A: Temperature ────────────────────────────────────────────
        print("\n[Model A — LightGBM_T]")
        print("  Physical rationale: T_Indoor reacts quickly and directly")
        print("  to AC_State and setpoint_proxy within 30 min of AC turning on.")
        X_train_T, y_train_T = get_xy(train_df, T_TARGET_COLS)
        models_T = train_model(X_train_T, y_train_T, LGBM_PARAMS_T, "LightGBM_T")

        # ── Model B: Humidity ───────────────────────────────────────────────
        print("\n[Model B — LightGBM_RH]")
        print("  Physical rationale: RH_Indoor has higher inertia, driven")
        print("  primarily by RH_Outdoor and GHI. AC_State is indirect.")
        X_train_RH, y_train_RH = get_xy(train_df, RH_TARGET_COLS)
        models_RH = train_model(X_train_RH, y_train_RH, LGBM_PARAMS_RH, "LightGBM_RH")

        env_model = DualLGBMEnvModel(models_T, models_RH)
        env_model.save(args.model_dir)

        # ── Feature importance ──────────────────────────────────────────────
        imp_T = feature_importance_report(models_T, T_TARGET_COLS, "LightGBM_T")
        imp_RH = feature_importance_report(models_RH, RH_TARGET_COLS, "LightGBM_RH")
        report["feature_importance"] = {"LightGBM_T": imp_T, "LightGBM_RH": imp_RH}

        # ── Sanity check: physical ordering of feature importance ───────────
        print("\n  [Sanity check] Physical ordering of top-3 features:")
        top3_T = list(imp_T.keys())[:3]
        top3_RH = list(imp_RH.keys())[:3]
        expected_T = {"AC_State", "setpoint_proxy", "T_Outdoor", "T_Indoor", "T_Indoor_lag1"}
        expected_RH = {"RH_Outdoor", "GHI", "RH_Indoor", "RH_Indoor_lag1"}
        t_ok = any(f in expected_T for f in top3_T)
        rh_ok = any(f in expected_RH for f in top3_RH)
        print(f"  LightGBM_T  top-3: {top3_T}  → {'✓ AC/setpoint dominant' if t_ok else '⚠ unexpected — check data'}")
        print(f"  LightGBM_RH top-3: {top3_RH} → {'✓ outdoor RH/GHI dominant' if rh_ok else '⚠ unexpected — check data'}")

        # ── Train-set metrics (overfitting check) ───────────────────────────
        X_tr_T, y_tr_T = get_xy(train_df, T_TARGET_COLS)
        X_tr_RH, y_tr_RH = get_xy(train_df, RH_TARGET_COLS)
        report["metrics"]["train_T"] = evaluate(models_T, X_tr_T, y_tr_T, "LightGBM_T", "train")
        report["metrics"]["train_RH"] = evaluate(models_RH, X_tr_RH, y_tr_RH, "LightGBM_RH", "train")

    # ── Holdout evaluation ──────────────────────────────────────────────────
    X_h_T, y_h_T = get_xy(holdout_df, T_TARGET_COLS)
    X_h_RH, y_h_RH = get_xy(holdout_df, RH_TARGET_COLS)

    if not args.eval_only:
        report["metrics"]["holdout_T"] = evaluate(
            env_model.models_T, X_h_T, y_h_T, "LightGBM_T", "holdout AP41–49"
        )
        report["metrics"]["holdout_RH"] = evaluate(
            env_model.models_RH, X_h_RH, y_h_RH, "LightGBM_RH", "holdout AP41–49"
        )
    else:
        report["metrics"]["holdout_T"] = evaluate(
            env_model.models_T, X_h_T, y_h_T, "LightGBM_T", "holdout AP41–49"
        )
        report["metrics"]["holdout_RH"] = evaluate(
            env_model.models_RH, X_h_RH, y_h_RH, "LightGBM_RH", "holdout AP41–49"
        )

    # ── Energy proxy stats on holdout ───────────────────────────────────────
    proxy_mean = holdout_df["energy_proxy_kWh"].mean()
    proxy_max = holdout_df["energy_proxy_kWh"].max()
    report["energy_proxy_stats_holdout"] = {
        "mean_kWh_per_step": round(float(proxy_mean), 4),
        "max_kWh_per_step": round(float(proxy_max), 4),
        "formula": f"AC_State × {P_RATED_KW} kW × {STEP_MINUTES}/60 h",
        "limitation": "No power meter in CAMaRSEC. Replace with smart plug in Phase 8.",
    }

    # ── Save report ─────────────────────────────────────────────────────────
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report saved → {args.report}")
    print("\nDone.")


if __name__ == "__main__":
    main()
