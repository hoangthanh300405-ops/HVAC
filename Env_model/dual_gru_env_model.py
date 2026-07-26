"""
dual_gru_env_model.py
=====================
Phase 2 — Dual GRU Surrogate Environment

Two independent GRU networks trained on separate physical targets:
  - GRU_T  : T_Indoor at t+30 min, t+60 min
  - GRU_RH : RH_Indoor at t+30 min, t+60 min

Energy is not modelled by a learned model. It is an explicit proxy:
  energy_proxy = AC_State_t × P_RATED_KW  (kWh over 15-min step)

Why GRU over LightGBM:
  - GRU natively captures temporal dependencies over a sequence window
    without hand-crafted lag features.
  - Better suited for autocorrelated indoor climate signals (T/RH drift
    over hours, not just one lag step).
  - Comparable inference speed with a shallow hidden layer (≤ 64 units).

Sequence window:
  SEQ_LEN = 8 steps = 2 hours of history fed as context at each step.

Horizons align with previous pipeline (15-min resolution):
  +2 steps (30 min), +4 steps (60 min).

Train/test split:
  AP01–AP40  → training
  AP41–AP49  → holdout evaluation (no leakage)

Usage
-----
  python dual_gru_env_model.py --data path/to/final_dataset_with_setpoint_proxy.csv
  python dual_gru_env_model.py --data ... --eval-only --model-dir outputs/env_model_gru
  python dual_gru_env_model.py --data ... --epochs 60 --hidden 64 --seq-len 12
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STEP_MINUTES: int = 15
HORIZONS: tuple[int, ...] = (2, 4)          # steps → +30 min, +60 min
HORIZON_LABELS: tuple[str, ...] = ("t30", "t60")

P_RATED_KW: float = 1.0                     # kWh proxy per step when AC on
TRAIN_AP_MAX: int = 40                       # AP01–AP40 train; AP41–AP49 holdout

# Physical output clamps applied in DualGRUEnvModel.predict().
#
# LightGBM's leaf-value predictions are automatically bounded by the range
# of targets seen during training and cannot extrapolate past them. A GRU
# with a linear output head has no such guarantee — an under-trained
# network, an unusual input sequence, or numerical drift early in RL
# training can produce values outside physical bounds (e.g. RH < 0 or
# RH > 100). hvac_reward.py's reward_from_env_model() intentionally hard-
# raises on out-of-range RH ("no silent fallback") because Phase 4 treats
# an invalid forecast as a bug, not something to quietly paper over. That
# is the right behaviour for a bug in the *forecast values themselves*,
# but it also means one bad GRU output would crash an entire Phase 6
# training run. Clamping here keeps predictions physically valid while
# still surfacing genuinely broken predictions (e.g. NaN) upstream.
TEMP_CLIP_RANGE_C: tuple[float, float] = (5.0, 50.0)
RH_CLIP_RANGE_PCT: tuple[float, float] = (0.0, 100.0)

# GRU defaults (overridable via CLI)
DEFAULT_SEQ_LEN: int = 8                     # 2 h look-back window
DEFAULT_HIDDEN: int = 64                     # GRU hidden units
DEFAULT_LAYERS: int = 2                      # stacked GRU depth
DEFAULT_DROPOUT: float = 0.2
DEFAULT_EPOCHS: int = 50
DEFAULT_BATCH: int = 256
DEFAULT_LR: float = 1e-3
DEFAULT_PATIENCE: int = 8                    # early stopping patience

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
# Raw features (before sequence stacking). No manual lag columns needed —
# the GRU learns temporal dependencies from the sequence context.
INPUT_FEATURES: list[str] = [
    "T_Indoor",
    "RH_Indoor",
    "T_Outdoor",
    "RH_Outdoor",
    "AC_State",
    "setpoint_proxy",
    "GHI",
    "Occupancy_State",
    "hour_sin",
    "hour_cos",
]

T_TARGET_COLS: list[str]  = ["T_Indoor_t30",  "T_Indoor_t60"]
RH_TARGET_COLS: list[str] = ["RH_Indoor_t30", "RH_Indoor_t60"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _extract_ap_number(location_id: str) -> int:
    try:
        return int("".join(filter(str.isdigit, location_id.split("_")[0])))
    except (ValueError, IndexError):
        return 0


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    """
    Load dataset and engineer target columns + cyclic time features.
    No lag columns are added — the GRU sequence window handles temporal context.
    """
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    if "GHI" not in df.columns:
        print("  [info] GHI column not found — filling with 0.")
        df["GHI"] = 0.0

    # Cyclic hour encoding
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Energy proxy (explicit, not learned)
    df["energy_proxy_kWh"] = df["AC_State"].fillna(0) * P_RATED_KW * (STEP_MINUTES / 60)

    # Shifted targets per location (no leakage across apartments)
    for steps, label in zip(HORIZONS, HORIZON_LABELS):
        df[f"T_Indoor_{label}"]  = df.groupby("location_id")["T_Indoor"].shift(-steps)
        df[f"RH_Indoor_{label}"] = df.groupby("location_id")["RH_Indoor"].shift(-steps)

    df["ap_number"] = df["location_id"].apply(_extract_ap_number)

    print(f"  Rows loaded : {len(df):,}")
    print(f"  Locations   : {df['location_id'].nunique()}")
    print(f"  Date range  : {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


def split_train_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train   = df[df["ap_number"] <= TRAIN_AP_MAX].copy()
    holdout = df[df["ap_number"] >  TRAIN_AP_MAX].copy()
    print(f"  Train   : {len(train):,} rows  (AP01–AP{TRAIN_AP_MAX})")
    print(f"  Holdout : {len(holdout):,} rows (AP{TRAIN_AP_MAX+1}–AP49)")
    return train, holdout


# ---------------------------------------------------------------------------
# Sequence Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """
    Builds overlapping windows of length `seq_len` per apartment.
    Each sample:
      X  : (seq_len, n_features)  — feature sequence
      y  : (n_targets,)           — targets at the last step of the window

    Windows do NOT cross apartment boundaries.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_cols: list[str],
        seq_len: int,
        scaler_X: StandardScaler | None = None,
        fit_scaler: bool = False,
    ) -> None:
        required = INPUT_FEATURES + target_cols
        mask = df[required].notna().all(axis=1)
        df = df[mask].reset_index(drop=True)

        # Fit or apply feature scaler
        raw_X = df[INPUT_FEATURES].values.astype(np.float32)
        if fit_scaler:
            self.scaler_X = StandardScaler()
            X_scaled = self.scaler_X.fit_transform(raw_X)
        else:
            assert scaler_X is not None, "Provide fitted scaler_X for val/test sets."
            self.scaler_X = scaler_X
            X_scaled = self.scaler_X.transform(raw_X)

        Y = df[target_cols].values.astype(np.float32)

        self.sequences: list[np.ndarray] = []
        self.targets:   list[np.ndarray] = []

        # Build windows per apartment (no cross-AP bleed)
        for _, grp in df.groupby("location_id", sort=False):
            idx = grp.index.tolist()
            if len(idx) < seq_len:
                continue
            local_X = X_scaled[idx]
            local_Y = Y[idx]
            for i in range(seq_len - 1, len(idx)):
                self.sequences.append(local_X[i - seq_len + 1 : i + 1])
                self.targets.append(local_Y[i])

        self._X = np.stack(self.sequences)      # (N, seq_len, n_feat)
        self._Y = np.stack(self.targets)        # (N, n_targets)
        print(f"    Dataset: {len(self._X):,} windows  "
              f"| seq_len={seq_len}  | n_features={len(INPUT_FEATURES)}  "
              f"| targets={target_cols}")

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._X[idx]),
            torch.from_numpy(self._Y[idx]),
        )


# ---------------------------------------------------------------------------
# GRU Model
# ---------------------------------------------------------------------------

class GRUEnvModel(nn.Module):
    """
    Stacked GRU with a linear head for direct multi-horizon output.

    Architecture:
      Input  → GRU (n_layers, hidden_size, dropout) → last hidden state
             → Linear(hidden_size, n_targets)

    Using the last hidden state (not all time steps) is natural for
    multi-step-ahead regression: we want the state summary after seeing
    the full window.
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        hidden_size: int = DEFAULT_HIDDEN,
        n_layers: int = DEFAULT_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_targets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, h_n = self.gru(x)          # h_n: (n_layers, batch, hidden)
        last_h = h_n[-1]              # top-layer last hidden: (batch, hidden)
        return self.head(last_h)      # (batch, n_targets)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_gru(
    model: GRUEnvModel,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs: int,
    lr: float,
    patience: int,
    model_name: str,
) -> list[float]:
    """
    Adam + MSELoss training loop with early stopping.
    Returns list of validation losses per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state: dict | None = None
    no_improve = 0
    val_losses: list[float] = []

    model.to(DEVICE)
    print(f"\n  Training {model_name} on {DEVICE} ...")
    print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>10}  {'LR':>10}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*10}  {'─'*10}")

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)  # type: ignore[arg-type]

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)      # type: ignore[arg-type]

        scheduler.step(val_loss)
        val_losses.append(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:>5}  {train_loss:>12.5f}  {val_loss:>10.5f}  {current_lr:>10.2e}")

        # ── early stopping ──
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}  (best val loss={best_val_loss:.5f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return val_losses


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_gru(
    model: GRUEnvModel,
    loader: DataLoader,
    target_cols: list[str],
    model_name: str,
    split_label: str,
) -> dict:
    """Return MAE, RMSE, R² per target column."""
    model.eval()
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(DEVICE)).cpu().numpy()
            all_pred.append(pred)
            all_true.append(yb.numpy())

    pred_arr = np.concatenate(all_pred, axis=0)
    true_arr = np.concatenate(all_true, axis=0)

    results: dict[str, dict] = {}
    print(f"\n{'─'*60}")
    print(f"  {model_name}  [{split_label}]")
    print(f"{'─'*60}")
    for i, col in enumerate(target_cols):
        p, t = pred_arr[:, i], true_arr[:, i]
        mae  = mean_absolute_error(t, p)
        rmse = mean_squared_error(t, p) ** 0.5
        r2   = r2_score(t, p)
        results[col] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}
        unit = "°C" if "T_" in col else "%"
        print(f"  {col:<22}  MAE={mae:.3f}{unit}  RMSE={rmse:.3f}{unit}  R²={r2:.4f}")

    return results


# ---------------------------------------------------------------------------
# Inference interface (used by reward function / RL env)
# ---------------------------------------------------------------------------

class DualGRUEnvModel:
    """
    Stateful wrapper around the two trained GRU networks.

    Maintains a rolling feature buffer of length `seq_len` per call to
    `step()`. Call `reset(initial_obs)` at the start of each episode.

    API mirrors the old DualLGBMEnvModel for drop-in compatibility with
    the reward function and Gymnasium wrapper.

    predict(state_dict) → dict with keys:
        T_Indoor_t30, T_Indoor_t60,
        RH_Indoor_t30, RH_Indoor_t60,
        energy_proxy_t30_kWh, energy_proxy_t60_kWh
    """

    def __init__(
        self,
        model_T:  GRUEnvModel,
        model_RH: GRUEnvModel,
        scaler_T:  StandardScaler,
        scaler_RH: StandardScaler,
        seq_len:  int = DEFAULT_SEQ_LEN,
    ) -> None:
        self.model_T  = model_T.eval().to(DEVICE)
        self.model_RH = model_RH.eval().to(DEVICE)
        self.scaler_T  = scaler_T
        self.scaler_RH = scaler_RH
        self.seq_len   = seq_len
        self._buffer: list[list[float]] = []  # rolling feature history

    def _state_to_row(self, state: dict) -> list[float]:
        return [
            float(state["T_Indoor"]),
            float(state["RH_Indoor"]),
            float(state["T_Outdoor"]),
            float(state["RH_Outdoor"]),
            float(state["AC_State"]),
            float(state.get("setpoint_proxy", state["T_Indoor"])),
            float(state.get("GHI", 0.0)),
            float(state.get("Occupancy_State", 1.0)),
            float(state["hour_sin"]),
            float(state["hour_cos"]),
        ]

    def reset(self, initial_obs: dict) -> None:
        """
        Initialise the rolling buffer by repeating the first observation.
        Call at episode start.
        """
        row = self._state_to_row(initial_obs)
        self._buffer = [row] * self.seq_len

    def predict(self, state: dict) -> dict:
        """
        Single-step prediction. Updates internal buffer.

        Parameters
        ----------
        state : dict with keys matching INPUT_FEATURES
                (setpoint_proxy, GHI, Occupancy_State are optional).

        Returns
        -------
        dict with T, RH predictions at each horizon, plus energy proxy.
        """
        row = self._state_to_row(state)
        # Update rolling buffer
        self._buffer.append(row)
        if len(self._buffer) > self.seq_len:
            self._buffer.pop(0)

        # Pad with first element if buffer not yet full (warm-up)
        buf = self._buffer
        if len(buf) < self.seq_len:
            buf = [self._buffer[0]] * (self.seq_len - len(buf)) + list(buf)

        seq = np.array(buf, dtype=np.float32)  # (seq_len, n_feat)

        # Apply scalers (both share the same feature space)
        seq_T  = self.scaler_T.transform(seq)
        seq_RH = self.scaler_RH.transform(seq)

        x_T  = torch.from_numpy(seq_T[None]).to(DEVICE)   # (1, seq, feat)
        x_RH = torch.from_numpy(seq_RH[None]).to(DEVICE)

        output: dict[str, float] = {}
        with torch.no_grad():
            pred_T  = self.model_T(x_T).cpu().numpy()[0]
            pred_RH = self.model_RH(x_RH).cpu().numpy()[0]

        for i, label in enumerate(HORIZON_LABELS):
            t_val = float(np.clip(pred_T[i], *TEMP_CLIP_RANGE_C))
            rh_val = float(np.clip(pred_RH[i], *RH_CLIP_RANGE_PCT))
            if not np.isfinite(pred_T[i]) or not np.isfinite(pred_RH[i]):
                raise ValueError(
                    f"GRU produced non-finite prediction at horizon {label}: "
                    f"T_raw={pred_T[i]}, RH_raw={pred_RH[i]}. This indicates a "
                    "genuine model/numerical problem (e.g. NaN weights), not "
                    "just an out-of-range value, and is not clamped."
                )
            output[f"T_Indoor_{label}"]  = t_val
            output[f"RH_Indoor_{label}"] = rh_val

        # Energy proxy (explicit formula, not learned)
        ac = float(state["AC_State"])
        dt = STEP_MINUTES / 60
        for steps, label in zip(HORIZONS, HORIZON_LABELS):
            output[f"energy_proxy_{label}_kWh"] = ac * P_RATED_KW * dt * steps

        return output

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_T.state_dict(),  model_dir / "gru_T.pt")
        torch.save(self.model_RH.state_dict(), model_dir / "gru_RH.pt")

        import pickle
        with open(model_dir / "scaler_T.pkl",  "wb") as f:
            pickle.dump(self.scaler_T,  f)
        with open(model_dir / "scaler_RH.pkl", "wb") as f:
            pickle.dump(self.scaler_RH, f)

        meta = {
            "seq_len": self.seq_len,
            "n_features": len(INPUT_FEATURES),
            "input_features": INPUT_FEATURES,
            "hidden_size": self.model_T.gru.hidden_size,
            "n_layers": self.model_T.gru.num_layers,
        }
        (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  Models saved to {model_dir}/")

    @classmethod
    def load(cls, model_dir: Path, device: torch.device = DEVICE) -> "DualGRUEnvModel":
        import pickle

        meta_path = model_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {model_dir}. Run training first.")

        meta = json.loads(meta_path.read_text())
        seq_len    = meta["seq_len"]
        n_features = meta["n_features"]
        hidden     = meta["hidden_size"]
        n_layers   = meta["n_layers"]

        model_T  = GRUEnvModel(n_features, len(T_TARGET_COLS),  hidden, n_layers)
        model_RH = GRUEnvModel(n_features, len(RH_TARGET_COLS), hidden, n_layers)
        model_T.load_state_dict(torch.load(model_dir / "gru_T.pt",  map_location=device))
        model_RH.load_state_dict(torch.load(model_dir / "gru_RH.pt", map_location=device))

        with open(model_dir / "scaler_T.pkl",  "rb") as f:
            scaler_T  = pickle.load(f)
        with open(model_dir / "scaler_RH.pkl", "rb") as f:
            scaler_RH = pickle.load(f)

        print(f"  Models loaded from {model_dir}/")
        return cls(model_T, model_RH, scaler_T, scaler_RH, seq_len)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Dual GRU surrogate env model.")
    p.add_argument(
        "--data", type=Path,
        default=Path("Data/Final_data/final_dataset_with_setpoint_proxy.csv"),
    )
    p.add_argument(
        "--model-dir", type=Path,
        default=Path("outputs/env_model_gru"),
    )
    p.add_argument(
        "--report", type=Path,
        default=Path("outputs/env_model_gru/training_report.json"),
    )
    p.add_argument("--seq-len",  type=int,   default=DEFAULT_SEQ_LEN)
    p.add_argument("--hidden",   type=int,   default=DEFAULT_HIDDEN)
    p.add_argument("--layers",   type=int,   default=DEFAULT_LAYERS)
    p.add_argument("--dropout",  type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--epochs",   type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch",    type=int,   default=DEFAULT_BATCH)
    p.add_argument("--lr",       type=float, default=DEFAULT_LR)
    p.add_argument("--patience", type=int,   default=DEFAULT_PATIENCE)
    p.add_argument("--val-ratio", type=float, default=0.1,
                   help="Fraction of train APs to use as validation (time-split within each AP).")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; load saved models and run holdout evaluation only.")
    return p.parse_args()


def _time_split_val(train_df: pd.DataFrame, val_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Within each AP, hold out the last `val_ratio` fraction of rows for
    validation. This respects temporal ordering and prevents data leakage.
    """
    val_parts, tr_parts = [], []
    for _, grp in train_df.groupby("location_id", sort=False):
        n = len(grp)
        cutoff = int(n * (1 - val_ratio))
        tr_parts.append(grp.iloc[:cutoff])
        val_parts.append(grp.iloc[cutoff:])
    return pd.concat(tr_parts), pd.concat(val_parts)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Dual GRU Surrogate Environment — Phase 2")
    print(f"  Device   : {DEVICE}")
    print(f"  seq_len  : {args.seq_len} steps ({args.seq_len * STEP_MINUTES} min look-back)")
    print(f"  hidden   : {args.hidden}  |  layers: {args.layers}  |  dropout: {args.dropout}")
    print(f"  Horizons : {[f'+{h*STEP_MINUTES}min' for h in HORIZONS]}")
    print(f"  Train AP : AP01–AP{TRAIN_AP_MAX}  |  Holdout: AP{TRAIN_AP_MAX+1}–AP49")
    print("=" * 60)

    df = load_and_prepare(args.data)
    train_df, holdout_df = split_train_holdout(df)

    report: dict = {
        "model_type": "GRU",
        "seq_len": args.seq_len,
        "hidden_size": args.hidden,
        "n_layers": args.layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch,
        "lr": args.lr,
        "device": str(DEVICE),
        "horizons_minutes": [h * STEP_MINUTES for h in HORIZONS],
        "p_rated_kw": P_RATED_KW,
        "energy_note": (
            "Energy is a proxy: AC_State × P_rated × Δt. "
            "No power meter in CAMaRSEC. Phase 8 replaces with smart plug."
        ),
        "train_ap_range": f"AP01–AP{TRAIN_AP_MAX}",
        "holdout_ap_range": f"AP{TRAIN_AP_MAX+1}–AP49",
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "input_features": INPUT_FEATURES,
        "metrics": {},
    }

    if args.eval_only:
        print("\nEval-only mode: loading saved models ...")
        dual_model = DualGRUEnvModel.load(args.model_dir)

        # Rebuild holdout datasets using saved scalers
        ds_h_T  = SequenceDataset(
            holdout_df, T_TARGET_COLS,  args.seq_len,
            scaler_X=dual_model.scaler_T,  fit_scaler=False
        )
        ds_h_RH = SequenceDataset(
            holdout_df, RH_TARGET_COLS, args.seq_len,
            scaler_X=dual_model.scaler_RH, fit_scaler=False
        )
        hl_T  = DataLoader(ds_h_T,  batch_size=args.batch, shuffle=False)
        hl_RH = DataLoader(ds_h_RH, batch_size=args.batch, shuffle=False)

        report["metrics"]["holdout_T"]  = evaluate_gru(
            dual_model.model_T,  hl_T,  T_TARGET_COLS,  "GRU_T",  "holdout AP41–49"
        )
        report["metrics"]["holdout_RH"] = evaluate_gru(
            dual_model.model_RH, hl_RH, RH_TARGET_COLS, "GRU_RH", "holdout AP41–49"
        )

    else:
        tr_df, val_df = _time_split_val(train_df, args.val_ratio)
        print(f"\n  Val split: {len(val_df):,} rows "
              f"(last {args.val_ratio*100:.0f}% per AP within train set)")

        # ── Model T ──────────────────────────────────────────────────────────
        print("\n[Model A — GRU_T  (Temperature)]")
        print("  Physical rationale: T_Indoor responds quickly to AC_State")
        print(f"  and setpoint_proxy. GRU sees {args.seq_len * STEP_MINUTES}-min history.")

        ds_tr_T  = SequenceDataset(tr_df,  T_TARGET_COLS, args.seq_len, fit_scaler=True)
        ds_val_T = SequenceDataset(val_df, T_TARGET_COLS, args.seq_len,
                                   scaler_X=ds_tr_T.scaler_X, fit_scaler=False)
        ds_h_T   = SequenceDataset(holdout_df, T_TARGET_COLS, args.seq_len,
                                   scaler_X=ds_tr_T.scaler_X, fit_scaler=False)

        tl_T  = DataLoader(ds_tr_T,  batch_size=args.batch, shuffle=True,  drop_last=True)
        vl_T  = DataLoader(ds_val_T, batch_size=args.batch, shuffle=False)
        hl_T  = DataLoader(ds_h_T,   batch_size=args.batch, shuffle=False)

        gru_T = GRUEnvModel(len(INPUT_FEATURES), len(T_TARGET_COLS),
                            args.hidden, args.layers, args.dropout)
        print(f"  Parameters: {sum(p.numel() for p in gru_T.parameters()):,}")
        train_gru(gru_T, tl_T, vl_T, args.epochs, args.lr, args.patience, "GRU_T")

        # ── Model RH ─────────────────────────────────────────────────────────
        print("\n[Model B — GRU_RH  (Humidity)]")
        print("  Physical rationale: RH_Indoor has higher inertia, driven")
        print(f"  by RH_Outdoor and latent moisture. {args.seq_len * STEP_MINUTES}-min context matters more.")

        ds_tr_RH  = SequenceDataset(tr_df,  RH_TARGET_COLS, args.seq_len, fit_scaler=True)
        ds_val_RH = SequenceDataset(val_df, RH_TARGET_COLS, args.seq_len,
                                    scaler_X=ds_tr_RH.scaler_X, fit_scaler=False)
        ds_h_RH   = SequenceDataset(holdout_df, RH_TARGET_COLS, args.seq_len,
                                    scaler_X=ds_tr_RH.scaler_X, fit_scaler=False)

        tl_RH = DataLoader(ds_tr_RH,  batch_size=args.batch, shuffle=True,  drop_last=True)
        vl_RH = DataLoader(ds_val_RH, batch_size=args.batch, shuffle=False)
        hl_RH = DataLoader(ds_h_RH,   batch_size=args.batch, shuffle=False)

        gru_RH = GRUEnvModel(len(INPUT_FEATURES), len(RH_TARGET_COLS),
                             args.hidden, args.layers, args.dropout)
        print(f"  Parameters: {sum(p.numel() for p in gru_RH.parameters()):,}")
        train_gru(gru_RH, tl_RH, vl_RH, args.epochs, args.lr, args.patience, "GRU_RH")

        # ── Build dual model wrapper ──────────────────────────────────────────
        dual_model = DualGRUEnvModel(
            gru_T, gru_RH,
            ds_tr_T.scaler_X, ds_tr_RH.scaler_X,
            seq_len=args.seq_len,
        )
        dual_model.save(args.model_dir)

        # ── Evaluation ────────────────────────────────────────────────────────
        report["metrics"]["train_T"]  = evaluate_gru(
            gru_T,  tl_T,  T_TARGET_COLS,  "GRU_T",  "train"
        )
        report["metrics"]["val_T"]    = evaluate_gru(
            gru_T,  vl_T,  T_TARGET_COLS,  "GRU_T",  "val"
        )
        report["metrics"]["holdout_T"] = evaluate_gru(
            gru_T,  hl_T,  T_TARGET_COLS,  "GRU_T",  "holdout AP41–49"
        )
        report["metrics"]["train_RH"] = evaluate_gru(
            gru_RH, tl_RH, RH_TARGET_COLS, "GRU_RH", "train"
        )
        report["metrics"]["val_RH"]   = evaluate_gru(
            gru_RH, vl_RH, RH_TARGET_COLS, "GRU_RH", "val"
        )
        report["metrics"]["holdout_RH"] = evaluate_gru(
            gru_RH, hl_RH, RH_TARGET_COLS, "GRU_RH", "holdout AP41–49"
        )

    # ── Energy proxy stats ───────────────────────────────────────────────────
    proxy_mean = holdout_df["energy_proxy_kWh"].mean()
    proxy_max  = holdout_df["energy_proxy_kWh"].max()
    report["energy_proxy_stats_holdout"] = {
        "mean_kWh_per_step": round(float(proxy_mean), 4),
        "max_kWh_per_step":  round(float(proxy_max),  4),
        "formula": f"AC_State × {P_RATED_KW} kW × {STEP_MINUTES}/60 h",
        "limitation": "No power meter in CAMaRSEC. Replace in Phase 8.",
    }

    # ── Save report ──────────────────────────────────────────────────────────
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report saved → {args.report}")
    print("\nDone.")


if __name__ == "__main__":
    main()