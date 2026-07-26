from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from GPro.preference import ProbitPreferenceGP
    from GPro.posterior import Laplace
    from scipy.stats import norm

    GPRO_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on optional package.
    ProbitPreferenceGP = None
    Laplace = object
    norm = None
    GPRO_IMPORT_ERROR = exc


DEFAULT_INPUT_FILE = Path("Data/Final_data/final_dataset_with_setpoint_proxy.csv")
DEFAULT_OUTPUT_FILE = Path("Data/Final_data/bayesian_preference_results.csv")
DEFAULT_REPORT_FILE = Path("Data/Final_data/bayesian_preference_report.json")
DEFAULT_CHINESE_PRIOR_FILE = Path("chinese_comfort_subset.csv")
CHINA_TSV_COLUMN = "D1.TSV"
CHINA_TCV_COLUMN = "D2.TCV"
CHINA_T_COLUMN = "E1.Indoor Air Temperature (℃)"
CHINA_RH_COLUMN = "E2.Indoor Relative Humidity (%)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate T* and RH* using Bayesian preference learning."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_FILE)
    parser.add_argument(
        "--prior-source",
        choices=["china", "fixed"],
        default="china",
        help=(
            "Use Guangdong Chinese comfort survey as Bayesian prior by default. "
            "Use fixed to keep the manual prior-t/rh values."
        ),
    )
    parser.add_argument("--china-prior-file", type=Path, default=DEFAULT_CHINESE_PRIOR_FILE)
    parser.add_argument("--china-comfort-tsv-abs", type=float, default=0.5)
    parser.add_argument("--china-comfort-tcv-max", type=float, default=2.0)
    parser.add_argument("--china-prior-t-min", type=float, default=18.0)
    parser.add_argument("--china-prior-t-max", type=float, default=35.0)
    parser.add_argument("--china-prior-rh-min", type=float, default=30.0)
    parser.add_argument("--china-prior-rh-max", type=float, default=95.0)
    parser.add_argument("--min-china-prior-rows", type=int, default=100)
    parser.add_argument(
        "--method",
        choices=["auto", "gpro", "grid"],
        default="grid",
        help=(
            "Corrected default is grid because lightweight GPro extrapolated to "
            "grid boundaries on this dataset. GPro remains available explicitly."
        ),
    )
    parser.add_argument("--t-min", type=float, default=23.0)
    parser.add_argument("--t-max", type=float, default=32.0)
    parser.add_argument("--t-step", type=float, default=0.25)
    parser.add_argument("--rh-min", type=float, default=55.0)
    parser.add_argument("--rh-max", type=float, default=85.0)
    parser.add_argument("--rh-step", type=float, default=1.0)
    parser.add_argument("--prior-t-mean", type=float, default=26.0)
    parser.add_argument("--prior-t-sigma", type=float, default=3.0)
    parser.add_argument("--prior-rh-mean", type=float, default=65.0)
    parser.add_argument("--prior-rh-sigma", type=float, default=12.0)
    parser.add_argument("--setpoint-sigma", type=float, default=3.5)
    parser.add_argument("--comfort-t-sigma", type=float, default=1.8)
    parser.add_argument("--comfort-rh-sigma", type=float, default=9.0)
    parser.add_argument("--k-temp", type=float, default=1.2)
    parser.add_argument("--k-rh", type=float, default=0.08)
    parser.add_argument("--setpoint-weight", type=float, default=0.25)
    parser.add_argument("--ac-on-weight", type=float, default=0.7)
    parser.add_argument("--ac-off-weight", type=float, default=0.4)
    parser.add_argument("--stable-weight", type=float, default=0.15)
    parser.add_argument("--max-stable-per-location", type=int, default=60)
    parser.add_argument("--max-gpro-pairs-per-location", type=int, default=40)
    parser.add_argument("--min-setpoint-observations", type=int, default=5)
    parser.add_argument(
        "--location-limit",
        type=int,
        default=None,
        help="Optional debug limit for number of locations to process.",
    )
    parser.add_argument(
        "--room-filter",
        type=str,
        default="BR",
        help=(
            "Room type to keep after loading the dataset (default: 'BR' = bedroom). "
            "Pass an empty string or 'all' to disable filtering and keep all rooms."
        ),
    )
    return parser.parse_args()


def apply_prior(args: argparse.Namespace) -> dict[str, object]:
    if args.prior_source == "fixed":
        return {
            "source": "fixed",
            "prior_t_mean": float(args.prior_t_mean),
            "prior_t_sigma": float(args.prior_t_sigma),
            "prior_rh_mean": float(args.prior_rh_mean),
            "prior_rh_sigma": float(args.prior_rh_sigma),
            "rows_used": None,
            "file": None,
        }

    prior = compute_chinese_guangdong_prior(args)
    args.prior_t_mean = prior["prior_t_mean"]
    args.prior_t_sigma = prior["prior_t_sigma"]
    args.prior_rh_mean = prior["prior_rh_mean"]
    args.prior_rh_sigma = prior["prior_rh_sigma"]
    return prior


def compute_chinese_guangdong_prior(args: argparse.Namespace) -> dict[str, object]:
    required_columns = {
        CHINA_TSV_COLUMN,
        CHINA_TCV_COLUMN,
        CHINA_T_COLUMN,
        CHINA_RH_COLUMN,
    }
    df = pd.read_csv(
        args.china_prior_file,
        usecols=lambda column: column in required_columns,
    )
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(
            f"Missing required Chinese prior columns in {args.china_prior_file}: "
            f"{sorted(missing)}"
        )

    t = pd.to_numeric(df[CHINA_T_COLUMN], errors="coerce")
    rh = pd.to_numeric(df[CHINA_RH_COLUMN], errors="coerce")
    tsv = pd.to_numeric(df[CHINA_TSV_COLUMN], errors="coerce")
    tcv = pd.to_numeric(df[CHINA_TCV_COLUMN], errors="coerce")
    comfort_mask = (
        t.between(args.china_prior_t_min, args.china_prior_t_max)
        & rh.between(args.china_prior_rh_min, args.china_prior_rh_max)
        & tsv.abs().le(args.china_comfort_tsv_abs)
        & tcv.le(args.china_comfort_tcv_max)
    )
    comfort = pd.DataFrame({"T": t[comfort_mask], "RH": rh[comfort_mask]}).dropna()
    if len(comfort) < args.min_china_prior_rows:
        raise ValueError(
            f"Only {len(comfort)} Chinese comfort rows matched the prior filter; "
            f"need at least {args.min_china_prior_rows}."
        )

    prior_t_sigma = max(float(comfort["T"].std(ddof=1)), 1.0)
    prior_rh_sigma = max(float(comfort["RH"].std(ddof=1)), 5.0)
    return {
        "source": "china_guangdong_comfort",
        "file": str(args.china_prior_file),
        "rows_total": int(len(df)),
        "rows_used": int(len(comfort)),
        "temperature_column": CHINA_T_COLUMN,
        "humidity_column": CHINA_RH_COLUMN,
        "comfort_filter": {
            "abs_TSV_lte": float(args.china_comfort_tsv_abs),
            "TCV_lte": float(args.china_comfort_tcv_max),
            "T_range": [float(args.china_prior_t_min), float(args.china_prior_t_max)],
            "RH_range": [float(args.china_prior_rh_min), float(args.china_prior_rh_max)],
        },
        "prior_t_mean": float(comfort["T"].mean()),
        "prior_t_sigma": prior_t_sigma,
        "prior_rh_mean": float(comfort["RH"].mean()),
        "prior_rh_sigma": prior_rh_sigma,
    }


class CompatLaplace(Laplace):
    """Laplace approximation compatible with modern NumPy scalar assignment."""

    def __call__(self, f, M, K):
        if norm is None:
            raise ImportError("scipy is required for GPro CompatLaplace.")

        def z(f_values: np.ndarray, preferences: np.ndarray) -> np.ndarray:
            preferred, worse = preferences[:, 0], preferences[:, 1]
            return (
                (f_values[preferred] - f_values[worse])
                / np.sqrt(2 * self.s_eval)
            ).flatten()

        def delta(f_values: np.ndarray, preferences: np.ndarray, kernel: np.ndarray):
            n_values = len(f_values)
            b = np.zeros(n_values)
            for i in range(n_values):
                preferred_here = preferences[:, 0] == i
                worse_here = preferences[:, 1] == i
                z_preferred = z(f_values, preferences[preferred_here, :])
                z_worse = z(f_values, preferences[worse_here, :])
                pos = norm.pdf(z_preferred) / norm.cdf(z_preferred)
                neg = norm.pdf(z_worse) / norm.cdf(z_worse)
                b[i] = (sum(pos) - sum(neg)) / np.sqrt(2 * self.s_eval)

            c = np.zeros((n_values, n_values))
            diag_obs = (norm.pdf(0) / norm.cdf(0)) ** 2 / 2 / self.s_eval
            unique_preferences = np.unique(preferences, axis=0)
            for i in range(unique_preferences.shape[0]):
                preferred, worse = unique_preferences[i, 0], unique_preferences[i, 1]
                z_pair = z(f_values, unique_preferences[[i], :])
                pdf_z = norm.pdf(z_pair)
                cdf_z = norm.cdf(z_pair)
                c_pair = (pdf_z / cdf_z) ** 2 + pdf_z / cdf_z * z_pair
                c_pair = float(np.ravel(c_pair)[0])
                c[preferred][worse] -= c_pair / 2 / self.s_eval
                c[worse][preferred] -= c_pair / 2 / self.s_eval
                c[preferred][preferred] += diag_obs
                c[worse][worse] += diag_obs

            kernel_f = np.linalg.solve(kernel, f_values)
            gradient = kernel_f.flatten() - b
            hessian = np.linalg.inv(kernel) + c
            return gradient, hessian

        f_new = f
        f_old = f
        eps = self.tol + 1
        iteration = 0
        while iteration < self.max_iter and eps > self.tol:
            gradient, hessian = delta(f_old, M, K)
            f_new = f_old - self.eta * np.linalg.solve(hessian, gradient).reshape(-1, 1)
            eps = np.linalg.norm(f_new - f_old, ord=2)
            f_old = f_new
            iteration += 1

        _, hessian = delta(f_new, M, K)
        c = hessian - np.linalg.inv(K)
        return f_new, c


def read_input(path: Path, room_filter: str = "BR") -> pd.DataFrame:
    required_columns = {
        "timestamp",
        "apartment_id",
        "room",
        "location_id",
        "T_Indoor",
        "indoor_RH",
        "AC_State",
        "occupancy",
        "setpoint_proxy",
    }
    df = pd.read_csv(
        path,
        usecols=lambda column: column in required_columns,
        parse_dates=["timestamp"],
    )
    df = df.rename(columns={"indoor_RH": "RH_Indoor", "occupancy": "Occupancy_State_SMTH"})
    
    missing = {"timestamp", "apartment_id", "room", "location_id", "T_Indoor", "RH_Indoor", "AC_State", "Occupancy_State_SMTH", "setpoint_proxy"}.difference(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in {path}: {sorted(missing)}")

    for column in ["T_Indoor", "RH_Indoor", "AC_State", "Occupancy_State_SMTH", "setpoint_proxy"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["AC_State"] = df["AC_State"].fillna(0).astype(int)
    df["Occupancy_State_SMTH"] = df["Occupancy_State_SMTH"].fillna(0).astype(int)

    # ---------------------------------------------------------------------------
    # Room filter — keep only the requested room type (default: BR = bedroom).
    #
    # Lý do 1 — BR signal sạch hơn (signal cleaner):
    #   Phòng ngủ (BR) là không gian sử dụng cá nhân, thường chỉ có 1 người hoặc
    #   một hộ gia đình nhỏ. Hành vi bật/tắt AC và thay đổi setpoint trong BR phản
    #   ánh trực tiếp sở thích nhiệt của người dùng mà không bị pha trộn bởi nhu
    #   cầu của nhiều người khác. Ngược lại, phòng khách (LR) thường có nhiều
    #   người với sở thích khác nhau, nên AC_State và setpoint_proxy phản ánh sự
    #   thỏa hiệp tập thể chứ không phải preference cá nhân.
    #
    # Lý do 2 — LR noisy hơn (living room noisier):
    #   Trong dataset CAMaRSEC/AP1_BR, tín hiệu AC_State tại LR có tỷ lệ chuyển
    #   đổi on/off cao hơn và không nhất quán hơn BR do: (a) lưu lượng người ra
    #   vào lớn hơn, (b) hoạt động nấu ăn/sinh hoạt tạo nhiễu nhiệt cục bộ,
    #   (c) cửa sổ/cửa ra vào thường xuyên mở hơn làm T_Indoor biến động mạnh.
    #   Noise này làm suy giảm chất lượng likelihood p(AC_on|θ,x) và p(AC_off|θ,x)
    #   trong bước Bayesian update, dẫn đến posterior T*, RH* kém ổn định.
    #
    # Lý do 3 — Phase 2 consistency (nhất quán với surrogate environment model):
    #   Surrogate environment model ở Phase 2 được huấn luyện chủ yếu trên dữ liệu
    #   BR (thermal dynamics phòng ngủ — thể tích nhỏ, ít cửa sổ, ít nguồn nhiệt
    #   bên ngoài). Nếu Phase 3 dùng cả LR để suy luận T* và RH*, thì preference
    #   được học sẽ không tương thích với môi trường mà RL agent sẽ vận hành —
    #   tạo ra sự không nhất quán giữa inferred preference và reward function
    #   trong Phase 4.
    #
    # Thesis note — ac_off likelihood approximation:
    #   Trong implementation hiện tại, sự kiện AC_off được mô hình hóa bằng
    #   Gaussian comfort likelihood quanh (T_Indoor, RH_Indoor) tại thời điểm tắt
    #   máy, với giả định rằng người dùng tắt AC vì đã đạt đến trạng thái thoải
    #   mái. Đây là một xấp xỉ vì trên thực tế người dùng có thể tắt AC vì nhiều
    #   lý do khác (ngủ quên bật, ra ngoài, tiết kiệm điện). Mô hình chấp nhận
    #   xấp xỉ này với trọng số ac_off_weight thấp hơn ac_on_weight để giảm tác
    #   động của các false-positive comfort signal. Tương tự, proxy_setpoint được
    #   dùng thay cho setpoint thực vì dataset Hà Nội không có trường setpoint
    #   trực tiếp — đây cũng là một xấp xỉ cần được thừa nhận trong phần
    #   Limitations của luận văn.
    # ---------------------------------------------------------------------------
    _filter = room_filter.strip().lower()
    if _filter and _filter != "all":
        before = len(df)
        df = df[df["room"].str.upper() == room_filter.strip().upper()]
        after = len(df)
        if after == 0:
            raise ValueError(
                f"room_filter='{room_filter}' removed all rows from {path}. "
                f"Check that the 'room' column contains the value '{room_filter}'."
            )
        dropped = before - after
        if dropped > 0:
            # Informational only — callers can suppress by passing room_filter=""
            import warnings
            warnings.warn(
                f"read_input: room_filter='{room_filter}' dropped {dropped} rows "
                f"({dropped / before:.1%} of total). {after} rows kept.",
                stacklevel=2,
            )

    return df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)


def add_behavior_columns(df: pd.DataFrame) -> pd.DataFrame:
    group = df.groupby("location_id", sort=False)
    previous_ac = group["AC_State"].shift(1)
    df["ac_on_event"] = (previous_ac.eq(0) & df["AC_State"].eq(1)).astype(int)
    df["ac_off_event"] = (previous_ac.eq(1) & df["AC_State"].eq(0)).astype(int)
    df["cooling_episode_id"] = group["ac_on_event"].cumsum()
    df.loc[df["AC_State"].ne(1), "cooling_episode_id"] = np.nan

    temp_delta = df["T_Indoor"] - group["T_Indoor"].shift(1)
    rh_delta = df["RH_Indoor"] - group["RH_Indoor"].shift(1)
    no_switch = previous_ac.notna() & df["AC_State"].eq(previous_ac)
    df["stable_comfort_period"] = (
        df["Occupancy_State_SMTH"].eq(1)
        & no_switch
        & temp_delta.abs().le(0.3)
        & rh_delta.abs().le(2.0)
    ).astype(int)
    return df


def make_grid(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_values = np.round(np.arange(args.t_min, args.t_max + args.t_step / 2, args.t_step), 4)
    rh_values = np.round(np.arange(args.rh_min, args.rh_max + args.rh_step / 2, args.rh_step), 4)
    t_grid = t_values[:, None]
    rh_grid = rh_values[None, :]
    return t_values, rh_values, t_grid, rh_grid


def log_sigmoid(value: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -np.clip(value, -60, 60))


def evenly_sample(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    positions = np.linspace(0, len(frame) - 1, max_rows).round().astype(int)
    return frame.iloc[np.unique(positions)]


def location_evidence(location_df: pd.DataFrame, max_stable: int) -> dict[str, pd.DataFrame | np.ndarray]:
    valid_context = location_df.dropna(subset=["T_Indoor", "RH_Indoor"])
    setpoint_rows = location_df.dropna(subset=["setpoint_proxy", "cooling_episode_id"])
    if not setpoint_rows.empty:
        setpoints = (
            setpoint_rows.groupby("cooling_episode_id", sort=False)["setpoint_proxy"]
            .first()
            .dropna()
            .to_numpy()
        )
    else:
        setpoints = np.array([], dtype=float)

    ac_on = valid_context[valid_context["ac_on_event"].eq(1)][["T_Indoor", "RH_Indoor"]]
    ac_off = valid_context[
        valid_context["ac_off_event"].eq(1)
        & valid_context["Occupancy_State_SMTH"].eq(1)
    ][["T_Indoor", "RH_Indoor"]]
    stable = valid_context[valid_context["stable_comfort_period"].eq(1)][
        ["T_Indoor", "RH_Indoor"]
    ]
    stable = evenly_sample(stable, max_stable)

    return {
        "setpoints": setpoints,
        "ac_on": ac_on,
        "ac_off": ac_off,
        "stable": stable,
    }


def invalid_profile_record(
    location_df: pd.DataFrame,
    evidence: dict[str, pd.DataFrame | np.ndarray],
    args: argparse.Namespace,
    method: str,
    reason: str,
    gpro_pairs: int = 0,
) -> dict[str, float | int | str]:
    first = location_df.iloc[0]
    return {
        "apartment_id": first["apartment_id"],
        "room": first["room"],
        "location_id": first["location_id"],
        "T_star": np.nan,
        "sigma_T": np.nan,
        "RH_star": np.nan,
        "sigma_RH": np.nan,
        "T_mean": np.nan,
        "RH_mean": np.nan,
        "posterior_map_probability": np.nan,
        "rows": int(len(location_df)),
        "setpoint_observations": int(len(evidence["setpoints"])),
        "ac_on_events": int(len(evidence["ac_on"])),
        "ac_off_events": int(len(evidence["ac_off"])),
        "stable_observations_used": int(len(evidence["stable"])),
        "method": method,
        "gpro_pairs": gpro_pairs,
        "profile_status": reason,
        "valid_for_reward": 0,
        "min_setpoint_observations": int(args.min_setpoint_observations),
    }


def estimate_location_grid(
    location_df: pd.DataFrame,
    args: argparse.Namespace,
    t_values: np.ndarray,
    rh_values: np.ndarray,
    t_grid: np.ndarray,
    rh_grid: np.ndarray,
) -> dict[str, float | int | str]:
    log_posterior = (
        -0.5 * ((t_grid - args.prior_t_mean) / args.prior_t_sigma) ** 2
        -0.5 * ((rh_grid - args.prior_rh_mean) / args.prior_rh_sigma) ** 2
    )

    evidence = location_evidence(location_df, args.max_stable_per_location)
    if len(evidence["setpoints"]) < args.min_setpoint_observations:
        return invalid_profile_record(
            location_df,
            evidence,
            args,
            method="grid_bayes",
            reason="insufficient_setpoint_evidence",
        )

    for setpoint in evidence["setpoints"]:
        log_posterior += args.setpoint_weight * (
            -0.5 * ((t_grid - setpoint) / args.setpoint_sigma) ** 2
        )

    for row in evidence["ac_on"].itertuples(index=False):
        z = args.k_temp * (row.T_Indoor - t_grid) + args.k_rh * (row.RH_Indoor - rh_grid)
        log_posterior += args.ac_on_weight * log_sigmoid(z)

    for row in evidence["ac_off"].itertuples(index=False):
        log_posterior += args.ac_off_weight * (
            -0.5 * ((t_grid - row.T_Indoor) / args.comfort_t_sigma) ** 2
            -0.5 * ((rh_grid - row.RH_Indoor) / args.comfort_rh_sigma) ** 2
        )

    for row in evidence["stable"].itertuples(index=False):
        log_posterior += args.stable_weight * (
            -0.5 * ((t_grid - row.T_Indoor) / args.comfort_t_sigma) ** 2
            -0.5 * ((rh_grid - row.RH_Indoor) / args.comfort_rh_sigma) ** 2
        )

    max_log = float(np.max(log_posterior))
    posterior = np.exp(log_posterior - max_log)
    posterior /= posterior.sum()

    map_t_index, map_rh_index = np.unravel_index(
        np.argmax(posterior),
        posterior.shape,
    )
    t_marginal = posterior.sum(axis=1)
    rh_marginal = posterior.sum(axis=0)

    t_mean = float(np.sum(t_values * t_marginal))
    rh_mean = float(np.sum(rh_values * rh_marginal))
    sigma_t = float(np.sqrt(np.sum(((t_values - t_mean) ** 2) * t_marginal)))
    sigma_rh = float(np.sqrt(np.sum(((rh_values - rh_mean) ** 2) * rh_marginal)))

    first = location_df.iloc[0]
    return {
        "apartment_id": first["apartment_id"],
        "room": first["room"],
        "location_id": first["location_id"],
        "T_star": float(t_values[map_t_index]),
        "sigma_T": sigma_t,
        "RH_star": float(rh_values[map_rh_index]),
        "sigma_RH": sigma_rh,
        "T_mean": t_mean,
        "RH_mean": rh_mean,
        "posterior_map_probability": float(posterior[map_t_index, map_rh_index]),
        "rows": int(len(location_df)),
        "setpoint_observations": int(len(evidence["setpoints"])),
        "ac_on_events": int(len(evidence["ac_on"])),
        "ac_off_events": int(len(evidence["ac_off"])),
        "stable_observations_used": int(len(evidence["stable"])),
        "method": "grid_bayes",
        "gpro_pairs": 0,
        "profile_status": "low_confidence" if len(evidence["setpoints"]) < 20 else "valid",
        "valid_for_reward": 1,
        "min_setpoint_observations": int(args.min_setpoint_observations),
    }


def clip_point(
    point: tuple[float, float],
    args: argparse.Namespace,
) -> tuple[float, float]:
    temperature, humidity = point
    temperature = min(max(temperature, args.t_min), args.t_max)
    humidity = min(max(humidity, args.rh_min), args.rh_max)
    return round(float(temperature), 3), round(float(humidity), 3)


def add_preference_pair(
    point_to_index: dict[tuple[float, float], int],
    points: list[tuple[float, float]],
    pairs: list[tuple[int, int]],
    preferred: tuple[float, float],
    worse: tuple[float, float],
    args: argparse.Namespace,
) -> None:
    preferred = clip_point(preferred, args)
    worse = clip_point(worse, args)
    if preferred == worse:
        return

    for point in [preferred, worse]:
        if point not in point_to_index:
            point_to_index[point] = len(points)
            points.append(point)
    pairs.append((point_to_index[preferred], point_to_index[worse]))


def build_gpro_preferences(
    location_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    point_to_index: dict[tuple[float, float], int] = {}
    points: list[tuple[float, float]] = []
    pairs: list[tuple[int, int]] = []

    evidence = location_evidence(location_df, args.max_stable_per_location)
    setpoint_rows = location_df.dropna(subset=["setpoint_proxy", "cooling_episode_id"])
    if not setpoint_rows.empty:
        setpoint_rows = setpoint_rows.groupby("cooling_episode_id", sort=False).first()

    for row in setpoint_rows.itertuples(index=False):
        if pd.isna(row.T_Indoor) or pd.isna(row.RH_Indoor):
            continue
        add_preference_pair(
            point_to_index,
            points,
            pairs,
            preferred=(row.setpoint_proxy, row.RH_Indoor),
            worse=(row.T_Indoor, row.RH_Indoor),
            args=args,
        )

    for row in evidence["ac_on"].itertuples(index=False):
        add_preference_pair(
            point_to_index,
            points,
            pairs,
            preferred=(row.T_Indoor - 1.5, row.RH_Indoor - 8.0),
            worse=(row.T_Indoor, row.RH_Indoor),
            args=args,
        )

    for row in evidence["ac_off"].itertuples(index=False):
        add_preference_pair(
            point_to_index,
            points,
            pairs,
            preferred=(row.T_Indoor, row.RH_Indoor),
            worse=(row.T_Indoor - 2.0, row.RH_Indoor),
            args=args,
        )
        add_preference_pair(
            point_to_index,
            points,
            pairs,
            preferred=(row.T_Indoor, row.RH_Indoor),
            worse=(row.T_Indoor + 2.0, row.RH_Indoor + 8.0),
            args=args,
        )

    for row in evidence["stable"].itertuples(index=False):
        add_preference_pair(
            point_to_index,
            points,
            pairs,
            preferred=(row.T_Indoor, row.RH_Indoor),
            worse=(row.T_Indoor + 2.0, row.RH_Indoor + 8.0),
            args=args,
        )

    if len(pairs) > args.max_gpro_pairs_per_location:
        indices = np.linspace(0, len(pairs) - 1, args.max_gpro_pairs_per_location)
        pairs = [pairs[index] for index in np.unique(indices.round().astype(int))]

    if pairs:
        compact_point_to_index: dict[tuple[float, float], int] = {}
        compact_points: list[tuple[float, float]] = []
        compact_pairs: list[tuple[int, int]] = []
        for preferred_old, worse_old in pairs:
            compact_pair = []
            for old_index in [preferred_old, worse_old]:
                point = points[old_index]
                if point not in compact_point_to_index:
                    compact_point_to_index[point] = len(compact_points)
                    compact_points.append(point)
                compact_pair.append(compact_point_to_index[point])
            compact_pairs.append((compact_pair[0], compact_pair[1]))
        points = compact_points
        pairs = compact_pairs

    stats = {
        "setpoint_observations": int(len(evidence["setpoints"])),
        "ac_on_events": int(len(evidence["ac_on"])),
        "ac_off_events": int(len(evidence["ac_off"])),
        "stable_observations_used": int(len(evidence["stable"])),
        "gpro_pairs": int(len(pairs)),
    }

    return (
        np.asarray(points, dtype="float64"),
        np.asarray(pairs, dtype="int64"),
        stats,
    )


def estimate_location_gpro(
    location_df: pd.DataFrame,
    args: argparse.Namespace,
    t_values: np.ndarray,
    rh_values: np.ndarray,
) -> dict[str, float | int | str]:
    if ProbitPreferenceGP is None:
        raise ImportError(
            "GPro is not installed. Install it with `pip install GPro` or run with "
            "`--method grid`."
        ) from GPRO_IMPORT_ERROR

    evidence_for_validation = location_evidence(location_df, args.max_stable_per_location)
    if len(evidence_for_validation["setpoints"]) < args.min_setpoint_observations:
        return invalid_profile_record(
            location_df,
            evidence_for_validation,
            args,
            method="gpro",
            reason="insufficient_setpoint_evidence",
        )

    x_train, preferences, stats = build_gpro_preferences(location_df, args)
    if len(x_train) < 2 or len(preferences) < 1:
        raise ValueError("Not enough preference pairs for GPro.")

    def scale_points(points: np.ndarray) -> np.ndarray:
        scaled = points.copy().astype("float64")
        scaled[:, 0] = (scaled[:, 0] - args.t_min) / (args.t_max - args.t_min)
        scaled[:, 1] = (scaled[:, 1] - args.rh_min) / (args.rh_max - args.rh_min)
        return scaled

    x_train_scaled = scale_points(x_train)
    model = ProbitPreferenceGP(
        alpha=1e-5,
        post_approx=CompatLaplace(s_eval=1e-5, max_iter=500, eta=0.01, tol=1e-4),
    )
    model.fit(x_train_scaled, preferences, f_prior=None)

    mesh_t, mesh_rh = np.meshgrid(t_values, rh_values, indexing="ij")
    grid = np.column_stack([mesh_t.ravel(), mesh_rh.ravel()])
    utility, utility_std = model.predict(scale_points(grid), return_y_std=True)
    utility = np.asarray(utility).reshape(len(t_values), len(rh_values))
    utility_std = np.asarray(utility_std).reshape(len(t_values), len(rh_values))

    map_t_index, map_rh_index = np.unravel_index(np.argmax(utility), utility.shape)
    weights = np.exp(utility - np.max(utility))
    weights /= weights.sum()
    t_marginal = weights.sum(axis=1)
    rh_marginal = weights.sum(axis=0)
    t_mean = float(np.sum(t_values * t_marginal))
    rh_mean = float(np.sum(rh_values * rh_marginal))
    sigma_t = float(np.sqrt(np.sum(((t_values - t_mean) ** 2) * t_marginal)))
    sigma_rh = float(np.sqrt(np.sum(((rh_values - rh_mean) ** 2) * rh_marginal)))

    first = location_df.iloc[0]
    return {
        "apartment_id": first["apartment_id"],
        "room": first["room"],
        "location_id": first["location_id"],
        "T_star": float(t_values[map_t_index]),
        "sigma_T": sigma_t,
        "RH_star": float(rh_values[map_rh_index]),
        "sigma_RH": sigma_rh,
        "T_mean": t_mean,
        "RH_mean": rh_mean,
        "posterior_map_probability": float(weights[map_t_index, map_rh_index]),
        "rows": int(len(location_df)),
        "setpoint_observations": stats["setpoint_observations"],
        "ac_on_events": stats["ac_on_events"],
        "ac_off_events": stats["ac_off_events"],
        "stable_observations_used": stats["stable_observations_used"],
        "method": "gpro",
        "gpro_pairs": stats["gpro_pairs"],
        "gpro_utility_at_map": float(utility[map_t_index, map_rh_index]),
        "gpro_std_at_map": float(utility_std[map_t_index, map_rh_index]),
        "profile_status": "low_confidence" if stats["setpoint_observations"] < 20 else "valid",
        "valid_for_reward": 1,
        "min_setpoint_observations": int(args.min_setpoint_observations),
    }


def bayesian_preference_learning(args: argparse.Namespace) -> pd.DataFrame:
    df = add_behavior_columns(read_input(args.input, room_filter=args.room_filter))
    t_values, rh_values, t_grid, rh_grid = make_grid(args)
    records = []
    grouped_locations = df.groupby("location_id", sort=True)
    if args.location_limit is not None:
        grouped_locations = list(grouped_locations)[: args.location_limit]

    for _, location_df in grouped_locations:
        use_gpro = args.method == "gpro" or (
            args.method == "auto" and ProbitPreferenceGP is not None
        )
        if use_gpro:
            try:
                records.append(estimate_location_gpro(location_df, args, t_values, rh_values))
                continue
            except Exception:
                if args.method == "gpro":
                    raise

        records.append(
            estimate_location_grid(
                location_df,
                args,
                t_values,
                rh_values,
                t_grid,
                rh_grid,
            )
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    prior = apply_prior(args)
    results = bayesian_preference_learning(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    valid_results = results[results["valid_for_reward"].eq(1)].copy()
    invalid_results = results[results["valid_for_reward"].ne(1)].copy()
    t_boundary = int(
        valid_results["T_star"].isin([float(args.t_min), float(args.t_max)]).sum()
    )
    rh_boundary = int(
        valid_results["RH_star"].isin([float(args.rh_min), float(args.rh_max)]).sum()
    )

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "locations": int(len(results)),
        "valid_locations": int(len(valid_results)),
        "invalid_locations": int(len(invalid_results)),
        "mean_T_star": None if valid_results.empty else float(valid_results["T_star"].mean()),
        "mean_RH_star": None if valid_results.empty else float(valid_results["RH_star"].mean()),
        "mean_sigma_T": None if valid_results.empty else float(valid_results["sigma_T"].mean()),
        "mean_sigma_RH": None if valid_results.empty else float(valid_results["sigma_RH"].mean()),
        "t_boundary_locations": t_boundary,
        "rh_boundary_locations": rh_boundary,
        "invalid_status_counts": invalid_results["profile_status"].value_counts().to_dict(),
        "requested_method": args.method,
        "methods_used": results["method"].value_counts().to_dict(),
        "gpro_available": ProbitPreferenceGP is not None,
        "gpro_import_error": None if GPRO_IMPORT_ERROR is None else str(GPRO_IMPORT_ERROR),
        "prior": prior,
        "config": {
            "t_grid": [args.t_min, args.t_max, args.t_step],
            "rh_grid": [args.rh_min, args.rh_max, args.rh_step],
            "prior_source": args.prior_source,
            "max_stable_per_location": args.max_stable_per_location,
            "max_gpro_pairs_per_location": args.max_gpro_pairs_per_location,
            "min_setpoint_observations": args.min_setpoint_observations,
            "prior_t_mean": args.prior_t_mean,
            "prior_t_sigma": args.prior_t_sigma,
            "prior_rh_mean": args.prior_rh_mean,
            "prior_rh_sigma": args.prior_rh_sigma,
            "setpoint_weight": args.setpoint_weight,
            "setpoint_sigma": args.setpoint_sigma,
            "ac_on_weight": args.ac_on_weight,
            "ac_off_weight": args.ac_off_weight,
            "stable_weight": args.stable_weight,
        },
        "fixes_applied": [
            "Corrected default method is grid_bayes, not lightweight GPro, because GPro extrapolated to grid boundaries on this dataset.",
            "Temperature grid narrowed from [18, 32] to [23, 29] to avoid unsupported extrapolation outside observed cooling-response setpoints.",
            "Humidity grid narrowed from [40, 90] to [55, 85] to reduce boundary artifacts.",
            "Locations with fewer than min_setpoint_observations are marked invalid and should not be used directly in Phase 4 reward.",
            "setpoint_proxy remains preference-only evidence and must not be used as Phase 2/5 environment state input because it is look-forward.",
            "Guangdong Chinese thermal-comfort rows now define the regional prior; Vietnam/CAMaRSEC behavior data updates the posterior for each location.",
        ],
        "model": {
            "gpro": (
                "Uses GPro.preference.ProbitPreferenceGP on inferred pairwise "
                "preferences when available. T_star/RH_star are selected by "
                "maximum predicted utility over the configured grid."
            ),
            "grid_bayes_prior_T": f"Normal({args.prior_t_mean}, {args.prior_t_sigma}^2)",
            "grid_bayes_prior_RH": f"Normal({args.prior_rh_mean}, {args.prior_rh_sigma}^2)",
            "prior_role": "Chinese Guangdong comfort survey defines population/regional prior.",
            "evidence_role": "Vietnam/CAMaRSEC HVAC behavior updates posterior per apartment-room.",
            "setpoint_likelihood": "Gaussian around setpoint_proxy for T*",
            "ac_on_likelihood": "sigmoid(k_temp*(T_in - T*) + k_rh*(RH_in - RH*))",
            "ac_off_likelihood": "Gaussian comfort evidence around observed T/RH",
            "stable_likelihood": "Low-weight Gaussian comfort evidence around observed T/RH",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved Bayesian preference results to {args.output}")
    print(f"Saved report to {args.report}")
    print(f"Locations: {len(results)}")


if __name__ == "__main__":
    main()
