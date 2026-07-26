"""
Tien xu ly Chinese Thermal Comfort Dataset (Class I/II/III)
Loc theo:
  - Tinh/thanh pho khi hau nong am giong Viet Nam: Quang Dong, Quang Tay,
    Hai Nam, Van Nam, Phuc Kien
  - Chi giu Summer Season
Gop 3 file thanh 1 dataset chung.
"""
import pandas as pd
import numpy as np

INPUT_FILES = [
    "/mnt/user-data/uploads/Chinese-Thermal-Comfort-Dataset-Class-I.csv",
    "/mnt/user-data/uploads/Chinese-Thermal-Comfort-Dataset-Class-II.csv",
    "/mnt/user-data/uploads/Chinese-Thermal-Comfort-Dataset-Class-III.csv",
]

# Cac tinh/thanh co khi hau nong am gan giong Viet Nam nhat
TARGET_PROVINCES = [
    "Guangdong",   # Quang Dong
    "Guangxi",     # Quang Tay
    "Hainan",      # Hai Nam
    "Yunnan",      # Van Nam
    "Fujian",      # Phuc Kien
]

KEEP_SEASON = "Summer Season"

RENAME_MAP = {
    "ID": "id",
    "A1.Code": "code",
    "A2.Date": "date",
    "A3.Data Contributor": "data_contributor",
    "A4.Season": "season",
    "A5.City": "city",
    "A6.Climate Zone": "climate_zone",
    "B1.Building Type": "building_type",
    "B2.Building Function": "building_function",
    "B3.Floors": "floors",
    "B4.Building Operation Mode": "building_operation_mode",
    "B5.Room (Length\xa1\xc1Width)": "room_dimension",
    "B5.Room Height (m)": "room_height_m",
    "C1.Sex": "sex",
    "C2.Age": "age",
    "C3.Height\xa3\xa8cm\xa3\xa9": "height_cm",
    "C4.Weight\xa3\xa8kg\xa3\xa9": "weight_kg",
    "C5.Living Years": "living_years",
    "D1.TSV": "tsv",
    "D2.TCV": "tcv",
    "D3.TAV": "tav",
    "D5.Clothing Insulation (clo)": "clothing_insulation_clo",
    "D6.Metabolic Rate (met)": "metabolic_rate_met",
    "E1.Indoor Air Temperature (\xa1\xe6)": "indoor_air_temp_c",
    "E2.Indoor Relative Humidity (%)": "indoor_rh_pct",
    "E3.Indoor Air Velocity (m/s)": "indoor_air_velocity_ms",
    "E4.Globe Temperature (\xa1\xe6)": "globe_temp_c",
    "F1.Operative Temperature (\xa1\xe6)": "operative_temp_c",
    "F2.Mean Radiant Temperature (\xa1\xe6)": "mean_radiant_temp_c",
    "F4.PMV": "pmv",
    "F5.PPD": "ppd",
    "G1.Real-Time Outdoor Temperature (\xa1\xe6)": "outdoor_temp_realtime_c",
    "G2.Mean Daily Outdoor Temperature (\xa1\xe6)": "outdoor_temp_daily_mean_c",
    "G3.Monthly Mean Outdoor Temperature (\xa1\xe6)": "outdoor_temp_monthly_mean_c",
    "G5.Mean Daily Outdoor Relative Humidity (%)": "outdoor_rh_daily_mean_pct",
    "G6.Mean Daily Outdoor Air Velocity (m/s)": "outdoor_air_velocity_daily_mean_ms",
}

CORE_COLUMNS = [
    "source_class", "id", "code", "date", "season", "city", "climate_zone",
    "province",
    "building_type", "building_function", "floors", "building_operation_mode",
    "sex", "age", "height_cm", "weight_kg",
    "tsv", "tcv", "tav", "clothing_insulation_clo", "metabolic_rate_met",
    "indoor_air_temp_c", "indoor_rh_pct", "indoor_air_velocity_ms",
    "globe_temp_c", "operative_temp_c", "mean_radiant_temp_c",
    "pmv", "ppd",
    "outdoor_temp_realtime_c", "outdoor_temp_daily_mean_c",
    "outdoor_temp_monthly_mean_c", "outdoor_rh_daily_mean_pct",
    "outdoor_air_velocity_daily_mean_ms",
]


def extract_province(city_str):
    if pd.isna(city_str):
        return np.nan
    s = str(city_str)
    if "Province" in s:
        return s.split("Province")[0].strip().rstrip(",").strip()
    return s.strip()


def load_and_clean(path, source_class):
    df = pd.read_csv(path, header=1, encoding="latin1")
    df = df.rename(columns=lambda c: c.strip())
    df = df.rename(columns=RENAME_MAP)
    df["source_class"] = source_class
    df["province"] = df["city"].apply(extract_province)

    keep_cols = [c for c in CORE_COLUMNS if c in df.columns]
    df = df[keep_cols]
    return df


def main():
    frames = []
    class_names = ["Class-I", "Class-II", "Class-III"]
    for path, cname in zip(INPUT_FILES, class_names):
        df = load_and_clean(path, cname)
        frames.append(df)
        print(f"{cname}: {df.shape[0]} rows loaded")

    full = pd.concat(frames, ignore_index=True)
    print(f"\nTong sau khi gop: {full.shape[0]} rows")

    province_mask = full["province"].apply(
        lambda p: isinstance(p, str) and any(tp in p for tp in TARGET_PROVINCES)
    )
    filtered = full[province_mask].copy()
    print(f"Sau khi loc theo tinh (Guangdong/Guangxi/Hainan/Yunnan/Fujian): {filtered.shape[0]} rows")

    filtered = filtered[filtered["season"] == KEEP_SEASON].copy()
    print(f"Sau khi loc Summer Season: {filtered.shape[0]} rows")

    before_na = filtered.shape[0]
    filtered = filtered.dropna(subset=["indoor_air_temp_c", "indoor_rh_pct"])
    print(f"Sau khi loai NA indoor_air_temp_c/indoor_rh_pct: {filtered.shape[0]} rows (loai {before_na - filtered.shape[0]})")

    before_dup = filtered.shape[0]
    filtered = filtered.drop_duplicates()
    print(f"Sau khi loai trung lap: {filtered.shape[0]} rows (loai {before_dup - filtered.shape[0]})")

    print("\n--- Phan bo theo tinh/thanh pho ---")
    print(filtered["city"].value_counts())

    print("\n--- Thong ke mo ta nhiet do/do am trong nha ---")
    print(filtered[["indoor_air_temp_c", "indoor_rh_pct"]].describe())

    out_path = "/home/claude/work/Chinese_Thermal_Comfort_VN_Climate_Filtered.csv"
    filtered.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nDa luu: {out_path}")


if __name__ == "__main__":
    main()
