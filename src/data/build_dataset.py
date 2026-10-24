"""Create six-hour TIHM feature windows and cautious agitation labels."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

WINDOW = "6h"
ROOMS = ["Back Door", "Bathroom", "Bedroom", "Front Door", "Fridge Door",
         "Hallway", "Kitchen", "Lounge", "Other"]
DATE_FORMAT = "%d-%m-%Y %H:%M"

def locate_csv(raw_dir: Path, stem: str, required=True) -> Optional[Path]:
    for p in raw_dir.rglob("*.csv"):
        if p.stem.lower() == stem.lower():
            return p
    if required:
        raise FileNotFoundError(f"Could not find {stem}.csv under {raw_dir}")
    return None

def read_csv(raw_dir: Path, stem: str, required=True):
    p = locate_csv(raw_dir, stem, required)
    return None if p is None else pd.read_csv(p, encoding="utf-8-sig", low_memory=False)


def parse_dates(s: pd.Series) -> pd.Series:
    # TIHM timestamps use ISO format, e.g. 2019-04-11 12:00:22.
    out = pd.to_datetime(
        s, format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )

    # Fallback for any valid timestamps using another format.
    mask = out.isna() & s.notna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(
            s.loc[mask], format="mixed", errors="coerce", dayfirst=False
        )

    return out


def clean_base(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = {"patient_id", "date"} - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")
    out = df.copy()
    out["patient_id"] = out["patient_id"].astype("string").str.strip()
    out["date"] = parse_dates(out["date"])
    out = out.dropna(subset=["patient_id", "date"])
    out = out[out["patient_id"].ne("")]
    out["window_start"] = out["date"].dt.floor(WINDOW)
    return out

def safe_name(x) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_").lower() or "unknown"

def process_activity(raw_dir: Path) -> pd.DataFrame:
    df = clean_base(read_csv(raw_dir, "Activity"), "Activity")
    if "location_name" not in df:
        raise ValueError("Activity.csv needs location_name")
    known = set(ROOMS) - {"Other"}
    df["location_name"] = df["location_name"].fillna("Other").astype(str).str.strip()
    df.loc[~df["location_name"].isin(known), "location_name"] = "Other"
    g = df.groupby(["patient_id", "window_start", "location_name"]).size().rename("n").reset_index()
    wide = g.pivot_table(index=["patient_id", "window_start"], columns="location_name", values="n", fill_value=0)
    for room in ROOMS:
        if room not in wide:
            wide[room] = 0
    wide = wide[ROOMS].rename(columns={r: "activity_count_" + safe_name(r) for r in ROOMS})
    count_cols = list(wide.columns)
    wide["total_activity_records"] = wide[count_cols].sum(axis=1)
    wide["active_location_count"] = (wide[count_cols] > 0).sum(axis=1)
    wide["has_activity_data"] = 1
    return wide.reset_index()

def process_physiology(raw_dir: Path):
    df = read_csv(raw_dir, "Physiology", required=False)
    if df is None:
        return None
    df = clean_base(df, "Physiology")
    if not {"device_type", "value"}.issubset(df.columns):
        raise ValueError("Physiology.csv needs device_type and value")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value", "device_type"])
    units = df["unit"].astype(str).str.strip() if "unit" in df else pd.Series("", index=df.index)
    df["feature"] = [safe_name(d) + ("_" + safe_name(u) if u and u.lower() != "nan" else "")
                     for d, u in zip(df["device_type"], units)]
    agg = df.groupby(["patient_id", "window_start", "feature"])["value"].agg(["mean", "min", "max", "count"]).unstack("feature")
    agg.columns = [f"phys_{stat}_{feature}" for stat, feature in agg.columns]
    agg["has_physiology_data"] = 1
    return agg.reset_index()

def process_sleep(raw_dir: Path):
    df = read_csv(raw_dir, "Sleep", required=False)
    if df is None:
        return None
    df = clean_base(df, "Sleep")
    keys = ["patient_id", "window_start"]
    out = pd.DataFrame(index=df.groupby(keys).size().index)
    for col in ("heart_rate", "respiratory_rate"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            agg = df.groupby(keys)[col].agg(["mean", "min", "max", "std"])
            agg.columns = [f"sleep_{col}_{s}" for s in agg.columns]
            out = out.join(agg, how="outer")
    if "state" in df:
        temp = df.assign(_state=df["state"].fillna("UNKNOWN").map(safe_name))
        counts = temp.groupby(keys + ["_state"]).size().unstack(fill_value=0)
        counts.columns = ["sleep_state_count_" + str(c) for c in counts.columns]
        out = out.join(counts, how="outer")
    if "snoring" in df:
        snore = df["snoring"].astype(str).str.lower().isin(["true", "1", "yes"])
        temp = df.assign(_snore=snore.astype(float))
        out = out.join(temp.groupby(keys)["_snore"].mean().rename("sleep_snoring_fraction"), how="outer")
    out["has_sleep_data"] = 1
    return out.reset_index()

def process_demographics(raw_dir: Path):
    df = read_csv(raw_dir, "Demographics", required=False)
    if df is None:
        return None
    if "patient_id" not in df:
        raise ValueError("Demographics.csv needs patient_id")
    cols = [c for c in ["patient_id", "age", "sex"] if c in df]
    df = df[cols].copy()
    df["patient_id"] = df["patient_id"].astype("string").str.strip()
    df = df.dropna(subset=["patient_id"]).drop_duplicates("patient_id", keep="last")
    if "age" in df:
        df["age_band"] = df.pop("age").astype("string").fillna("unknown")
    if "sex" in df:
        df["sex"] = df["sex"].astype("string").fillna("unknown").str.strip().str.lower()
    # Encode categories numerically for this starter, with explicit stable mappings.
    for col in ("age_band", "sex"):
        if col in df:
            df[col] = df[col].astype("category").cat.codes.astype(float)
    return df

def build_labels(raw_dir: Path):
    df = clean_base(read_csv(raw_dir, "Labels"), "Labels")
    if "type" not in df:
        raise ValueError("Labels.csv needs a 'type' column")
    df["label_type"] = df["type"].astype("string").str.strip().str.casefold()
    pos = df[df["label_type"].eq("agitation")][["patient_id", "window_start"]].drop_duplicates().copy()
    pos["agitation_target"] = 1.0
    pos["label_status"] = "positive"
    audit = df.groupby("label_type").size().rename("record_count").reset_index()
    audit["is_agitation_type"] = audit["label_type"].eq("agitation")
    return pos, audit

def add_time_features(df):
    df = df.copy()
    dt = df["window_start"]
    hour = dt.dt.hour.astype(float)
    dow = dt.dt.dayofweek.astype(float)
    df["hour_sin"] = np.sin(2*np.pi*hour/24)
    df["hour_cos"] = np.cos(2*np.pi*hour/24)
    df["day_of_week"] = dow
    df["day_sin"] = np.sin(2*np.pi*dow/7)
    df["day_cos"] = np.cos(2*np.pi*dow/7)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    feat = process_activity(args.raw_dir)
    for extra in [process_physiology(args.raw_dir), process_sleep(args.raw_dir)]:
        if extra is not None:
            feat = feat.merge(extra, on=["patient_id", "window_start"], how="outer", validate="one_to_one")
    feat = add_time_features(feat)
    for flag in ["has_activity_data", "has_physiology_data", "has_sleep_data"]:
        if flag not in feat:
            feat[flag] = 0
        feat[flag] = feat[flag].fillna(0).astype(int)
    for c in feat.columns:
        if c.startswith("activity_count_") or c in ("total_activity_records", "active_location_count"):
            feat[c] = feat[c].fillna(0)
    labels, audit = build_labels(args.raw_dir)
    feat = feat.merge(labels, on=["patient_id", "window_start"], how="left", validate="one_to_one")
    feat["label_status"] = feat["label_status"].fillna("unknown")
    # Do not convert unknown targets to zero.
    feat["agitation_target"] = feat["agitation_target"].astype("Float64")
    demo = process_demographics(args.raw_dir)
    if demo is not None:
        feat = feat.merge(demo, on="patient_id", how="left", validate="many_to_one")
    feat = feat.sort_values(["patient_id", "window_start"]).reset_index(drop=True)
    feat.to_csv(args.out_dir / "member2_features_labels.csv", index=False)
    audit.to_csv(args.out_dir / "member2_label_audit.csv", index=False)
    exclude = {"patient_id", "window_start", "agitation_target", "label_status"}
    cols = [c for c in feat.select_dtypes(include=[np.number]).columns if c not in exclude]
    (args.out_dir / "member2_feature_columns.json").write_text(json.dumps(cols, indent=2), encoding="utf-8")
    print(f"Saved features: {args.out_dir / 'member2_features_labels.csv'}")
    print(f"Rows={len(feat)}; positives={(feat.label_status == 'positive').sum()}; unknown={(feat.label_status == 'unknown').sum()}")
    print("Unknown windows were not labelled negative.")

if __name__ == "__main__":
    main()
