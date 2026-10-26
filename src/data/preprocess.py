
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Paths and settings
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "src" / "data" / "raw" / "TIHM_Dataset" / "Dataset"
if not RAW.exists():
    RAW = ROOT / "data" / "raw" / "TIHM_Dataset" / "Dataset"
OUT = ROOT / "data" / "processed"
WINDOW = "6h"

ROOMS = [
    "Back Door", "Bathroom", "Bedroom", "Fridge Door",
    "Front Door", "Hallway", "Kitchen", "Lounge",
]


def load_csv(filename):
    path = RAW / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return pd.read_csv(path)


def clean_base(df, filename):
    required = {"patient_id", "date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")

    df = df.copy()
    df["patient_id"] = df["patient_id"].astype("string").str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["patient_id", "date"])
    df["window_start"] = df["date"].dt.floor(WINDOW)
    return df


# --------------------------------------------------
# 1. Activity features
# --------------------------------------------------
def process_activity():
    df = clean_base(load_csv("Activity.csv"), "Activity.csv")

    if "location_name" not in df.columns:
        raise ValueError("Activity.csv needs location_name")

    df["location_name"] = df["location_name"].astype("string").str.strip()
    df = df.dropna(subset=["location_name"])

    # Count triggers for each room in each six-hour window.
    counts = (
        df.groupby(["patient_id", "window_start", "location_name"])
        .size()
        .unstack("location_name", fill_value=0)
    )

    # Ensure all expected room features exist.
    counts = counts.reindex(columns=ROOMS, fill_value=0)
    counts.columns = [
        "activity_count_" + room.lower().replace(" ", "_")
        for room in counts.columns
    ]

    counts["total_activity_records"] = counts.sum(axis=1)
    counts["active_location_count"] = (
        df.groupby(["patient_id", "window_start"])["location_name"]
        .nunique()
    )

    # Trigger counts are observations, not proof of full sensor coverage.
    counts["has_activity_records"] = True

    return counts.reset_index()


# --------------------------------------------------
# 2. Optional physiology features
# --------------------------------------------------
def process_physiology():
    df = clean_base(load_csv("Physiology.csv"), "Physiology.csv")

    required = {"device_type", "value", "unit"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Physiology.csv is missing columns: {required - set(df.columns)}"
        )

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["device_type", "value"]).copy()

    measurement = (
        df["device_type"].astype(str).str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    unit = (
        df["unit"].fillna("unknown").astype(str).str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    df["feature"] = measurement + "_" + unit

    result = (
        df.groupby(["patient_id", "window_start", "feature"])["value"]
        .agg(["mean", "min", "max", "count"])
        .unstack("feature")
    )
    result.columns = [
        f"phys_{stat}_{feature}" for stat, feature in result.columns
    ]
    return result.reset_index()


# --------------------------------------------------
# 3. Optional sleep features
# --------------------------------------------------
def process_sleep():
    path = RAW / "Sleep.csv"
    if not path.exists():
        return None

    df = clean_base(pd.read_csv(path), "Sleep.csv")

    result = (
        df.groupby(["patient_id", "window_start"])
        .size()
        .rename("sleep_records")
        .to_frame()
    )

    for col in ["heart_rate", "respiratory_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            stats = df.groupby(
                ["patient_id", "window_start"]
            )[col].agg(["mean", "min", "max"])
            stats.columns = [f"sleep_{col}_{s}" for s in stats.columns]
            result = result.join(stats, how="outer")

    if "state" in df.columns:
        states = pd.crosstab(
            [df["patient_id"], df["window_start"]],
            df["state"].fillna("UNKNOWN")
        )
        states.columns = [
            "sleep_state_" +
            str(state).lower().replace(" ", "_") + "_records"
            for state in states.columns
        ]
        result = result.join(states, how="outer")

    return result.reset_index()


# --------------------------------------------------
# 4. Build time-based features
# --------------------------------------------------
def add_time_features(df):
    df = df.sort_values(["patient_id", "window_start"]).copy()

    hour = df["window_start"].dt.hour
    df["window_hour"] = hour
    df["window_sin"] = (2 * 3.141592653589793 * hour / 24).map(
        lambda x: __import__("math").sin(x)
    )
    df["window_cos"] = (2 * 3.141592653589793 * hour / 24).map(
        lambda x: __import__("math").cos(x)
    )
    df["day_of_week"] = df["window_start"].dt.dayofweek

    # Compare with the previous available window for the same participant.
    # Missing previous observations remain missing; they are not set to zero.
    activity_cols = [
        c for c in df.columns if c.startswith("activity_count_")
    ] + ["total_activity_records", "active_location_count"]

    for col in activity_cols:
        df[f"{col}_change"] = df.groupby("patient_id")[col].diff()

    return df


# --------------------------------------------------
# 5. Main pipeline
# --------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)

    activity = process_activity()
    physiology = process_physiology()
    sleep = process_sleep()

    # Activity windows form the primary timeline.
    data = activity.merge(
        physiology,
        on=["patient_id", "window_start"],
        how="outer",
        validate="one_to_one",
    )

    if sleep is not None:
        data = data.merge(
            sleep,
            on=["patient_id", "window_start"],
            how="outer",
            validate="one_to_one",
        )

    data["has_activity_data"] = data["has_activity_records"].fillna(False)
    data["has_activity_records"] = data["has_activity_records"].fillna(False)

    phys_count_cols = [
        c for c in data.columns if c.startswith("phys_count_")
    ]
    data["has_physiology_data"] = (
        data[phys_count_cols].fillna(0).sum(axis=1) > 0
        if phys_count_cols
        else False
    )

    if sleep is not None:
        data["has_sleep_data"] = data["sleep_records"].notna()
    else:
        data["has_sleep_data"] = False

    # Do not fill missing physiological or sleep values with zero.
    # Fill room counts only for rows where activity records were observed;
    # rows with no activity records remain flagged as unavailable.
    room_cols = [
        c for c in data.columns if c.startswith("activity_count_")
    ]
    observed = data["has_activity_records"]
    data.loc[observed, room_cols] = data.loc[observed, room_cols].fillna(0)

    data = add_time_features(data)
    data = data.sort_values(
        ["patient_id", "window_start"]
    ).reset_index(drop=True)

    # Preserve original event types. These are not HAR ground-truth labels.
    events = clean_base(load_csv("Labels.csv"), "Labels.csv")
    events = events.sort_values(["patient_id", "date"])

    feature_path = OUT / "anomaly_detection_features.csv"
    event_path = OUT / "event_records_cleaned.csv"

    data.to_csv(feature_path, index=False)
    events.to_csv(event_path, index=False)

    print("Preprocessing complete.")
    print("Feature rows:", len(data))
    print("Participants:", data["patient_id"].nunique())
    print("Windows with activity records:", int(observed.sum()))
    print("Windows with physiology:", int(data["has_physiology_data"].sum()))
    print("Windows with sleep:", int(data["has_sleep_data"].sum()))
    print("Event records preserved:", len(events))
    print("Saved features:", feature_path)
    print("Saved events:", event_path)


if __name__ == "__main__":
    main()
