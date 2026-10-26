from pathlib import Path
import pandas as pd

from src.data.build_dataset import (
    process_activity,
    process_physiology,
    process_sleep,
    build_labels,
)

raw_dir = Path("src/data/raw/TIHM_Dataset/Dataset")

activity = process_activity(raw_dir)
physiology = process_physiology(raw_dir)
sleep = process_sleep(raw_dir)

features = activity.merge(
    physiology, on=["patient_id", "window_start"], how="outer"
)
features = features.merge(
    sleep, on=["patient_id", "window_start"], how="outer"
)

labels, audit = build_labels(raw_dir)

keys = ["patient_id", "window_start"]

feature_keys = features[keys].drop_duplicates()
label_keys = labels[keys].drop_duplicates()

overlap = label_keys.merge(
    feature_keys, on=keys, how="inner"
)

unmatched = label_keys.merge(
    feature_keys, on=keys, how="left", indicator=True
)
unmatched = unmatched[unmatched["_merge"] == "left_only"]

print("Feature windows:", len(feature_keys))
print("Distinct positive label windows:", len(label_keys))
print("Positive windows matching features:", len(overlap))
print("Positive windows without matching features:", len(unmatched))
print("\nFeature date range:")
print(features["window_start"].min(), "to", features["window_start"].max())
print("\nLabel date range:")
print(labels["window_start"].min(), "to", labels["window_start"].max())
print("\nExamples of unmatched labels:")
print(unmatched[keys].head(10).to_string(index=False))
