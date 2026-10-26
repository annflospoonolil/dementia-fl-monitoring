"""Dataset and window-eligibility utilities for activity anomaly detection.

Domain Rules:
1. Sensor Recording Gaps: Missing activity records or windows with `has_activity_data == False`
   represent unavailable data or sensor dropouts, NOT zero activity. Such windows must be
   flagged as ineligible and excluded from autoencoder training.
2. Incomplete Sensor Coverage: The presence of activity records (`has_activity_data == True`)
   proves that sensor triggers occurred within the window, but does NOT prove complete or
   continuous sensor coverage across all rooms or throughout the entire 6-hour period.
3. Presumed-Normal Training Data: Unlabelled activity windows are treated as presumed-normal
   training baseline data, NOT confirmed normal. Absence of an agitation label does not prove
   absence of agitation or unusual behaviour. Unknown labels must remain unknown.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ActivityWindowDataset(Dataset):
    """PyTorch Dataset yielding eligible activity feature vectors for autoencoder training and evaluation.

    Preserves participant ID, timestamps, and label status without using agitation labels
    as reconstruction targets.
    """

    def __init__(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        participant_ids: Optional[Union[List[str], np.ndarray, pd.Series]] = None,
        timestamps: Optional[Union[List, np.ndarray, pd.Series]] = None,
        label_status: Optional[Union[List[str], np.ndarray, pd.Series]] = None,
    ):
        if isinstance(features, pd.DataFrame):
            x = features.to_numpy(dtype=np.float32)
        else:
            x = np.asarray(features, dtype=np.float32)

        if x.ndim != 2:
            raise ValueError(f"Features must have 2 dimensions [samples, features], got shape {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError("Feature matrix contains non-finite values (NaN or Inf)")

        self.features = torch.from_numpy(x)
        self.num_samples = len(x)

        if participant_ids is not None:
            pids = list(participant_ids)
            if len(pids) != self.num_samples:
                raise ValueError("participant_ids length does not match feature rows")
            self.participant_ids = pids
        else:
            self.participant_ids = ["unknown"] * self.num_samples

        if timestamps is not None:
            ts = list(timestamps)
            if len(ts) != self.num_samples:
                raise ValueError("timestamps length does not match feature rows")
            self.timestamps = ts
        else:
            self.timestamps = [None] * self.num_samples

        if label_status is not None:
            ls = list(label_status)
            if len(ls) != self.num_samples:
                raise ValueError("label_status length does not match feature rows")
            self.label_status = ls
        else:
            self.label_status = ["unknown"] * self.num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        return {
            "features": self.features[index],
            "x": self.features[index],
            "participant_id": self.participant_ids[index],
            "window_start": str(self.timestamps[index]),
            "label_status": self.label_status[index],
        }


def filter_eligible_windows(
    df: pd.DataFrame,
    feature_columns: List[str],
    require_activity_data: bool = True,
    exclude_positive_labels: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Filter candidate windows for anomaly detection based on explicit data availability.

    Rules:
    - Rows missing activity data (has_activity_data == False or NaN) are excluded as sensor gaps.
    - Zero activity counts are NOT fabricated for unobserved sensor windows.
    - Preserved unlabelled windows (label_status == 'unknown') are treated as presumed-normal data.
    - True agitation rows (label_status == 'positive') can be held out from training to avoid
      contaminating the presumed-normal baseline.

    Returns:
        eligible_df: Subset of DataFrame meeting eligibility requirements.
        audit: Summary dictionary documenting counts of included and excluded rows.
    """
    total_rows = len(df)
    working = df.copy()

    # 1. Check activity data availability flag
    if require_activity_data:
        if "has_activity_data" in working.columns:
            has_act = working["has_activity_data"].fillna(False).astype(bool)
        elif "has_activity_records" in working.columns:
            has_act = working["has_activity_records"].fillna(False).astype(bool)
        else:
            has_act = pd.Series(True, index=working.index)
    else:
        has_act = pd.Series(True, index=working.index)

    # 2. Check completeness of required feature columns (no missing values allowed)
    valid_features = working[feature_columns].notna().all(axis=1)

    # 3. Label status filter
    if "label_status" in working.columns:
        status = working["label_status"].fillna("unknown").astype(str).str.lower()
    else:
        status = pd.Series("unknown", index=working.index)

    if exclude_positive_labels:
        # Exclude known agitation windows from the presumed-normal training baseline
        eligible_mask = has_act & valid_features & (status != "positive")
    else:
        eligible_mask = has_act & valid_features

    eligible_df = working.loc[eligible_mask].copy()

    audit = {
        "total_candidate_windows": total_rows,
        "eligible_windows": int(eligible_mask.sum()),
        "excluded_sensor_gaps": int((~has_act).sum()),
        "excluded_invalid_features": int((has_act & ~valid_features).sum()),
        "excluded_positive_agitation": int(
            (has_act & valid_features & (status == "positive")).sum()
            if exclude_positive_labels
            else 0
        ),
        "presumed_normal_unlabelled": int(
            (eligible_mask & (status == "unknown")).sum()
        ),
    }

    return eligible_df, audit


def split_participant_data(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform a reproducible temporal split per participant to prevent test leakage.

    Splits each participant's chronological sequence into:
    - Train: Earliest windows (presumed-normal baseline training)
    - Validation: Subsequent windows (threshold calibration)
    - Test: Latest windows (anomaly evaluation)
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be less than 1.0")

    train_dfs, val_dfs, test_dfs = [], [], []

    # Ensure chronological order per participant
    sort_cols = ["patient_id"]
    if "window_start" in df.columns:
        sort_cols.append("window_start")
    ordered = df.sort_values(sort_cols)

    p_col = "patient_id" if "patient_id" in ordered.columns else None

    if p_col is not None:
        for _, group in ordered.groupby(p_col):
            n = len(group)
            n_train = int(np.floor(train_frac * n))
            n_val = int(np.floor(val_frac * n))

            # Ensure minimal partition allocation if possible
            if n_train == 0 and n > 0:
                n_train = 1
            if n_train + n_val >= n and n > n_train:
                n_val = max(0, n - n_train - 1)

            train_dfs.append(group.iloc[:n_train])
            val_dfs.append(group.iloc[n_train : n_train + n_val])
            test_dfs.append(group.iloc[n_train + n_val :])
    else:
        n = len(ordered)
        n_train = int(np.floor(train_frac * n))
        n_val = int(np.floor(val_frac * n))
        train_dfs.append(ordered.iloc[:n_train])
        val_dfs.append(ordered.iloc[n_train : n_train + n_val])
        test_dfs.append(ordered.iloc[n_train + n_val :])

    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame(columns=df.columns)

    return train_df, val_df, test_df
