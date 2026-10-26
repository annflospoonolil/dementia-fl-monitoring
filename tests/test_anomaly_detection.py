import numpy as np
import pandas as pd
import pytest

from src.models.anomaly_detector import AnomalyThresholdCalibrator
from src.training.anomaly_dataset import filter_eligible_windows, split_participant_data


def test_threshold_calibration_methods():
    """Verify quantile and parametric threshold calculations on baseline errors."""
    errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Quantile method (90th percentile)
    cal_q = AnomalyThresholdCalibrator(method="quantile", quantile=0.90)
    thresh_q = cal_q.fit_global(errors)
    assert np.isclose(thresh_q, 0.91, atol=0.05)

    # Parametric method (mean + 1 std)
    cal_p = AnomalyThresholdCalibrator(method="parametric", std_multiplier=1.0)
    thresh_p = cal_p.fit_global(errors)
    assert thresh_p > np.mean(errors)


def test_personalized_calibration_with_fallback():
    """Verify participant-specific threshold calibration and fallback for sparse data."""
    calibrator = AnomalyThresholdCalibrator(
        method="quantile", quantile=0.95, min_participant_samples=10
    )

    participant_errors = {
        # Participant with sufficient historical windows (n=20)
        "p_sufficient": np.linspace(0.1, 0.5, 20),
        # Participant with sparse historical windows (n=3, less than min_participant_samples=10)
        "p_sparse": np.array([0.2, 0.25, 0.3]),
    }

    calibrator.fit_personalized(participant_errors)

    assert "p_sufficient" in calibrator.personalized_thresholds
    assert "p_sparse" in calibrator.personalized_thresholds
    assert "p_sparse" in calibrator.fallback_participants
    assert "p_sufficient" not in calibrator.fallback_participants

    # Fallback participant must receive the global cohort threshold
    assert calibrator.personalized_thresholds["p_sparse"] == calibrator.global_threshold

    # Scoring verifies origin reporting
    res = calibrator.score(np.array([0.4, 0.4]), participant_ids=["p_sufficient", "p_sparse"])
    assert res.threshold_types[0] == "personalized"
    assert res.threshold_types[1] == "global_fallback"


def test_zero_test_data_leakage():
    """Verify that anomaly thresholds are fitted strictly on training/validation data."""
    rng = np.random.default_rng(42)
    train_errors = rng.normal(loc=0.2, scale=0.05, size=100)
    test_errors = rng.normal(loc=0.8, scale=0.2, size=50)  # Distinct distribution

    calibrator = AnomalyThresholdCalibrator(method="quantile", quantile=0.95)
    calibrator.fit_global(train_errors)

    calibrated_thresh = calibrator.global_threshold

    # Score test data without refitting
    result = calibrator.score(test_errors)

    # Threshold must remain strictly the one derived from train_errors
    assert np.all(result.thresholds == calibrated_thresh)
    assert calibrator.global_threshold == calibrated_thresh


def test_filter_eligible_windows_rules():
    """Verify handling of missing activity gaps, label preservation, and coverage rules.

    Requirements:
    - Sensor gaps (missing activity) must NOT be converted to zero activity.
    - Unlabelled windows are treated as presumed-normal data, NOT confirmed normal.
    - Unknown labels remain unknown.
    - has_activity_data = True does not assume complete sensor coverage.
    """
    df = pd.DataFrame({
        "patient_id": ["p1", "p1", "p2", "p2", "p3"],
        "has_activity_data": [True, False, True, True, False],
        "activity_count_bedroom": [12.0, np.nan, 5.0, 0.0, np.nan],
        "activity_count_kitchen": [8.0, np.nan, 2.0, 0.0, np.nan],
        "label_status": ["unknown", "unknown", "positive", "unknown", "unknown"],
    })

    feature_cols = ["activity_count_bedroom", "activity_count_kitchen"]

    # Filter for training (excluding positive agitation to keep presumed-normal baseline clean)
    eligible_train, audit = filter_eligible_windows(
        df,
        feature_columns=feature_cols,
        require_activity_data=True,
        exclude_positive_labels=True,
    )

    # Sensor gaps must be excluded, not filled with zero
    assert audit["excluded_sensor_gaps"] == 2
    assert audit["excluded_positive_agitation"] == 1
    # 2 presumed-normal unlabelled windows retained
    assert audit["presumed_normal_unlabelled"] == 2
    assert len(eligible_train) == 2

    # Verify that unknown labels are preserved as unknown, never converted to confirmed normal/negative
    assert list(eligible_train["label_status"]) == ["unknown", "unknown"]


def test_participant_split_leakage_prevention():
    """Verify temporal per-participant splitting produces valid partitions without shuffle leakage."""
    dates = pd.date_range("2020-01-01", periods=10, freq="6h")
    df = pd.DataFrame({
        "patient_id": ["p1"] * 10,
        "window_start": dates,
        "feature_1": range(10),
    })

    train_df, val_df, test_df = split_participant_data(df, train_frac=0.6, val_frac=0.2)

    assert len(train_df) == 6
    assert len(val_df) == 2
    assert len(test_df) == 2

    # Chronological integrity: max(train) < min(val) < min(test)
    assert train_df["window_start"].max() < val_df["window_start"].min()
    assert val_df["window_start"].max() < test_df["window_start"].min()
