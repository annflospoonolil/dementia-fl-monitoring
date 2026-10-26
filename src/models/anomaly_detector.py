"""Anomaly scoring and threshold calibration for activity window autoencoders.

Design Principles:
1. Reconstruction-error Anomaly Scoring: Anomaly score s_i is the mean squared reconstruction
   error across feature dimensions for window i.
2. Zero Test-Data Leakage: Thresholds are fitted exclusively on training or validation baseline data.
   The test split is evaluated using fixed pre-calibrated thresholds.
3. Presumed-Normal Baseline: Unlabelled training windows are treated as presumed-normal data rather
   than confirmed normal. Agitation labels are not used as training targets, and unlabelled windows
   are not assumed to be confirmed negatives.
4. Personalized Calibration with Documented Fallback: Thresholds can be calibrated per-participant.
   When a participant has fewer than min_participant_samples in their baseline, the detector
   falls back to the global cohort threshold.
"""

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, Union
import numpy as np


@dataclass
class AnomalyEvaluationResult:
    reconstruction_errors: np.ndarray
    thresholds: np.ndarray
    is_anomaly: np.ndarray
    threshold_types: List[str]


class AnomalyThresholdCalibrator:
    """Calibrates anomaly thresholds and scores eligible activity windows."""

    def __init__(
        self,
        method: str = "quantile",
        quantile: float = 0.95,
        std_multiplier: float = 2.0,
        min_participant_samples: int = 10,
    ):
        if method not in ("quantile", "parametric"):
            raise ValueError(f"Method must be 'quantile' or 'parametric', got {method}")
        if not (0.0 < quantile < 1.0):
            raise ValueError(f"Quantile must be in (0, 1), got {quantile}")
        if std_multiplier <= 0:
            raise ValueError("std_multiplier must be positive")
        if min_participant_samples < 1:
            raise ValueError("min_participant_samples must be >= 1")

        self.method = method
        self.quantile = float(quantile)
        self.std_multiplier = float(std_multiplier)
        self.min_participant_samples = int(min_participant_samples)

        self.global_threshold: Optional[float] = None
        self.personalized_thresholds: Dict[str, float] = {}
        self.participant_sample_counts: Dict[str, int] = {}
        self.fallback_participants: List[str] = []

    def _compute_single_threshold(self, errors: np.ndarray) -> float:
        """Compute threshold from an array of baseline reconstruction errors."""
        arr = np.asarray(errors, dtype=np.float64)
        if arr.size == 0:
            raise ValueError("Cannot calibrate threshold on empty errors")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Errors contain non-finite values (NaN or Inf)")

        if self.method == "quantile":
            return float(np.quantile(arr, self.quantile))
        else:
            mean = np.mean(arr)
            std = np.std(arr)
            return float(mean + self.std_multiplier * std)

    def fit_global(self, baseline_errors: np.ndarray) -> float:
        """Fit a global cohort threshold on training/validation presumed-normal errors.

        Must be called on training/validation data only to prevent test leakage.
        """
        arr = np.asarray(baseline_errors, dtype=np.float64).reshape(-1)
        self.global_threshold = self._compute_single_threshold(arr)
        return self.global_threshold

    def fit_personalized(
        self,
        participant_errors: Mapping[str, Union[np.ndarray, List[float]]],
        global_fallback_errors: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Fit personalized thresholds for participants with sufficient historical data.

        If a participant has fewer than min_participant_samples, their threshold falls back
        to the global threshold.
        """
        if self.global_threshold is None:
            if global_fallback_errors is not None:
                self.fit_global(global_fallback_errors)
            else:
                # Combine all available participant errors to establish global baseline
                all_errs = [
                    np.asarray(errs, dtype=np.float64).reshape(-1)
                    for errs in participant_errors.values()
                    if len(errs) > 0
                ]
                if not all_errs:
                    raise ValueError("No baseline errors provided to establish global threshold")
                self.fit_global(np.concatenate(all_errs))

        self.personalized_thresholds = {}
        self.fallback_participants = []

        for pid, errs in participant_errors.items():
            arr = np.asarray(errs, dtype=np.float64).reshape(-1)
            count = int(arr.size)
            self.participant_sample_counts[pid] = count

            if count >= self.min_participant_samples:
                thresh = self._compute_single_threshold(arr)
                self.personalized_thresholds[pid] = thresh
            else:
                # Fall back to global cohort threshold due to insufficient history
                self.personalized_thresholds[pid] = self.global_threshold
                self.fallback_participants.append(pid)

        return dict(self.personalized_thresholds)

    def get_threshold(self, participant_id: Optional[str] = None) -> Tuple[float, str]:
        """Retrieve the applicable threshold and its origin (personalized vs fallback vs global)."""
        if self.global_threshold is None and not self.personalized_thresholds:
            raise RuntimeError("Calibrator has not been fitted. Call fit_global or fit_personalized first.")

        if participant_id is not None and participant_id in self.personalized_thresholds:
            is_fallback = participant_id in self.fallback_participants
            thresh = self.personalized_thresholds[participant_id]
            origin = "global_fallback" if is_fallback else "personalized"
            return thresh, origin

        if self.global_threshold is not None:
            return self.global_threshold, "global"

        raise RuntimeError(f"No threshold available for participant '{participant_id}'.")

    def score(
        self,
        reconstruction_errors: np.ndarray,
        participant_ids: Optional[Union[List[str], np.ndarray]] = None,
    ) -> AnomalyEvaluationResult:
        """Score eligible activity windows and apply calibrated thresholds.

        Parameters:
            reconstruction_errors: Array of per-window MSE anomaly scores.
            participant_ids: Optional participant identifier per window for personalized scoring.

        Returns:
            AnomalyEvaluationResult containing errors, applied thresholds, binary flags,
            and threshold origin types.
        """
        errs = np.asarray(reconstruction_errors, dtype=np.float64).reshape(-1)
        n = len(errs)
        if n == 0:
            raise ValueError("Cannot score empty reconstruction errors")

        thresholds = np.zeros(n, dtype=np.float64)
        threshold_types: List[str] = []

        if participant_ids is not None:
            pids = list(participant_ids)
            if len(pids) != n:
                raise ValueError(f"participant_ids length ({len(pids)}) does not match errors ({n})")
            for i, pid in enumerate(pids):
                thresh, t_type = self.get_threshold(pid)
                thresholds[i] = thresh
                threshold_types.append(t_type)
        else:
            thresh, t_type = self.get_threshold(None)
            thresholds.fill(thresh)
            threshold_types = [t_type] * n

        is_anomaly = errs >= thresholds

        return AnomalyEvaluationResult(
            reconstruction_errors=errs,
            thresholds=thresholds,
            is_anomaly=is_anomaly,
            threshold_types=threshold_types,
        )
