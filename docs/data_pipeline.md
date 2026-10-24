# Data Pipeline — TIHM Activity Data

## 1. Objective

Prepare TIHM activity records and agitation labels for research into federated learning for human activity recognition. The pipeline preserves unknown labels rather than assuming that unlabelled windows represent normal activity.

## 2. Pipeline stages

1. **Load source data:** Read the original activity and label CSV files from `data/raw/TIHM_Dataset/Dataset/`.
2. **Clean records:** Parse timestamps, standardize participant identifiers, and remove records that cannot be used because required fields are invalid.
3. **Create time windows:** Assign timestamps to six-hour windows.
4. **Aggregate activity:** Count activity records by participant, window, and location.
5. **Prepare labels:** Match recorded agitation events to participant windows.
6. **Preserve uncertainty:** Mark windows with recorded agitation as `positive` and windows without a recorded agitation label as `unknown`.
7. **Join features and labels:** Combine the activity aggregates with the agitation-label information.
8. **Export outputs:** Save the processed feature table, cleaned agitation records, and pipeline summary under `data/processed/`.

## 3. Main output files

| File                                  | Purpose                                                   |
| ------------------------------------- | --------------------------------------------------------- |
| `activity_6h_features_and_labels.csv` | Aggregated activity features and label information        |
| `agitation_events_cleaned.csv`        | Cleaned agitation records                                 |
| `member1_data_summary.txt`            | Summary of preprocessing results                          |
| `tihm_6h_label_audit.csv`             | Earlier feasibility audit of activity coverage and labels |

## 4. Current validation results

- Cleaned activity records: 1,030,559
- Cleaned agitation records: 135
- Participants with activity data: 56
- Participant-window rows: 10,159
- Positive-labelled windows: 135
- Unknown-labelled windows: 10,024
- Duplicate participant-window keys: 0
- Positive-labelled windows without activity records: 2

These counts describe the current run and should be regenerated whenever the source data or pipeline changes.

## 5. Data integrity rules

- Raw source files must not be overwritten.
- Unknown labels must not be converted automatically to negative labels.
- Zero-filled activity features must be interpreted alongside `has_activity_data`.
- Participant identifiers and window timestamps must be retained for auditing and leakage prevention.
- Feature generation and label construction must be reproducible from the source files.

## 6. Limitations and next steps

The output is a sparse table, not a complete participant-by-time grid. The six-hour window is a preprocessing choice and is not established as a clinically optimal event horizon.

Before model training, the project must define a defensible approach to unknown labels, establish participant-aware evaluation splits, and specify how false alerts and missed events will be evaluated.

Federated learning and PPO-based client selection are downstream stages; this pipeline alone does not demonstrate model accuracy or the effectiveness of PPO.
