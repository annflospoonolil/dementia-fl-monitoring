# Dementia FL Monitoring

A research-oriented project for monitoring activity patterns associated with agitation and unusual behaviour using TIHM activity data, with a future focus on federated learning and anomaly detection.

## Project goal

The project prepares time-windowed activity and contextual sensor data from the TIHM dataset so that downstream machine learning models can learn patterns related to anomalous or agitation-associated activity.

The current implementation focuses on data preparation and feature generation rather than final model training.

## Current status

The project is in the preprocessing / feature-engineering stage.

What has been implemented in [src/data/preprocess.py](D:/CET/finalyearproject/dementia-fl-monitoring/src/data/preprocess.py):

- Loads raw TIHM CSV files from the dataset directory
- Validates required columns and raises clear errors if source files are missing or malformed
- Cleans participant IDs and timestamps
- Removes invalid records with missing patient IDs or unusable dates
- Constructs six-hour time windows using timestamp flooring
- Aggregates activity counts by patient and time window for each room/location
- Builds room-level features such as:
  - activity_count_back_door
  - activity_count_bathroom
  - activity_count_bedroom
  - activity_count_fridge_door
  - activity_count_front_door
  - activity_count_hallway
  - activity_count_kitchen
  - activity_count_lounge
  - total_activity_records
  - active_location_count
- Handles optional physiology data when present and extracts aggregated statistics per feature
- Handles optional sleep data when present and keeps sleep summaries and state counts
- Adds time-based features such as hour-of-day, cyclic sine/cosine encoding, and day-of-week values
- Computes change features between consecutive windows for the same patient
- Merges activity, physiology, and sleep tables on patient_id and window_start
- Flags whether activity, physiology, or sleep data existed in each window
- Preserves source event records instead of converting them into assumed negatives
- Exports processed feature data and cleaned event records to [data/processed](D:/CET/finalyearproject/dementia-fl-monitoring/data/processed)

## Outputs generated

The pipeline currently writes the following files to [data/processed](D:/CET/finalyearproject/dementia-fl-monitoring/data/processed):

- anomaly_detection_features.csv
- event_records_cleaned.csv

These files are the current machine-readable outputs from the preprocessing stage.

## Dataset and documentation

This project includes documentation for the processed table and pipeline assumptions:

- [docs/data_pipeline.md](D:/CET/finalyearproject/dementia-fl-monitoring/docs/data_pipeline.md)
- [docs/data_dictionary.md](D:/CET/finalyearproject/dementia-fl-monitoring/docs/data_dictionary.md)

These documents explain:

- the six-hour windowing strategy
- the distinction between positive and unknown labels
- the importance of preserving uncertain labels
- limitations of the current preprocessing approach

## Validation status

The preprocessing script has been executed successfully in the project environment.

The run completed and produced feature rows and event outputs without an exception. The data pipeline is therefore operational at the preprocessing stage.

## Is the project ready for model training?

Not yet, at least not for a production-quality or research-grade unusual-activity model.

The project is ready for exploratory model experiments, but not fully ready for final training and evaluation because:

- there is no trained model pipeline implemented yet
- there is no dedicated train/validation/test split logic
- there is no model evaluation report or metrics harness
- the label handling strategy for unknown windows must still be defined clearly for training
- there is no explicit anomaly-detection model architecture in place yet
- there are no automated tests covering model correctness or data integrity

In practical terms: the project has reached the stage where you can begin baseline experiments such as:

- simple supervised classification using positive vs known data
- anomaly detection models on activity features
- patient-aware cross-validation experiments
- feature ablation studies

But a robust final model for identifying unusual activities still requires additional work beyond preprocessing.

## Suggested next steps

1. Define the final label strategy for unknown windows
2. Create train/validation/test splits by patient to avoid leakage
3. Implement a baseline anomaly-detection or supervised model
4. Add metrics such as precision, recall, F1, false alarm rate, and missed-event rate
5. Add experiment tracking and reproducibility scripts
6. Extend toward federated learning if the project goal is FL-based monitoring

## Run the preprocessing pipeline

From the project root:

```bash
python src/data/preprocess.py
```

or, if the project virtual environment is active:

```bash
python .\src\data\preprocess.py
```

## Repository structure

- [src](D:/CET/finalyearproject/dementia-fl-monitoring/src) — source code
- [data](D:/CET/finalyearproject/dementia-fl-monitoring/data) — raw and processed datasets
- [docs](D:/CET/finalyearproject/dementia-fl-monitoring/docs) — project documentation
- [tests](D:/CET/finalyearproject/dementia-fl-monitoring/tests) — currently empty / not yet implemented
- [results](D:/CET/finalyearproject/dementia-fl-monitoring/results) — experiment outputs
- [experiments](D:/CET/finalyearproject/dementia-fl-monitoring/experiments) — experimental notebooks/scripts

## Bottom line

This project has successfully built the foundation for activity-based feature extraction and data preparation, but it is not yet in a complete stage for final unusual-activity model training. It is a good starting point for the next stage of model development and experimentation.
