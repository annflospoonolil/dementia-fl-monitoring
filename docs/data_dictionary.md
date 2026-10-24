# Data Dictionary — TIHM Activity Features

## 1. Dataset overview

The processed dataset is derived from the TIHM activity and label data. Activity records are aggregated into six-hour windows for each participant. Recorded agitation labels are joined to these windows.

- **Processed file:** `data/processed/activity_6h_features_and_labels.csv`
- **Window duration:** 6 hours
- **Participants with activity data:** 56
- **Rows:** 10,159
- **Positive-labelled windows:** 135
- **Unknown-label windows:** 10,024

These figures describe the current preprocessing output and should be updated if the pipeline changes.

## 2. Participant and time identifiers

| Column         | Meaning                                            |
| -------------- | -------------------------------------------------- |
| `patient_id`   | Participant identifier from the source dataset     |
| `window_start` | Start timestamp of the six-hour aggregation window |

## 3. Activity features

Each `activity_count_<location>` column contains the number of activity records attributed to that location during the participant's six-hour window.

| Column                       | Meaning                                                             |
| ---------------------------- | ------------------------------------------------------------------- |
| `activity_count_back_door`   | Activity record count for the back door                             |
| `activity_count_bathroom`    | Activity record count for the bathroom                              |
| `activity_count_bedroom`     | Activity record count for the bedroom                               |
| `activity_count_fridge_door` | Activity record count for the fridge door                           |
| `activity_count_front_door`  | Activity record count for the front door                            |
| `activity_count_hallway`     | Activity record count for the hallway                               |
| `activity_count_kitchen`     | Activity record count for the kitchen                               |
| `activity_count_lounge`      | Activity record count for the lounge                                |
| `total_activity_records`     | Sum of the location-specific activity counts                        |
| `active_location_count`      | Number of locations with at least one activity record in the window |

## 4. Data availability and labels

| Column                 | Meaning                                                                             |
| ---------------------- | ----------------------------------------------------------------------------------- |
| `has_activity_data`    | Indicates whether the window has source activity records                            |
| `has_agitation_record` | Indicates whether an agitation record was matched to the window                     |
| `label_status`         | `positive` for a recorded agitation label; `unknown` when no such label is recorded |
| `agitation_target`     | `1` for positive-labelled windows; missing for unknown-labelled windows             |

**Important:** An unknown label does not mean that the participant was not agitated. Unknown rows must not be silently converted to negative labels.

A positive six-hour window indicates that an agitation record was associated with that window; it does not establish the precise event onset or duration.

## 5. Missing activity data

Two positive-labelled windows have no corresponding activity records. Their activity feature values may be zero-filled by the pipeline, but `has_activity_data` must remain false for those rows. These zero values represent unavailable activity observations, not confirmed inactivity.

## 6. Limitations

- The table is sparse and does not represent every possible six-hour window for every participant.
- The six-hour aggregation is a preprocessing choice, not a clinically validated event horizon.
- The labels do not establish that every unlabelled window is a confirmed negative.
- The data dictionary documents the current processed table; it does not establish clinical validity or predictive performance.
