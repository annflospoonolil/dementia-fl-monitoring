import pandas as pd
from src.data.build_dataset import build_labels

def test_agitation_only(tmp_path):
    pd.DataFrame({
        "patient_id": ["p1", "p1", "p2"],
        "date": ["11-04-2019 12:00", "11-04-2019 13:00", "11-04-2019 13:00"],
        "type": ["Agitation", "Blood pressure", "Blood pressure"],
    }).to_csv(tmp_path / "Labels.csv", index=False)
    pos, audit = build_labels(tmp_path)
    assert len(pos) == 1
    assert pos.iloc[0]["agitation_target"] == 1.0
    assert "blood pressure" in set(audit["label_type"])
