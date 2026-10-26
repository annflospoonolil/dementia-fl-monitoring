import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from src.models.mlp import MLPClassifier
from src.models.adapter import ModelAdapter, TrainConfig
from src.training.dataset import WindowDataset

def test_output_shape():
    assert MLPClassifier(5)(torch.zeros(4, 5)).shape == (4, 1)

def test_weights():
    a = ModelAdapter(MLPClassifier(5))
    w = a.get_weights()
    a.set_weights(w)
    first = next(iter(w))
    bad = dict(w)
    bad[first] = np.zeros((999,), dtype=np.float32)
    with pytest.raises(ValueError):
        a.set_weights(bad)

def test_train_eval():
    x = np.random.default_rng(1).normal(size=(40, 5)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)
    loader = DataLoader(WindowDataset(x, y), batch_size=8, shuffle=True)
    a = ModelAdapter(MLPClassifier(5))
    assert a.train_local(loader, TrainConfig())["num_samples"] == 40
    assert 0 <= a.evaluate(loader)["f1"] <= 1
