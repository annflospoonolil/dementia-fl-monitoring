import numpy as np
from torch.utils.data import DataLoader
from src.models.mlp import MLPClassifier
from src.models.adapter import ModelAdapter, TrainConfig
from src.training.dataset import WindowDataset

def main():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(256, 12)).astype("float32")
    y = ((x[:, 0] + .6*x[:, 1]) > 0).astype("float32")
    ds = WindowDataset(x, y)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    adapter = ModelAdapter(MLPClassifier(input_dim=x.shape[1]))
    print("Train:", adapter.train_local(loader, TrainConfig(local_epochs=3)))
    print("Synthetic metrics only:", adapter.evaluate(DataLoader(ds, batch_size=64)))
    weights = adapter.get_weights()
    adapter.set_weights(weights)
    print("Weight round-trip passed.")

if __name__ == "__main__":
    main()
