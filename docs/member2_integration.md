# Member 2 / Member 3 interface

```python
from src.models.mlp import MLPClassifier
from src.models.adapter import ModelAdapter, TrainConfig
adapter = ModelAdapter(MLPClassifier(input_dim=INPUT_DIM))
initial = adapter.get_weights()
adapter.set_weights(initial)
stats = adapter.train_local(train_loader, TrainConfig(local_epochs=1))
metrics = adapter.evaluate(validation_loader)
updated = adapter.get_weights()
```

FedAvg uses sample-weighted parameter averaging. Every client must have identical parameter names/shapes. Exclude clients without usable labelled data. If data contains only positive labels, binary classification cannot be evaluated meaningfully; agree a valid negative-label protocol before real experiments.
