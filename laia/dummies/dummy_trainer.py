import pytorch_lightning as pl


class DummyTrainer(pl.Trainer):
    def __init__(self, **kwargs):
        defaults = {
            "enable_checkpointing": False,
            "logger": True,
            "enable_model_summary": False,
            "max_epochs": 1,
            "limit_train_batches": 10,
            "limit_val_batches": 10,
            "limit_test_batches": 10,
            "enable_progress_bar": False,
            "deterministic": True,
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
