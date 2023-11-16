import sys
import lightning as L

from data.trec.datamodule import TrecDataModule
from model.lstm import LSTMClassifier


def train(tokenizer_ckpt_path):
    hparams = {
        "batch_size": 32,
        "num_workers": 2,
        "embedding_dim": 256,
        "hidden_dim": 128,
        "num_layers": 2,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "dropout": 0.01,
        "bidirectional": True,
        "fine": True,
    }
    dm = TrecDataModule(tokenizer_ckpt_path, **hparams)
    dm.prepare_data()
    model = LSTMClassifier(
        dm.tokenizer.get_vocab_size(),
        num_classes=50 if hparams["fine"] else 6,
        **hparams,
    )

    trainer = L.Trainer(
        max_epochs=20,
        fast_dev_run=False,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    train(sys.argv[1])
