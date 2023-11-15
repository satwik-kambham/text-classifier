import sys
import lightning as L

from data.trec.datamodule import TrecDataModule
from model.lstm import LSTMClassifier


def train(tokenizer_ckpt_path):
    dm = TrecDataModule(tokenizer_ckpt_path)
    dm.prepare_data()
    model = LSTMClassifier(
        dm.tokenizer.get_vocab_size(),
        embedding_dim=256,
        hidden_dim=128,
        num_classes=6,
        num_layers=2,
    )

    trainer = L.Trainer(
        max_epochs=10,
        fast_dev_run=False,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    train(sys.argv[1])
