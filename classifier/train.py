import sys
import lightning as L

from data.trec.datamodule import TrecDataModule
from model.lstm import LSTMClassifier
from model.lstm_with_attention import LSTMWithAttentionClassifier
from model.transformer import TransformerClassifier


def train(model_name, tokenizer_ckpt_path):
    if model_name == "lstm":
        hparams = {
            "model_name": "lstm",
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
    elif model_name == "lstm_with_attention":
        hparams = {
            "model_name": "lstm_with_attention",
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
        model = LSTMWithAttentionClassifier(
            dm.tokenizer.get_vocab_size(),
            num_classes=50 if hparams["fine"] else 6,
            **hparams,
        )
    elif model_name == "transformer":
        hparams = {
            "model_name": "transformer",
            "batch_size": 32,
            "num_workers": 2,
            "d_model": 256,
            "num_layers": 2,
            "num_heads": 8,
            "dff": 512,
            "maximum_position_encoding": 1000,
            "lr": 1e-4,
            "weight_decay": 1e-2,
            "dropout": 0.1,
            "fine": True,
        }
        dm = TrecDataModule(tokenizer_ckpt_path, **hparams)
        dm.prepare_data()
        model = TransformerClassifier(
            input_vocab_size=dm.tokenizer.get_vocab_size(),
            num_classes=50 if hparams["fine"] else 6,
            **hparams,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    trainer = L.Trainer(
        max_epochs=30,
        fast_dev_run=False,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])
