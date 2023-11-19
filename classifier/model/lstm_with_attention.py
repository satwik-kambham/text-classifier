import torch
import torch.nn as nn
import lightning as L
import torchmetrics as tm


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.WQuery = nn.Linear(hidden_dim, hidden_dim)
        self.WKey = nn.Linear(hidden_dim, hidden_dim)
        self.WValue = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        query = torch.tanh(self.WQuery(x))
        key = torch.tanh(self.WKey(x))

        attention_weights = torch.softmax(self.WValue(query + key), dim=1)

        return (attention_weights * x).sum(dim=1), attention_weights


class LSTMWithAttentionClassifier(L.LightningModule):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_classes,
        lr=1e-3,
        weight_decay=1e-2,
        num_layers=1,
        bidirectional=False,
        dropout=0.0,
        padding_idx=3,
        fine=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.fine = fine

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.attention = Attention(
            hidden_dim * (1 + bidirectional),
        )
        self.fc = nn.Linear(
            hidden_dim * (1 + bidirectional),
            num_classes,
        )

        self.criteria = nn.CrossEntropyLoss()
        self.accuracy = tm.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        x, attention_weights = self.attention(x)
        x = self.fc(x)
        return x, attention_weights

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        coarse = batch["coarse"]
        fine = batch["fine"]
        logits, _ = self(input_ids)
        loss = self.criteria(logits, fine if self.fine else coarse)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        coarse = batch["coarse"]
        fine = batch["fine"]
        logits, _ = self(input_ids)
        loss = self.criteria(logits, fine if self.fine else coarse)
        self.log("val_loss", loss)
        pred = logits.argmax(dim=1)
        self.accuracy(pred, fine if self.fine else coarse)
        self.log("val_acc", self.accuracy, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
