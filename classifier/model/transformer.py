import torch
import torch.nn as nn
import lightning as L
import torchmetrics as tm


def scalar_dot_product_attention(query, key, value):
    matmul_qk = torch.matmul(query, key.transpose(-2, -1))
    scaled_attention_logits = matmul_qk / torch.sqrt(
        torch.tensor(query.size(-1), dtype=torch.float32)
    )
    attention_weights = scaled_attention_logits.softmax(dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def forward(self, x):
        num_batches = x.size(0)
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        query = query.view(num_batches, -1, self.num_heads, self.depth).transpose(1, 2)
        key = key.view(num_batches, -1, self.num_heads, self.depth).transpose(1, 2)
        value = value.view(num_batches, -1, self.num_heads, self.depth).transpose(1, 2)

        scaled_attention, attention_weights = scalar_dot_product_attention(
            query, key, value
        )

        scaled_attention = (
            scaled_attention.transpose(1, 2)
            .contiguous()
            .view(num_batches, -1, self.d_model)
        )

        output = self.dense(scaled_attention)

        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff):
        super().__init__()
        self.dense1 = nn.Linear(d_model, dff)
        self.dense2 = nn.Linear(dff, d_model)

    def forward(self, x):
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, dff)
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, attention_weights = self.mha(x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2, attention_weights


def positional_encoding(position, d_model):
    angle_rates = 1 / torch.pow(10000, (2 * (torch.arange(d_model) // 2)) / d_model)
    angle_rads = torch.arange(position).unsqueeze(1) * angle_rates.unsqueeze(0)
    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads.unsqueeze(0)
    return pos_encoding


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        maximum_position_encoding,
        dropout,
        padding_idx,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(
            input_vocab_size,
            d_model,
            padding_idx=padding_idx,
        )
        self.pos_encoding = positional_encoding(
            maximum_position_encoding,
            d_model,
        )
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    num_heads,
                    dff,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_weights = {}
        seq_len = x.size(1)
        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        for i in range(self.num_layers):
            x, attention_weights[f"encoder_block{i + 1}"] = self.encoder_blocks[i](x)
        return x, attention_weights


class TransformerClassifier(L.LightningModule):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        input_vocab_size,
        maximum_position_encoding,
        num_classes,
        padding_idx=3,
        lr=1e-3,
        weight_decay=1e-2,
        dropout=0.01,
        fine=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.fine = fine

        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            maximum_position_encoding,
            dropout,
            padding_idx,
        )
        self.dense = nn.Linear(d_model, num_classes)

        self.criteria = nn.CrossEntropyLoss()
        self.accuracy = tm.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )

    def forward(self, input_ids):
        x, attention_weights = self.encoder(input_ids)
        x = x[:, 0, :]
        x = self.dense(x)
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
