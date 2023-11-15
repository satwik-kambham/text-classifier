import torch
from torch.utils.data import DataLoader
import lightning as L
from tokenizers import Tokenizer

from data.trec.dataset import TrecDataset


class TrecDataModule(L.LightningDataModule):
    def __init__(
        self,
        tokenizer_ckpt_path,
        batch_size=32,
    ):
        super().__init__()
        self.tokenizer_ckpt_path = tokenizer_ckpt_path
        self.batch_size = batch_size

    def prepare_data(self):
        # download, tokenize, etc...
        self.train_ds = TrecDataset(split="train")
        self.val_ds = TrecDataset(split="test")
        self.tokenizer = Tokenizer.from_file(self.tokenizer_ckpt_path)

    def setup(self, stage):
        pass

    def collate_fn(self, batch):
        text, coarse, fine = zip(*batch)
        encodings = self.tokenizer.encode_batch(text)
        ids = [encoding.ids for encoding in encodings]
        return {
            "input_ids": torch.tensor(ids),
            "coarse": torch.tensor(coarse),
            "fine": torch.tensor(fine),
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
        )
