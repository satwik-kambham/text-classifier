from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from tokenizers import Tokenizer

from dataset import TrecDataset


class TrecDataModule(LightningDataModule):
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
        # TODO: implement collate_fn
        text, coarse, fine = zip(*batch)
        out = self.tokenizer.encode_batch(text)
        return out

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
