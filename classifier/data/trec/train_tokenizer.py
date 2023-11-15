from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import normalizers, pre_tokenizers
from tokenizers.normalizers import StripAccents
from tokenizers.pre_tokenizers import Whitespace

from datasets import load_dataset

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.normalizer = normalizers.Sequence([StripAccents()])
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace()])
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    show_progress=True,
)

dataset = load_dataset("trec", split="train+test")


def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]


tokenizer.train_from_iterator(
    batch_iterator(),
    trainer=trainer,
    length=len(dataset),
)

tokenizer.save("tokenizer.json")
