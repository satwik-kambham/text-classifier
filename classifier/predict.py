import sys
import torch
from tokenizers import Tokenizer

from model.lstm import LSTMClassifier
from model.lstm_with_attention import LSTMWithAttentionClassifier
from model.transformer import TransformerClassifier

COARSE_LABELS = [
    "ABBR (0): Abbreviation",
    "ENTY (1): Entity",
    "DESC (2): Description and abstract concept",
    "HUM (3): Human being",
    "LOC (4): Location",
    "NUM (5): Numeric value",
]

FINE_LABELS = [
    "ABBR (0): Abbreviation",
    "ABBR (1): Expression abbreviated",
    "ENTY (2): Animal",
    "ENTY (3): Organ of body",
    "ENTY (4): Color",
    "ENTY (5): Invention, book and other creative piece",
    "ENTY (6): Currency name",
    "ENTY (7): Disease and medicine",
    "ENTY (8): Event",
    "ENTY (9): Food",
    "ENTY (10): Musical instrument",
    "ENTY (11): Language",
    "ENTY (12): Letter like a-z",
    "ENTY (13): Other entity",
    "ENTY (14): Plant",
    "ENTY (15): Product",
    "ENTY (16): Religion",
    "ENTY (17): Sport",
    "ENTY (18): Element and substance",
    "ENTY (19): Symbols and sign",
    "ENTY (20): Techniques and method",
    "ENTY (21): Equivalent term",
    "ENTY (22): Vehicle",
    "ENTY (23): Word with a special property",
    "DESC (24): Definition of something",
    "DESC (25): Description of something",
    "DESC (26): Manner of an action",
    "DESC (27): Reason",
    "HUM (28): Group or organization of persons",
    "HUM (29): Individual",
    "HUM (30): Title of a person",
    "HUM (31): Description of a person",
    "LOC (32): City",
    "LOC (33): Country",
    "LOC (34): Mountain",
    "LOC (35): Other location",
    "LOC (36): State",
    "NUM (37): Postcode or other code",
    "NUM (38): Number of something",
    "NUM (39): Date",
    "NUM (40): Distance, linear measure",
    "NUM (41): Price",
    "NUM (42): Order, rank",
    "NUM (43): Other number",
    "NUM (44): Lasting time of something",
    "NUM (45): Percent, fraction",
    "NUM (46): Speed",
    "NUM (47): Temperature",
    "NUM (48): Size, area and volume",
    "NUM (49): Weight",
]


def load_ckpts(
    model_name,
    tokenizer_ckpt_path,
    model_ckpt_path,
):
    tokenizer = Tokenizer.from_file(tokenizer_ckpt_path)

    if model_name == "lstm":
        model = LSTMClassifier.load_from_checkpoint(
            model_ckpt_path,
            map_location="cpu",
        )
    elif model_name == "lstm_with_attention":
        model = LSTMWithAttentionClassifier.load_from_checkpoint(
            model_ckpt_path,
            map_location="cpu",
        )
    elif model_name == "transformer":
        model = TransformerClassifier.load_from_checkpoint(
            model_ckpt_path,
            map_location="cpu",
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return tokenizer, model, model_name


def infer(text, tokenizer, model, model_name):
    encoding = tokenizer.encode(text)
    ids = torch.tensor([encoding.ids])
    if model_name == "lstm":
        logits = model(ids)
    elif model_name == "lstm_with_attention":
        logits, _ = model(ids)
    elif model_name == "transformer":
        logits, _ = model(ids)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    label = logits.argmax(dim=1).item()
    return label


if __name__ == "__main__":
    tokenizer, model, model_name = load_ckpts(
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
    )
    inp = ""
    while inp != "exit":
        inp = input("Enter text: ")
        label = infer(inp, tokenizer, model, model_name)
        if model.fine:
            print(FINE_LABELS[label])
        else:
            print(COARSE_LABELS[label])
