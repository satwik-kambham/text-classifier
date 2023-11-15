from torch.utils.data import Dataset
from datasets import load_dataset


class TrecDataset(Dataset):
    """
    TREC dataset
    Coarse labels: 6
    Fine labels: 50

    Coarse labels:
    'ABBR' (0): Abbreviation.
    'ENTY' (1): Entity.
    'DESC' (2): Description and abstract concept.
    'HUM' (3): Human being.
    'LOC' (4): Location.
    'NUM' (5): Numeric value.

    Fine labels:

    ABBREVIATION:
        'ABBR:abb' (0): Abbreviation.
        'ABBR:exp' (1): Expression abbreviated.
    ENTITY:
        'ENTY:animal' (2): Animal.
        'ENTY:body' (3): Organ of body.
        'ENTY:color' (4): Color.
        'ENTY:cremat' (5): Invention, book and other creative piece.
        'ENTY:currency' (6): Currency name.
        'ENTY:dismed' (7): Disease and medicine.
        'ENTY:event' (8): Event.
        'ENTY:food' (9): Food.
        'ENTY:instru' (10): Musical instrument.
        'ENTY:lang' (11): Language.
        'ENTY:letter' (12): Letter like a-z.
        'ENTY:other' (13): Other entity.
        'ENTY:plant' (14): Plant.
        'ENTY:product' (15): Product.
        'ENTY:religion' (16): Religion.
        'ENTY:sport' (17): Sport.
        'ENTY:substance' (18): Element and substance.
        'ENTY:symbol' (19): Symbols and sign.
        'ENTY:techmeth' (20): Techniques and method.
        'ENTY:termeq' (21): Equivalent term.
        'ENTY:veh' (22): Vehicle.
        'ENTY:word' (23): Word with a special property.
    DESCRIPTION:
        'DESC:def' (24): Definition of something.
        'DESC:desc' (25): Description of something.
        'DESC:manner' (26): Manner of an action.
        'DESC:reason' (27): Reason.
    HUMAN:
        'HUM:gr' (28): Group or organization of persons
        'HUM:ind' (29): Individual.
        'HUM:title' (30): Title of a person.
        'HUM:desc' (31): Description of a person.
    LOCATION:
        'LOC:city' (32): City.
        'LOC:country' (33): Country.
        'LOC:mount' (34): Mountain.
        'LOC:other' (35): Other location.
        'LOC:state' (36): State.
    NUMERIC:
        'NUM:code' (37): Postcode or other code.
        'NUM:count' (38): Number of something.
        'NUM:date' (39): Date.
        'NUM:dist' (40): Distance, linear measure.
        'NUM:money' (41): Price.
        'NUM:ord' (42): Order, rank.
        'NUM:other' (43): Other number.
        'NUM:period' (44): Lasting time of something
        'NUM:perc' (45): Percent, fraction.
        'NUM:speed' (46): Speed.
        'NUM:temp' (47): Temperature.
        'NUM:volsize' (48): Size, area and volume.
        'NUM:weight' (49): Weight.
    """

    def __init__(self, split="train"):
        self.data = load_dataset("trec", split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        coarse = self.data[idx]["coarse_label"]
        fine = self.data[idx]["fine_label"]
        return text, coarse, fine
