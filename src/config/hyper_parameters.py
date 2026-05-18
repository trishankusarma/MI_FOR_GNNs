from dataclasses import dataclass

@dataclass
class HyperParameters:
    # model.py
    in_channels:int = 2
    out_channels:int = 10

    # train.py
    num_epochs: int = 5
    lr: float = 1e-3
    weight_decay:float = 1e-2

    # circuit size
    k: int = 2