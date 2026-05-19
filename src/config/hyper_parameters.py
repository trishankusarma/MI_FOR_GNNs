from dataclasses import dataclass, field

@dataclass
class HyperParameters:
    # dataset_generation.py
    num_graphs:int  = 10000
    min_nodes       = 200
    max_nodes       = 300
    edge_prob       = 0.6
    transform       = None 
    pre_transform   = None

    # model.py
    in_channels:int = 2
    hidden_layers: list[int] = field(default_factory=lambda: [64])
    out_channels:int = 1

    # train.py
    num_epochs: int = 20
    lr: float = 1e-3
    weight_decay:float = 1e-2

    # circuit size
    k: int = 10