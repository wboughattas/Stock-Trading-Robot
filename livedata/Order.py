from dataclasses import dataclass


@dataclass
class Order:
    buy: int
    sell: int
