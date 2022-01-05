from dataclasses import dataclass
from util import epoch_to_str


@dataclass
class Trade:
    exchange: str
    symbol: str
    timestamp: int = 0
    price: float = 0
    volume: float = 0
    minute: int = 0

    def __str__(self):
        return '{} : {} : {} : {} : {} : {} : {} : {}'.format(self.exchange,
                                                              self.symbol,
                                                              self.timestamp,
                                                              epoch_to_str(self.timestamp, 'us'),
                                                              self.minute,
                                                              epoch_to_str(self.minute, 'm'),
                                                              self.price,
                                                              self.volume)
