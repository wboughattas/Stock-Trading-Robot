from dataclasses import dataclass

from util import epoch_to_str


@dataclass
class Quote:
    exchange: str
    symbol: str
    minute: int = 0
    open: float = 0
    high: float = 0
    low: float = 0
    close: float = 0
    volume: float = 0

    def __str__(self):
        return '{} : {} : {} : {} : {} : {} : {} : {} : {}'.format(self.exchange,
                                                                   self.symbol,
                                                                   self.minute,
                                                                   epoch_to_str(self.minute, 'm'),
                                                                   self.open,
                                                                   self.high,
                                                                   self.low,
                                                                   self.close,
                                                                   self.volume)
