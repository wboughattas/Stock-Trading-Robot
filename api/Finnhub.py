import json
from dataclasses import replace
from livedata.Quote import Quote
from livedata.Trade import Trade
from queries.ingest_livedata import ingest_trade, ingest_quote
from util import round_time


class Finnhub(object):
    book = {}

    def __init__(self, token, org, conf, write_client):
        self.url = 'wss://ws.finnhub.io?token=' + token
        self.org = org
        self.conf = conf
        self.write_client = write_client
        self.on_open = self.on_open
        self.on_message = self.on_message
        self.on_close = self.on_close
        self.on_error = self.on_error

    def on_open(self, ws):
        for (exchange, symbols) in self.conf['finnhub'].items():
            for symbol in symbols:
                message = {
                    "type": "subscribe",
                    "symbol": exchange + ':' + symbol
                }
                ws.send(json.dumps(message))

    def on_message(self, ws, message):
        current_trades = json.loads(message)

        message_trades = len(current_trades['data'])
        if message_trades > 0:
            for new_trade in current_trades['data']:
                # initialize temporary variables
                stock_pair = new_trade['s']
                exchange, symbol = stock_pair.split(':')

                # add 'sx:stock' to book
                if stock_pair not in self.book.keys():
                    self.book[stock_pair] = {
                        'current_trade': None,
                        'previous_trade': None,
                        'current_quote': None
                    }

                # initialize temporary variables
                current_trade = self.book[stock_pair]['current_trade']
                previous_trade = self.book[stock_pair]['previous_trade']
                current_quote = self.book[stock_pair]['current_quote']

                # initialize and update previous_trade
                if current_trade:
                    previous_trade = Trade(current_trade.exchange,
                                           current_trade.symbol,
                                           current_trade.timestamp,
                                           current_trade.price,
                                           current_trade.volume,
                                           current_trade.minute)
                    # update previous_trade
                    self.book[stock_pair]['previous_trade'] = previous_trade

                # initialize and update current_trade
                current_trade = Trade(exchange,
                                      symbol,
                                      new_trade['t'] * 1000,
                                      new_trade['p'],
                                      new_trade['v'],
                                      round_time(new_trade['t']))

                # update current_trade
                self.book[stock_pair]['current_trade'] = current_trade
                if previous_trade:
                    if current_trade.timestamp <= previous_trade.timestamp:
                        # simultaneous trades are saved at 1microsecond difference to preserve uniqueness of timestamp
                        current_trade = replace(current_trade, timestamp=previous_trade.timestamp + 1)
                        # update current_trade
                        self.book[stock_pair]['current_trade'] = current_trade

                # import point to trade-book
                ingest_trade(self.org, self.write_client, current_trade)

                if isinstance(current_quote, Quote):
                    # accumulate quote volume
                    cumulative_volume = current_quote.volume + current_trade.volume
                    current_quote = replace(current_quote, volume=cumulative_volume)

                    # update quote close price
                    current_quote = replace(current_quote, close=current_trade.price)

                    # update quote high price
                    if current_trade.price > current_quote.high:
                        current_quote = replace(current_quote, high=current_trade.price)

                    # update quote low price
                    if current_trade.price < current_quote.low:
                        current_quote = replace(current_quote, low=current_trade.price)

                    # update current_quote
                    self.book[stock_pair]['current_quote'] = current_quote

                    # import to quote-book
                    ingest_quote(self.org, self.write_client, current_quote, '1m')

                if current_trade and previous_trade:
                    if current_trade.minute > previous_trade.minute:
                        # initialize new Quote (open price initialized)
                        current_quote = Quote(current_trade.exchange,
                                              current_trade.symbol,
                                              current_trade.minute,
                                              current_trade.price,
                                              current_trade.price,
                                              current_trade.price,
                                              current_trade.price,
                                              current_trade.volume)
                        # update current_quote
                        self.book[stock_pair]['current_quote'] = current_quote

                # testing
                print(self.book['BINANCE:ETHUSDT'])

    @staticmethod
    def on_close(ws):
        print('closed')

    @staticmethod
    def on_error(ws, error):
        print('error: ', error)
