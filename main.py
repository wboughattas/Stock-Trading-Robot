import os
from dataclasses import dataclass, replace
from pathlib import Path
import strict_rfc3339
import websocket
import json
import datetime
import time
from influxdb_client.client.flux_table import FluxStructureEncoder
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client import InfluxDBClient, BucketRetentionRules, Point
from conf import conf
from Livedata import *
from util import epoch_to_str, round_time
from util.backup import export
from util.config import get_env

# stock info
from util.connect_influxdb import connect_influxdb

stock_exchange = "BINANCE"
stock_symbol = "BTCUSDT"

# trade-book
current_trade = None
previous_trade = None

# quote-book
current_quote = None


def on_open(ws):
    message = {
        "type": "subscribe",
        "symbol": ':'.join([stock_exchange, stock_symbol])
    }
    ws.send(json.dumps(message))


def on_message(ws, message):
    global current_trade, previous_trade, current_quote
    current_trades = json.loads(message)

    message_trades = len(current_trades['data'])
    if message_trades > 0:
        for new_trade in current_trades['data']:
            if current_trade is not None:
                previous_trade = Trade(current_trade.exchange,
                                       current_trade.symbol,
                                       current_trade.timestamp,
                                       current_trade.price,
                                       current_trade.volume,
                                       current_trade.minute)
            # initialize new Trade
            current_trade = Trade(stock_exchange,
                                  stock_symbol,
                                  new_trade['t'] * 1000,
                                  new_trade['p'],
                                  new_trade['v'],
                                  round_time(new_trade['t']))
            if current_trade.timestamp <= previous_trade.timestamp:
                current_trade = replace(current_trade, timestamp=previous_trade.timestamp + 1)

            # import point to trade-book
            write_api.write('trade-book', 'trade-data',
                            Point('trade_data')
                            .tag('exchange', current_trade.exchange)
                            .tag('symbol', current_trade.symbol)
                            .field('price', float(current_trade.price))
                            .field('volume', float(current_trade.volume))
                            .time(strict_rfc3339.timestamp_to_rfc3339_utcoffset(current_trade.timestamp / 1000 / 1000)))

            if isinstance(current_quote, Quote):
                # override quote volume
                cumulative_volume = current_quote.volume + current_trade.volume
                current_quote = replace(current_quote, volume=cumulative_volume)

                # override quote close price
                current_quote = replace(current_quote, close=current_trade.price)

                # override quote high price
                if current_trade.price > current_quote.high:
                    current_quote = replace(current_quote, high=current_trade.price)

                # override quote low price
                if current_trade.price < current_quote.low:
                    current_quote = replace(current_quote, low=current_trade.price)

                # import to quote-book
                write_api.write('quote-book', 'trade-data',
                                Point('quote_data')
                                .tag('market', current_quote.exchange)
                                .tag('symbol', current_quote.symbol)
                                .tag('ticker', '1m')
                                .field('open', float(current_quote.open))
                                .field('high', float(current_quote.high))
                                .field('low', float(current_quote.low))
                                .field('close', float(current_quote.close))
                                .time(strict_rfc3339.timestamp_to_rfc3339_utcoffset(current_quote.minute)))

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

            print('trade', current_trade)
            print('quote', current_quote)


def on_close(ws):
    print("### closed ###")


def on_error(ws, error):
    print('error: ', error)


if __name__ == '__main__':
    socket, token, org = get_env()

    # connect to InfluxDB2.0
    clientDB = connect_influxdb(token, org)
    write_api = clientDB.write_api(write_options=SYNCHRONOUS)
    
    print('verification complete')

    web_socket = websocket.WebSocketApp(socket,
                                        on_open=on_open,
                                        on_message=on_message,
                                        on_close=on_close,
                                        on_error=on_error)
    print(web_socket)
    web_socket.run_forever()
