import strict_rfc3339
from influxdb_client import Point


def ingest_trade(org, write_client, trade):
    write_client.write('trade-book', org,
                       Point('trade_data')
                       .tag('exchange', trade.exchange)
                       .tag('symbol', trade.symbol)
                       .field('price', float(trade.price))
                       .field('volume', float(trade.volume))
                       .time(strict_rfc3339.timestamp_to_rfc3339_utcoffset(trade.timestamp / 1000 / 1000)))


def ingest_quote(org, write_client, quote, ticker):
    write_client.write('quote-book', org,
                       Point('quote_data')
                       .tag('market', quote.exchange)
                       .tag('symbol', quote.symbol)
                       .tag('ticker', ticker)
                       .field('open', float(quote.open))
                       .field('high', float(quote.high))
                       .field('low', float(quote.low))
                       .field('close', float(quote.close))
                       .time(strict_rfc3339.timestamp_to_rfc3339_utcoffset(quote.minute)))
