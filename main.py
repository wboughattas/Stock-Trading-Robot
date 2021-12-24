import websocket
import json
from datetime import datetime
from pytz import timezone
from connections import socket, token, org, bucket
from influxdb_client import InfluxDBClient, Point, WritePrecision, WriteOptions
from influxdb_client.client.write_api import SYNCHRONOUS

minutes_processed = {}
minute_candlesticks = []
current_trade = None
previous_trade = None

stock_exchange = "BINANCE"
stock_symbol = "BTCUSDT"


def on_open(ws):
    message = {
        "type": "subscribe",
        "symbol": ':'.join([stock_exchange, stock_symbol])
    }
    ws.send(json.dumps(message))


def on_message(ws, message):
    global current_trade, previous_trade

    current_trades = json.loads(message)
    if len(current_trades['data']) > 0:
        for idx, new_data in enumerate(current_trades['data']):
            previous_trade = current_trade
            current_trade = new_data
            trade_time = datetime.fromtimestamp(current_trade['t'] / 1000, timezone('US/Eastern'))
            trade_time_ms = trade_time.strftime('%Y-%m-%d %H:%M:%S.%f')
            trade_time_min = trade_time.strftime('%Y-%m-%d %H:%M')
            print('{} : {} : {} : {}'.format(current_trade['s'], trade_time_ms, current_trade['p'], current_trade['v']))
            # ingest to trade_db

            if trade_time_min not in minutes_processed:
                minutes_processed[trade_time_min] = True

                if len(minute_candlesticks) > 0:
                    minute_candlesticks[-1]['close'] = previous_trade['p']

                minute_candlesticks.append({
                    "minute": trade_time_min,
                    "open": current_trade['p'],
                    "high": current_trade['p'],
                    "low": current_trade['p'],
                    "volume": 0
                })

            if len(minute_candlesticks) > 0:
                current_candlestick = minute_candlesticks[-1]
                current_candlestick['volume'] += current_trade['v']
                if current_trade['p'] > current_candlestick['high']:
                    current_candlestick['high'] = current_trade['p']
                if current_trade['p'] < current_candlestick['low']:
                    current_candlestick['low'] = current_trade['p']

            print("== Candlesticks ==")
            for candlestick in minute_candlesticks:
                print(candlestick)
                # ingest to intraday_db


def on_close(ws):
    print("### closed ###")


def on_error(ws, error):
    print('error: ', error)


if __name__ == '__main__':
    web_socket = websocket.WebSocketApp(socket,
                                        on_open=on_open,
                                        on_message=on_message,
                                        on_close=on_close,
                                        on_error=on_error)
    # web_socket.run_forever()

    # flux write query
    write = \
        'import "experimental/csv"\
        relativeToNow = (tables=<-) =>\
            tables\
                |> elapsed()\
                |> sort(columns: ["_time"], desc: true)\
                |> cumulativeSum(columns: ["elapsed"])\
                |> map(fn: (r) => ({ r with _time: time(v: int(v: now()) - (r.elapsed * 1000000000))}))\
        csv.from(url: "https://influx-testdata.s3.amazonaws.com/noaa.csv")\
            |> relativeToNow()\
            |> to(bucket: "test-bucket", org: "test-org")'

    # flux read query
    read = \
        'from(bucket: "temp")\
            |> range(start: -3h)\
            |> filter(fn: (r) => r._measurement == "average_temperature")\
            |> filter(fn: (r) => r._field == "degrees")\
            |> filter(fn: (r) => r.location == "coyote_creek")'

    # establish a connection
    client = InfluxDBClient(url="http://localhost:8086", token=token, org=org)

    # instantiate the WriteAPI and QueryAPI
    # write_api = client.write_api()
    query_api = client.query_api()

    # create and write the point
    # write_api.write(bucket=bucket, org=org, record=p)
    query_api.query(query=write)
    # return the table and print the result
    result = query_api.query(org=org, query=read)
    for table in result:
        for record in table.records:
            print(record.get_value(), record.get_field())

    print()
