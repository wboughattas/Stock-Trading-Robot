import websocket
import json
from datetime import datetime
from pytz import timezone
import mysqlPy
from connections import SOCKET, MYSQL_host_name, MYSQL_user_name, MYSQL_user_password, MYSQL_db_name

minutes_processed = {}
minute_candlesticks = []
current_trade = None
previous_trade = None


def on_open(ws):
    message = {
        "type": "subscribe",
        "symbol": "BINANCE:BTCUSDT"
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
    web_socket = websocket.WebSocketApp(SOCKET,
                                        on_open=on_open,
                                        on_message=on_message,
                                        on_close=on_close,
                                        on_error=on_error)
    sql_connection = mysqlPy.create_db_connection(MYSQL_host_name, MYSQL_user_name, MYSQL_user_password, MYSQL_db_name)

    # web_socket.run_forever()

    print()
