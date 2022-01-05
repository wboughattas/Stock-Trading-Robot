import os

import websocket
from influxdb_client.client.write_api import SYNCHRONOUS

from api.Influxdb import InfluxDB
from util.config import get_env, get_conf
from api import *

if __name__ == '__main__':
    # update os environments
    get_env()

    # get configurations
    config = get_conf('influxdb', 'livedata')

    # connect to InfluxDB2.0
    clientDB = InfluxDB("http://localhost:8086", os.getenv('influxdb_token'), os.getenv('org'),
                        config).connect_influxdb()

    # execute check-up on buckets given configuration (auto add/delete/update)
    clientDB.verify_buckets(clientDB.buckets_api(), clientDB.query_api(), config['desired_buckets'])

    # initialize write client
    write_api = clientDB.write_api(write_options=SYNCHRONOUS)

    # initialize finnhub object given configuration and connect to websocket
    clientFB = Finnhub(os.getenv('finnhub_token'), clientDB, config, write_api)
    web_socket = websocket.WebSocketApp(url=clientFB.url,
                                        on_open=clientFB.on_open,
                                        on_message=clientFB.on_message,
                                        on_close=clientFB.on_close,
                                        on_error=clientFB.on_error)

    # run in loop
    web_socket.run_forever()
