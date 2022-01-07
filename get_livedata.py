import os
import sys

from influxdb_client.client.write_api import SYNCHRONOUS
from api import *
from util.config import get_env, get_conf

# sys.stdout = None


if __name__ == '__main__':
    # update os environments
    get_env()

    # get configurations
    config = get_conf('influxdb', 'livedata')

    # todo: disable verbose
    # todo: test performance (add attribute: ingestion time to db)
    # todo: add quote attribute: average price and maybe more indicators? (check tradingView + research)
    # todo: update requirements

    # connect to InfluxDB2.0
    clientDB = Influxdb.InfluxDB("http://localhost:8086", os.getenv('influxdb_token'), os.getenv('org'), config, verbose=False)

    # execute check-up on buckets given configuration (auto add/delete/update)
    clientDB.verify_buckets(clientDB.buckets_api(), clientDB.query_api(), config['desired_buckets'])
    # initialize write client
    write_api = clientDB.write_api(write_options=SYNCHRONOUS)

    # initialize finnhub object given configuration and connect to websocket
    clientFB = Finnhub.Finnhub(os.getenv('finnhub_token'), clientDB.org, config, write_api)

    # run in loop
    clientFB.run_forever()
