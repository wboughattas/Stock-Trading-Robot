import os
from influxdb_client.client.write_api import SYNCHRONOUS
from api import *
from util.config import get_env, get_conf

if __name__ == '__main__':
    # update os environments
    get_env()

    # get configurations
    config = get_conf('influxdb', 'livedata')

    # todo: disable verbose
    # connect to InfluxDB2.0
    with Influxdb.InfluxDB("http://localhost:8086", os.getenv('influxdb_token'), os.getenv('org'), config) as clientDB:
        # execute check-up on buckets given configuration (auto add/delete/update)
        clientDB.verify_buckets(clientDB.buckets_api(), clientDB.query_api(), config['desired_buckets'])

        # initialize write client
        write_api = clientDB.write_api(write_options=SYNCHRONOUS)

        # initialize finnhub object given configuration and connect to websocket
        clientFB = Finnhub.Finnhub(os.getenv('finnhub_token'), clientDB.org, config, write_api)

        # run in loop
        clientFB.run_forever()
