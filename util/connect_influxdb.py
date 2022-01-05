import json

from influxdb_client import InfluxDBClient
from influxdb_client.client.flux_table import FluxStructureEncoder
from influxdb_client.client.write_api import SYNCHRONOUS

from conf import conf
from util.backup import export


def connect_influxdb(token, org):
    try:
        clientDB = InfluxDBClient(url="http://localhost:8086", token=token, org=org)
        buckets_api = clientDB.buckets_api()
        existing_buckets = buckets_api.find_buckets().buckets
        query_api = clientDB.query_api()
        verify_buckets(org, buckets_api, query_api, existing_buckets, conf.desired_buckets)
        return clientDB
    except Exception as e:
        # todo: logging
        raise e


def verify_buckets(org, buckets_api_, query_api_, existing_buckets_, desired_buckets_):
    existing_bucket_names = [bucket.name for bucket in existing_buckets_]

    for desired_bucket_name in desired_buckets_.keys():
        if desired_bucket_name in existing_bucket_names:
            bucket = next(bucket for bucket in existing_buckets_ if bucket.name == desired_bucket_name)
            if bucket.retention_rules[0] == desired_buckets_[desired_bucket_name]:
                continue
            else:
                # if not empty, store bucket data
                tables = query_api_.query('from(bucket:"{}") |> range(start: 0)'.format(desired_bucket_name))
                if bool(tables):
                    print('export bucket as json')
                    output = json.dumps(tables, cls=FluxStructureEncoder, indent=2)
                    export(output, desired_bucket_name)
                else:
                    print('empty bucket')

                # delete bucket with bad retention rules
                buckets_api_.delete_bucket(bucket)
                # create new bucket
                buckets_api_.create_bucket(bucket_name=desired_bucket_name,
                                           retention_rules=desired_buckets_[desired_bucket_name], org=org)
        else:
            buckets_api_.create_bucket(bucket_name=desired_bucket_name,
                                       retention_rules=desired_buckets_[desired_bucket_name], org=org)
