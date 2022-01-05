import json
from influxdb_client import InfluxDBClient, BucketRetentionRules
from influxdb_client.client.flux_table import FluxStructureEncoder
from util.backup import export


class InfluxDB(InfluxDBClient):
    def __init__(self, url, token, org, conf, **kwargs) -> None:
        """
        Initialize custom influxDB client instance (child of InfluxDBClient)
        :param url:
        :param token:
        :param org:
        :param conf:
        """
        super().__init__(url, token, org, **kwargs)
        self.url = url
        self.token = token
        self.org = org
        self.conf = conf

    def verify_buckets(self, buckets_api_, query_api_, desired_buckets_):
        try:
            existing_buckets_ = buckets_api_.find_buckets().buckets
            existing_bucket_names = [bucket.name for bucket in existing_buckets_]

            for desired_bucket_name in desired_buckets_.keys():
                if desired_bucket_name in existing_bucket_names:
                    bucket = next(bucket for bucket in existing_buckets_ if bucket.name == desired_bucket_name)
                    if bucket.retention_rules[0] == BucketRetentionRules(**desired_buckets_[desired_bucket_name]):
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
                                                   retention_rules=BucketRetentionRules(
                                                       **desired_buckets_[desired_bucket_name]),
                                                   org=self.org)
                else:
                    buckets_api_.create_bucket(bucket_name=desired_bucket_name,
                                               retention_rules=BucketRetentionRules(
                                                   **desired_buckets_[desired_bucket_name]),
                                               org=self.org)
        except Exception as e:
            raise e
