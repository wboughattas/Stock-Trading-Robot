from influxdb_client import BucketRetentionRules

desired_buckets = {
    'order-book': BucketRetentionRules(type="expire", every_seconds=0, shard_group_duration_seconds=604800),
    'quote-book': BucketRetentionRules(type="expire", every_seconds=0, shard_group_duration_seconds=604800),
    'trade-book': BucketRetentionRules(type="expire", every_seconds=0, shard_group_duration_seconds=604800)
}

live_data = {
    'finnhub': {
        'BINANCE': ('BTCUSDT', 'ETHUSDT'),
        'KUKOIN': ('BTCUSDT', 'ETHUSDT')
    },
    'alpaca': {
        'BINANCE': ('BTCUSDT', 'ETHUSDT'),
        'KUKOIN': ('BTCUSDT', 'ETHUSDT')
    }
}
