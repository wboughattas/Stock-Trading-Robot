desired_buckets = {
    'order-book': {'type': "expire", 'every_seconds': 0, 'shard_group_duration_seconds': 604800},
    'quote-book': {'type': "expire", 'every_seconds': 0, 'shard_group_duration_seconds': 604800},
    'trade-book': {'type': "expire", 'every_seconds': 0, 'shard_group_duration_seconds': 604800}
}
