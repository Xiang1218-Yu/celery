"""Tests for per-request execution profile handling on the worker side."""
from kombu.utils.limits import TokenBucket

from celery.worker.consumer.consumer import Consumer


class test_ExecutionProfileBuckets:

    def setup_method(self):
        # bucket_for_execution_profile only needs the per-consumer bucket
        # store, so a non-initialized Consumer instance is sufficient.
        self.consumer = Consumer.__new__(Consumer)
        self.consumer.execution_profile_buckets = {}

    def test_no_snapshot(self):
        assert self.consumer.bucket_for_execution_profile(None) is None
        assert self.consumer.bucket_for_execution_profile({}) is None

    def test_snapshot_without_rate(self):
        assert self.consumer.bucket_for_execution_profile(
            {'name': 'p', 'rate_limit': None}) is None

    def test_bucket_cached_per_name_and_rate(self):
        profile = {
            'name': 'tenant-a', 'rate_limit': '1/m', 'priority': 7,
            'time_limit': 60, 'soft_time_limit': 45,
        }
        bucket = self.consumer.bucket_for_execution_profile(profile)
        assert isinstance(bucket, TokenBucket)
        # the same (name, rate) snapshot reuses the bucket so the rate is
        # actually enforced across consecutive messages
        assert self.consumer.bucket_for_execution_profile(dict(profile)) \
            is bucket

    def test_rate_change_creates_new_bucket(self):
        old = {'name': 'tenant-a', 'rate_limit': '1/m'}
        new = {'name': 'tenant-a', 'rate_limit': '100/s'}
        old_bucket = self.consumer.bucket_for_execution_profile(old)
        new_bucket = self.consumer.bucket_for_execution_profile(new)
        assert new_bucket is not old_bucket
        # in-flight messages carrying the old snapshot keep the old bucket
        assert self.consumer.bucket_for_execution_profile(old) is old_bucket
