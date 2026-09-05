from unittest.mock import Mock, patch

import pytest

from celery import Celery, signals
from celery.app.capabilities import (
    CapabilityRegistry, normalize_capabilities, satisfies,
)
from celery.exceptions import NoQualifiedWorkerError


def test_normalize_capabilities():
    assert normalize_capabilities(None) == ()
    assert normalize_capabilities('gpu') == ('gpu',)
    assert normalize_capabilities(['b', 'a', 'a', None, '']) == ('a', 'b')
    assert normalize_capabilities({'x', 'y'}) == ('x', 'y')


def test_satisfies():
    assert satisfies(['gpu', 'cpu'], ['gpu'])
    assert not satisfies(['cpu'], ['gpu'])
    assert satisfies(None, None)
    assert satisfies(['gpu'], None)


class test_CapabilityRegistry:

    def setup_method(self):
        self.app = Celery(set_as_current=False)
        self.app.conf.task_capability_state_ttl = 0
        self.reg = CapabilityRegistry(self.app)

    def test_remember_forget_snapshot(self):
        self.reg.remember('w1', ['gpu', 'gpu'])
        self.reg.remember('w2', ('ffmpeg',))
        assert self.reg.workers['w1'] == frozenset({'gpu'})
        assert self.reg.workers['w2'] == frozenset({'ffmpeg'})
        self.reg.forget('w1')
        assert 'w1' not in self.reg.workers

    def test_qualified_workers_requires_subset(self):
        self.reg.remember('gpu-worker', ['gpu', 'cuda'])
        self.reg.remember('plain-worker', [])
        assert self.reg.qualified_workers(['gpu'], refresh=False) == {
            'gpu-worker'}
        assert self.reg.qualified_workers(
            ['gpu', 'ffmpeg'], refresh=False) == set()

    def test_refresh_from_inspect_replies(self):
        inspect = self.app.control.inspect.return_value = Mock()
        inspect.capabilities.return_value = {
            'w1': {'ok': ['gpu']},
            'w2': {'ok': ['ffmpeg', 'gpu']},
        }
        with patch('celery.app.control.Control.inspect',
                   return_value=inspect):
            workers = self.reg.refresh(force=True)
        assert workers['w1'] == frozenset({'gpu'})
        assert workers['w2'] == frozenset({'gpu', 'ffmpeg'})
        assert self.reg.qualified_workers(['gpu'], refresh=False) == {
            'w1', 'w2'}

    def test_refresh_without_replies_yields_empty_view(self):
        inspect = self.app.control.inspect.return_value = Mock()
        inspect.capabilities.return_value = None
        with patch('celery.app.control.Control.inspect',
                   return_value=inspect):
            with pytest.raises(NoQualifiedWorkerError) as ei:
                self.reg.ensure_qualified(['gpu'], task='t')
        assert ei.value.required == ('gpu',)
        assert ei.value.available == {}


class test_capability_routing:

    def test_no_capabilities_publishes_unchecked(self, app):
        @app.task
        def add(x, y):
            return x + y

        with patch.object(app.capabilities, 'ensure_qualified') as gate:
            add.apply_async((2, 2))
            gate.assert_not_called()

    def test_matching_worker_publishes(self, app):
        @app.task
        def add(x, y):
            return x + y

        app.capabilities.remember('w1', ['gpu'])
        add.apply_async((2, 2), capabilities=['gpu'])  # no exception

    def test_no_qualified_worker_raises_and_records_failure(self, app):
        @app.task
        def add(x, y):
            return x + y

        received = []
        signals.task_routing_rejected.connect(
            lambda **kw: received.append(kw), weak=False)
        app.conf.task_capability_state_ttl = 0

        with pytest.raises(NoQualifiedWorkerError) as ei:
            add.apply_async((2, 2), capabilities=['gpu'])
        assert ei.value.required == ('gpu',)
        assert received and received[0]['task_id']
        assert received[0]['required_capabilities'] == ['gpu']

        result = app.AsyncResult(received[0]['task_id'])
        assert result.state == 'FAILURE'
        assert isinstance(result.result, NoQualifiedWorkerError)

    def test_routing_can_be_disabled(self, app):
        app.conf.task_capability_routing = False

        @app.task
        def add(x, y):
            return x + y

        with patch.object(app.capabilities, 'ensure_qualified') as gate:
            add.apply_async((2, 2), capabilities=['gpu'])
            gate.assert_not_called()

    def test_task_class_declared_capabilities(self, app):
        @app.task(capabilities=['gpu'])
        def add(x, y):
            return x + y

        app.conf.task_capability_state_ttl = 0
        with pytest.raises(NoQualifiedWorkerError):
            add.apply_async((2, 2))

    def test_capabilities_header_is_sorted_list(self, app):
        @app.task
        def add(x, y):
            return x + y

        app.capabilities.remember('w1', ['gpu'])
        captured = {}
        signals.before_task_publish.connect(
            lambda headers=None, **kw: captured.update(headers),
            weak=False)
        add.apply_async((2, 2), capabilities=['z', 'a'])
        assert captured['capabilities'] == ['a', 'z']
