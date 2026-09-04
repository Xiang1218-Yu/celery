"""Tests for named execution profiles."""
import pytest

from celery import chain, chord, group
from celery.app.profiles import ExecutionProfile, ExecutionProfileRegistry
from celery.app.task import Context
from celery.exceptions import UnknownExecutionProfile

PROFILE = {
    'rate_limit': '100/m',
    'priority': 7,
    'time_limit': 60,
    'soft_time_limit': 45,
}


class test_ExecutionProfile:

    def test_snapshot_frozen(self):
        p = ExecutionProfile('p', **PROFILE)
        snap = p.snapshot()
        p.update(priority=1)
        assert snap == {'name': 'p', **PROFILE}
        assert p.priority == 1

    @pytest.mark.parametrize('fields', [
        {'time_limit': 5, 'soft_time_limit': 10},
        {'rate_limit': '10/x'},
        {'rate_limit': ''},
        {'priority': 'high'},
        {'priority': 1.5},
        {'time_limit': 0},
        {'time_limit': -1},
    ])
    def test_invalid_fields_rejected(self, fields):
        with pytest.raises(ValueError):
            ExecutionProfile('p', **fields)

    def test_partial_update_keeps_other_fields(self):
        p = ExecutionProfile('p', **PROFILE)
        p.update(priority=2)
        assert p.priority == 2
        assert p.rate_limit == PROFILE['rate_limit']
        assert p.time_limit == PROFILE['time_limit']
        assert p.soft_time_limit == PROFILE['soft_time_limit']


class test_ExecutionProfileRegistry:

    def setup_method(self):
        self.registry = ExecutionProfileRegistry()

    def test_add_and_query(self):
        self.registry.add('gold', **PROFILE)
        assert 'gold' in self.registry
        assert len(self.registry) == 1
        assert self.registry.names() == ['gold']
        profile = self.registry['gold']
        assert profile.name == 'gold'
        assert profile.rate_limit == '100/m'
        snap = self.registry.snapshot('gold')
        assert snap == {'name': 'gold', **PROFILE}

    def test_get_unknown_raises(self):
        with pytest.raises(UnknownExecutionProfile):
            self.registry.get('nope')
        with pytest.raises(UnknownExecutionProfile):
            self.registry.snapshot('nope')

    def test_add_duplicate_and_exist_ok(self):
        self.registry.add('gold', **PROFILE)
        with pytest.raises(ValueError):
            self.registry.add('gold', priority=1)
        self.registry.add('gold', priority=1, exist_ok=True)
        assert self.registry['gold'].priority == 1

    def test_update_and_delete(self):
        self.registry.add('gold', **PROFILE)
        self.registry.update('gold', rate_limit='10/s', priority=9,
                             time_limit=30, soft_time_limit=20)
        assert self.registry.snapshot('gold')['rate_limit'] == '10/s'
        self.registry.delete('gold')
        assert self.registry.names() == []
        with pytest.raises(UnknownExecutionProfile):
            self.registry.delete('gold')

    def test_iterate_and_clear(self):
        self.registry.add('a', priority=1)
        self.registry.add('b', priority=2)
        assert sorted(self.registry) == ['a', 'b']
        assert {p.name for p in self.registry.all()} == {'a', 'b'}
        self.registry.clear()
        assert len(self.registry) == 0

    def test_resolve(self):
        self.registry.add('gold', **PROFILE)
        # None -> no selection
        assert self.registry.resolve(None) is None
        # name -> snapshot
        assert self.registry.resolve('gold') == {'name': 'gold', **PROFILE}
        # profile object -> snapshot
        assert self.registry.resolve(self.registry.get('gold')) == \
            {'name': 'gold', **PROFILE}
        # already resolved snapshot mapping -> validated copy
        snap = {'name': 'worker-sent', 'rate_limit': '5/h', 'priority': 2,
                'time_limit': 12, 'soft_time_limit': 8}
        assert self.registry.resolve(snap) == snap
        # malformed snapshot rejected
        with pytest.raises(ValueError):
            self.registry.resolve({'rate_limit': 'bad'})
        # unsupported type
        with pytest.raises(TypeError):
            self.registry.resolve(123)


class test_publish_with_execution_profile:

    def setup_method(self):
        self.sent = []

        def record(producer, name, message, **opts):
            self.sent.append((name, message, opts))

        self.record = record

    @property
    def last(self):
        return self.sent[-1]

    def test_profile_snapshot_in_message(self, app):
        app.execution_profiles.add('gold', **PROFILE)

        @app.task
        def add(x, y):
            return x + y

        app.amqp.send_task_message = self.record
        add.apply_async((2, 3), execution_profile='gold')

        _, message, opts = self.last
        assert message.headers['execution_profile'] == \
            {'name': 'gold', **PROFILE}
        # time limits flow through the existing timelimit header
        assert message.headers['timelimit'] == [60, 45]
        # priority flows through message properties
        assert opts['priority'] == 7

    def test_explicit_options_take_precedence(self, app):
        app.execution_profiles.add('gold', **PROFILE)

        @app.task
        def add(x, y):
            return x + y

        app.amqp.send_task_message = self.record
        add.apply_async((2, 3), execution_profile='gold',
                        priority=1, time_limit=10, soft_time_limit=5)
        _, message, opts = self.last
        assert message.headers['timelimit'] == [10, 5]
        assert opts['priority'] == 1
        # the profile is still recorded for tracing/rate limiting
        assert message.headers['execution_profile']['name'] == 'gold'

    def test_no_profile_keeps_defaults(self, app):

        @app.task
        def add(x, y):
            return x + y

        app.amqp.send_task_message = self.record
        add.apply_async((4, 4))
        _, message, opts = self.last
        assert message.headers['execution_profile'] is None
        assert message.headers['timelimit'] == [None, None]
        assert 'priority' not in opts

    def test_update_only_affects_subsequent_messages(self, app):
        app.execution_profiles.add('gold', **PROFILE)

        @app.task
        def add(x, y):
            return x + y

        app.amqp.send_task_message = self.record
        add.apply_async((1, 1), execution_profile='gold')
        _, first, first_opts = self.sent[-1]

        app.execution_profiles.update('gold', rate_limit='10/s', priority=9,
                                      time_limit=30, soft_time_limit=20)
        add.apply_async((2, 2), execution_profile='gold')
        _, second, second_opts = self.last

        # the first message keeps the old snapshot
        assert first.headers['execution_profile'] == \
            {'name': 'gold', **PROFILE}
        assert first.headers['timelimit'] == [60, 45]
        assert first_opts['priority'] == 7
        # subsequent messages get the new snapshot
        assert second.headers['execution_profile']['rate_limit'] == '10/s'
        assert second.headers['timelimit'] == [30, 20]
        assert second_opts['priority'] == 9

    def test_unknown_profile_raises(self, app):

        @app.task
        def add(x, y):
            return x + y

        app.amqp.send_task_message = self.record
        with pytest.raises(UnknownExecutionProfile):
            add.apply_async((1, 1), execution_profile='missing')

    def test_snapshot_mapping_passes_through_without_registry(self, app):
        # this is how retries/canvas tasks re-publish on a worker
        # that does not have the named profile registered

        @app.task
        def mul(x, y):
            return x * y

        snap = {'name': 'tenant-a', 'rate_limit': '5/h', 'priority': 2,
                'time_limit': 12, 'soft_time_limit': 8}
        app.amqp.send_task_message = self.record
        mul.apply_async((2, 3), execution_profile=snap)
        _, message, opts = self.last
        assert message.headers['execution_profile'] == snap
        assert message.headers['timelimit'] == [12, 8]
        assert opts['priority'] == 2

    def test_chain_successors_inherit_snapshot(self, app):
        app.execution_profiles.add('gold', **PROFILE)

        @app.task
        def add(x, y):
            return x + y

        app.amqp.send_task_message = self.record
        chain(add.s(2, 2) | add.s(4)).apply_async(execution_profile='gold')
        _, message, _ = self.last
        assert message.headers['execution_profile']['name'] == 'gold'
        successor = message.body[2]['chain'][0]
        assert successor['options']['execution_profile']['name'] == 'gold'

    def test_callback_inherits_snapshot(self, app):
        app.execution_profiles.add('gold', **PROFILE)

        @app.task
        def add(x, y):
            return x + y

        app.amqp.send_task_message = self.record
        add.apply_async((1, 2), link=add.s(9), execution_profile='gold')
        _, message, _ = self.last
        callback = message.body[2]['callbacks'][0]
        assert callback['options']['execution_profile']['name'] == 'gold'

    def test_group_members_inherit_snapshot(self, app):
        from celery import group
        app.execution_profiles.add('gold', **PROFILE)

        @app.task
        def add(x, y):
            return x + y

        app.amqp.send_task_message = self.record
        group(add.s(1, 1), add.s(2, 2)).apply_async(execution_profile='gold')
        member_messages = [m for _, m, _ in self.sent]
        assert len(member_messages) == 2
        for message in member_messages:
            assert message.headers['execution_profile']['name'] == 'gold'

    def test_chord_body_carries_snapshot(self, app):
        app.execution_profiles.add('gold', **PROFILE)

        @app.task
        def add(x, y):
            return x + y

        app.amqp.send_task_message = self.record
        captured = {}
        app.backend.apply_chord = (
            lambda header_args, body, **kw: captured.setdefault('body', body))
        app.backend.ensure_chords_allowed = lambda: None

        chord(add.s(1, 1))(add.s(100), execution_profile='gold')

        # body applied later by chord-unlock on a (possibly different)
        # worker: it must carry the frozen snapshot, not the profile name
        body_snapshot = captured['body'].options['execution_profile']
        assert body_snapshot == {'name': 'gold', **PROFILE}
        _, message, _ = self.last
        assert message.headers['execution_profile']['name'] == 'gold'

    def test_retry_republishes_snapshot(self, app):

        @app.task(bind=True)
        def retrying(self):
            return None

        request = Context({
            'id': 'id', 'task': 't',
            'execution_profile': {'name': 'gold', **PROFILE},
        })
        options = request.as_execution_options()
        assert options['execution_profile'] == {'name': 'gold', **PROFILE}

        # signature built from the request re-publishes the snapshot
        app.amqp.send_task_message = self.record
        sig = retrying.signature_from_request(request=request)
        sig.apply_async()
        _, message, _ = self.last
        assert message.headers['execution_profile'] == \
            {'name': 'gold', **PROFILE}

    def test_eager_request_carries_resolved_snapshot(self, app):
        app.conf.task_always_eager = True
        app.execution_profiles.add('gold', **PROFILE)

        seen = []

        @app.task
        def add(x, y):
            seen.append(add.request.execution_profile)
            return x + y

        try:
            add.apply_async((2, 3), execution_profile='gold')
        finally:
            app.conf.task_always_eager = False
        assert seen[0] == {'name': 'gold', **PROFILE}
