"""Tests for the persistent task event journal."""
import threading
from decimal import Decimal
from unittest.mock import Mock

import pytest

from celery import states
from celery.events import EventJournal, JournalEntry
from celery.events.journal import (
    FileJournalStorage, JournalStorage, MemoryJournalStorage,
)


def make_event(type, uuid=None, hostname=None, ts=None, **extra):
    event = {'type': type}
    if ts is not None:
        event['timestamp'] = ts
    if uuid is not None:
        event['uuid'] = uuid
    if hostname is not None:
        event['hostname'] = hostname
    event.update(extra)
    return event


LIFECYCLE = [
    # task-sent is emitted by the client, which also supplies a hostname
    # (see EventDispatcher.send); State aggregation requires it.
    make_event('task-sent', uuid='t1', hostname='client1', ts=100.0,
               name='tasks.add', clock=1),
    make_event('task-received', uuid='t1', hostname='w1', ts=101.0,
               name='tasks.add', clock=2),
    make_event('task-started', uuid='t1', hostname='w1', ts=102.0,
               pid=42, clock=3),
    make_event('task-succeeded', uuid='t1', hostname='w1', ts=103.0,
               result=42, runtime=1.0, clock=4),
]


@pytest.fixture(params=['memory', 'file', 'sqlite'])
def journal_kind(request):
    return request.param


@pytest.fixture
def make_journal(journal_kind, tmp_path):
    paths = {
        'file': str(tmp_path / 'events.log'),
        'sqlite': str(tmp_path / 'events.db'),
    }

    def make(kind=None, **kwargs):
        kind = kind or journal_kind
        if kind == 'memory':
            return EventJournal.memory(**kwargs)
        if kind == 'file':
            return EventJournal.file(paths['file'], **kwargs)
        if kind == 'sqlite':
            return EventJournal.sqlite(paths['sqlite'], **kwargs)
        raise ValueError(kind)

    make.kind = journal_kind
    make.path = paths.get(journal_kind)
    return make


class test_EventJournalCore:

    def test_append_assigns_monotonic_cursors(self, make_journal):
        journal = make_journal()
        seqs = [journal.append(make_event('task-sent', uuid='t', ts=1.0))
                for _ in range(5)]
        assert seqs == [1, 2, 3, 4, 5]
        assert journal.last_cursor == 5

    def test_append_many(self, make_journal):
        journal = make_journal()
        seqs = journal.append_many(
            [make_event('task-sent', uuid='t', ts=1.0) for _ in range(3)])
        assert seqs == [1, 2, 3]
        assert journal.last_cursor == 3

    def test_entry_fields_indexed(self, make_journal):
        journal = make_journal()
        journal.append(make_event('task-succeeded', uuid='t9',
                                  hostname='w9', ts=123.0, result=1))
        entry = journal.read()[0]
        assert isinstance(entry, JournalEntry)
        assert entry.seq == 1
        assert entry.type == 'task-succeeded'
        assert entry.group == 'task'
        assert entry.task_id == 't9'
        assert entry.worker == 'w9'
        assert entry.timestamp == 123.0
        assert entry.event['result'] == 1

    def test_read_batch_resumes_from_cursor_without_duplicates(
            self, make_journal):
        journal = make_journal()
        for i in range(6):
            journal.append(make_event('task-sent', uuid='t',
                                      ts=float(i)))
        batch = journal.read(limit=2)
        assert [e.seq for e in batch] == [1, 2]
        batch = journal.read(after_cursor=2, limit=2)
        assert [e.seq for e in batch] == [3, 4]
        # resuming from the last returned cursor never yields it again:
        # seq 3/4 are skipped, the unseen 5/6 are returned exactly once
        rest = journal.read(after_cursor=4, limit=10)
        assert [e.seq for e in rest] == [5, 6]
        assert journal.read(after_cursor=6) == []

    def test_replay_pages_through_all_entries(self, make_journal):
        journal = make_journal(page_size=2)
        for i in range(5):
            journal.append(make_event('task-sent', uuid='t',
                                      ts=float(i)))
        seqs = [e.seq for e in journal.replay()]
        assert seqs == [1, 2, 3, 4, 5]

    def test_replay_resumes_and_respects_limit(self, make_journal):
        journal = make_journal(page_size=2)
        for i in range(6):
            journal.append(make_event('task-sent', uuid='t',
                                      ts=float(i)))
        assert [e.seq for e in journal.replay(after_cursor=2, limit=2)] == \
            [3, 4]
        assert [e.seq for e in journal.replay(after_cursor=4)] == [5, 6]

    def test_filter_by_task(self, make_journal):
        journal = make_journal()
        journal.append(make_event('task-sent', uuid='a', ts=1.0))
        journal.append(make_event('task-sent', uuid='b', ts=2.0))
        journal.append(make_event('task-succeeded', uuid='a', ts=3.0))
        journal.append(make_event('worker-online', hostname='w1', ts=4.0))

        assert {e.task_id for e in journal.read(task='a')} == {'a'}
        assert {e.task_id for e in journal.read(task=('a', 'b'))} == \
            {'a', 'b'}
        assert journal.read(task=('nonexistent',)) == []

    def test_filter_by_worker(self, make_journal):
        journal = make_journal()
        journal.append(make_event('worker-heartbeat', hostname='w1',
                                  ts=1.0))
        journal.append(make_event('worker-heartbeat', hostname='w2',
                                  ts=2.0))
        journal.append(make_event('task-received', uuid='t',
                                  hostname='w2', ts=3.0))

        entries = journal.read(worker='w2')
        assert len(entries) == 2
        assert all(e.worker == 'w2' for e in entries)
        assert {e.worker for e in journal.read(worker=('w1', 'w2'))} == \
            {'w1', 'w2'}

    def test_filter_by_event_type_and_group(self, make_journal):
        journal = make_journal()
        journal.append(make_event('task-sent', uuid='t', ts=1.0))
        journal.append(make_event('task-succeeded', uuid='t', ts=2.0))
        journal.append(make_event('task-failed', uuid='t', ts=3.0))
        journal.append(make_event('worker-online', hostname='w', ts=4.0))

        assert [e.type for e in journal.read(type='task-succeeded')] == \
            ['task-succeeded']
        assert {e.type for e in journal.read(
            type=('task-succeeded', 'task-failed'))} == \
            {'task-succeeded', 'task-failed'}
        assert {e.group for e in journal.read(group='worker')} == {'worker'}
        assert len(journal.read(group='task')) == 3

    def test_filter_by_time_range(self, make_journal):
        journal = make_journal()
        for ts in range(100, 600, 100):
            journal.append(make_event('task-sent', uuid='t',
                                      ts=float(ts)))
        assert [e.timestamp for e in journal.read(since=300.0)] == \
            [300.0, 400.0, 500.0]
        assert [e.timestamp for e in journal.read(until=300.0)] == \
            [100.0, 200.0, 300.0]
        assert [e.timestamp for e in journal.read(
            since=200.0, until=400.0)] == [200.0, 300.0, 400.0]

    def test_combined_filters(self, make_journal):
        journal = make_journal()
        journal.append(make_event('task-sent', uuid='a', hostname='w1',
                                  ts=1.0))
        journal.append(make_event('task-succeeded', uuid='a',
                                  hostname='w1', ts=2.0))
        journal.append(make_event('task-succeeded', uuid='b',
                                  hostname='w2', ts=3.0))
        entries = journal.read(task='a', type='task-succeeded',
                               since=2.0)
        assert len(entries) == 1
        assert entries[0].seq == 2

    def test_payload_is_snapshotted_at_append(self, make_journal):
        journal = make_journal()
        event = make_event('task-sent', uuid='t', ts=1.0, name='before')
        journal.append(event)
        event['name'] = 'after'
        event['extra'] = 'mutated'
        stored = journal.read()[0].event
        assert stored['name'] == 'before'
        assert 'extra' not in stored

    def test_decimal_timestamp_is_normalized(self, make_journal):
        journal = make_journal()
        journal.append(make_event('task-sent', uuid='t',
                                  ts=Decimal('123.45')))
        entry = journal.read()[0]
        assert entry.timestamp == pytest.approx(123.45)
        assert isinstance(entry.timestamp, float)

    def test_timestamp_falls_back_to_local_received(self, make_journal):
        journal = make_journal()
        journal.append({'type': 'task-sent', 'uuid': 't',
                        'local_received': 999.0})
        assert journal.read()[0].timestamp == 999.0


class test_JournalPersistence:

    def test_recovers_after_reopen(self, make_journal):
        if make_journal.kind == 'memory':
            pytest.skip('memory backend is not persistent')
        journal = make_journal()
        for event in LIFECYCLE:
            journal.append(dict(event))
        assert journal.last_cursor == 4
        journal.close()

        reopened = make_journal()
        assert reopened.last_cursor == 4
        entries = reopened.read()
        assert [e.type for e in entries] == [
            'task-sent', 'task-received', 'task-started',
            'task-succeeded']
        assert entries[-1].event['result'] == 42
        # cursors keep advancing past the recovered high-water mark
        assert reopened.append(
            make_event('worker-heartbeat', hostname='w1', ts=104.0)) == 5

    def test_replay_from_cursor_across_reopen(self, make_journal):
        if make_journal.kind == 'memory':
            pytest.skip('memory backend is not persistent')
        journal = make_journal()
        for event in LIFECYCLE:
            journal.append(dict(event))
        journal.close()

        reopened = make_journal()
        assert [e.seq for e in reopened.read(after_cursor=2)] == [3, 4]

    def test_context_manager_closes_storage(self, make_journal):
        if make_journal.kind == 'memory':
            pytest.skip('memory backend is not persistent')
        with make_journal() as journal:
            journal.append(make_event('task-sent', uuid='t', ts=1.0))
        # storage closed -> reopening still recovers the data
        reopened = make_journal()
        assert reopened.last_cursor == 1

    def test_file_partial_tail_line_is_truncated(self, tmp_path):
        path = str(tmp_path / 'events.log')
        journal = EventJournal.file(path)
        journal.append(make_event('task-sent', uuid='t1', ts=1.0))
        journal.append(make_event('task-sent', uuid='t2', ts=2.0))
        journal.close()
        # simulate a crash mid-write: append a partial JSON line
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write('{"s":3,"t":3.0,"y":"task-sen')

        reopened = EventJournal.file(path)
        entries = reopened.read()
        assert [e.task_id for e in entries] == ['t1', 't2']
        # cursor continues past the partial record's seq
        assert reopened.append(
            make_event('task-sent', uuid='t3', ts=3.0)) == 3


class test_JournalRetention:

    def test_prune_by_cursor_keeps_monotonic_cursors(self, make_journal):
        journal = make_journal()
        for i in range(5):
            journal.append(make_event('task-sent', uuid='t',
                                      ts=float(i)))
        removed = journal.prune(before_cursor=3)
        assert removed == 2
        assert [e.seq for e in journal.read()] == [3, 4, 5]
        # cursors are never reused after pruning
        assert journal.append(
            make_event('task-sent', uuid='t', ts=5.0)) == 6
        assert [e.seq for e in journal.read()] == [3, 4, 5, 6]

    def test_prune_by_time(self, make_journal):
        journal = make_journal()
        for ts in (100.0, 200.0, 300.0, 400.0):
            journal.append(make_event('task-sent', uuid='t', ts=ts))
        removed = journal.prune(before_time=300.0)
        assert removed == 2
        assert [e.timestamp for e in journal.read()] == [300.0, 400.0]

    def test_prune_both_boundaries_intersect(self, make_journal):
        journal = make_journal()
        for ts in (100.0, 200.0, 300.0, 400.0):
            journal.append(make_event('task-sent', uuid='t', ts=ts))
        # an event must be older than BOTH boundaries (seq<2 AND ts<300),
        # so seq 2 (ts=200) is kept even though it is older than 300
        removed = journal.prune(before_cursor=2, before_time=300.0)
        assert removed == 1
        assert [e.seq for e in journal.read()] == [2, 3, 4]

    def test_prune_without_boundary_is_noop(self, make_journal):
        journal = make_journal()
        for i in range(3):
            journal.append(make_event('task-sent', uuid='t',
                                      ts=float(i)))
        assert journal.prune() == 0
        assert journal.last_cursor == 3

    def test_prune_all_then_append_keeps_monotonic_cursor(
            self, make_journal):
        journal = make_journal()
        for i in range(3):
            journal.append(make_event('task-sent', uuid='t',
                                      ts=float(i)))
        assert journal.prune(before_cursor=10) == 3
        assert journal.read() == []
        assert journal.append(
            make_event('task-sent', uuid='t', ts=9.0)) == 4
        # a client resuming from the old cursor must not silently skip
        # the new event once it replays from a pruned position
        assert [e.seq for e in journal.read(after_cursor=0)] == [4]

    def test_retain_max_entries(self, make_journal):
        journal = make_journal()
        for i in range(10):
            journal.append(make_event('task-sent', uuid='t',
                                      ts=100.0 + i))
        removed = journal.retain(max_entries=3)
        assert removed == 7
        assert [e.seq for e in journal.read()] == [8, 9, 10]

    def test_retain_max_age(self, make_journal):
        journal = make_journal()
        for ts in (100.0, 200.0, 300.0, 400.0):
            journal.append(make_event('task-sent', uuid='t', ts=ts))
        removed = journal.retain(max_age=150.0, now=400.0)
        assert removed == 2
        assert [e.timestamp for e in journal.read()] == [300.0, 400.0]

    def test_compact_preserves_entries_and_advances_cursor(
            self, make_journal):
        journal = make_journal()
        for i in range(6):
            journal.append(make_event('task-sent', uuid='t',
                                      ts=float(i)))
        journal.prune(before_cursor=4)
        stats = journal.compact()
        assert 'reclaimed' in stats
        entries = journal.read()
        assert [e.seq for e in entries] == [4, 5, 6]
        assert journal.append(
            make_event('task-sent', uuid='t', ts=6.0)) == 7

    def test_compact_on_disk_after_reopen(self, make_journal):
        if make_journal.kind == 'memory':
            pytest.skip('memory backend is not persistent')
        journal = make_journal()
        for i in range(4):
            journal.append(make_event('task-sent', uuid='t',
                                      ts=float(i)))
        journal.prune(before_cursor=3)
        journal.compact()
        journal.close()

        reopened = make_journal()
        assert [e.seq for e in reopened.read()] == [3, 4]
        assert reopened.last_cursor == 4


class test_PluggableStorage:

    def test_custom_storage_backend(self):
        class SpyStorage(JournalStorage):
            def __init__(self):
                self.appended = 0
                self.closed = False

            def append(self, records):
                seqs = list(range(self.appended + 1,
                                  self.appended + 1 + len(records)))
                self.appended += len(records)
                return seqs

            def read(self, after_cursor, limit, filters):
                return []

            def last_cursor(self):
                return self.appended

            def prune(self, before_cursor=None, before_time=None):
                return 0

            def compact(self):
                return {'reclaimed': 0}

            def close(self):
                self.closed = True

        storage = SpyStorage()
        journal = EventJournal(storage=storage)
        journal.append(make_event('task-sent', uuid='t', ts=1.0))
        assert storage.appended == 1
        journal.close()
        assert storage.closed

    def test_default_storage_is_memory(self):
        journal = EventJournal()
        assert isinstance(journal.storage, MemoryJournalStorage)
        journal.close()


class test_ReceiverIntegration:

    def _receiver(self, app, journal=None, handlers=None):
        connection = Mock()
        connection.transport_cls = 'memory'
        return app.events.Receiver(
            connection,
            handlers=handlers if handlers is not None else {},
            node_id='celery.tests',
            journal=journal,
        )

    def test_default_receiver_has_no_journal(self, app):
        r = self._receiver(app)
        assert r.journal is None
        assert r.catchup(42) == 42

    def test_live_events_are_journaled_and_dispatched(self, app):
        journal = EventJournal.memory()
        seen = []
        r = self._receiver(app, journal=journal,
                           handlers={'*': seen.append})
        r._receive(make_event('task-sent', uuid='t1', ts=1.0), object())
        assert len(seen) == 1
        assert journal.last_cursor == 1
        entry = journal.read()[0]
        assert entry.type == 'task-sent'
        assert entry.task_id == 't1'
        # receiver normalizes the event (local_received added)
        assert 'local_received' in entry.event
        journal.close()

    def test_batched_messages_are_journaled(self, app):
        journal = EventJournal.memory()
        r = self._receiver(app, journal=journal)
        r._receive([
            make_event('task-sent', uuid='t1', ts=1.0),
            make_event('task-sent', uuid='t2', ts=2.0),
        ], Mock())
        assert journal.last_cursor == 2
        assert [e.task_id for e in journal.read()] == ['t1', 't2']
        journal.close()

    def test_catchup_replays_history_to_handlers(self, app):
        journal = EventJournal.memory()
        for event in LIFECYCLE:
            journal.append(dict(event))
        dispatched = []
        r = self._receiver(app, journal=journal,
                           handlers={'*': dispatched.append})
        last = r.catchup(0)
        assert last == 4
        assert [e['type'] for e in dispatched] == [
            'task-sent', 'task-received', 'task-started',
            'task-succeeded']
        # resuming from the returned cursor dispatches nothing again
        assert r.catchup(last) == 4
        assert len(dispatched) == 4
        journal.close()

    def test_catchup_resume_after_live_event_has_no_duplicates(self, app):
        journal = EventJournal.memory()
        for event in LIFECYCLE:
            journal.append(dict(event))
        dispatched = []
        r = self._receiver(app, journal=journal,
                           handlers={'*': dispatched.append})
        # consumer had processed up to cursor 3 before reconnecting
        assert r.catchup(3) == 4
        assert [e['type'] for e in dispatched] == ['task-succeeded']
        # a live event arriving afterwards gets seq 5
        r._receive(make_event('worker-heartbeat', hostname='w1',
                              ts=104.0), object())
        assert journal.last_cursor == 5
        assert len(dispatched) == 2
        # resume from cursor 4: only the heartbeat is delivered, once
        dispatched.clear()
        assert r.catchup(4) == 5
        assert [e['type'] for e in dispatched] == ['worker-heartbeat']
        assert r.catchup(5) == 5
        assert len(dispatched) == 1
        journal.close()

    def test_catchup_supports_filters(self, app):
        journal = EventJournal.memory()
        for event in LIFECYCLE:
            journal.append(dict(event))
        journal.append(make_event('worker-online', hostname='w1',
                                  ts=104.0))
        dispatched = []
        r = self._receiver(app, journal=journal,
                           handlers={'*': dispatched.append})
        last = r.catchup(0, group='worker')
        assert last == 5
        assert [e['type'] for e in dispatched] == ['worker-online']
        journal.close()

    def test_catchup_limit(self, app):
        journal = EventJournal.memory()
        for event in LIFECYCLE:
            journal.append(dict(event))
        dispatched = []
        r = self._receiver(app, journal=journal,
                           handlers={'*': dispatched.append})
        last = r.catchup(0, limit=2)
        assert last == 2
        assert len(dispatched) == 2
        journal.close()

    def test_state_aggregation_rebuilds_after_restart(self, app, tmp_path):
        """The headline use case: task lifecycle survives process restart."""
        db_path = str(tmp_path / 'state.db')

        def fresh_pipeline():
            journal = EventJournal.sqlite(db_path)
            state = app.events.State()
            connection = Mock()
            connection.transport_cls = 'memory'
            receiver = app.events.Receiver(
                connection, handlers={'*': state.event},
                node_id='celery.tests', journal=journal)
            return journal, state, receiver

        # first "process": events arrive live
        journal, state, receiver = fresh_pipeline()
        for event in LIFECYCLE:
            receiver._receive(dict(event), object())
        assert state.tasks['t1'].state == states.SUCCESS
        assert state.tasks['t1'].result == 42
        journal.close()

        # second "process": empty in-memory state, same durable journal
        journal2, state2, receiver2 = fresh_pipeline()
        assert len(state2.tasks) == 0
        last = receiver2.catchup(0)
        assert last == len(LIFECYCLE)
        task = state2.tasks['t1']
        assert task.state == states.SUCCESS
        assert task.result == 42
        assert task.worker.hostname == 'w1'
        journal2.close()

    def test_app_events_journal_factory(self, app):
        journal = app.events.Journal(storage=MemoryJournalStorage())
        assert isinstance(journal, EventJournal)
        assert journal.app is app
        journal.append(make_event('task-sent', uuid='t', ts=1.0))
        assert journal.last_cursor == 1
        journal.close()


class test_JournalThreadSafety:

    def test_concurrent_appends_get_unique_cursors(self, make_journal):
        journal = make_journal()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    journal.append(make_event('task-sent', uuid='t',
                                              ts=1.0))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        entries = journal.read(limit=1000)
        seqs = [e.seq for e in entries]
        assert len(seqs) == 200
        assert len(set(seqs)) == 200
        assert seqs == sorted(seqs)
