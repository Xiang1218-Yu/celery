"""Persistent, pluggable task event journal.

The broker event stream (:mod:`celery.events.receiver`) only delivers
events to consumers that are currently connected: whenever the monitoring
process restarts or the network drops, the events that were emitted in the
meantime are lost, and there is no way to reconcile live events with
historical queries.

This module provides a durable append-only event log that solves that:

* Every received event is appended under a **monotonic cursor** (``seq``,
  starting at ``1``). Cursors are never reused, not even after pruning,
  compaction or a process restart.
* Events can be **replayed in batches from any cursor**
  (``seq > after_cursor``), so a consumer resuming after a reconnect never
  sees the same event twice and never misses one either.
* Reads can be filtered by **task** (``uuid``), **worker**
  (``hostname``), **event type**/group and a **time range**.
* The log survives process restarts; storage backends are pluggable via
  the :class:`JournalStorage` interface.  Two durable backends ship in the
  box: :class:`FileJournalStorage` (append-only JSON-lines) and
  :class:`SqliteJournalStorage` (indexed SQLite database), plus
  :class:`MemoryJournalStorage` for tests and ephemeral use.
* Bounded :meth:`EventJournal.prune` (retention) and
  :meth:`EventJournal.compact` operations reclaim space without ever
  crossing the caller-supplied boundary.

Typical usage::

    journal = EventJournal.sqlite('/var/lib/celery/events.db')
    receiver = app.events.Receiver(connection, handlers={'*': on_event},
                                   journal=journal)
    # ...after a restart/reconnect, rebuild what was missed:
    last_seen = receiver.catchup(last_seen)        # replays to handlers
    # historical queries share the same log:
    entries = journal.read(task=task_id, since=start, until=end)
"""
import json
import os
import sqlite3
import tempfile
import threading
import time
from collections import namedtuple

from celery.utils.log import get_logger

from .event import group_from

__all__ = (
    'JournalEntry', 'EventJournal',
    'JournalStorage', 'MemoryJournalStorage',
    'FileJournalStorage', 'SqliteJournalStorage',
)

logger = get_logger(__name__)

#: A single event as read back from the journal.
JournalEntry = namedtuple('JournalEntry', (
    'seq',        # monotonic cursor (int, >= 1)
    'timestamp',  # event timestamp (float, seconds since epoch)
    'type',       # event type, e.g. ``task-succeeded``
    'group',      # event group, e.g. ``task`` / ``worker``
    'task_id',    # task uuid (``uuid`` field) or ``None``
    'worker',     # worker hostname (``hostname`` field) or ``None``
    'event',      # the original event payload (dict)
))

#: Normalized record used internally between journal and storage backends.
_Record = namedtuple('_Record', (
    'seq', 'timestamp', 'type', 'group', 'task_id', 'worker', 'payload',
))


def _as_set(value):
    """Normalize a filter value into a set of accepted values."""
    if isinstance(value, str):
        return {value}
    return set(value)


def _matches(rec, task=None, worker=None, type=None, group=None,
             since=None, until=None):
    """Return ``True`` if a record passes all supplied filters."""
    if task is not None and rec.task_id not in _as_set(task):
        return False
    if worker is not None and (
            rec.worker is None or rec.worker not in _as_set(worker)):
        return False
    if type is not None and rec.type not in _as_set(type):
        return False
    if group is not None and rec.group not in _as_set(group):
        return False
    if since is not None and rec.timestamp < since:
        return False
    if until is not None and rec.timestamp > until:
        return False
    return True


def _prunable(rec, before_cursor, before_time):
    """Return ``True`` if a record falls outside the retention boundary.

    When both boundaries are supplied the record must be older than
    *both* (intersection), so retention never deletes more than asked.
    """
    if before_cursor is None and before_time is None:
        return False
    if before_cursor is not None and rec.seq >= before_cursor:
        return False
    if before_time is not None and rec.timestamp >= before_time:
        return False
    return True


class JournalStorage:
    """Abstract storage backend for the event journal.

    Implementations must guarantee that assigned cursors (``seq``) are
    strictly monotonic and **never reused**, even after
    :meth:`prune`/:meth:`compact` or a process restart.

    Records handed to :meth:`append` are dicts with the keys
    ``timestamp`` (float), ``type`` (str), ``group`` (str),
    ``task_id`` (str|None), ``worker`` (str|None) and ``payload``
    (a JSON-encoded string of the original event).
    """

    def open(self):
        """Open/recover the storage. Returns ``self``."""
        return self

    def close(self):
        """Close the storage, making sure all data is durable."""

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def append(self, records):
        """Append a batch of records, returning the list of assigned seqs."""
        raise NotImplementedError()

    def read(self, after_cursor, limit, filters):
        """Return records with ``seq > after_cursor`` matching *filters*.

        The result is ordered by ``seq`` ascending and capped at *limit*
        when it is not ``None``.
        """
        raise NotImplementedError()

    def last_cursor(self):
        """Return the highest cursor ever assigned (``0`` if empty)."""
        raise NotImplementedError()

    def prune(self, before_cursor=None, before_time=None):
        """Delete records older than the supplied boundaries.

        Returns the number of deleted records. With both arguments the
        record must be older than both boundaries.
        """
        raise NotImplementedError()

    def compact(self):
        """Reclaim space. Returns a dict of implementation-defined stats."""
        raise NotImplementedError()


class MemoryJournalStorage(JournalStorage):
    """Non-persistent in-memory storage (useful for tests)."""

    def __init__(self):
        self._records = []
        self._next_seq = 1
        self._lock = threading.Lock()

    def append(self, records):
        seqs = []
        with self._lock:
            for rec in records:
                seq = self._next_seq
                self._next_seq += 1
                self._records.append(_Record(
                    seq, rec['timestamp'], rec['type'], rec['group'],
                    rec['task_id'], rec['worker'], rec['payload']))
                seqs.append(seq)
        return seqs

    def read(self, after_cursor, limit, filters):
        out = []
        with self._lock:
            for rec in self._records:
                if rec.seq <= after_cursor:
                    continue
                if not _matches(rec, **filters):
                    continue
                out.append(rec)
                if limit is not None and len(out) >= limit:
                    break
        return out

    def last_cursor(self):
        with self._lock:
            return self._next_seq - 1

    def prune(self, before_cursor=None, before_time=None):
        if before_cursor is None and before_time is None:
            return 0
        with self._lock:
            kept = [r for r in self._records
                    if not _prunable(r, before_cursor, before_time)]
            removed = len(self._records) - len(kept)
            self._records = kept
        return removed

    def compact(self):
        with self._lock:
            return {'records': len(self._records), 'reclaimed': 0}


class FileJournalStorage(JournalStorage):
    """Append-only JSON-lines file storage.

    Each line of *path* is a JSON object containing the cursor, indexed
    fields and the full event payload.  A sidecar meta file
    (``<path>.meta``) remembers the high-water cursor so that cursors
    keep advancing even after the log has been pruned/compacted down to
    zero records.

    Writes are flushed (and, with ``sync=True``, ``fsync``-ed) on every
    append; compaction/prune rewrite the file atomically via
    temp-file + ``os.replace``. A partially written tail line left by a
    crash is detected and truncated on :meth:`open`.
    """

    def __init__(self, path, sync=True):
        self.path = os.path.abspath(path)
        self.meta_path = self.path + '.meta'
        self.sync = sync
        self._lock = threading.Lock()
        self._fh = None
        self._next_seq = 1

    def open(self):
        with self._lock:
            max_seq, valid_offset = 0, 0
            if os.path.exists(self.path):
                with open(self.path, 'r', encoding='utf-8') as fh:
                    # readline() (unlike file iteration) allows tell(), used
                    # to truncate a partial tail left by a crash.
                    while True:
                        line = fh.readline()
                        if not line:
                            break
                        stripped = line.strip()
                        if not stripped:
                            valid_offset = fh.tell()
                            continue
                        try:
                            obj = json.loads(stripped)
                            seq = obj['s']
                        except (ValueError, KeyError, TypeError):
                            # Partial tail from a crash mid-write: stop and
                            # truncate everything from this offset below.
                            break
                        max_seq = max(max_seq, seq)
                        valid_offset = fh.tell()
                size = os.path.getsize(self.path)
                if valid_offset < size:
                    with open(self.path, 'r+', encoding='utf-8') as fh:
                        fh.truncate(valid_offset)
            meta_next = self._read_meta().get('next_seq', 0)
            self._next_seq = max(max_seq + 1, meta_next)
            self._fh = open(self.path, 'a', encoding='utf-8')
            return self

    def close(self):
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                    if self.sync:
                        os.fsync(self._fh.fileno())
                except OSError:
                    pass
                try:
                    self._fh.close()
                except OSError:
                    pass
                self._fh = None
            self._write_meta()

    def append(self, records):
        if self._fh is None:
            raise RuntimeError('Journal storage is not open')
        seqs, lines = [], []
        with self._lock:
            for rec in records:
                seq = self._next_seq
                self._next_seq += 1
                seqs.append(seq)
                lines.append(
                    '{"s":%d,"t":%s,"y":%s,"g":%s,"u":%s,"h":%s,"d":%s}' % (
                        seq,
                        json.dumps(rec['timestamp']),
                        json.dumps(rec['type']),
                        json.dumps(rec['group']),
                        json.dumps(rec['task_id']),
                        json.dumps(rec['worker']),
                        rec['payload'] if rec['payload'] is not None
                        else 'null'))
            self._fh.write('\n'.join(lines) + '\n')
            self._fh.flush()
            if self.sync:
                os.fsync(self._fh.fileno())
        return seqs

    def _scan(self):
        if self._fh is not None:
            self._fh.flush()
        if not os.path.exists(self.path):
            return []
        records = []
        with open(self.path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append(_Record(
                        obj['s'], obj['t'], obj['y'], obj.get('g'),
                        obj.get('u'), obj.get('h'),
                        json.dumps(obj.get('d'), default=str)))
                except (ValueError, KeyError, TypeError):
                    break
        return records

    def read(self, after_cursor, limit, filters):
        with self._lock:
            out = []
            for rec in self._scan():
                if rec.seq <= after_cursor:
                    continue
                if not _matches(rec, **filters):
                    continue
                out.append(rec)
                if limit is not None and len(out) >= limit:
                    break
            return out

    def last_cursor(self):
        with self._lock:
            return self._next_seq - 1

    def prune(self, before_cursor=None, before_time=None):
        if before_cursor is None and before_time is None:
            return 0
        with self._lock:
            kept, removed = [], 0
            for rec in self._scan():
                if _prunable(rec, before_cursor, before_time):
                    removed += 1
                else:
                    kept.append(rec)
            if removed:
                self._rewrite(kept)
            return removed

    def compact(self):
        with self._lock:
            records = self._scan()
            size_before = (os.path.getsize(self.path)
                           if os.path.exists(self.path) else 0)
            self._rewrite(records)
            size_after = (os.path.getsize(self.path)
                          if os.path.exists(self.path) else 0)
            return {
                'records': len(records),
                'bytes_before': size_before,
                'bytes_after': size_after,
                'reclaimed': size_before - size_after,
            }

    def _rewrite(self, records):
        directory = os.path.dirname(self.path) or '.'
        fd, tmp = tempfile.mkstemp(dir=directory, prefix='.journal-',
                                   suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                for rec in records:
                    fh.write(
                        '{"s":%d,"t":%s,"y":%s,"g":%s,"u":%s,"h":%s,'
                        '"d":%s}\n' % (
                            rec.seq,
                            json.dumps(rec.timestamp),
                            json.dumps(rec.type),
                            json.dumps(rec.group),
                            json.dumps(rec.task_id),
                            json.dumps(rec.worker),
                            rec.payload if rec.payload is not None
                            else 'null'))
                fh.flush()
                os.fsync(fh.fileno())
            # Persist the high-water mark before swapping the data file in,
            # so cursors are never reused even if all records were pruned.
            self._write_meta()
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
        self._fh = open(self.path, 'a', encoding='utf-8')

    def _read_meta(self):
        try:
            with open(self.meta_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_meta(self):
        directory = os.path.dirname(self.meta_path) or '.'
        fd, tmp = tempfile.mkstemp(dir=directory, prefix='.journal-meta-',
                                   suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                json.dump({'version': 1, 'next_seq': self._next_seq}, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.meta_path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


class SqliteJournalStorage(JournalStorage):
    """Indexed SQLite storage.

    Uses an ``AUTOINCREMENT`` primary key so cursors are monotonic and
    never reused after pruning. Columns for task uuid, worker hostname,
    event type/group and timestamp are indexed for efficient filtering.
    """

    TABLE = 'celery_event_journal'

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS {table} (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,
        type TEXT NOT NULL,
        grp TEXT,
        task_id TEXT,
        worker TEXT,
        payload TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS {table}_task   ON {table} (task_id);
    CREATE INDEX IF NOT EXISTS {table}_worker ON {table} (worker);
    CREATE INDEX IF NOT EXISTS {table}_type   ON {table} (type);
    CREATE INDEX IF NOT EXISTS {table}_grp    ON {table} (grp);
    CREATE INDEX IF NOT EXISTS {table}_ts     ON {table} (ts);
    """

    def __init__(self, path, timeout=30.0):
        self.path = path
        self.timeout = timeout
        self._lock = threading.RLock()
        self._conn = None

    def open(self):
        with self._lock:
            self._conn = sqlite3.connect(
                self.path, timeout=self.timeout, check_same_thread=False)
            self._conn.executescript(self._SCHEMA.format(table=self.TABLE))
            self._conn.commit()
            return self

    def close(self):
        with self._lock:
            if self._conn is not None:
                self._conn.commit()
                self._conn.close()
                self._conn = None

    def append(self, records):
        seqs = []
        with self._lock:
            cur = self._conn.cursor()
            for rec in records:
                cur.execute(
                    'INSERT INTO {0} (ts, type, grp, task_id, worker, '
                    'payload) VALUES (?, ?, ?, ?, ?, ?)'.format(self.TABLE),
                    (rec['timestamp'], rec['type'], rec['group'],
                     rec['task_id'], rec['worker'], rec['payload']))
                seqs.append(cur.lastrowid)
            self._conn.commit()
        return seqs

    @staticmethod
    def _add_in(clause, params, column, value):
        if value is None:
            return
        if isinstance(value, str):
            clause.append('{0} = ?'.format(column))
            params.append(value)
        else:
            values = list(value)
            if not values:
                # An empty allow-list matches nothing.
                clause.append('0')
                return
            clause.append('{0} IN ({1})'.format(
                column, ', '.join('?' for _ in values)))
            params.extend(values)

    def read(self, after_cursor, limit, filters):
        clause = ['seq > ?']
        params = [after_cursor]
        self._add_in(clause, params, 'task_id', filters.get('task'))
        self._add_in(clause, params, 'worker', filters.get('worker'))
        self._add_in(clause, params, 'type', filters.get('type'))
        self._add_in(clause, params, 'grp', filters.get('group'))
        if filters.get('since') is not None:
            clause.append('ts >= ?')
            params.append(filters['since'])
        if filters.get('until') is not None:
            clause.append('ts <= ?')
            params.append(filters['until'])
        sql = ('SELECT seq, ts, type, grp, task_id, worker, payload '
               'FROM {0} WHERE {1} ORDER BY seq'.format(
                   self.TABLE, ' AND '.join(clause)))
        if limit is not None:
            sql += ' LIMIT ?'
            params.append(int(limit))
        with self._lock:
            return [_Record(*row) for row in self._conn.execute(sql, params)]

    def last_cursor(self):
        with self._lock:
            row = self._conn.execute(
                'SELECT seq FROM sqlite_sequence WHERE name = ?',
                (self.TABLE,)).fetchone()
            return row[0] if row else 0

    def prune(self, before_cursor=None, before_time=None):
        if before_cursor is None and before_time is None:
            return 0
        clause, params = [], []
        if before_cursor is not None:
            clause.append('seq < ?')
            params.append(before_cursor)
        if before_time is not None:
            clause.append('ts < ?')
            params.append(before_time)
        with self._lock:
            cur = self._conn.execute(
                'DELETE FROM {0} WHERE {1}'.format(
                    self.TABLE, ' AND '.join(clause)), params)
            self._conn.commit()
            return cur.rowcount

    def compact(self):
        with self._lock:
            self._conn.commit()
            page_size = self._conn.execute(
                'PRAGMA page_size').fetchone()[0]
            before = self._conn.execute('PRAGMA page_count').fetchone()[0]
            self._conn.execute('VACUUM')
            after = self._conn.execute('PRAGMA page_count').fetchone()[0]
            return {
                'pages_before': before,
                'pages_after': after,
                'bytes_before': before * page_size,
                'bytes_after': after * page_size,
                'reclaimed': (before - after) * page_size,
            }


class EventJournal:
    """Durable append-only task event log.

    Arguments:
        storage (JournalStorage): Backend to use. Defaults to an in-memory
            backend; use :meth:`file` / :meth:`sqlite` classmethods for a
            persistent journal, or pass any custom
            :class:`JournalStorage` implementation.
        page_size (int): Batch size used when paging through the backend
            in :meth:`replay`.

    Cursors are opaque monotonically increasing integers. Reads always
    use strict ``seq > after_cursor`` semantics, so resuming from the
    last processed cursor can never return the same event twice.
    """

    #: App-compatible default app (set by ``app.subclass_with_self``).
    app = None

    def __init__(self, storage=None, app=None, page_size=1000):
        self.app = app or self.app
        self.page_size = page_size
        self._storage = storage if storage is not None \
            else MemoryJournalStorage()
        self._lock = threading.RLock()
        self._closed = False
        self._storage.open()

    @classmethod
    def memory(cls, **kwargs):
        """Ephemeral in-memory journal."""
        return cls(storage=MemoryJournalStorage(), **kwargs)

    @classmethod
    def file(cls, path, sync=True, **kwargs):
        """Persistent JSON-lines file journal at *path*."""
        return cls(storage=FileJournalStorage(path, sync=sync), **kwargs)

    @classmethod
    def sqlite(cls, path, **kwargs):
        """Persistent SQLite journal at *path* (``':memory:'`` allowed)."""
        return cls(storage=SqliteJournalStorage(path), **kwargs)

    def append(self, event):
        """Append one event dict, returning its assigned cursor."""
        return self.append_many([event])[0]

    def append_many(self, events):
        """Append a batch of event dicts, returning their cursors."""
        records = [self._prepare(event) for event in events]
        with self._lock:
            return self._storage.append(records)

    @staticmethod
    def _prepare(event):
        ev_type = event['type']
        timestamp = event.get('timestamp')
        if timestamp is None:
            timestamp = event.get('local_received')
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError):
            timestamp = time.time()
        # Snapshot the payload at append time: handlers may mutate the
        # event dict afterwards, and some JSON libraries hand us Decimals.
        payload = json.dumps(event, default=str, ensure_ascii=False)
        return {
            'timestamp': timestamp,
            'type': ev_type,
            'group': group_from(ev_type),
            'task_id': event.get('uuid'),
            'worker': event.get('hostname'),
            'payload': payload,
        }

    @property
    def last_cursor(self):
        """Highest cursor ever assigned (``0`` for an empty journal)."""
        with self._lock:
            return self._storage.last_cursor()

    @property
    def storage(self):
        return self._storage

    def read(self, after_cursor=0, limit=None, task=None, worker=None,
             type=None, group=None, since=None, until=None):
        """Read a batch of entries from the journal.

        Only entries with ``seq > after_cursor`` are returned, ordered by
        cursor. Filters:

        * ``task``: task uuid (str or iterable of uuids),
        * ``worker``: worker hostname (str or iterable),
        * ``type``: exact event type, e.g. ``'task-succeeded'``
          (str or iterable),
        * ``group``: event group, e.g. ``'task'`` (str or iterable),
        * ``since``/``until``: inclusive timestamp range (seconds since
          epoch).
        """
        filters = {
            'task': task, 'worker': worker, 'type': type, 'group': group,
            'since': since, 'until': until,
        }
        with self._lock:
            records = self._storage.read(
                int(after_cursor or 0), limit, filters)
        return [self._to_entry(rec) for rec in records]

    def replay(self, after_cursor=0, limit=None, page_size=None, **filters):
        """Yield entries from *after_cursor*, paging through storage.

        Same filters as :meth:`read`. The generator stops once the live
        end of the journal is reached; call it again with the last seen
        cursor after a reconnect to resume without gaps or duplicates.
        """
        page_size = page_size or self.page_size
        remaining = limit
        cursor = int(after_cursor or 0)
        while True:
            batch_limit = page_size if remaining is None \
                else min(page_size, remaining)
            entries = self.read(cursor, limit=batch_limit, **filters)
            if not entries:
                return
            for entry in entries:
                yield entry
                cursor = entry.seq
            if remaining is not None:
                remaining -= len(entries)
                if remaining <= 0:
                    return

    def prune(self, before_cursor=None, before_time=None):
        """Delete entries older than the supplied (inclusive-safe) boundary.

        ``before_cursor=N`` drops entries with ``seq < N``;
        ``before_time=T`` drops entries with ``timestamp < T``. When both
        are given an entry must be older than both. Returns the number of
        deleted entries.
        """
        with self._lock:
            return self._storage.prune(before_cursor=before_cursor,
                                       before_time=before_time)

    def retain(self, max_entries=None, max_age=None, now=None):
        """Apply a retention policy, returning the number dropped entries.

        Keeps at most *max_entries* newest events and/or events no older
        than *max_age* seconds. Boundaries are derived and then passed to
        :meth:`prune`, so deletion stays bounded.
        """
        before_cursor = None
        before_time = None
        if max_entries is not None:
            with self._lock:
                last = self._storage.last_cursor()
            if last > max_entries:
                before_cursor = last - max_entries + 1
        if max_age is not None:
            before_time = (now if now is not None else time.time()) - max_age
        if before_cursor is None and before_time is None:
            return 0
        return self.prune(before_cursor=before_cursor,
                          before_time=before_time)

    def compact(self):
        """Reclaim storage space (VACUUM / atomic rewrite)."""
        with self._lock:
            return self._storage.compact()

    @staticmethod
    def _to_entry(rec):
        try:
            event = json.loads(rec.payload)
        except (TypeError, ValueError):
            event = None
        if not isinstance(event, dict):
            event = {}
        return JournalEntry(
            rec.seq, rec.timestamp, rec.type, rec.group,
            rec.task_id, rec.worker, event)

    def close(self):
        with self._lock:
            if not self._closed:
                self._storage.close()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
