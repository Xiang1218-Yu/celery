"""Capability-based routing.

Workers in a heterogeneous cluster may declare a set of *capability*
tags describing what kind of tasks they are able to execute (e.g.
``"gpu"``, ``"ffmpeg"``).  Task messages can in turn carry a set of
required capability tags, and publishing is refused when none of the
currently online workers declares every required tag.

This module holds:

* :func:`normalize_capabilities` / :func:`satisfies` - small helpers
  used both on the worker and publisher side;
* :class:`CapabilityRegistry` - an in-memory view of the capabilities
  of the online workers.  It is fed by remote control inspect replies
  (refreshed on demand with a short TTL) and, when running inside a
  worker, by the heartbeats/gossip stream so that the view stays up to
  date without extra broker round-trips.
"""
import threading
from time import monotonic

from celery.exceptions import NoQualifiedWorkerError
from celery.utils.log import get_logger

__all__ = (
    'CapabilityRegistry', 'normalize_capabilities', 'satisfies',
)

logger = get_logger(__name__)


def normalize_capabilities(capabilities):
    """Normalize a capability declaration to a sorted tuple of strings.

    Accepts ``None``/empty values (no capabilities), a single string or
    any iterable of strings.  Duplicates and falsy entries are removed
    and the result is sorted so that the wire representation and
    routing outcomes are stable.
    """
    if capabilities is None:
        return ()
    if isinstance(capabilities, str):
        capabilities = (capabilities,)
    return tuple(sorted({
        str(capability)
        for capability in capabilities
        if capability
    }))


def satisfies(worker_capabilities, required_capabilities):
    """Return True if a worker satisfies all required capabilities."""
    required = set(required_capabilities or ())
    if not required:
        return True
    return required.issubset(set(worker_capabilities or ()))


class CapabilityRegistry:
    """In-memory view of online workers' declared capabilities.

    The view maps worker hostnames to a :class:`frozenset` of declared
    capability tags.  It can be populated two ways:

    * :meth:`refresh` broadcasts the ``capabilities`` inspect command to
      the cluster and merges the replies (cached for
      :setting:`task_capability_state_ttl` seconds);
    * :meth:`remember` / :meth:`forget` are used by the worker's gossip
      consumer whenever a heartbeat arrives or a worker goes away.
    """

    def __init__(self, app):
        self.app = app
        self._lock = threading.Lock()
        self._workers = {}
        self._last_refresh = 0.0

    def remember(self, hostname, capabilities):
        """Record the capability set advertised by a worker."""
        caps = frozenset(normalize_capabilities(capabilities))
        if hostname:
            with self._lock:
                self._workers[hostname] = caps
        return caps

    def forget(self, hostname):
        """Remove a worker from the view (e.g. on heartbeat loss)."""
        with self._lock:
            self._workers.pop(hostname, None)

    def clear(self):
        """Forget all workers."""
        with self._lock:
            self._workers.clear()
            self._last_refresh = 0.0

    @property
    def workers(self):
        """Snapshot mapping hostname -> frozenset of capabilities."""
        with self._lock:
            return dict(self._workers)

    @property
    def last_refresh(self):
        """``time.monotonic()`` of the last inspect refresh."""
        return self._last_refresh

    def refresh(self, timeout=None, force=False):
        """Refresh the view by querying online workers.

        Returns the current snapshot.  A refresh is skipped while the
        cached view is younger than :setting:`task_capability_state_ttl`
        seconds unless ``force`` is set.
        """
        now = monotonic()
        ttl = float(self.app.conf.task_capability_state_ttl or 0.0)
        with self._lock:
            fresh = bool(self._last_refresh) and (now - self._last_refresh) < ttl
        if fresh and not force:
            return self.workers

        if timeout is None:
            timeout = self.app.conf.task_capability_inspect_timeout
        replies = None
        try:
            replies = self.app.control.inspect(timeout=timeout).capabilities()
        except Exception as exc:  # pylint: disable=broad-except
            # Control is not available on every transport (e.g. SQS);
            # callers decide what to do with an empty view.  Do not let
            # a control-plane failure break task publishing entirely.
            logger.warning(
                'Capability registry refresh failed: %r', exc)
        if replies:
            for hostname, reply in replies.items():
                self.remember(hostname, self._extract_capabilities(reply))
        with self._lock:
            self._last_refresh = monotonic()
        return self.workers

    @staticmethod
    def _extract_capabilities(reply):
        if isinstance(reply, dict):
            if 'ok' in reply:
                return reply['ok']
            if 'capabilities' in reply:
                return reply['capabilities']
            return ()
        if isinstance(reply, (list, tuple, set, frozenset)):
            return reply
        return ()

    def qualified_workers(self, required, refresh=True):
        """Return the hostnames of workers satisfying ``required``."""
        required = normalize_capabilities(required)
        if not required:
            return set()
        workers = self.refresh() if refresh else self.workers
        return {
            hostname for hostname, caps in workers.items()
            if satisfies(caps, required)
        }

    def ensure_qualified(self, required, task=None):
        """Raise :exc:`NoQualifiedWorkerError` if no worker qualifies.

        Returns the set of qualified worker hostnames otherwise.
        """
        required = normalize_capabilities(required)
        if not required:
            return set()
        qualified = self.qualified_workers(required)
        if qualified:
            logger.debug(
                'Task %r requiring %r can run on %r',
                task, list(required), sorted(qualified))
            return qualified
        available = {
            hostname: sorted(caps)
            for hostname, caps in self.workers.items()
        }
        raise NoQualifiedWorkerError(
            task=task, required=required, available=available)
