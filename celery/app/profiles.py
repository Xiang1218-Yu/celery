"""Named execution profiles.

Execution profiles bundle a set of per-request execution constraints --
a rate limit, a priority and hard/soft time limits -- under a name, so
that multi-tenant workers can apply a whole set of constraints per task
request without registering a new task type for every combination.

Profiles live in an app-level registry, available as
:attr:`celery.Celery.execution_profiles`::

    app.execution_profiles.add(
        'tenant-gold', rate_limit='100/m', priority=7,
        time_limit=60, soft_time_limit=45,
    )
    task.apply_async(execution_profile='tenant-gold')

Profiles are *named, queryable, updatable and removable* at runtime.

When a task is published with a selected profile, a *snapshot* of the
profile is frozen into the task message.  Workers execute the task using
the snapshot carried by the message they receive, which means:

* Changes to a profile only affect tasks published after the change;
  tasks already in flight keep the snapshot they were published with.
* Retries and canvas-derived tasks (chain/group/chord successors and
  callbacks) keep the selection of the task they originate from, even
  though they are re-published by the worker, where the profile name
  might not be registered.
* Tasks published without selecting a profile keep using the existing
  defaults (task attributes and settings), exactly as before.
"""
import threading
from collections.abc import Mapping

from celery.exceptions import UnknownExecutionProfile
from celery.utils.time import rate

__all__ = ('ExecutionProfile', 'ExecutionProfileRegistry')

#: Constraints carried by an execution profile.
PROFILE_FIELDS = ('rate_limit', 'priority', 'time_limit', 'soft_time_limit')

_UNSET = object()


def _validate_name(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Execution profile name must be a non-empty string')
    return name


def _validate_rate_limit(value):
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            raise ValueError('Rate limit must not be empty')
        try:
            parsed = rate(value)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(
                f'Invalid rate limit {value!r}: use e.g. "100/s", "50/m" or "10/h".'
            ) from exc
        if not parsed:
            raise ValueError(f'Invalid rate limit {value!r}')
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return value
    raise ValueError(
        f'Rate limit must be a rate string (e.g. "100/m") or a number, '
        f'not {value!r}.'
    )


def _validate_priority(value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'Priority must be an integer, not {value!r}.')
    return value


def _validate_time_limit(value, what):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(
            f'{what} must be a positive number of seconds, not {value!r}.'
        )
    return value


def _validate_fields(rate_limit, priority, time_limit, soft_time_limit):
    """Validate a full set of profile constraints.

    Returns the normalized values as a tuple.
    """
    rate_limit = _validate_rate_limit(rate_limit)
    priority = _validate_priority(priority)
    time_limit = _validate_time_limit(time_limit, 'Hard time limit')
    soft_time_limit = _validate_time_limit(soft_time_limit, 'Soft time limit')
    if time_limit is not None and soft_time_limit is not None \
            and soft_time_limit > time_limit:
        raise ValueError(
            'Soft time limit must be less than or equal to the hard time limit'
        )
    return rate_limit, priority, time_limit, soft_time_limit


class ExecutionProfile:
    """A named, reusable set of per-request execution constraints.

    Arguments:
        name (str): Unique name used to select the profile.
        rate_limit (str, int, float): Rate limit applied to tasks
            published with this profile, e.g. ``"100/m"``.
            :const:`None` leaves the task-type/worker default in place.
        priority (int): Message priority.  :const:`None` leaves the
            default in place.
        time_limit (int, float): Hard time limit in seconds.
        soft_time_limit (int, float): Soft time limit in seconds.
    """

    def __init__(self, name, rate_limit=None, priority=None,
                 time_limit=None, soft_time_limit=None):
        self.name = _validate_name(name)
        self.rate_limit, self.priority, self.time_limit, \
            self.soft_time_limit = _validate_fields(
                rate_limit, priority, time_limit, soft_time_limit)

    def update(self, rate_limit=_UNSET, priority=_UNSET,
               time_limit=_UNSET, soft_time_limit=_UNSET, **kwargs):
        """Update one or more constraints.

        Only the supplied constraints change; the others keep their
        current value.  Raises :exc:`ValueError` if the new values are
        invalid.
        """
        fields = {
            'rate_limit': self.rate_limit if rate_limit is _UNSET else rate_limit,
            'priority': self.priority if priority is _UNSET else priority,
            'time_limit': self.time_limit if time_limit is _UNSET else time_limit,
            'soft_time_limit': (
                self.soft_time_limit if soft_time_limit is _UNSET
                else soft_time_limit),
        }
        self.rate_limit, self.priority, self.time_limit, \
            self.soft_time_limit = _validate_fields(**fields)
        return self

    def snapshot(self):
        """Return a frozen copy of the constraints as a plain dict.

        The snapshot is what travels inside task messages; it must stay
        serializable and must not be mutated after publication.
        """
        return {
            'name': self.name,
            'rate_limit': self.rate_limit,
            'priority': self.priority,
            'time_limit': self.time_limit,
            'soft_time_limit': self.soft_time_limit,
        }

    as_dict = snapshot

    def __eq__(self, other):
        if not isinstance(other, ExecutionProfile):
            return NotImplemented
        return self.snapshot() == other.snapshot()

    def __repr__(self):
        return (
            f'<ExecutionProfile: {self.name!r} rate_limit={self.rate_limit!r} '
            f'priority={self.priority!r} time_limit={self.time_limit!r} '
            f'soft_time_limit={self.soft_time_limit!r}>'
        )


class ExecutionProfileRegistry:
    """Registry of named :class:`ExecutionProfile` instances for an app.

    Obtained via :attr:`celery.Celery.execution_profiles`; profiles are
    shared by every publisher of the application and are safe to mutate
    from multiple threads.  Worker processes do not need the profiles to
    be registered: task messages carry the resolved snapshot.
    """

    def __init__(self, app=None):
        self.app = app
        self._profiles = {}
        self._lock = threading.RLock()

    def add(self, name, rate_limit=None, priority=None,
            time_limit=None, soft_time_limit=None, *, exist_ok=False):
        """Create and register a named execution profile.

        Arguments:
            name (str): Unique profile name.
            rate_limit (str, int, float): Optional rate limit
                (e.g. ``"100/m"``).
            priority (int): Optional message priority.
            time_limit (int, float): Optional hard time limit in seconds.
            soft_time_limit (int, float): Optional soft time limit in
                seconds.
            exist_ok (bool): If :const:`True`, replace an existing
                profile with the same name instead of raising.

        Raises:
            ValueError: If a profile with this name already exists and
                ``exist_ok`` is :const:`False`, or if any constraint is
                invalid.
        """
        name = _validate_name(name)
        with self._lock:
            if name in self._profiles and not exist_ok:
                raise ValueError(
                    f'Execution profile {name!r} already exists; '
                    f'use update() to change it or add(..., exist_ok=True).'
                )
            profile = ExecutionProfile(
                name, rate_limit=rate_limit, priority=priority,
                time_limit=time_limit, soft_time_limit=soft_time_limit,
            )
            self._profiles[name] = profile
            return profile

    #: Alias for :meth:`add`.
    register = add

    def get(self, name):
        """Return the profile registered under ``name``.

        Raises:
            celery.exceptions.UnknownExecutionProfile: if no such
                profile is registered.
        """
        try:
            with self._lock:
                return self._profiles[name]
        except KeyError:
            raise UnknownExecutionProfile(name)

    def snapshot(self, name):
        """Return the frozen snapshot dict for the profile ``name``.

        This is what gets embedded in task messages.
        """
        return self.get(name).snapshot()

    def update(self, name, **fields):
        """Update constraints of an existing profile.

        Only affects tasks published after the update; already published
        tasks carry their own snapshot.
        """
        with self._lock:
            profile = self.get(name)
            profile.update(**fields)
            return profile

    def delete(self, name):
        """Remove a profile from the registry.

        Raises:
            celery.exceptions.UnknownExecutionProfile: if no such
                profile is registered.
        """
        with self._lock:
            try:
                del self._profiles[name]
            except KeyError:
                raise UnknownExecutionProfile(name)

    #: Alias for :meth:`delete`.
    remove = delete

    def names(self):
        """Return a sorted list of all registered profile names."""
        with self._lock:
            return sorted(self._profiles)

    def all(self):
        """Return a list of all registered profiles."""
        with self._lock:
            return list(self._profiles.values())

    def items(self):
        with self._lock:
            return list(self._profiles.items())

    def clear(self):
        """Remove every registered profile."""
        with self._lock:
            self._profiles.clear()

    def resolve(self, profile):
        """Resolve a profile selection into a frozen snapshot dict.

        Accepts:

        * :const:`None` -- no profile selected (returns :const:`None`);
        * a profile name (:class:`str`) -- looked up in this registry;
        * an :class:`ExecutionProfile` -- snapshotted directly;
        * a :class:`~collections.abc.Mapping` -- an already resolved
          snapshot (e.g. propagated from a received task message by
          retries or canvas tasks on the worker), validated and copied.

        Raises:
            celery.exceptions.UnknownExecutionProfile: if a name is
                given that is not registered.
            ValueError: if an inline snapshot contains invalid values.
            TypeError: if ``profile`` is of an unsupported type.
        """
        if profile is None:
            return None
        if isinstance(profile, str):
            return self.snapshot(profile)
        if isinstance(profile, ExecutionProfile):
            return profile.snapshot()
        if isinstance(profile, Mapping):
            snapshot = {
                'name': profile.get('name'),
                'rate_limit': profile.get('rate_limit'),
                'priority': profile.get('priority'),
                'time_limit': profile.get('time_limit'),
                'soft_time_limit': profile.get('soft_time_limit'),
            }
            _validate_fields(
                snapshot['rate_limit'], snapshot['priority'],
                snapshot['time_limit'], snapshot['soft_time_limit'],
            )
            if snapshot['name'] is not None:
                _validate_name(snapshot['name'])
            return snapshot
        raise TypeError(
            f'Unsupported execution profile {profile!r}: use a registered '
            f'profile name, an ExecutionProfile, or a snapshot mapping.'
        )

    def __getitem__(self, name):
        return self.get(name)

    def __contains__(self, name):
        with self._lock:
            return name in self._profiles

    def __iter__(self):
        with self._lock:
            return iter(list(self._profiles))

    def __len__(self):
        with self._lock:
            return len(self._profiles)

    def __repr__(self):
        with self._lock:
            return f'<ExecutionProfileRegistry: {sorted(self._profiles)!r}>'
