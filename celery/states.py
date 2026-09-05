"""Built-in task states.

.. _states:

States
------

See :ref:`task-states`.

.. _statesets:

Sets
----

.. state:: READY_STATES

READY_STATES
~~~~~~~~~~~~

Set of states meaning the task result is ready (has been executed).

.. state:: UNREADY_STATES

UNREADY_STATES
~~~~~~~~~~~~~~

Set of states meaning the task result is not ready (hasn't been executed).

.. state:: EXCEPTION_STATES

EXCEPTION_STATES
~~~~~~~~~~~~~~~~

Set of states meaning the task returned an exception.

.. state:: PROPAGATE_STATES

PROPAGATE_STATES
~~~~~~~~~~~~~~~~

Set of exception states that should propagate exceptions to the user.

.. state:: ALL_STATES

ALL_STATES
~~~~~~~~~~

Set of all possible states.

Misc
----

"""

__all__ = (
    'PENDING', 'RECEIVED', 'STARTED', 'SUCCESS', 'FAILURE',
    'REVOKED', 'RETRY', 'IGNORED', 'READY_STATES', 'UNREADY_STATES',
    'EXCEPTION_STATES', 'PROPAGATE_STATES', 'precedence', 'state',
    # Saga (compensation workflow) states
    'SAGA_RUNNING', 'SAGA_COMPENSATING', 'SAGA_SUCCEEDED',
    'STEP_DONE', 'STEP_SKIPPED',
    'PENDING_COMPENSATION', 'COMPENSATING', 'COMPENSATED',
    'COMPENSATION_FAILED',
    'SAGA_STATES', 'SAGA_TERMINAL_STATES', 'COMPENSATION_TERMINAL_STATES',
)

#: State precedence.
#: None represents the precedence of an unknown state.
#: Lower index means higher precedence.
PRECEDENCE = [
    'SUCCESS',
    'FAILURE',
    None,
    'REVOKED',
    'STARTED',
    'RECEIVED',
    'REJECTED',
    'RETRY',
    'PENDING',
]

#: Hash lookup of PRECEDENCE to index
PRECEDENCE_LOOKUP = dict(zip(PRECEDENCE, range(0, len(PRECEDENCE))))
NONE_PRECEDENCE = PRECEDENCE_LOOKUP[None]


def precedence(state: str) -> int:
    """Get the precedence index for state.

    Lower index means higher precedence.
    """
    try:
        return PRECEDENCE_LOOKUP[state]
    except KeyError:
        return NONE_PRECEDENCE


class state(str):
    """Task state.

    State is a subclass of :class:`str`, implementing comparison
    methods adhering to state precedence rules::

        >>> from celery.states import state, PENDING, SUCCESS

        >>> state(PENDING) < state(SUCCESS)
        True

    Any custom state is considered to be lower than :state:`FAILURE` and
    :state:`SUCCESS`, but higher than any of the other built-in states::

        >>> state('PROGRESS') > state(STARTED)
        True

        >>> state('PROGRESS') > state('SUCCESS')
        False
    """

    def __gt__(self, other: str) -> bool:
        return precedence(self) < precedence(other)

    def __ge__(self, other: str) -> bool:
        return precedence(self) <= precedence(other)

    def __lt__(self, other: str) -> bool:
        return precedence(self) > precedence(other)

    def __le__(self, other: str) -> bool:
        return precedence(self) >= precedence(other)


#: Task state is unknown (assumed pending since you know the id).
PENDING = 'PENDING'
#: Task was received by a worker (only used in events).
RECEIVED = 'RECEIVED'
#: Task was started by a worker (:setting:`task_track_started`).
STARTED = 'STARTED'
#: Task succeeded
SUCCESS = 'SUCCESS'
#: Task failed
FAILURE = 'FAILURE'
#: Task was revoked.
REVOKED = 'REVOKED'
#: Task was rejected (only used in events).
REJECTED = 'REJECTED'
#: Task is waiting for retry.
RETRY = 'RETRY'
IGNORED = 'IGNORED'

READY_STATES = frozenset({SUCCESS, FAILURE, REVOKED})
UNREADY_STATES = frozenset({PENDING, RECEIVED, STARTED, REJECTED, RETRY})
EXCEPTION_STATES = frozenset({RETRY, FAILURE, REVOKED})
PROPAGATE_STATES = frozenset({FAILURE, REVOKED})

ALL_STATES = frozenset({
    PENDING, RECEIVED, STARTED, SUCCESS, FAILURE, RETRY, REVOKED,
})


# --- Saga (compensation workflow) states ---------------------------------

#: Saga forward flow is still running (no failure observed yet).
SAGA_RUNNING = 'SAGA_RUNNING'
#: Saga forward flow failed and compensations are being dispatched.
SAGA_COMPENSATING = 'SAGA_COMPENSATING'
#: Saga forward flow completed successfully; no compensation needed.
SAGA_SUCCEEDED = 'SAGA_SUCCEEDED'

#: Forward step completed successfully (its compensation is not yet needed).
STEP_DONE = 'DONE'
#: Forward step never completed (did not run or failed before commit);
#: it has no side effects that need compensating.
STEP_SKIPPED = 'SKIPPED'

#: Forward step completed and its compensation still has to run
#: ("still awaiting compensation" / 仍待补偿).
PENDING_COMPENSATION = 'PENDING_COMPENSATION'
#: The compensation for this step is currently running.
COMPENSATING = 'COMPENSATING'
#: The step has been compensated successfully (已补偿).
COMPENSATED = 'COMPENSATED'
#: The compensation for this step failed (补偿失败). This is a stable
#: terminal state; the saga can be resumed to retry the compensation.
COMPENSATION_FAILED = 'COMPENSATION_FAILED'

#: All states a saga document may be in.
SAGA_STATES = frozenset({
    SAGA_RUNNING, SAGA_COMPENSATING, SAGA_SUCCEEDED,
    COMPENSATED, COMPENSATION_FAILED,
})

#: Saga states that will not transition any further without an explicit
#: resume request.
SAGA_TERMINAL_STATES = frozenset({
    SAGA_SUCCEEDED, COMPENSATED, COMPENSATION_FAILED,
})

#: Stable terminal states for an individual step compensation result.
COMPENSATION_TERMINAL_STATES = frozenset({COMPENSATED, COMPENSATION_FAILED})
