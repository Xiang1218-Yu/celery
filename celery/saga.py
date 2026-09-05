"""Composable compensation (saga) workflows for canvas.

A :class:`saga` is a chain-like canvas where every forward *step* (a leaf
task signature) may carry a serializable *compensation* signature::

    from celery import saga

    workflow = saga(
        charge_card.s(amount).on_compensation(refund.s(amount)),
        ship_order.s(order_id).on_compensation(cancel_shipment.s(order_id)),
    )
    res = workflow.apply_async()

While the forward flow runs, the saga persists its plan (the ordered list
of steps together with their compensation definitions and frozen task ids)
in the result backend.  Every completed step is recorded as well, so the
compensation state survives task retries, worker / backend restarts and
nested ``chain`` / ``group`` / ``chord`` structures.

When a step fails, an internal errback task triggers the saga coordinator,
which dispatches the compensation signatures in **reverse completion
order**.  Every compensation ends in one of the stable terminal states
:data:`~celery.states.COMPENSATED` (已补偿),
:data:`~celery.states.COMPENSATION_FAILED` (补偿失败) or stays
:data:`~celery.states.PENDING_COMPENSATION` (仍待补偿) until the saga is
resumed.

The forward success flow, ordinary ``link`` / ``link_error`` callbacks and
errbacks, as well as eager and asynchronous result trees keep their
existing semantics -- the saga only *adds* internal callbacks.
"""

from kombu.utils.uuid import uuid

from celery._state import current_app
from celery.canvas import (
    Signature,
    _chain,
    _chord,
    group,
    maybe_signature,
    signature,
)
from celery.states import (
    COMPENSATED,
    COMPENSATING,
    COMPENSATION_FAILED,
    FAILURE,
    IGNORED,
    PENDING,
    PENDING_COMPENSATION,
    RECEIVED,
    REJECTED,
    RETRY,
    REVOKED,
    SAGA_COMPENSATING,
    SAGA_RUNNING,
    SAGA_SUCCEEDED,
    STARTED,
    STEP_DONE,
    STEP_SKIPPED,
    SUCCESS,
    COMPENSATION_TERMINAL_STATES,
    SAGA_TERMINAL_STATES,
)
from celery.utils.functional import is_list, maybe_list
from celery.utils.log import get_logger

__all__ = (
    'saga', 'Saga', 'get_saga_state', 'resume_saga',
    'SagaCompensationDeferred',
    'COMPENSATE_TASK_NAME', 'MARK_DONE_TASK_NAME',
    'MARK_COMP_TASK_NAME', 'SUCCEED_TASK_NAME',
)

logger = get_logger(__name__)

#: Names of the built-in tasks that drive saga bookkeeping.
COMPENSATE_TASK_NAME = 'celery.saga.compensate'
MARK_DONE_TASK_NAME = 'celery.saga.mark_done'
MARK_COMP_TASK_NAME = 'celery.saga.mark_comp'
SUCCEED_TASK_NAME = 'celery.saga.succeed'

_IN_FLIGHT_STATES = frozenset({RETRY, STARTED, RECEIVED})


class SagaCompensationDeferred(Exception):
    """Raised by the coordinator when some steps are still in flight.

    The coordinator task catches this and retries, so compensation resumes
    once every step has reached a terminal task state.
    """

    def __init__(self, saga_id):
        self.saga_id = saga_id
        super().__init__(
            f'Saga {saga_id} still has steps in flight; '
            f'deferring compensation')


# ---------------------------------------------------------------------------
# Backend backed saga document
# ---------------------------------------------------------------------------

def _state_key(saga_id):
    """Backend (task-meta) key under which the saga document is stored."""
    return f'celery-saga-state-{saga_id}'


def _comp_task_id(saga_id, seq):
    """Deterministic task id base for a step's compensation."""
    return f'{_state_key(saga_id)}:comp:{seq}'


def _comp_attempt_task_id(saga_id, seq, attempt):
    """Deterministic task id for one compensation attempt (dedupe key)."""
    return f'{_comp_task_id(saga_id, seq)}:{attempt}'


def load_saga_doc(app, saga_id):
    """Load the persisted saga document, or :const:`None` if absent."""
    backend = app.backend
    try:
        # Use the low-level accessor directly: saga state is also written
        # when tasks run eagerly, where get_task_meta() would warn.
        meta = backend._get_task_meta_for(_state_key(saga_id))
    except Exception:  # pragma: no cover - backend unavailable
        logger.exception('Saga %s: failed to load state document', saga_id)
        return None
    if not isinstance(meta, dict):
        return None
    if meta.get('status') == PENDING and not meta.get('result'):
        return None
    doc = meta.get('result')
    if isinstance(doc, dict) and doc.get('saga_id') == saga_id:
        return doc
    return None


def save_saga_doc(app, doc):
    """Persist the saga document (idempotent, last-write-wins)."""
    app.backend.store_result(
        _state_key(doc['saga_id']), doc, state=doc['state'])
    return doc


def _find_step(doc, seq):
    for step in doc.get('steps', ()):  # noqa
        if step.get('seq') == seq:
            return step
    return None


def _reconcile(doc):
    """Compute the saga level state from the individual step states."""
    step_states = [s.get('state') for s in doc.get('steps', ())]
    if COMPENSATION_FAILED in step_states:
        doc['state'] = COMPENSATION_FAILED
    elif any(s in (PENDING_COMPENSATION, COMPENSATING) for s in step_states):
        doc['state'] = SAGA_COMPENSATING
    else:
        doc['state'] = COMPENSATED
    return doc


def _reconcile_and_save(app, doc):
    return save_saga_doc(app, _reconcile(doc))


# ---------------------------------------------------------------------------
# Signature tree walking
# ---------------------------------------------------------------------------

def _iter_leaf_signatures(sig):
    """Yield leaf task signatures from a (possibly nested) canvas node."""
    if isinstance(sig, _chord):
        header = sig.tasks
        if isinstance(header, group):
            header_tasks = header.tasks
        else:
            header_tasks = maybe_list(header) or []
        for task in header_tasks:
            yield from _iter_leaf_signatures(task)
        if sig.body is not None:
            yield from _iter_leaf_signatures(sig.body)
    elif isinstance(sig, group):
        for task in sig.tasks:
            yield from _iter_leaf_signatures(task)
    elif isinstance(sig, _chain):
        for task in sig.tasks:
            yield from _iter_leaf_signatures(task)
    elif isinstance(sig, Signature):
        yield sig


def _hooks_contain(hooks, task_name):
    for hook in maybe_list(hooks) or []:
        name = hook.get('task') if isinstance(hook, dict) \
            else getattr(hook, 'task', None)
        if name == task_name:
            return True
    return False


def _internal_hook_names():
    return frozenset({
        COMPENSATE_TASK_NAME, MARK_DONE_TASK_NAME,
        MARK_COMP_TASK_NAME, SUCCEED_TASK_NAME,
    })


def _is_internal_hook(hook):
    name = hook.get('task') if isinstance(hook, dict) \
        else getattr(hook, 'task', None)
    return name in _internal_hook_names()


def _hook_saga_id(hook):
    """Return the saga id embedded in an internal hook payload."""
    args = hook.get('args') if isinstance(hook, dict) \
        else getattr(hook, 'args', ())
    args = args or ()
    if args and isinstance(args[0], dict):
        return args[0].get('saga_id')
    return None


def _strip_internal_hooks(options, saga_id=None):
    """Remove internal saga hooks (optionally only stale ones).

    Canvas composition (``|`` / ``unchain_tasks``) and cloning can carry
    saga bookkeeping hooks from a previous construction -- re-attaching
    them blindly would double-fire callbacks and reference old saga ids.
    """
    for key in ('link', 'link_error'):
        hooks = options.get(key)
        if hooks is None:
            continue
        kept = []
        for hook in maybe_list(hooks) or []:
            if not _is_internal_hook(hook):
                kept.append(hook)
            elif saga_id is not None and _hook_saga_id(hook) == saga_id:
                kept.append(hook)
        if kept:
            options[key] = kept
        else:
            options.pop(key, None)


# ---------------------------------------------------------------------------
# Saga canvas primitive
# ---------------------------------------------------------------------------

@Signature.register_type(name='saga')
class Saga(_chain):
    """Chain-like canvas primitive that tracks compensations.

    See :mod:`celery.saga` for the full documentation.
    """

    @classmethod
    def from_dict(cls, d, app=None):
        tasks = d['kwargs']['tasks']
        if isinstance(tasks, tuple):  # aaaargh
            tasks = d['kwargs']['tasks'] = list(tasks)
        tasks = [maybe_signature(task, app=app) for task in tasks]
        saga_id = d['kwargs'].get('saga_id')
        return cls(tasks, app=app, saga_id=saga_id, **d['options'])

    def __init__(self, *tasks, **options):
        saga_id = options.pop('saga_id', None)
        super().__init__(*tasks, **options)
        self.subtask_type = 'saga'
        saga_id = saga_id or self.kwargs.get('saga_id') or uuid()
        self.kwargs['saga_id'] = saga_id
        # Nested sagas are flattened: the outermost saga owns the single
        # compensation state for every leaf step.
        self.kwargs['tasks'] = self._flatten_sagas(self.tasks)
        self._install_saga_hooks()

    @property
    def saga_id(self):
        return self.kwargs['saga_id']

    def _flatten_sagas(self, tasks):
        flat = []
        for task in tasks:
            task = maybe_signature(task, app=self._app)
            if isinstance(task, Saga):
                flat.extend(self._flatten_sagas(task.tasks))
            else:
                flat.append(task)
        return flat

    def _saga_sig(self, task_name, payload=None):
        payload = {'saga_id': self.saga_id, **(payload or {})}
        return signature(task_name, args=(payload,), immutable=True,
                         app=self._app)

    def _install_saga_hooks(self):
        """Attach the internal coordinator / success hooks.

        They live in the chain options so they survive composition with
        other canvases (``|``) and are propagated by the existing chain
        machinery.  Per-step hooks are attached later in
        :meth:`prepare_steps`.  Any internal hooks left over from a
        previous construction (composition / cloning with another saga
        id) are dropped first.
        """
        _strip_internal_hooks(self.options, self.saga_id)
        self.append_to_list_option(
            'link_error', self._saga_sig(COMPENSATE_TASK_NAME))
        self.append_to_list_option(
            'link', self._saga_sig(SUCCEED_TASK_NAME))

    # -- plan / hooks -----------------------------------------------------

    def _saga_plan(self, prepared):
        """Build the ordered step plan from prepared (frozen) tasks.

        ``prepared`` is returned in reverse execution order by
        :meth:`_chain.prepare_steps`, so reversing it yields forward
        (completion) order.
        """
        steps = []
        for node in reversed(prepared):
            for leaf in _iter_leaf_signatures(node):
                task_id = leaf.options.get('task_id')
                if not task_id:
                    continue
                steps.append({
                    'seq': len(steps),
                    'task_id': task_id,
                    'task_name': leaf.get('task'),
                    'compensation': leaf.get('compensation'),
                    'state': PENDING,
                })
        return steps

    def _persist_saga_plan(self, steps, app=None):
        app = app or self.app
        if load_saga_doc(app, self.saga_id) is not None:
            # Already persisted (freeze + run, retry, resume, ...).
            return
        save_saga_doc(app, {
            'saga_id': self.saga_id,
            'state': SAGA_RUNNING,
            'steps': steps,
        })

    def _attach_step_hooks(self, prepared, steps):
        """Attach per-step bookkeeping callbacks to frozen leaf signatures.

        * ``mark_done`` (success callback) on every step -- records
          completion, works with eager workers that store no task results.
        * the compensation coordinator (errback) on every step -- directly
          covers chord header members and nested canvases.
        * user supplied callbacks/errbacks are propagated the same way.
        * the saga success marker (plus user chain callbacks) is attached
          to the *last* forward node only.
        """
        by_id = {step['task_id']: step for step in steps}
        user_links = [h for h in maybe_list(self.options.get('link')) or []
                      if not _is_internal_hook(h)]
        user_errbacks = [
            h for h in maybe_list(self.options.get('link_error')) or []
            if not _is_internal_hook(h)]

        last_node = prepared[0] if prepared else None
        for node in prepared:
            is_last = node is last_node
            for leaf in _iter_leaf_signatures(node):
                step = by_id.get(leaf.options.get('task_id'))
                if step is None:
                    continue
                # Drop internal hooks carried over from a previous
                # construction (composition / clone with another saga id)
                # and install the ones matching this saga.
                _strip_internal_hooks(leaf.options, self.saga_id)
                payload = {'seq': step['seq']}
                leaf.link(self._saga_sig(MARK_DONE_TASK_NAME, payload))
                leaf.link_error(self._saga_sig(COMPENSATE_TASK_NAME))
                for errback in user_errbacks:
                    leaf.link_error(maybe_signature(errback, app=self._app,
                                                    clone=True))
                if is_last:
                    leaf.link(self._saga_sig(SUCCEED_TASK_NAME))
                    for callback in user_links:
                        leaf.link(maybe_signature(callback, app=self._app,
                                                  clone=True))

    def prepare_steps(self, args, kwargs, tasks, root_id=None,
                      parent_id=None, link_error=None, app=None,
                      last_task_id=None, group_id=None, chord_body=None,
                      clone=True, from_dict=Signature.from_dict,
                      group_index=None):
        prepared, results = super().prepare_steps(
            args, kwargs, tasks, root_id=root_id, parent_id=parent_id,
            link_error=link_error, app=app, last_task_id=last_task_id,
            group_id=group_id, chord_body=chord_body, clone=clone,
            from_dict=from_dict, group_index=group_index)
        try:
            steps = self._saga_plan(prepared)
            self._persist_saga_plan(steps, app)
            self._attach_step_hooks(prepared, steps)
        except Exception:  # pragma: no cover - never break the forward flow
            logger.exception(
                'Saga %s: failed to persist plan or attach step hooks',
                self.saga_id)
        return prepared, results

    # -- execution --------------------------------------------------------

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        _attach_saga_id(result, self.saga_id)
        return result

    def apply(self, args=None, kwargs=None, **options):
        # Eager execution: build the fully prepared nodes (frozen task ids,
        # plan persisted, per-step hooks attached -- including nodes that
        # group->chord upgrading only produces in prepared form) and run
        # them inline in forward completion order.
        from celery.result import EagerResult  # noqa: PLC0415

        args = args if args else ()
        kwargs = kwargs if kwargs else {}
        try:
            prepared, _ = self.prepare_steps(
                args, kwargs, self.tasks, app=self.app, clone=True)
        except Exception:  # pragma: no cover
            logger.exception(
                'Saga %s: failed to prepare before eager execution',
                self.saga_id)
            return super().apply(args, kwargs, **options)

        # Chain level links/errbacks are already installed on the prepared
        # leaf signatures; drop them from the per-task options so they are
        # not merged over the per-step hooks.
        run_options = dict(self.options, **options) if options else \
            dict(self.options)
        run_options.pop('link', None)
        run_options.pop('link_error', None)

        last = None
        for task in reversed(prepared):
            res = task.apply(
                last and (last.get(),), **dict(run_options))
            res.parent, last = last, res
            if isinstance(res, EagerResult) and res.state in (IGNORED, REJECTED):
                break
        _attach_saga_id(last, self.saga_id)
        return last

    def __new__(cls, *tasks, **kwargs):
        # Normalise a single iterable argument (list/generator) to the
        # tuple of steps and delegate to __init__ -- unlike chain() we do
        # not build the instance through | composition, which would seed
        # intermediate sagas with stale (wrong saga id) bookkeeping hooks.
        if not kwargs and tasks and (len(tasks) != 1 or is_list(tasks[0])):
            tasks = (tuple(tasks[0]),) if len(tasks) == 1 else tasks
        return super().__new__(cls, *tasks, **kwargs)

    def __repr__(self):
        if not self.tasks:
            return f'<saga@{id(self):#x}: empty>'
        return 'saga(' + ', '.join(repr(t) for t in self.tasks) + ')'


def _attach_saga_id(result, saga_id):
    if result is None:
        return
    try:
        result.saga_id = saga_id
    except Exception:  # pragma: no cover
        pass


# Public name for the canvas primitive (mirrors ``chord = _chord``).
saga = Saga


# ---------------------------------------------------------------------------
# Bookkeeping logic (invoked by the built-in tasks in app/builtins.py)
# ---------------------------------------------------------------------------

def mark_step_done(app, payload):
    """Record that a forward step completed successfully."""
    saga_id = payload['saga_id']
    doc = load_saga_doc(app, saga_id)
    if doc is None or doc['state'] != SAGA_RUNNING:
        return None
    step = _find_step(doc, payload['seq'])
    if step is not None and step.get('state') == PENDING:
        step['state'] = STEP_DONE
        return save_saga_doc(app, doc)
    return doc


def mark_saga_succeeded(app, payload):
    """Mark the saga as succeeded once every step is done.

    Deliberately conservative: the marker may fire more than once (eager
    chains pass chain callbacks to every inline task), so we only transition
    when *all* steps have actually completed.
    """
    saga_id = payload['saga_id']
    doc = load_saga_doc(app, saga_id)
    if doc is None or doc['state'] != SAGA_RUNNING:
        return doc
    steps = doc.get('steps') or []
    if steps and all(s.get('state') == STEP_DONE for s in steps):
        doc['state'] = SAGA_SUCCEEDED
        return save_saga_doc(app, doc)
    return doc


def mark_compensation_result(app, payload):
    """Record the terminal result of a single compensation."""
    saga_id = payload['saga_id']
    doc = load_saga_doc(app, saga_id)
    if doc is None:
        return None
    step = _find_step(doc, payload['seq'])
    if step is None:
        return doc
    status = payload.get('status')
    if status not in COMPENSATION_TERMINAL_STATES:
        return doc
    if step.get('state') in COMPENSATION_TERMINAL_STATES:
        # Idempotent: keep the first terminal state (at-least-once delivery
        # may fire the markers more than once).
        return doc
    step['state'] = status
    if status == COMPENSATION_FAILED and payload.get('error'):
        step['error'] = payload['error']
    return _reconcile_and_save(app, doc)


def _task_state(app, task_id):
    """Best-effort lookup of a task's state directly in the backend.

    Uses the low-level meta accessor so this also works when tasks run
    eagerly (where task results are not stored by default).
    """
    try:
        meta = app.backend._get_task_meta_for(task_id)
    except Exception:  # pragma: no cover - backend unavailable
        return PENDING
    if isinstance(meta, dict):
        return meta.get('status', PENDING)
    return PENDING


def _forward_state(app, step):
    """Best-effort lookup of the forward task's current state."""
    return _task_state(app, step['task_id'])


def _dispatch_compensation(app, doc, step):
    """Dispatch (or reconcile) a single step's compensation."""
    saga_id = doc['saga_id']
    seq = step['seq']
    # Always operate on the latest persisted document: markers (running
    # inline in eager mode, or as separate tasks later) may have advanced
    # sibling steps concurrently, and writing a stale in-memory copy would
    # roll those transitions back.
    doc = load_saga_doc(app, saga_id) or doc
    step = _find_step(doc, seq)
    if step is None or step.get('state') == COMPENSATED:
        return doc

    # Deterministic attempt id: identical across duplicate coordinators
    # (dedupes redelivery); bumped on an explicit resume after a terminal
    # COMPENSATION_FAILED so the retry gets a fresh task id instead of
    # replaying the old failed result.
    attempt = int(step.get('comp_attempt', 0))
    if step.get('state') == COMPENSATION_FAILED:
        attempt += 1
    step['comp_attempt'] = attempt

    comp_task_id = _comp_attempt_task_id(saga_id, seq, attempt)
    existing = _task_state(app, comp_task_id)
    if existing == SUCCESS:
        step['state'] = COMPENSATED
        _reconcile_and_save(app, doc)
        return doc
    if existing in (FAILURE, REVOKED):
        step['state'] = COMPENSATION_FAILED
        _reconcile_and_save(app, doc)
        return doc

    comp_sig = signature(step['compensation'], app=app)
    comp_sig.set(task_id=comp_task_id)
    marker_payload = {'saga_id': saga_id, 'seq': seq}
    # NOTE: link()/link_error() return the *linked* signature, not self.
    comp_sig.link(signature(
        MARK_COMP_TASK_NAME,
        args=({**marker_payload, 'status': COMPENSATED},),
        immutable=True, app=app))
    comp_sig.link_error(signature(
        MARK_COMP_TASK_NAME,
        args=({**marker_payload, 'status': COMPENSATION_FAILED},),
        immutable=True, app=app))

    step['state'] = COMPENSATING
    save_saga_doc(app, doc)
    # In eager mode this runs inline and the markers above update the
    # document immediately; in async mode the markers run as separate tasks
    # and redelivery / resume reconciles any gaps.
    comp_sig.apply_async()
    return doc


def run_compensation(app, saga_id):
    """Run (or resume) reverse compensation for a saga.

    Idempotent: safe to invoke multiple times concurrently or after a
    worker / backend restart.
    """
    doc = load_saga_doc(app, saga_id)
    if doc is None:
        return None
    if doc['state'] in SAGA_TERMINAL_STATES \
            and doc['state'] != COMPENSATION_FAILED:
        # Fully terminal (succeeded / all compensated): nothing left to do.
        # COMPENSATION_FAILED is also a stable terminal state, but explicit
        # resume_saga() calls are allowed to re-run the coordinator to
        # retry failed compensations, so it falls through below.
        return doc

    eager = bool(app.conf.task_always_eager)

    # 1. Reconcile the forward outcome of every step that has not been
    #    explicitly recorded yet (covers lost completion markers).
    uncertain = False
    for step in doc['steps']:
        if step.get('state') != PENDING:
            continue
        state = _forward_state(app, step)
        if state == SUCCESS:
            step['state'] = STEP_DONE
        elif state in (FAILURE, REVOKED):
            step['state'] = STEP_SKIPPED
        elif state in _IN_FLIGHT_STATES:
            # The task may still commit side effects; wait for it.
            uncertain = True
        else:
            # Never dispatched / result expired: no side effects assumed.
            step['state'] = STEP_SKIPPED

    if uncertain:
        save_saga_doc(app, doc)
        if not eager:
            # The coordinator task retries; eager flows are synchronous so
            # no step can be in flight here.
            raise SagaCompensationDeferred(saga_id)

    pending = [
        step for step in doc['steps']
        if step.get('compensation')
        and step.get('state') in (STEP_DONE, COMPENSATING, COMPENSATION_FAILED)
    ]
    if not pending:
        # Nothing to undo: all completed steps are compensation-free.
        doc['state'] = COMPENSATED
        return save_saga_doc(app, doc)

    doc['state'] = SAGA_COMPENSATING
    save_saga_doc(app, doc)

    # 2. Dispatch compensations in reverse completion order.
    for step in sorted(pending, key=lambda s: s['seq'], reverse=True):
        _dispatch_compensation(app, doc, step)

    # 3. Final reconcile (reload to pick up markers applied inline/eagerly).
    doc = load_saga_doc(app, saga_id) or doc
    return _reconcile_and_save(app, doc)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_saga_state(saga_id, app=None):
    """Return the persisted saga state document, or :const:`None`.

    The document has the shape::

        {
          'saga_id': str,
          'state': SAGA_RUNNING | SAGA_COMPENSATING | SAGA_SUCCEEDED
                    | COMPENSATED | COMPENSATION_FAILED,
          'steps': [
            {'seq': int, 'task_id': str, 'task_name': str,
             'compensation': <serialized signature> | None,
             'state': PENDING | STEP_DONE | STEP_SKIPPED
                      | PENDING_COMPENSATION | COMPENSATING
                      | COMPENSATED | COMPENSATION_FAILED},
            ...
          ],
        }
    """
    app = app or current_app
    doc = load_saga_doc(app, saga_id)
    if doc is not None and doc.get('state') == SAGA_RUNNING:
        steps = doc.get('steps') or []
        if steps and all(s.get('state') == STEP_DONE for s in steps):
            # Success marker got lost between completion and delivery.
            doc['state'] = SAGA_SUCCEEDED
            save_saga_doc(app, doc)
    return doc


def resume_saga(saga_id, app=None, **options):
    """Resume (or retry) compensation for a saga.

    Applies the coordinator task again.  The coordinator is idempotent, so
    this is safe to call after worker / backend restarts or after a
    compensation failed and the underlying issue has been fixed.
    """
    app = app or current_app
    sig = signature(
        COMPENSATE_TASK_NAME, args=({'saga_id': saga_id},),
        immutable=True, app=app)
    return sig.apply_async(**options)
