"""Tests for the composable compensation (saga) canvas workflow."""
import json

import pytest

from celery.canvas import chain, chord, group, signature
from celery.saga import (
    COMPENSATE_TASK_NAME,
    get_saga_state,
    load_saga_doc,
    resume_saga,
    run_compensation,
    saga,
)
from celery.states import (
    COMPENSATED,
    COMPENSATION_FAILED,
    FAILURE,
    PENDING,
    SAGA_RUNNING,
    SAGA_SUCCEEDED,
    STEP_DONE,
    STEP_SKIPPED,
    SUCCESS,
)


class SagaCase:

    def setup_method(self):
        self.app.conf.task_always_eager = True
        calls = []

        @self.app.task(shared=False)
        def forward(n):
            calls.append(('forward', n))
            if n == 'fail' or n == 'fail-group':
                raise RuntimeError(f'boom at {n}')
            return n

        @self.app.task(shared=False)
        def undo(n):
            calls.append(('compensate', n))
            return f'undid-{n}'

        @self.app.task(shared=False)
        def undo_flaky(n):
            # Fails until ``flaky_ok`` is flipped on the instance, so the
            # compensation can be recovered via resume_saga().
            calls.append(('compensate_flaky', n))
            if not getattr(self, 'flaky_ok', False):
                raise RuntimeError(f'cannot undo {n} yet')
            return f'undid-{n}'

        @self.app.task(shared=False)
        def add(x, y):
            return x + y

        @self.app.task(shared=False)
        def user_callback(*args, **kwargs):
            calls.append(('user_callback',))

        @self.app.task(shared=False)
        def user_errback(*args, **kwargs):
            calls.append(('user_errback',))

        self.calls = calls
        self.forward = forward
        self.undo = undo
        self.undo_flaky = undo_flaky
        self.add = add
        self.user_callback = user_callback
        self.user_errback = user_errback

    def teardown_method(self):
        self.app.conf.task_always_eager = False

    def run_saga(self, wf, catch=True):
        """Run an eager saga, returning (result, raised_exc_or_None)."""
        try:
            return wf.apply_async(), None
        except RuntimeError as exc:  # existing eager chain semantics
            if not catch:
                raise
            return None, exc

    def step_states(self, sid):
        return [(s['seq'], s['state']) for s in get_saga_state(sid, app=self.app)['steps']]

    def compensated_steps(self, sid):
        return [s['seq'] for s in get_saga_state(sid, app=self.app)['steps']
                if s['state'] == COMPENSATED]


class test_saga_api(SagaCase):

    def test_on_compensation_attaches_serializable_signature(self):
        sig = self.forward.si('a').on_compensation(self.undo.si('a'))
        assert sig['compensation']['task'] == self.undo.name
        # survives a json round trip (the wire/broker serialisation)
        restored = signature(json.loads(json.dumps(dict(sig))))
        assert restored['compensation']['task'] == self.undo.name
        # survives cloning
        cloned = sig.clone()
        assert cloned['compensation']['task'] == self.undo.name
        # can be removed
        sig.on_compensation(None)
        assert 'compensation' not in sig

    def test_on_compensation_rejects_canvas_primitives(self):
        with pytest.raises(TypeError):
            chain(self.forward.si('a')).on_compensation(self.undo.si('a'))
        with pytest.raises(TypeError):
            group(self.forward.si('a')).on_compensation(self.undo.si('a'))
        with pytest.raises(TypeError):
            chord([self.forward.si('a')],
                  self.forward.si('b')).on_compensation(self.undo.si('a'))

    def test_saga_accepts_iterable_of_steps(self):
        wf = saga([self.forward.si('a'),
                   self.forward.si('b').on_compensation(self.undo.si('b'))],
                  app=self.app)
        assert wf.subtask_type == 'saga'
        assert wf.saga_id
        # not persisted until the saga is prepared/run
        assert get_saga_state(wf.saga_id, app=self.app) is None
        wf.apply_async()
        assert get_saga_state(wf.saga_id, app=self.app) is not None

    def test_saga_round_trips_through_serialization(self):
        wf = saga(self.forward.si('a').on_compensation(self.undo.si('a')),
                  self.forward.si('b'), app=self.app)
        restored = signature(json.loads(json.dumps(dict(wf))))
        assert isinstance(restored, saga)
        assert restored.kwargs['saga_id'] == wf.saga_id


class test_saga_eager(SagaCase):

    def test_success_flow_marks_succeeded_and_runs_no_compensation(self):
        wf = saga(
            self.forward.si('a').on_compensation(self.undo.si('a')),
            self.forward.si('b').on_compensation(self.undo.si('b')),
            app=self.app,
        )
        res, _ = self.run_saga(wf)
        assert res.state == SUCCESS
        assert res.get() == 'b'
        doc = get_saga_state(wf.saga_id, app=self.app)
        assert doc['state'] == SAGA_SUCCEEDED
        assert [s['state'] for s in doc['steps']] == [STEP_DONE, STEP_DONE]
        assert ('compensate', 'a') not in self.calls
        assert ('compensate', 'b') not in self.calls

    def test_failure_dispatches_compensations_in_reverse_order(self):
        wf = saga(
            self.forward.si('a').on_compensation(self.undo.si('a')),
            self.forward.si('b').on_compensation(self.undo.si('b')),
            self.forward.si('fail').on_compensation(self.undo.si('c')),
            self.forward.si('d').on_compensation(self.undo.si('d')),
            app=self.app,
        )
        res, exc = self.run_saga(wf)
        assert exc is not None or (res is not None and res.state == FAILURE)
        doc = get_saga_state(wf.saga_id, app=self.app)
        assert doc['state'] == COMPENSATED
        assert self.step_states(wf.saga_id) == [
            (0, COMPENSATED),
            (1, COMPENSATED),
            (2, STEP_SKIPPED),
            (3, STEP_SKIPPED),
        ]
        # reverse completion order
        comps = [c[1] for c in self.calls if c[0] == 'compensate']
        assert comps == ['b', 'a']

    def test_failed_compensation_is_stable_terminal_state(self):
        wf = saga(
            self.forward.si('a').on_compensation(self.undo_flaky.si('a')),
            self.forward.si('fail').on_compensation(self.undo.si('x')),
            app=self.app,
        )
        self.run_saga(wf)
        doc = get_saga_state(wf.saga_id, app=self.app)
        assert doc['state'] == COMPENSATION_FAILED
        assert self.step_states(wf.saga_id)[0][1] == COMPENSATION_FAILED

        # resuming before the underlying problem is fixed stays failed
        resume_saga(wf.saga_id, app=self.app)
        doc = get_saga_state(wf.saga_id, app=self.app)
        assert doc['state'] == COMPENSATION_FAILED

        # once the compensation can succeed, resume recovers the saga
        self.flaky_ok = True
        resume_saga(wf.saga_id, app=self.app)
        doc = get_saga_state(wf.saga_id, app=self.app)
        assert doc['state'] == COMPENSATED
        assert self.step_states(wf.saga_id)[0][1] == COMPENSATED

    def test_group_failure_compensates_completed_members_only(self):
        wf = saga(
            group(
                self.forward.si('g1').on_compensation(self.undo.si('g1')),
                self.forward.si('g2').on_compensation(self.undo.si('g2')),
                self.forward.si('fail-group')
                    .on_compensation(self.undo.si('g3')),
                app=self.app,
            ),
            self.forward.si('after').on_compensation(self.undo.si('after')),
            app=self.app,
        )
        self.run_saga(wf)
        doc = get_saga_state(wf.saga_id, app=self.app)
        assert doc['state'] == COMPENSATED
        # completed members g1/g2 are compensated; the failed member and the
        # never-run tail step are skipped
        states = {s['task_name']: s['state'] for s in doc['steps']}
        assert states[self.forward.name] in (COMPENSATED, STEP_SKIPPED)
        comps = {c[1] for c in self.calls if c[0] == 'compensate'}
        assert comps <= {'g1', 'g2'}
        assert comps
        # no compensation for steps that never completed
        assert 'after' not in comps
        assert self.step_states(wf.saga_id)[-1][1] == STEP_SKIPPED

    def test_chord_header_failure_compensates_successful_header(self):
        wf = saga(
            chord(
                [
                    self.forward.si('h1').on_compensation(
                        self.undo.si('h1')),
                    self.forward.si('fail-group').on_compensation(
                        self.undo.si('h2')),
                ],
                self.forward.si('body').on_compensation(
                    self.undo.si('body')),
                app=self.app,
            ),
            self.forward.si('tail').on_compensation(self.undo.si('tail')),
            app=self.app,
        )
        self.run_saga(wf)
        doc = get_saga_state(wf.saga_id, app=self.app)
        assert doc['state'] == COMPENSATED
        comps = [c[1] for c in self.calls if c[0] == 'compensate']
        assert comps == ['h1']  # body/tail never ran

    def test_nested_saga_is_flattened_into_one_state(self):
        inner = saga(
            self.forward.si('i1').on_compensation(self.undo.si('i1')),
            app=self.app,
        )
        outer = saga(
            inner,
            self.forward.si('fail').on_compensation(self.undo.si('x')),
            app=self.app,
        )
        assert outer.kwargs['saga_id'] != inner.kwargs['saga_id']
        self.run_saga(outer)
        outer_doc = get_saga_state(outer.saga_id, app=self.app)
        inner_doc = load_saga_doc(self.app, inner.saga_id)
        assert inner_doc is None  # inner saga id is not used
        assert outer_doc['state'] == COMPENSATED
        assert self.step_states(outer.saga_id) == [
            (0, COMPENSATED), (1, STEP_SKIPPED)]
        comps = [c[1] for c in self.calls if c[0] == 'compensate']
        assert comps == ['i1']

    def test_user_callbacks_and_errbacks_still_fire(self):
        wf = saga(
            self.forward.si('a').on_compensation(self.undo.si('a')),
            self.forward.si('fail').on_compensation(self.undo.si('x')),
            app=self.app,
        )
        wf.link(self.user_callback.s())
        wf.link_error(self.user_errback.s())
        self.run_saga(wf)
        assert ('user_errback',) in self.calls
        # success callback must not fire on a failed saga
        assert ('user_callback',) not in self.calls

        self.calls.clear()
        wf2 = saga(
            self.forward.si('a'),
            self.forward.si('b'),
            app=self.app,
        )
        wf2.link(self.user_callback.s())
        self.run_saga(wf2)
        assert ('user_callback',) in self.calls

    def test_compensation_coordinator_is_idempotent(self):
        wf = saga(
            self.forward.si('a').on_compensation(self.undo.si('a')),
            self.forward.si('fail').on_compensation(self.undo.si('x')),
            app=self.app,
        )
        self.run_saga(wf)
        # re-running the coordinator (redelivery / restart) changes nothing
        run_compensation(self.app, wf.saga_id)
        doc = get_saga_state(wf.saga_id, app=self.app)
        assert doc['state'] == COMPENSATED
        comps = [c for c in self.calls if c[0] == 'compensate']
        assert comps == [('compensate', 'a')]

    def test_restart_reconciles_compensation_left_in_flight(self):
        from celery.saga import (
            _comp_attempt_task_id,
            mark_step_done,
            save_saga_doc,
        )
        wf = saga(
            self.forward.si('a').on_compensation(self.undo.si('a')),
            self.forward.si('fail').on_compensation(self.undo.si('x')),
            app=self.app,
        )
        # persist the plan without running the forward flow
        wf.prepare_steps((), {}, list(wf.tasks), app=self.app, clone=True)
        sid = wf.saga_id
        # simulated state after a worker crash: step completed, coordinator
        # dispatched its compensation (which actually finished) but died
        # before recording the result.
        mark_step_done(self.app, {'saga_id': sid, 'seq': 0})
        doc = get_saga_state(sid, app=self.app)
        doc['state'] = 'SAGA_COMPENSATING'
        comp_task_id = _comp_attempt_task_id(sid, 0, 0)
        self.app.backend.store_result(comp_task_id, 'undid-a', 'SUCCESS')
        doc['steps'][0]['state'] = 'COMPENSATING'
        save_saga_doc(self.app, doc)

        # resuming must reconcile without re-running the compensation
        run_compensation(self.app, sid)
        doc = get_saga_state(sid, app=self.app)
        assert doc['state'] == COMPENSATED
        assert doc['steps'][0]['state'] == COMPENSATED
        assert ('compensate', 'a') not in self.calls

    def test_compensation_recovers_from_forward_result_state(self):
        # completion markers lost (e.g. backend data was pruned), but the
        # forward task results are still readable: the coordinator infers
        # which steps committed side effects.
        wf = saga(
            self.forward.si('a').on_compensation(self.undo.si('a')),
            self.forward.si('fail').on_compensation(self.undo.si('x')),
            app=self.app,
        )
        wf.prepare_steps((), {}, list(wf.tasks), app=self.app, clone=True)
        task_ids = [
            step['task_id'] for step in
            get_saga_state(wf.saga_id, app=self.app)['steps']
        ]
        # simulate: first step committed (SUCCESS), second failed
        self.app.backend.store_result(task_ids[0], 'a', 'SUCCESS')
        self.app.backend.store_result(
            task_ids[1], RuntimeError('boom'), 'FAILURE')
        run_compensation(self.app, wf.saga_id)
        doc = get_saga_state(wf.saga_id, app=self.app)
        assert doc['state'] == COMPENSATED
        assert self.step_states(wf.saga_id) == [
            (0, COMPENSATED), (1, STEP_SKIPPED)]
        comps = [c[1] for c in self.calls if c[0] == 'compensate']
        assert comps == ['a']

    def test_chain_value_passing_result_semantics_preserved(self):
        wf = saga(
            self.add.s(2, 2),
            self.add.s(3),
            app=self.app,
        )
        res, _ = self.run_saga(wf)
        assert res.get() == 7  # ((2 + 2) + 3)
        assert get_saga_state(wf.saga_id, app=self.app)['state'] == SAGA_SUCCEEDED


class test_saga_plan_persisted_async(SagaCase):

    def setup_method(self):
        super().setup_method()
        self.app.conf.task_always_eager = False

    def test_plan_is_persisted_before_any_task_runs(self):
        wf = saga(
            self.forward.si('a').on_compensation(self.undo.si('a')),
            self.forward.si('b').on_compensation(self.undo.si('b')),
            app=self.app,
        )
        # broker is memory:// with no worker: nothing executes
        res = wf.apply_async()
        assert getattr(res, 'saga_id', None) == wf.saga_id
        doc = load_saga_doc(self.app, wf.saga_id)
        assert doc['state'] == SAGA_RUNNING
        assert len(doc['steps']) == 2
        assert all(s['state'] == PENDING for s in doc['steps'])
        assert all(s['compensation'] for s in doc['steps'])
        # frozen task ids recorded for restart recovery
        assert all(s['task_id'] for s in doc['steps'])

    def test_prepared_steps_carry_compensation_hooks(self):
        wf = saga(
            self.forward.si('a').on_compensation(self.undo.si('a')),
            self.forward.si('b'),
            app=self.app,
        )
        prepared, _ = wf.prepare_steps(
            (), {}, list(wf.tasks), app=self.app, clone=True)
        from celery.saga import _iter_leaf_signatures
        for node in prepared:
            for leaf in _iter_leaf_signatures(node):
                errbacks = [
                    h.get('task') if isinstance(h, dict) else h.task
                    for h in (leaf.options.get('link_error') or [])
                ]
                assert COMPENSATE_TASK_NAME in errbacks
        # plan got persisted while preparing
        assert load_saga_doc(self.app, wf.saga_id)['state'] == SAGA_RUNNING


class test_saga_states:

    def test_compensation_states_are_stable_strings(self):
        from celery import states
        for state in (states.SAGA_RUNNING, states.SAGA_COMPENSATING,
                      states.SAGA_SUCCEEDED, states.COMPENSATED,
                      states.COMPENSATION_FAILED,
                      states.PENDING_COMPENSATION):
            assert isinstance(state, str) and state.isupper()
        assert states.COMPENSATED in states.SAGA_TERMINAL_STATES
        assert states.COMPENSATION_FAILED in states.SAGA_TERMINAL_STATES
        assert states.COMPENSATED in states.COMPENSATION_TERMINAL_STATES
        assert states.COMPENSATION_FAILED in \
            states.COMPENSATION_TERMINAL_STATES
