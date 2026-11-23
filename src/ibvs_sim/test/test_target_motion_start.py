"""Checks for the simulation target's ACTIVE start signal."""

from types import SimpleNamespace

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.empty_pb2 import Empty
import pytest

from ibvs_sim.target_motion_start import TargetMotionStart


def _relay(responses):
    calls = []
    logger = SimpleNamespace(info=lambda _: None, warning=lambda _: None)

    def request(*args):
        calls.append(args)
        return next(responses)

    node = SimpleNamespace(
        started=False, service='/world/test/paper_target/start',
        gz_node=SimpleNamespace(request=request), get_logger=lambda: logger,
    )
    return node, calls


@pytest.mark.parametrize('phase', (
    'WAIT_HOVER', 'VISUAL_SEARCH', 'VISUAL_READY', 'PRESTREAM', 'ABORT', 'LAND',
))
def test_other_interception_states_do_not_start_target(phase):
    node, calls = _relay(iter(()))
    TargetMotionStart._control_callback(node, SimpleNamespace(control_state=phase))
    assert not calls and not node.started


def test_active_starts_target_once_without_pausing_on_later_states():
    node, calls = _relay(iter(((True, Boolean(data=True)),)))
    for phase in ('PRESTREAM', 'ACTIVE', 'ACTIVE', 'ABORT', 'ACTIVE'):
        TargetMotionStart._control_callback(node, SimpleNamespace(control_state=phase))
    assert node.started and len(calls) == 1
    assert calls[0][0] == '/world/test/paper_target/start'
    assert isinstance(calls[0][1], Empty)


@pytest.mark.parametrize('response', ((False, None), (True, Boolean(data=False))))
def test_start_requires_gazebo_acknowledgement(response):
    node, calls = _relay(iter((response, (True, Boolean(data=True)))))
    active = SimpleNamespace(control_state='ACTIVE')
    TargetMotionStart._control_callback(node, active)
    assert not node.started
    TargetMotionStart._control_callback(node, active)
    assert node.started and len(calls) == 2
