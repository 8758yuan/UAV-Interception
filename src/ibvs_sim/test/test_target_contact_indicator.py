"""Regression tests for the target hit visual confirmation."""

from types import SimpleNamespace

import pytest
from gz.msgs10.serialized_map_pb2 import SerializedStepMap

from ibvs_sim.target_contact_indicator import (
    _green_visual_message,
    _has_expected_target_contact,
    _state_has_green_visual_command,
)


def test_green_request_identifies_visual_by_child_and_parent_names() -> None:
    request = _green_visual_message('target_visual', 'target_link')
    assert request.name == 'target_visual'
    assert request.parent_name == 'target_link'
    assert request.material.diffuse.g == pytest.approx(1.0)


def test_green_request_rejects_unsafe_names() -> None:
    with pytest.raises(ValueError):
        _green_visual_message('', 'target_link')


def test_state_confirmation_requires_applied_green_command() -> None:
    state = SerializedStepMap()
    component = state.state.entities[1].components[2]
    red = _green_visual_message('target_visual', 'target_link')
    red.material.diffuse.r, red.material.diffuse.g = 1.0, 0.0
    red.material.ambient.r, red.material.ambient.g = 1.0, 0.0
    component.component = red.SerializeToString()
    assert not _state_has_green_visual_command(
        state,
        'target_visual',
        'target_link',
    )
    component.component = _green_visual_message(
        'target_visual',
        'target_link',
    ).SerializeToString()
    assert _state_has_green_visual_command(
        state,
        'target_visual',
        'target_link',
    )


def test_indicator_only_accepts_expected_vehicle_contact() -> None:
    target_vehicle = SimpleNamespace(
        collision1=SimpleNamespace(name='target'),
        collision2=SimpleNamespace(name='vehicle::rotor'),
    )
    target_ground = SimpleNamespace(
        collision1=SimpleNamespace(name='target'),
        collision2=SimpleNamespace(name='ground::collision'),
    )
    assert _has_expected_target_contact(
        SimpleNamespace(contacts=[target_vehicle]),
        'target',
        'vehicle::',
    )
    assert not _has_expected_target_contact(
        SimpleNamespace(contacts=[target_ground]),
        'target',
        'vehicle::',
    )
