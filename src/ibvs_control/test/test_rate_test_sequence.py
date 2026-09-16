import pytest

from ibvs_control.rate_test_sequence import RateTestConfig, RateTestSequence


def test_all_signed_axis_responses_pass() -> None:
    """Matching measured FRD rates pass all six direction checks."""
    sequence = RateTestSequence(RateTestConfig())

    for index in range(1000):
        now_s = index * 0.02
        command = sequence.command(now_s)
        sequence.observe(command, now_s)

    assert sequence.complete
    assert sequence.passed
    assert len(sequence.results) == 6
    assert {result.name for result in sequence.results} == {
        'ROLL_POSITIVE',
        'ROLL_NEGATIVE',
        'PITCH_POSITIVE',
        'PITCH_NEGATIVE',
        'YAW_POSITIVE',
        'YAW_NEGATIVE',
    }
    assert all(result.samples >= 3 for result in sequence.results)


def test_wrong_response_sign_fails() -> None:
    """A measured response opposite to the command is reported explicitly."""
    config = RateTestConfig(
        prepare_duration_s=0.1,
        pulse_duration_s=0.2,
        settle_duration_s=0.1,
        response_ignore_s=0.0,
        minimum_samples=1,
    )
    sequence = RateTestSequence(config)

    sequence.command(0.0)
    command = sequence.command(0.1)
    sequence.observe(tuple(-value for value in command), 0.11)
    sequence.command(0.31)

    result = sequence.pop_results()[0]
    assert result.name == 'ROLL_POSITIVE'
    assert not result.passed
    assert result.reason == 'wrong_sign'


def test_position_control_is_requested_between_signed_pulses() -> None:
    """Every rate pulse is followed by an explicit hover-recovery segment."""
    config = RateTestConfig(
        prepare_duration_s=0.1,
        pulse_duration_s=0.2,
        settle_duration_s=0.3,
        response_ignore_s=0.0,
        minimum_samples=1,
    )
    sequence = RateTestSequence(config)

    sequence.command(0.0)
    assert sequence.current_name == 'PREPARE'
    assert sequence.uses_position_control

    sequence.command(0.1)
    assert sequence.current_name == 'ROLL_POSITIVE'
    assert not sequence.uses_position_control


def test_recovery_waits_for_external_stability_confirmation() -> None:
    """Elapsed recovery time alone must not start the next rate pulse."""
    config = RateTestConfig(
        prepare_duration_s=0.1,
        pulse_duration_s=0.2,
        settle_duration_s=0.3,
        response_ignore_s=0.0,
        minimum_samples=1,
    )
    sequence = RateTestSequence(config)

    sequence.command(0.0)
    command = sequence.command(0.1)
    sequence.observe(command, 0.11)
    sequence.command(0.31)
    assert sequence.current_name == 'ROLL_POSITIVE_RECOVER'

    assert sequence.command(1.0, recovery_ready=False) == (0.0, 0.0, 0.0)
    assert sequence.current_name == 'ROLL_POSITIVE_RECOVER'

    sequence.command(1.01, recovery_ready=True)
    assert sequence.current_name == 'ROLL_NEGATIVE'
    assert not sequence.uses_position_control


def test_missing_measurements_fail() -> None:
    """A pulse without enough angular-velocity samples cannot pass."""
    config = RateTestConfig(
        prepare_duration_s=0.1,
        pulse_duration_s=0.2,
        settle_duration_s=0.1,
        response_ignore_s=0.0,
    )
    sequence = RateTestSequence(config)

    sequence.command(0.0)
    sequence.command(0.1)
    sequence.command(0.31)

    result = sequence.pop_results()[0]
    assert not result.passed
    assert result.reason == 'insufficient_samples'


def test_unsafe_rate_is_rejected() -> None:
    """The P1 test refuses aggressive angular-rate configurations."""
    with pytest.raises(ValueError):
        RateTestSequence(RateTestConfig(roll_rate_rad_s=0.6))
