import math
import os
import sys

import pytest

# Add tests folder and project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sequence_packer import MaskingRatioCooldownScheduler, MaskingSchedule


@pytest.mark.parametrize(
    "initial, final, cooldown, start",
    [
        (0.3, 0.1, 100, 20),
        (0.5, 0.0, 50, 0),
        (1.0, 0.5, 10, 5),
    ],
)
def test_linear_masking_ratio_cooldown(initial, final, cooldown, start):
    sched = MaskingRatioCooldownScheduler(
        initial_masking_ratio=initial,
        final_masking_ratio=final,
        cooldown_tokens=cooldown,
        cooldown_start_tokens=start,
        schedule_type=MaskingSchedule.LINEAR,
    )
    # Before start: current_tokens < start should yield initial
    if start > 0:
        for t in [0, start // 2, start - 1]:
            assert sched(t) == pytest.approx(initial)
    # At start
    assert sched(start) == pytest.approx(initial)
    # Midpoint of cooldown
    mid = start + cooldown // 2
    expected_mid = initial - ((mid - start) / cooldown) * (initial - final)
    assert sched(mid) == pytest.approx(expected_mid)
    # At and after end of cooldown
    for t in [start + cooldown, start + cooldown + 1]:
        assert sched(t) == pytest.approx(final)


@pytest.mark.parametrize(
    "initial, final, cooldown, step_size",
    [
        (0.6, 0.0, 6, 0.2),
        (0.5, 0.2, 9, 0.1),
    ],
)
def test_stair_masking_ratio_cooldown(initial, final, cooldown, step_size):
    sched = MaskingRatioCooldownScheduler(
        initial_masking_ratio=initial,
        final_masking_ratio=final,
        cooldown_tokens=cooldown,
        cooldown_start_tokens=0,
        schedule_type=MaskingSchedule.STAIR,
        masking_ratio_step_size=step_size,
    )
    num_steps = math.ceil((initial - final) / step_size) if initial > final else 0
    # Test within cooldown range
    for t in range(cooldown):
        progress = t / cooldown
        step = math.floor(progress * num_steps) if num_steps > 0 else 0
        expected = max(initial - step * step_size, final)
        assert sched(t) == pytest.approx(expected)
    # At and after end of cooldown
    for t in [cooldown, cooldown + 1]:
        assert sched(t) == pytest.approx(final)


@pytest.mark.parametrize(
    "kwargs,error_msg",
    [
        # Stair without step size
        (
            {
                "initial_masking_ratio": 0.5,
                "final_masking_ratio": 0.2,
                "cooldown_tokens": 10,
                "schedule_type": MaskingSchedule.STAIR,
            },
            "must be provided",
        ),
        # Negative step size
        (
            {
                "initial_masking_ratio": 0.5,
                "final_masking_ratio": 0.2,
                "cooldown_tokens": 10,
                "schedule_type": MaskingSchedule.STAIR,
                "masking_ratio_step_size": -0.1,
            },
            "must be positive",
        ),
        # Cooldown tokens < 1
        ({"initial_masking_ratio": 0.5, "final_masking_ratio": 0.2, "cooldown_tokens": 0}, "greater than 1"),
    ],
)
def test_invalid_masking_ratio_cooldown_configs(kwargs, error_msg):
    # Expect ValueError for invalid configurations
    with pytest.raises(ValueError) as exc:
        MaskingRatioCooldownScheduler(**kwargs)
    assert error_msg in str(exc.value)
