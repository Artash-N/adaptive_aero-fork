from __future__ import annotations

import math
import pytest

from src.filters.adaptive_ukf import AdaptiveUKFConfig


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_motor_max_rpm_must_be_positive(value: float):
    with pytest.raises(ValueError, match="motor_max_rpm"):
        AdaptiveUKFConfig(motor_max_rpm=value)


@pytest.mark.parametrize(
    "field_name",
    [
        "pos_process_var",
        "vel_process_var",
        "att_process_var",
    ],
)
def test_base_variances_must_be_non_negative(field_name: str):
    kwargs = {field_name: -1e-6}
    with pytest.raises(ValueError, match=field_name):
        AdaptiveUKFConfig(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    [
        "alpha_pos_var",
        "alpha_vel_var",
        "alpha_att_var",
        "beta_pos_var",
        "beta_vel_var",
        "beta_att_var",
    ],
)
def test_adaptive_gains_must_be_non_negative(field_name: str):
    kwargs = {field_name: -1e-6}
    with pytest.raises(ValueError, match=field_name):
        AdaptiveUKFConfig(**kwargs)


@pytest.mark.parametrize(
    ("cap_name", "base_name"),
    [
        ("max_pos_var", "pos_process_var"),
        ("max_vel_var", "vel_process_var"),
        ("max_att_var", "att_process_var"),
    ],
)
def test_caps_must_be_non_negative_and_not_below_base(cap_name: str, base_name: str):
    with pytest.raises(ValueError, match=cap_name):
        AdaptiveUKFConfig(**{cap_name: -1e-9})

    with pytest.raises(ValueError, match=cap_name):
        AdaptiveUKFConfig(**{base_name: 1e-3, cap_name: 1e-4})


@pytest.mark.parametrize(
    "field_name",
    [
        "pos_process_var",
        "vel_process_var",
        "att_process_var",
        "vio_pos_var",
        "alpha_pos_var",
        "alpha_vel_var",
        "alpha_att_var",
        "beta_pos_var",
        "beta_vel_var",
        "beta_att_var",
        "motor_max_rpm",
        "max_pos_var",
        "max_vel_var",
        "max_att_var",
    ],
)
def test_finite_scalar_validation(field_name: str):
    kwargs = {field_name: math.inf}
    with pytest.raises(ValueError, match=field_name):
        AdaptiveUKFConfig(**kwargs)


def test_can_disable_finite_scalar_validation():
    cfg = AdaptiveUKFConfig(vio_pos_var=math.inf, validate_finite_scalars=False)
    assert math.isinf(cfg.vio_pos_var)
