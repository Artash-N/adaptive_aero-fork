from __future__ import annotations

from dataclasses import dataclass, fields
import math
import numpy as np

from src.filters.ukf_core import (
    UnscentedKalmanFilter,
    DroneStateSpaceModel,
    make_process_noise,
    make_position_measurement_noise,
)


@dataclass(slots=True)
class AdaptiveUKFConfig:
    # Base process noise floor
    pos_process_var: float = 3e-5
    vel_process_var: float = 2e-2
    att_process_var: float = 3e-5

    # Keep VIO measurement noise fixed
    vio_pos_var: float = 0.05 ** 2

    # Adaptive scaling for PROCESS noise Q
    # Q_k = Q_floor + alpha * normalized_sum_rpm_sq + beta * ||specific_force||
    alpha_pos_var: float = 2e-4
    alpha_vel_var: float = 5e-2
    alpha_att_var: float = 2e-4

    beta_pos_var: float = 1e-4
    beta_vel_var: float = 2e-2
    beta_att_var: float = 1e-4

    motor_max_rpm: float = 12000.0

    # Caps so Q does not explode
    max_pos_var: float = 5e-3
    max_vel_var: float = 5e-1
    max_att_var: float = 5e-3

    validate_finite_scalars: bool = True

    def __post_init__(self) -> None:
        if self.motor_max_rpm <= 0.0:
            raise ValueError("motor_max_rpm must be > 0")

        base_vars = {
            "pos_process_var": self.pos_process_var,
            "vel_process_var": self.vel_process_var,
            "att_process_var": self.att_process_var,
        }
        adaptive_gains = {
            "alpha_pos_var": self.alpha_pos_var,
            "alpha_vel_var": self.alpha_vel_var,
            "alpha_att_var": self.alpha_att_var,
            "beta_pos_var": self.beta_pos_var,
            "beta_vel_var": self.beta_vel_var,
            "beta_att_var": self.beta_att_var,
        }
        caps = {
            "max_pos_var": (self.max_pos_var, self.pos_process_var),
            "max_vel_var": (self.max_vel_var, self.vel_process_var),
            "max_att_var": (self.max_att_var, self.att_process_var),
        }

        for name, value in base_vars.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

        for name, value in adaptive_gains.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

        for name, (cap_value, base_value) in caps.items():
            if cap_value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            if cap_value < base_value:
                raise ValueError(f"{name} must be >= corresponding base variance")

        if self.validate_finite_scalars:
            for field in fields(self):
                value = getattr(self, field.name)
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError(f"{field.name} must be finite")

    def rpm_sq_norm(self, rpm_sq_sum: float) -> float:
        return float(rpm_sq_sum / (4.0 * self.motor_max_rpm ** 2))


class AdaptiveUKF:
    def __init__(self, cfg: AdaptiveUKFConfig | None = None) -> None:
        self.cfg = cfg or AdaptiveUKFConfig()
        self.model = DroneStateSpaceModel()
        self.ukf = UnscentedKalmanFilter(self.model)

        # Fixed VIO measurement covariance
        self.R = make_position_measurement_noise(self.cfg.vio_pos_var)

        # For logging/debugging
        self.last_pos_var: float = self.cfg.pos_process_var
        self.last_vel_var: float = self.cfg.vel_process_var
        self.last_att_var: float = self.cfg.att_process_var

    def initialize(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self.ukf.initialize(x0, P0)

    def compute_Q(
        self,
        rpm_sq_sum: float,
        specific_force_mag_mps2: float,
    ) -> np.ndarray:
        rpm_term = self.cfg.rpm_sq_norm(rpm_sq_sum)
        g_term = float(specific_force_mag_mps2)

        pos_var = (
            self.cfg.pos_process_var
            + self.cfg.alpha_pos_var * rpm_term
            + self.cfg.beta_pos_var * g_term
        )
        vel_var = (
            self.cfg.vel_process_var
            + self.cfg.alpha_vel_var * rpm_term
            + self.cfg.beta_vel_var * g_term
        )
        att_var = (
            self.cfg.att_process_var
            + self.cfg.alpha_att_var * rpm_term
            + self.cfg.beta_att_var * g_term
        )

        pos_var = float(np.clip(pos_var, self.cfg.pos_process_var, self.cfg.max_pos_var))
        vel_var = float(np.clip(vel_var, self.cfg.vel_process_var, self.cfg.max_vel_var))
        att_var = float(np.clip(att_var, self.cfg.att_process_var, self.cfg.max_att_var))

        self.last_pos_var = pos_var
        self.last_vel_var = vel_var
        self.last_att_var = att_var

        return make_process_noise(
            pos_var=pos_var,
            vel_var=vel_var,
            att_var=att_var,
        )

    def predict(
        self,
        imu_accel_mps2: np.ndarray,
        imu_gyro_radps: np.ndarray,
        dt: float,
        rpm_sq_sum: float,
        specific_force_mag_mps2: float,
    ) -> None:
        u = np.concatenate([imu_accel_mps2, imu_gyro_radps]).astype(float)
        Q = self.compute_Q(
            rpm_sq_sum=rpm_sq_sum,
            specific_force_mag_mps2=specific_force_mag_mps2,
        )
        self.ukf.predict(u=u, dt=dt, Q=Q)

    def update_vio(self, vio_pos_w_m: np.ndarray) -> None:
        self.ukf.update(z=np.asarray(vio_pos_w_m, dtype=float), R=self.R)

    def state(self) -> np.ndarray:
        return self.ukf.state.x.copy()

    def covariance(self) -> np.ndarray:
        return self.ukf.state.P.copy()
