"""Experimental object-independent Cartesian integral feedback.

The SONIC physical ablation did not validate this compensator; it is disabled
by default. Its bounded kinematic unit test is not a full-body success claim.
"""
import numpy as np

from .kinematics import G1Kinematics


class CartesianTracker:
    def __init__(self, sonic_root, reference_frame, tool_offsets, gain=1.0, max_offset=.04):
        self.desired = G1Kinematics(sonic_root, reference_frame, tool_offsets)
        self.measured = G1Kinematics(sonic_root, reference_frame, tool_offsets)
        self.solver = G1Kinematics(sonic_root, reference_frame, tool_offsets)
        self.gain, self.max_offset = gain, max_offset
        self.integral = np.zeros((2, 3))
        self.bias = np.zeros(29)
        self.elapsed = 0.0
        self.rejections = 0
        self.max_error = 0.0

    def update(self, nominal_q, measured_q, dt):
        self.elapsed += min(dt, .1)
        if self.elapsed >= .1:
            targets = self.desired.forward(nominal_q)
            actual = self.measured.forward(measured_q)
            error = np.array([t[:3, 3]-a[:3, 3] for t, a in zip(targets, actual)])
            self.max_error = max(self.max_error, float(np.linalg.norm(error, axis=1).max()))
            self.integral = np.clip(self.integral+self.gain*self.elapsed*error,
                                    -self.max_offset, self.max_offset)
            for t, offset in zip(targets, self.integral):
                t[:3, 3] += offset
            try:
                self.bias = self.solver.solve(targets, nominal_q, position_tolerance=.01)-nominal_q
            except ValueError:
                self.integral *= .5
                self.bias[:] = 0
                self.rejections += 1
            self.elapsed = 0.0
        # Compensation remains subject to the same URDF and packet rate limits.
        return np.clip(nominal_q+self.bias, self.solver.bounds[:, 0], self.solver.bounds[:, 1])
