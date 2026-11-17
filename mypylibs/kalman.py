"""Step 3: constant-velocity Kalman filter fed with optical-flow position + velocity.

State       x = [px, py, vx, vy]  (pixels, pixels/frame)
Measurement z = [px, py, vx, vy]  (anchor point moved by the object's median flow, and that flow; see main.py)
Time step   dt = 1 frame

Usage:
    sigma_pos, sigma_vel = estimate_measurement_noise(measurements, centroids)
    kf = KalmanTracker(first_measurement, sigma_pos, sigma_vel)
    for z in later_measurements:          # z is (position, velocity) or None
        state = kf.step(z)                # predict, + correct when z is not None
    future = kf.forecast(n)               # what-if: n more frames with no evidence
"""

from typing import Sequence

import cv2
import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float32]
Measurement = tuple[Vec, Vec] | None  # (position, velocity) from optical flow, None = not measured

SIGMA_ACC = 0.1  # px/frame^2, std of the unmodelled acceleration (process noise), tuned on the 3 videos
MIN_PAIRS = 10  # consecutive measured frames needed to estimate R from the video itself
DEFAULT_SIGMA_POS, DEFAULT_SIGMA_VEL = 10.0, 0.5  # px, px/frame


def estimate_measurement_noise(
    measurements: Sequence[Measurement], centroids: Sequence[Vec | None]
) -> tuple[Vec, Vec]:
    """Std of the OF position and velocity measurement noise, estimated from the video itself.

    Velocity: if it were perfect, v_t = v_{t-1} (constant velocity). Each residual is the difference
    of two independent noisy measurements, so its std is ~sqrt(2) * sigma.
    Position: the measured position is an anchor moved by the flow, so p_t - p_{t-1} - v_t is ~0 and
    says nothing. Instead we use how far the anchor sits from the raw keypoint centroid of the object.
    MAD instead of std, because a few keypoint glitches (100 px jumps) would otherwise dominate.
    """
    pairs = [(a, b) for a, b in zip(measurements, measurements[1:]) if a is not None and b is not None]
    offsets = [c - m[0] for m, c in zip(measurements, centroids) if m is not None and c is not None]
    if len(pairs) < MIN_PAIRS:  # too few to estimate (e.g. 04.mp4): fall back to typical values of 01-03
        return np.full(2, DEFAULT_SIGMA_POS, np.float32), np.full(2, DEFAULT_SIGMA_VEL, np.float32)
    res_vel = np.array([b[1] - a[1] for a, b in pairs])

    def robust_std(r: np.ndarray) -> Vec:
        return (1.4826 * np.median(np.abs(r - np.median(r, axis=0)), axis=0)).astype(np.float32)

    return robust_std(np.array(offsets)), robust_std(res_vel) / np.float32(np.sqrt(2))


class KalmanTracker:
    """Thin wrapper around cv2.KalmanFilter that exposes K and P, and can forecast without evidence."""

    def __init__(self, first: tuple[Vec, Vec], sigma_pos: Vec, sigma_vel: Vec, sigma_acc: float = SIGMA_ACC):
        kf = cv2.KalmanFilter(4, 4)  # 4 state vars, 4 measured vars, no control input (G disregarded)

        # F: constant velocity, p_{t+1} = p_t + v_t, v_{t+1} = v_t
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)

        # H: OF measures the whole state directly
        kf.measurementMatrix = np.eye(4, dtype=np.float32)

        # R: diagonal, from the OF noise of this video (x/y and position/velocity treated as independent)
        kf.measurementNoiseCov = np.diag(np.r_[sigma_pos, sigma_vel] ** 2).astype(np.float32)

        # Q: discrete white-noise acceleration. An unknown acceleration a ~ N(0, sigma_acc^2) acting over
        # one frame shifts position by a/2 and velocity by a, so Q = sigma_acc^2 * g g^T with g = [1/2, 1].
        q = sigma_acc**2 * np.array([[1 / 4, 1 / 2], [1 / 2, 1]])
        kf.processNoiseCov = np.kron(q, np.eye(2)).astype(np.float32)  # same model for x and y, uncorrelated

        # x0, P0: start at the first measurement, exactly as uncertain as that measurement is (P0 = R)
        kf.statePost = np.r_[first[0], first[1]].astype(np.float32).reshape(4, 1)
        kf.errorCovPost = kf.measurementNoiseCov.copy()

        self.kf = kf

    @property
    def state(self) -> Vec:
        """Current estimate [px, py, vx, vy]."""
        return self.kf.statePost.ravel().copy()

    @property
    def std(self) -> Vec:
        """Current 1-sigma uncertainty of [px, py, vx, vy] (sqrt of P's diagonal)."""
        return np.sqrt(np.diag(self.kf.errorCovPost)).copy()

    @property
    def gain(self) -> Vec:
        """K from the last correction (4x4): how far each state component moves towards each measurement."""
        return self.kf.gain.copy()

    def predict(self) -> Vec:
        """Advance one frame without evidence: x = F x, P = F P F^T + Q."""
        self.kf.predict()
        # cv2 only writes statePre/errorCovPre; copy them into *Post so that predict-only steps chain
        # (otherwise a 2nd predict() in a row would start again from the last corrected state).
        self.kf.statePost = self.kf.statePre.copy()
        self.kf.errorCovPost = self.kf.errorCovPre.copy()
        return self.state

    def correct(self, measurement: tuple[Vec, Vec]) -> Vec:
        """Fuse a measurement into the prediction (call predict() first).

        K = P H^T (H P H^T + R)^-1;  x = x + K (z - H x);  P = (I - K H) P
        """
        self.kf.correct(np.r_[measurement[0], measurement[1]].astype(np.float32).reshape(4, 1))
        return self.state

    def step(self, measurement: Measurement) -> Vec:
        """One frame: predict, then correct if there is a measurement."""
        self.predict()
        return self.state if measurement is None else self.correct(measurement)

    def forecast(self, n: int) -> Vec:
        """(n, 4) states the filter alone expects over the next n frames. Does not modify the tracker.

        Use it when the object reappears to get the "what if it had not come back" trajectory.
        """
        F, x, out = self.kf.transitionMatrix, self.kf.statePost.copy(), []
        for _ in range(n):
            x = F @ x
            out.append(x.ravel())
        return np.array(out)
