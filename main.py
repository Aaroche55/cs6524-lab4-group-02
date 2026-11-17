from pathlib import Path

import cv2
import numpy as np

from mypylibs.kalman import KalmanTracker, estimate_measurement_noise
from mypylibs.optical_flow import Points, track_moving_points

MIN_POINTS = 5  # fewer moving keypoints than this = no measurement (e.g. object hidden)


def optical_flow(video: Path) -> list[tuple[Points, Points] | None]:
    """Write out/XX_of.mp4 and return (position, velocity) per frame, or None if not measured."""
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(f"out/{video.stem}_of.mp4", cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), (w, h))

    _, frame = cap.read()
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    measurements: list[tuple[Points, Points] | None] = [None]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start, end = track_moving_points(prev, curr)

        for a, b in zip(start, end):
            cv2.arrowedLine(frame, a.astype(int), (a + 3 * (b - a)).astype(int), (0, 255, 0), 1)  # green = keypoint flow (3x longer)

        if len(start) >= MIN_POINTS:
            position, velocity = np.median(end, axis=0), np.median(end - start, axis=0)
            cv2.circle(frame, position.astype(int), 8, (0, 0, 255), -1)  # red = object
            measurements.append((position, velocity))
        else:
            measurements.append(None)

        out.write(frame)
        prev = curr

    cap.release()
    out.release()
    return measurements


def kalman(measurements: list[tuple[Points, Points] | None]) -> list[np.ndarray | None]:
    """Filtered [px, py, vx, vy] per frame (None before the first measurement)."""
    sigma_pos, sigma_vel = estimate_measurement_noise(measurements)
    first = next(i for i, m in enumerate(measurements) if m is not None)
    kf = KalmanTracker(measurements[first], sigma_pos, sigma_vel)
    states: list[np.ndarray | None] = [None] * first + [kf.state]
    for m in measurements[first + 1:]:
        states.append(kf.step(m))
    return states


if __name__ == "__main__":
    for video in sorted(Path("in").glob("*.mp4")):
        measurements = optical_flow(video)
        states = kalman(measurements)
        print(f"{video.name}: measured {sum(m is not None for m in measurements)}/{len(measurements)} frames, "
              f"final KF state {np.round(states[-1], 1)}")
