from pathlib import Path

import cv2
import numpy as np

from mypylibs.kalman import KalmanTracker, estimate_measurement_noise
from mypylibs.optical_flow import Group, Points, group_points, track_moving_points

MAX_TURN_DEGREES = 45  # the objects move in straight lines, so anything heading another way isn't the object
LOST_AFTER = 3  # frames without the object before we drop the anchor and re-acquire it from scratch
SEARCH_RADIUS = 150  # px: a group further than this from where the anchor should be is something else
PULL = 0.1  # fraction of the anchor-to-centroid gap closed each frame, so the anchor can't drift off the object


def optical_flow(video: Path) -> tuple[list[tuple[Points, Points] | None], list[Points | None]]:
    """Write out/XX_of.mp4 and return, per frame, the measured (position, velocity) or None.

    The measured position is a single anchor point that moves with the object's median flow, instead
    of the median of its keypoints: those keep joining/leaving the group, which made the median jump
    by up to 240 px between frames. The anchor is pulled slightly (PULL) towards that median so it
    doesn't drift away. Also returns the raw median (centroid) per frame, to estimate position noise.
    """
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        f"out/{video.stem}_of.mp4", cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), (w, h)
    )

    _, frame = cap.read()
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    measurements: list[tuple[Points, Points] | None] = [None]
    centroids: list[Points | None] = [None]
    points = np.empty((0, 2), np.float32)
    anchor: Points | None = None
    velocity: Points | None = None  # last measured velocity, kept across occlusions
    missing = 0  # frames since the object was last measured

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        start, end = track_moving_points(prev, curr, points)
        points = end

        for a, b in zip(start, end):
            cv2.arrowedLine(
                frame, a.astype(int), (a + 3 * (b - a)).astype(int), (0, 255, 0), 1
            )  # green = keypoint flow (3x longer)

        groups = group_points(start, end, frame.shape)
        if missing >= LOST_AFTER:
            anchor = None  # lost (e.g. behind the occluder): re-acquire when it reappears
        if anchor is None:
            obj = pick_object(groups, velocity)
            if obj is not None:
                anchor = obj[0].copy()
        else:
            obj = nearest_object(groups, anchor + velocity * (missing + 1), velocity)
            if obj is not None:
                anchor = anchor + velocity * missing + obj[1]  # coast over missed frames, then follow the flow
                anchor = anchor + PULL * (obj[0] - anchor)

        if obj is None:
            missing += 1
            measurements.append(None)
            centroids.append(None)
        else:
            missing = 0
            velocity = obj[1]
            measurements.append((anchor.copy(), velocity.copy()))
            centroids.append(obj[0])
            cv2.circle(frame, anchor.astype(int), 8, (0, 0, 255), -1)  # red = object (anchor)

        out.write(frame)
        prev = curr

    cap.release()
    out.release()
    return measurements, centroids


def pick_object(groups: list[Group], last_velocity: Points | None) -> Group | None:
    """The fastest group heading the same way the object last did.

    Ignores other moving things (the hand pulling back in video 03) and LK glitches at the occluder's
    edges (points sliding up/down the edge), since those move in a different direction.
    """
    if last_velocity is not None:
        groups = [g for g in groups if angle_between(g[1], last_velocity) < MAX_TURN_DEGREES]
    return max(groups, key=lambda g: np.linalg.norm(g[1]), default=None)


def nearest_object(groups: list[Group], expected: Points, velocity: Points) -> Group | None:
    """Once we follow the object: the group closest to where it should be, heading the same way."""
    groups = [
        g
        for g in groups
        if np.linalg.norm(g[0] - expected) < SEARCH_RADIUS and angle_between(g[1], velocity) < MAX_TURN_DEGREES
    ]
    return min(groups, key=lambda g: np.linalg.norm(g[0] - expected), default=None)


def angle_between(a: Points, b: Points) -> float:
    cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def kalman(measurements: list[tuple[Points, Points] | None], centroids: list[Points | None]) -> list[np.ndarray | None]:
    """Filtered [px, py, vx, vy] per frame (None before the first measurement)."""
    sigma_pos, sigma_vel = estimate_measurement_noise(measurements, centroids)
    first = next(i for i, m in enumerate(measurements) if m is not None)
    kf = KalmanTracker(measurements[first], sigma_pos, sigma_vel)
    states: list[np.ndarray | None] = [None] * first + [kf.state]
    for m in measurements[first + 1 :]:
        states.append(kf.step(m))
    return states


def reappearance(measurements: list[tuple[Points, Points] | None]) -> int:
    best, r, length = 0, 0, 0
    for i, m in enumerate(measurements):
        if m is None:
            length += 1
        else:
            if length > best and i - length > 0:
                best, r = length, i
            length = 0
    return r


def track(measurements: list[tuple[Points, Points] | None], centroids: list[Points | None]):
    sigma_pos, sigma_vel = estimate_measurement_noise(measurements, centroids)
    first = next(i for i, m in enumerate(measurements) if m is not None)
    r = reappearance(measurements)
    kf = KalmanTracker(measurements[first], sigma_pos, sigma_vel)
    states: list[np.ndarray | None] = [None] * first + [kf.state]
    predicted = None
    ghost = np.empty((0, 4))
    for i in range(first + 1, len(measurements)):
        kf.predict()
        if i == r:
            predicted = kf.state
            ghost = np.vstack([predicted, kf.forecast(len(measurements) - r - 1)])
        if measurements[i] is not None:
            kf.correct(measurements[i])
        states.append(kf.state)
    measured = [m is not None for m in measurements]
    if predicted is None:  # object never disappeared and came back (e.g. 04.mp4): nothing to compare
        return states, measured, len(measurements), ghost, float("nan"), float("nan")
    error =np.linalg.norm(predicted[:2] - measurements[r][0])
    real = np.array([s[:2] for s in states[r:]])
    drift = np.linalg.norm(ghost[-1, :2] - real[-1])
    return states, measured, r, ghost, error, drift


BLUE, RED, PINK = (255, 0, 0), (0, 0, 255), (255, 0, 255)  # BGR


def plot_trajectories(
    video: Path,
    states: list[np.ndarray | None],
    measurements: list[tuple[Points, Points] | None],
    r: int,
    ghost: np.ndarray,
) -> None:
    """Write out/XX_tr.mp4: the object's trail so far, coloured by where each point came from."""
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        f"out/{video.stem}_tr.mp4", cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS), (w, h)
    )

    last_seen = max(i for i, m in enumerate(measurements) if m is not None)  # object stops moving after this
    for i in range(len(states)):
        _, frame = cap.read()
        for j in range(min(i, last_seen) + 1):  # trail up to the current frame
            if states[j] is not None:
                cv2.circle(frame, states[j][:2].astype(int), 3, BLUE if measurements[j] is not None else RED, -1)
                if j % 10 == 0:  # velocity arrow every 10 frames (5x longer to be visible)
                    p, v = states[j][:2], states[j][2:]
                    cv2.arrowedLine(frame, p.astype(int), (p + 5 * v).astype(int), (255, 255, 255), 2)
            if j >= r:
                cv2.circle(frame, ghost[j - r][:2].astype(int), 3, PINK, -1)

        legend = [("seen", BLUE), ("hidden: filter's guess", RED), ("filter's guess if it had stayed hidden", PINK)]
        for k, (text, colour) in enumerate(legend):  # small, bottom-left, out of the way
            y = h - 15 - 25 * (len(legend) - 1 - k)
            cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        out.write(frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    for video in sorted(Path("in").glob("*.mp4")):
        measurements, centroids = optical_flow(video)
        states = kalman(measurements, centroids)
        print(
            f"{video.name}: measured {sum(m is not None for m in measurements)}/{len(measurements)} frames, "
            f"final KF state {np.round(states[-1], 1)}"
        )
        tracked_states, _, r, ghost, error, drift = track(measurements, centroids)
        print(f"{video.name}: reappears at frame {r}, error {error:.1f} px, ghost drift at end {drift:.1f} px")
        plot_trajectories(video, tracked_states, measurements, r, ghost)
