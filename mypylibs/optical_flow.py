"""Step 2: track the moving object with Lucas-Kanade optical flow."""

import cv2
import numpy as np
import numpy.typing as npt

Points = npt.NDArray[np.float32]  # shape (N, 2): one (x, y) per row


def track_moving_points(prev: cv2.typing.MatLike, curr: cv2.typing.MatLike, points: Points) -> tuple[Points, Points]:
    """Follow `points` (plus new corners on whatever moved) from prev to curr. Frames are greyscale.

    Returns where the moving points start (prev) and end (curr). Pass `end` back in as `points` next
    frame: following the same points keeps the measured position steady instead of jumping around.
    """
    points = np.vstack([points, new_corners(prev, curr, points)])
    if len(points) == 0:
        return points, points

    end, found = lucas_kanade(prev, curr, points)
    back, found_back = lucas_kanade(curr, prev, end)
    # Forward-backward check: if tracking the point back doesn't land where it started, LK got it wrong
    # (e.g. the point slid along the occluder's edge as the object went behind it).
    reliable = found & found_back & (np.linalg.norm(back - points, axis=1) < 1)
    moving = np.linalg.norm(end - points, axis=1) > 1  # px/frame
    keep = reliable & moving
    return points[keep], end[keep]


def new_corners(prev: cv2.typing.MatLike, curr: cv2.typing.MatLike, existing: Points) -> Points:
    """Corners where pixels changed (i.e. on the moving object), away from points we already follow."""
    # Only look where pixels changed, otherwise the busy background steals all the corners.
    changed = (cv2.absdiff(prev, curr) > 15).astype(np.uint8)
    mask = cv2.dilate(changed, np.ones((15, 15), np.uint8))
    for x, y in existing:
        cv2.circle(mask, (int(x), int(y)), 5, 0, -1)

    corners = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=5, mask=mask)
    return np.empty((0, 2), np.float32) if corners is None else corners.reshape(-1, 2)


def lucas_kanade(a: cv2.typing.MatLike, b: cv2.typing.MatLike, points: Points) -> tuple[Points, npt.NDArray[np.bool_]]:
    """Where `points` in frame a moved to in frame b, and whether LK found each one."""
    moved, found, _ = cv2.calcOpticalFlowPyrLK(a, b, points.reshape(-1, 1, 2), None, winSize=(21, 21), maxLevel=3)
    return moved.reshape(-1, 2), found.ravel() == 1


Group = tuple[Points, Points]  # (position, velocity) of one group of moving points


def group_points(start: Points, end: Points, frame_shape: tuple[int, ...], min_points: int = 5) -> list[Group]:
    """Split the moving points into groups of nearby points (one group per moving thing).

    Each group's position/velocity is the median of its points. Groups smaller than `min_points` are noise.
    """
    height, width = frame_shape[:2]
    inside = (end[:, 0] >= 0) & (end[:, 0] < width) & (end[:, 1] >= 0) & (end[:, 1] < height)
    start, end = start[inside], end[inside]

    canvas = np.zeros((height, width), np.uint8)
    for x, y in end:
        cv2.circle(canvas, (int(x), int(y)), 40, 255, -1)  # points closer than ~80 px merge into one blob
    _, labels = cv2.connectedComponents(canvas)
    ids = labels[end[:, 1].astype(int), end[:, 0].astype(int)]

    groups = []
    for i in np.unique(ids):
        mine = ids == i
        if mine.sum() >= min_points:
            groups.append((np.median(end[mine], axis=0), np.median(end[mine] - start[mine], axis=0)))
    return groups
