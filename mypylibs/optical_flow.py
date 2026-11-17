"""Step 2: track the moving object with Lucas-Kanade optical flow."""

import cv2
import numpy as np
import numpy.typing as npt

Points = npt.NDArray[np.float32]  # shape (N, 2): one (x, y) per row


def track_moving_points(prev: cv2.typing.MatLike, curr: cv2.typing.MatLike) -> tuple[Points, Points]:
    """Return where the moving keypoints start (prev) and end (curr). Frames are greyscale."""
    # Only look for corners where pixels changed, otherwise the busy background steals them all.
    changed = (cv2.absdiff(prev, curr) > 15).astype(np.uint8)
    mask = cv2.dilate(changed, np.ones((15, 15), np.uint8))

    start = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=5, mask=mask)
    if start is None:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    end, found, _ = cv2.calcOpticalFlowPyrLK(prev, curr, start, None, winSize=(21, 21), maxLevel=3)
    ok = found.ravel() == 1
    start, end = start.reshape(-1, 2)[ok], end.reshape(-1, 2)[ok]

    moved = np.linalg.norm(end - start, axis=1) > 1  # px/frame
    return start[moved], end[moved]
