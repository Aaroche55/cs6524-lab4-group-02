"""Regenerate the report figures in assets/latex/. Run after main.py: python -m mypylibs.report_figures"""

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402
from mypylibs.kalman import KalmanTracker, estimate_measurement_noise  # noqa: E402

OUT = Path("assets/latex")
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"


def frame(path: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    _, img = cap.read()
    cap.release()
    return img


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
    cv2.putText(img, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return img


def fig_bias() -> None:
    """01_of.mp4 as the orange enters (frame 30) and leaves (frame 66) the occluder."""
    enter = frame("out/01_of.mp4", 30)[0:420, 760:1180]
    leave = frame("out/01_of.mp4", 66)[0:420, 120:540]
    cv2.imwrite(str(OUT / "fig_bias.png"), np.hstack([label(enter, "t=30"), label(leave, "t=66")]))


def fig_tracks() -> None:
    """Trajectory overlays after reappearance (01 and 03)."""
    a = frame("out/01_tr.mp4", 90)[60:420]
    b = frame("out/03_tr.mp4", 130)[160:520]
    cv2.imwrite(str(OUT / "fig_tracks.png"), cv2.resize(np.vstack([label(a, "01, t=90"), label(b, "03, t=130")]), None, fx=0.6, fy=0.6))


def run(measurements, centroids):
    """Same loop as main.track, also keeping the 1-sigma position uncertainty."""
    sp, sv = estimate_measurement_noise(measurements, centroids)
    first = next(i for i, m in enumerate(measurements) if m is not None)
    r = main.reappearance(measurements)
    kf = KalmanTracker(measurements[first], sp, sv)
    xs, sd, ghost = {first: kf.state}, {first: kf.std}, None
    for i in range(first + 1, len(measurements)):
        kf.predict()
        if i == r:
            ghost = np.vstack([kf.state, kf.forecast(len(measurements) - r - 1)])
        if measurements[i] is not None:
            kf.correct(measurements[i])
        xs[i], sd[i] = kf.state, kf.std
    return xs, sd, ghost, r


def fig_xt(all_measurements) -> None:
    """x(t): measurements, filter estimate (+-2 sigma), and the no-reappearance forecast."""
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.2), sharex=False)
    for ax, (name, (m, centroids)) in zip(axes, all_measurements.items()):
        xs, sd, ghost, r = run(m, centroids)
        t = np.array(sorted(xs))
        x = np.array([xs[i][0] for i in t])
        s = np.array([sd[i][0] for i in t])
        seen = np.array([m[i] is not None for i in t])
        last_before = max(i for i in range(r) if m[i] is not None)
        ax.axvspan(last_before + 0.5, r - 0.5, color="#e6e6e3", lw=0)
        ax.fill_between(t, x - 2 * s, x + 2 * s, color=BLUE, alpha=0.18, lw=0)
        ax.plot(t, x, color=BLUE, lw=1.2, label="KF estimate (±2σ)")
        ax.plot(t[~seen], x[~seen], ".", color=ORANGE, ms=2.5, label="KF, prediction only")
        mt = [i for i in t if m[i] is not None]
        ax.plot(mt, [m[i][0][0] for i in mt], ".", color=GREY, ms=2.5, label="OF measurement")
        gt = np.arange(r, r + len(ghost))
        keep = (ghost[:, 0] > -200) & (ghost[:, 0] < 1500)
        ax.plot(gt[keep], ghost[keep, 0], "--", color=AQUA, lw=1.2, label="forecast if not back")
        ax.set_ylim(-50, 1350)
        ax.set_ylabel(f"{name}: x [px]", fontsize=7)
        ax.tick_params(labelsize=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e6e6e3", lw=0.5)
    axes[-1].set_xlabel("frame", fontsize=7)
    axes[0].legend(fontsize=5.5, loc="upper right", frameon=False, ncol=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_xt.png", dpi=300)


if __name__ == "__main__":
    measurements = {v.stem: main.optical_flow(v) for v in sorted(Path("in").glob("0[1-3].mp4"))}
    fig_bias()
    fig_tracks()
    fig_xt(measurements)
