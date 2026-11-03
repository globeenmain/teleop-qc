"""Signal-level checks on one episode, before any model looks at it.

- timestamps: gaps (dropped frames), repeats and reversals against the nominal frame rate;
- frames: frozen runs (the camera stalls and repeats an image), dark frames, sharpness;
- alignment: the lag between what a camera sees move and what the robot's joint encoders report, estimated by
  cross-correlation. It is the idea behind camera-IMU time calibration: a camera that runs k frames late shows
  the arm's motion k frames after the encoders do.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def luma(frames: np.ndarray) -> np.ndarray:
    """(n, h, w, 3) uint8 RGB frames -> (n, h, w) float32 luma (Rec. 601)."""
    return frames.astype(np.float32) @ _LUMA


def brightness(frames: np.ndarray) -> np.ndarray:
    return luma(frames).mean(axis=(1, 2))


def motion_energy(frames: np.ndarray, stride: int = 2) -> np.ndarray:
    """Mean absolute luma change between consecutive frames: n frames give n - 1 values."""
    y = luma(frames[:, ::stride, ::stride])
    return np.abs(np.diff(y, axis=0)).mean(axis=(1, 2))


def sharpness(frames: np.ndarray) -> np.ndarray:
    """Variance of the Laplacian of each frame: low values mean blur or a featureless image."""
    y = luma(frames)
    lap = -4 * y[:, 1:-1, 1:-1] + y[:, :-2, 1:-1] + y[:, 2:, 1:-1] + y[:, 1:-1, :-2] + y[:, 1:-1, 2:]
    return lap.var(axis=(1, 2))


def joint_speed(q: np.ndarray) -> np.ndarray:
    """Norm of the joint-position change between consecutive frames: n states give n - 1 values."""
    return np.linalg.norm(np.diff(np.asarray(q, dtype=np.float64), axis=0), axis=1)


def runs(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """[start, end) index ranges where mask holds for at least min_len consecutive samples."""
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True) if b - a >= min_len]


def frozen_runs(motion: np.ndarray, threshold: float = 0.25, min_len: int = 2) -> list[tuple[int, int]]:
    """Frame ranges [start, end) over which the image does not change.

    motion[i] compares frames i and i + 1, so min_len near-zero differences in a row cover min_len + 1 frames.
    """
    return [(a, b + 1) for a, b in runs(np.asarray(motion) < threshold, min_len)]


def timestamp_issues(stamps: np.ndarray, fps: float) -> dict[str, list[int]]:
    """Indices i where the step from frame i - 1 to frame i is a gap (> 1.5 frames), a repeat or goes back."""
    step = np.diff(np.asarray(stamps, dtype=np.float64)) * fps
    return {
        "gaps": [int(i) + 1 for i in np.flatnonzero(step > 1.5)],
        "repeats": [int(i) + 1 for i in np.flatnonzero(np.abs(step) < 0.5)],
        "reversals": [int(i) + 1 for i in np.flatnonzero(step < -0.5)],
    }


class Lag(NamedTuple):
    frames: int  # integer lag with the best correlation
    corr: float  # correlation at that lag
    subframe: float  # refined by a parabola through the peak and its two neighbours
    at_edge: bool  # the best lag sits on the edge of the search window: no real peak was found


def estimate_lag(a: np.ndarray, b: np.ndarray, max_lag: int, min_overlap: int = 20) -> Lag:
    """Lag L in samples that maximizes the Pearson correlation of a[t] and b[t + L].

    L > 0 means b shows each event L samples after a does (b lags a). Returns a zero lag with a NaN correlation
    when the signals are too short or flat to say anything.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        x, y = (a[: n - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[: n + lag])
        if len(x) >= min_overlap and x.std() > 0 and y.std() > 0:
            corr[i] = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr).all():
        return Lag(0, float("nan"), 0.0, False)
    i = int(np.nanargmax(corr))
    subframe = float(lags[i])
    if 0 < i < len(lags) - 1 and np.isfinite(corr[i - 1]) and np.isfinite(corr[i + 1]):
        curvature = corr[i - 1] - 2 * corr[i] + corr[i + 1]
        if curvature < 0:
            subframe += 0.5 * (corr[i - 1] - corr[i + 1]) / curvature
    return Lag(int(lags[i]), float(corr[i]), float(subframe), i in (0, len(lags) - 1))


@dataclass(frozen=True)
class StreamReport:
    """What the checks found on one camera stream of one episode."""

    camera: str
    frames: int
    expected_frames: int
    timestamp_gaps: int
    timestamp_repeats: int
    timestamp_reversals: int
    frozen_frames: int  # frames that repeat the previous image
    dark_frames: int
    median_sharpness: float
    lag_frames: int  # positive: the camera shows motion after the joint encoders report it
    lag_subframe: float
    lag_corr: float  # correlation at that lag; low values mean the arm's motion is barely visible
    lag_at_edge: bool

    def desync(self, offset: float = 0.0, tolerance: float = 2.0, min_corr: float = 0.5) -> bool:
        """The camera is off by at least `tolerance` frames from its usual `offset`, and the evidence is clear:
        a real correlation peak, inside the search window."""
        return (not self.lag_at_edge and self.lag_corr >= min_corr
                and abs(self.lag_subframe - offset) >= tolerance)

    def flags(self, offset: float = 0.0) -> list[str]:
        found = []
        if self.frames != self.expected_frames:
            found.append("frame_count")
        if self.timestamp_gaps or self.timestamp_repeats or self.timestamp_reversals:
            found.append("timestamps")
        if self.frozen_frames:
            found.append("frozen")
        if self.dark_frames:
            found.append("dark")
        if self.desync(offset):
            found.append("desync")
        return found

    def as_dict(self) -> dict:
        return {**asdict(self), "flags": ",".join(self.flags())}


def stream_report(camera: str, frames: np.ndarray, stamps: np.ndarray, robot_speed: np.ndarray, fps: float,
                  expected_frames: int, max_lag: int = 8, dark_threshold: float = 12.0) -> StreamReport:
    motion = motion_energy(frames)
    issues = timestamp_issues(stamps, fps)
    lag = estimate_lag(robot_speed, motion, max_lag)
    return StreamReport(
        camera=camera,
        frames=len(frames),
        expected_frames=expected_frames,
        timestamp_gaps=len(issues["gaps"]),
        timestamp_repeats=len(issues["repeats"]),
        timestamp_reversals=len(issues["reversals"]),
        frozen_frames=sum(b - a - 1 for a, b in frozen_runs(motion)),
        dark_frames=int((brightness(frames) < dark_threshold).sum()),
        median_sharpness=float(np.median(sharpness(frames))),
        lag_frames=lag.frames,
        lag_subframe=lag.subframe,
        lag_corr=lag.corr,
        lag_at_edge=lag.at_edge,
    )
