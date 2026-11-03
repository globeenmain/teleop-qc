"""Runs the checks over many episodes, opening each data or video file once and reading episodes in file order."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from itertools import groupby

import numpy as np
import pandas as pd
from huggingface_hub import HfFileSystem

from .droid import CAMERAS, FPS, Episode, VideoFile, read_proprio_many
from .signals import StreamReport, joint_speed, stream_report


def robot_speeds(episodes: Iterable[Episode], fs: HfFileSystem) -> dict[int, np.ndarray]:
    """Joint speed per frame for each episode, from the measured joint positions."""
    speeds = {}
    ordered = sorted(episodes, key=lambda e: e.data_path)
    for path, group in groupby(ordered, key=lambda e: e.data_path):
        group = list(group)
        table = read_proprio_many(path, [e.index for e in group], fs=fs)
        for index, rows in table.groupby("episode_index"):
            speeds[int(index)] = joint_speed(np.stack(rows["observation.state.joint_position"].to_numpy()))
    return speeds


def camera_offsets(reports: list[StreamReport], min_corr: float = 0.5) -> dict[str, float]:
    """Each camera's usual lag behind the joint encoders: the median over streams with a clear peak."""
    offsets = {}
    for camera in {r.camera for r in reports}:
        lags = [r.lag_subframe for r in reports
                if r.camera == camera and r.lag_corr >= min_corr and not r.lag_at_edge]
        offsets[camera] = float(np.median(lags)) if lags else 0.0
    return offsets


def run_signals(episodes: list[Episode], fs: HfFileSystem | None = None,
                log: Callable[[str], None] = print) -> tuple[pd.DataFrame, dict[str, float]]:
    """One row per (episode, camera) with the signal checks and decoding time, and each camera's usual lag.

    Desync is judged against the camera's own usual lag, estimated on the same run: a camera that always runs one
    frame late is a calibration constant, not a fault; an episode where it runs four frames late is a fault.
    """
    fs = fs or HfFileSystem()
    speeds = robot_speeds(episodes, fs)
    found: list[tuple[Episode, StreamReport, float]] = []
    for camera in CAMERAS:
        ordered = sorted(episodes, key=lambda e: (e.video(camera).path, e.video(camera).start))
        for path, group in groupby(ordered, key=lambda e: e.video(camera).path):
            with VideoFile(path, fs) as video:
                for episode in group:
                    span = episode.video(camera)
                    started = time.perf_counter()
                    frames, stamps = video.frames_between(span.start, span.end)
                    seconds = time.perf_counter() - started
                    report = stream_report(camera, frames, stamps - span.start, speeds[episode.index], FPS,
                                           expected_frames=episode.length)
                    found.append((episode, report, seconds))
            log(f"{camera}: {path.rsplit('/', 2)[-2]}/{path.rsplit('/', 1)[-1]} done ({len(found)} streams)")
    offsets = camera_offsets([r for _, r, _ in found])
    rows = [{"episode_index": e.index, "success": e.success, "task": e.task, **r.as_dict(),
             "flags": ",".join(r.flags(offsets[r.camera])), "decode_seconds": s} for e, r, s in found]
    return pd.DataFrame(rows), offsets


def signals_summary(df: pd.DataFrame, offsets: dict[str, float], min_corr: float = 0.3) -> pd.DataFrame:
    """Per camera: share of episodes with each flag, the camera's usual lag, how often the arm's motion is visible,
    and decoding throughput."""
    def share(flags: pd.Series, flag: str) -> float:
        return float(flags.apply(lambda found: flag in found).mean())

    out = []
    for camera, g in df.groupby("camera", sort=False):
        visible = g[g["lag_corr"] >= min_corr]
        flags = g["flags"].str.split(",")
        out.append({
            "camera": camera,
            "episodes": len(g),
            "frame_count_mismatch": share(flags, "frame_count"),
            "timestamp_issues": share(flags, "timestamps"),
            "frozen": share(flags, "frozen"),
            "dark": share(flags, "dark"),
            "desync": share(flags, "desync"),
            "usual_lag_frames": offsets.get(camera, float("nan")),
            "motion_visible": len(visible) / len(g),
            "median_corr": float(g["lag_corr"].median()),
            "decode_fps": float(g["frames"].sum() / g["decode_seconds"].sum()),
        })
    return pd.DataFrame(out)
