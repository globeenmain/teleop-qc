"""DROID episodes on the Hugging Face Hub (LeRobot v3.0 layout), read without downloading whole files.

DROID 1.0.1 packs hundreds of episodes into each parquet and MP4 file: one video file per camera holds about
300 episodes and weighs about 500 MB. The per-episode metadata says which file holds an episode and where its
frames start, so we open the files through fsspec (byte-range requests) and let PyAV seek to the frames we need.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Self

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO_ID = "lerobot/droid_1.0.1"
FPS = 15
CAMERAS = ("exterior_1_left", "exterior_2_left", "wrist_left")
_ROOT = f"datasets/{REPO_ID}"
_VIDEO_FIELDS = ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
META_COLUMNS = [
    "episode_index",
    "tasks",
    "length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "stats/is_episode_successful/mean",
    *(f"videos/observation.images.{cam}/{field}" for cam in CAMERAS for field in _VIDEO_FIELDS),
]


@dataclass(frozen=True)
class VideoSpan:
    """Where one camera stream of an episode lives: a video file of the dataset and a time range inside it."""

    camera: str
    chunk: int
    file: int
    start: float  # seconds from the start of the file
    end: float

    @property
    def path(self) -> str:
        return f"{_ROOT}/videos/observation.images.{self.camera}/chunk-{self.chunk:03d}/file-{self.file:03d}.mp4"

    def timestamps(self, k: int) -> np.ndarray:
        """k frame timestamps (file time) spread evenly over the episode, first and last frames included."""
        last = self.end - 1 / FPS
        return np.round(np.linspace(self.start, last, k) * FPS) / FPS


@dataclass(frozen=True)
class Episode:
    index: int
    task: str  # language instruction given to the teleoperator, "" when missing
    length: int  # frames
    success: bool  # label recorded by the human who collected the episode
    data_chunk: int
    data_file: int
    first_frame: int  # global index of the episode's first frame in the data files
    videos: tuple[VideoSpan, ...]

    def video(self, camera: str) -> VideoSpan:
        return next(v for v in self.videos if v.camera == camera)

    @property
    def data_path(self) -> str:
        return f"{_ROOT}/data/chunk-{self.data_chunk:03d}/file-{self.data_file:03d}.parquet"

    @property
    def seconds(self) -> float:
        return self.length / FPS


def _scalar(value) -> float:
    return float(np.asarray(value).ravel()[0])


def episodes_from_meta(df: pd.DataFrame) -> list[Episode]:
    """Episodes described by one parquet file of meta/episodes; rows without a clean 0/1 label are skipped."""
    episodes = []
    for row in df[META_COLUMNS].to_dict("records"):
        label = _scalar(row["stats/is_episode_successful/mean"])
        if label not in (0.0, 1.0):
            continue
        tasks = list(row["tasks"])
        videos = tuple(
            VideoSpan(
                camera=cam,
                chunk=int(row[f"videos/observation.images.{cam}/chunk_index"]),
                file=int(row[f"videos/observation.images.{cam}/file_index"]),
                start=float(row[f"videos/observation.images.{cam}/from_timestamp"]),
                end=float(row[f"videos/observation.images.{cam}/to_timestamp"]),
            )
            for cam in CAMERAS
        )
        episodes.append(
            Episode(
                index=int(row["episode_index"]),
                task=str(tasks[0]).strip() if tasks else "",
                length=int(row["length"]),
                success=label == 1.0,
                data_chunk=int(row["data/chunk_index"]),
                data_file=int(row["data/file_index"]),
                first_frame=int(row["dataset_from_index"]),
                videos=videos,
            )
        )
    return episodes


def load_episodes(meta_file: int = 0, fs: HfFileSystem | None = None) -> list[Episode]:
    """Episodes listed in meta/episodes/chunk-000/file-<meta_file>.parquet (about 14,000 per file)."""
    fs = fs or HfFileSystem()
    with fs.open(f"{_ROOT}/meta/episodes/chunk-000/file-{meta_file:03d}.parquet", "rb") as f:
        return episodes_from_meta(pd.read_parquet(f, columns=META_COLUMNS))


def sample_balanced(episodes: list[Episode], n: int, seed: int = 0, require_task: bool = True) -> list[Episode]:
    """n episodes drawn at random, half labelled successful and half failed, sorted by episode index."""
    pool = [e for e in episodes if e.task or not require_task]
    successes = [e for e in pool if e.success]
    failures = [e for e in pool if not e.success]
    n_fail = n // 2
    if len(failures) < n_fail or len(successes) < n - n_fail:
        raise ValueError(f"not enough episodes: {len(successes)} successes, {len(failures)} failures for n={n}")
    rng = random.Random(seed)
    return sorted(rng.sample(successes, n - n_fail) + rng.sample(failures, n_fail), key=lambda e: e.index)


def read_proprio(episode: Episode, columns: tuple[str, ...] = ("observation.state.joint_position",),
                 fs: HfFileSystem | None = None) -> pd.DataFrame:
    """Per-frame robot state of one episode, read from its data file (only the matching row groups)."""
    return read_proprio_many(episode.data_path, [episode.index], columns, fs)


def read_proprio_many(data_path: str, episode_indices: list[int],
                      columns: tuple[str, ...] = ("observation.state.joint_position",),
                      fs: HfFileSystem | None = None) -> pd.DataFrame:
    """Per-frame robot state of several episodes stored in the same data file, in one read."""
    fs = fs or HfFileSystem()
    with fs.open(data_path, "rb") as f:
        table = pq.read_table(
            f, columns=["episode_index", "frame_index", "timestamp", *columns],
            filters=[("episode_index", "in", list(episode_indices))],
        )
    return table.to_pandas().sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


class VideoFile:
    """One MP4 of the dataset, opened once and read by seeking. Frames come back as RGB uint8 arrays."""

    def __init__(self, path: str, fs: HfFileSystem | None = None, block_size: int = 1 << 20):
        self._file = (fs or HfFileSystem()).open(path, "rb", block_size=block_size)
        self._container = av.open(self._file)
        self._stream = self._container.streams.video[0]
        self._tol = 0.5 / FPS

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._container.close()
        self._file.close()

    def _seek(self, t: float) -> None:
        self._container.seek(int(t / self._stream.time_base), stream=self._stream, backward=True)

    def frames_at(self, timestamps) -> np.ndarray:
        """The frame shown at each timestamp (seconds, file time)."""
        frames = []
        for t in timestamps:
            self._seek(t)
            for frame in self._container.decode(self._stream):
                if frame.pts is not None and float(frame.pts * self._stream.time_base) >= t - self._tol:
                    frames.append(frame.to_ndarray(format="rgb24"))
                    break
            else:
                raise ValueError(f"no frame at {t:.3f}s in {self._file.path}")
        return np.stack(frames)

    def frames_between(self, start: float, end: float) -> tuple[np.ndarray, np.ndarray]:
        """Every frame with start <= pts < end, with its timestamp, decoded in one pass after a single seek."""
        self._seek(start)
        frames, stamps = [], []
        for frame in self._container.decode(self._stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * self._stream.time_base)
            if t < start - self._tol:
                continue
            if t >= end - self._tol:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
            stamps.append(t)
        if not frames:
            raise ValueError(f"no frame between {start:.3f}s and {end:.3f}s in {self._file.path}")
        return np.stack(frames), np.asarray(stamps)
