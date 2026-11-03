import numpy as np
import pandas as pd
import pytest

from teleop_qc import droid


def meta_row(index, success, task="Put the marker in the cup", start=0.0, length=150):
    row = {
        "episode_index": index,
        "tasks": np.array([task], dtype=object),
        "length": length,
        "data/chunk_index": 0,
        "data/file_index": 2,
        "dataset_from_index": 1000 + index,
        "stats/is_episode_successful/mean": np.array([success]),
    }
    for cam in droid.CAMERAS:
        row |= {
            f"videos/observation.images.{cam}/chunk_index": 0,
            f"videos/observation.images.{cam}/file_index": 7,
            f"videos/observation.images.{cam}/from_timestamp": start,
            f"videos/observation.images.{cam}/to_timestamp": start + length / droid.FPS,
        }
    return row


def test_episodes_from_meta_parses_labels_tasks_and_video_spans():
    df = pd.DataFrame([meta_row(0, 1.0), meta_row(1, 0.0, task=""), meta_row(2, 0.5)])
    episodes = droid.episodes_from_meta(df)
    assert [e.index for e in episodes] == [0, 1]  # the ambiguous 0.5 label is skipped
    first = episodes[0]
    assert first.success and first.task == "Put the marker in the cup"
    assert not episodes[1].success and episodes[1].task == ""
    span = first.video("wrist_left")
    assert span.path.endswith("videos/observation.images.wrist_left/chunk-000/file-007.mp4")
    assert first.data_path.endswith("data/chunk-000/file-002.parquet")
    assert first.seconds == pytest.approx(10.0)


def test_timestamps_land_on_frames_and_stay_inside_the_episode():
    span = droid.VideoSpan("wrist_left", 0, 0, start=12.0, end=12.0 + 150 / droid.FPS)
    t = span.timestamps(6)
    assert t[0] == pytest.approx(12.0)
    assert t[-1] == pytest.approx(12.0 + 149 / droid.FPS)
    assert np.allclose(t * droid.FPS, np.round(t * droid.FPS))


def test_sample_balanced_is_balanced_reproducible_and_skips_missing_tasks():
    rows = [meta_row(i, float(i % 3 != 0), task="" if i % 10 == 0 else "Open the drawer") for i in range(90)]
    episodes = droid.episodes_from_meta(pd.DataFrame(rows))
    sample = droid.sample_balanced(episodes, 20, seed=1)
    assert len(sample) == 20 and sum(e.success for e in sample) == 10
    assert all(e.task for e in sample)
    assert [e.index for e in sample] == [e.index for e in droid.sample_balanced(episodes, 20, seed=1)]
    with pytest.raises(ValueError):
        droid.sample_balanced(episodes, 80)


@pytest.mark.network
def test_reads_real_frames_from_the_hub():
    episode = droid.load_episodes(0)[0]
    span = episode.video("exterior_1_left")
    with droid.VideoFile(span.path) as video:
        frames = video.frames_at(span.timestamps(3))
        clip, stamps = video.frames_between(span.start, span.end)
    assert frames.shape == (3, 180, 320, 3)
    assert len(clip) == episode.length and np.allclose(np.diff(stamps), 1 / droid.FPS, atol=1e-3)
    proprio = droid.read_proprio(episode)
    assert len(proprio) == episode.length
