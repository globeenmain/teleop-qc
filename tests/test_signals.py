"""Fault injection: each check must find the fault we plant in an otherwise clean synthetic episode."""

import numpy as np

from teleop_qc import signals as sig

FPS = 15.0


def synthetic_episode(n=150, seed=0):
    """A smooth scene panned sideways, frame by frame, at the speed the robot's joints report."""
    rng = np.random.default_rng(seed)
    speed = np.abs(np.sin(np.linspace(0, 6 * np.pi, n))) * 3 + rng.uniform(0.2, 0.6, n)  # pixels per frame
    offsets = np.cumsum(speed)
    x = np.arange(160)[None, :] + offsets[:, None]  # sub-pixel pan: no two frames are identical
    rows = 1 + 0.2 * np.sin(np.arange(120) / 9.0)[None, :, None]
    gray = 127 + 50 * np.sin(2 * np.pi * x / 97) + 35 * np.sin(2 * np.pi * x / 41 + 1.0)
    frames = np.repeat((gray[:, None, :] * rows)[..., None], 3, axis=3).clip(0, 255).astype(np.uint8)
    q = offsets[:, None] * np.array([[0.01, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]])
    return frames, np.arange(n) / FPS, q


def test_clean_episode_raises_no_flag():
    frames, stamps, q = synthetic_episode()
    report = sig.stream_report("cam", frames, stamps, sig.joint_speed(q), FPS, expected_frames=len(frames))
    assert report.flags() == []
    assert report.lag_frames == 0 and report.lag_corr > 0.8


def test_camera_running_late_is_flagged_as_desync():
    frames, stamps, q = synthetic_episode()
    late = np.concatenate([np.repeat(frames[:1], 3, axis=0), frames[:-3]])  # 3 frames behind the encoders
    report = sig.stream_report("cam", late, stamps, sig.joint_speed(q), FPS, expected_frames=len(frames))
    assert report.lag_frames == 3
    assert "desync" in report.flags()


def test_frozen_camera_is_found():
    frames, stamps, q = synthetic_episode()
    frames[40:46] = frames[40]  # the camera repeats one image for 6 frames
    runs = sig.frozen_runs(sig.motion_energy(frames))
    assert (40, 46) in runs
    report = sig.stream_report("cam", frames, stamps, sig.joint_speed(q), FPS, expected_frames=len(frames))
    assert report.frozen_frames == 5 and "frozen" in report.flags()


def test_dropped_repeated_and_reversed_timestamps_are_found():
    # frame numbers: 5 and 6 dropped, 9 repeated, then a step back from 11 to 10 and a jump to 12
    stamps = np.array([0, 1, 2, 3, 4, 7, 8, 9, 9, 10, 11, 10, 12, 13]) / FPS
    issues = sig.timestamp_issues(stamps, FPS)
    assert issues == {"gaps": [5, 12], "repeats": [8], "reversals": [11]}


def test_dark_frames_and_frame_count_are_flagged():
    frames, stamps, q = synthetic_episode()
    frames[10:12] = 0
    report = sig.stream_report("cam", frames, stamps, sig.joint_speed(q), FPS, expected_frames=len(frames) + 4)
    assert report.dark_frames == 2
    assert {"dark", "frame_count"} <= set(report.flags())


def test_estimate_lag_sign_convention():
    rng = np.random.default_rng(3)
    a = rng.normal(size=200)
    b = np.roll(a, 4)  # b shows each value 4 samples after a
    assert sig.estimate_lag(a, b, max_lag=8).frames == 4
    assert sig.estimate_lag(b, a, max_lag=8).frames == -4


def test_subframe_lag_recovers_a_fractional_delay():
    t = np.arange(300.0)
    a = np.sin(t / 7) + 0.5 * np.sin(t / 3.1)
    b = np.sin((t - 2.4) / 7) + 0.5 * np.sin((t - 2.4) / 3.1)  # 2.4 samples late
    lag = sig.estimate_lag(a, b, max_lag=8)
    assert lag.frames == 2 and abs(lag.subframe - 2.4) < 0.2 and not lag.at_edge


def test_estimate_lag_on_flat_signal_says_nothing():
    lag = sig.estimate_lag(np.ones(100), np.arange(100.0), max_lag=5)
    assert lag.frames == 0 and np.isnan(lag.corr)


def test_desync_is_judged_against_the_cameras_usual_lag():
    frames, stamps, q = synthetic_episode()
    late = np.concatenate([np.repeat(frames[:1], 3, axis=0), frames[:-3]])
    report = sig.stream_report("cam", late, stamps, sig.joint_speed(q), FPS, expected_frames=len(frames))
    assert report.desync(offset=0.0)  # 3 frames off a camera that is usually on time
    assert not report.desync(offset=2.5)  # but normal for a camera that usually runs 2.5 frames late
    assert not report.desync(offset=0.0, min_corr=0.999)  # and not claimed without a clear correlation peak
