"""The judge report on a known set of verdicts."""

import numpy as np
import pandas as pd
import pytest

from teleop_qc.cli import report


def verdicts(p, labels, model="claude-opus-5-5"):
    return pd.DataFrame({"label": labels, "p_success": p, "model": model, "input_tokens": 2000,
                         "output_tokens": 200, "seconds_of_video": 36.0, "seconds": 5.0})


def test_report_counts_errors_per_class_and_cost():
    # 20 human successes, 4 judged failures; 20 human failures, 10 judged successes; one abstention
    p = [0.8] * 16 + [0.3] * 4 + [0.7] * 10 + [0.2] * 10 + [np.nan]
    labels = [True] * 20 + [False] * 21
    out = report(verdicts(p, labels))
    assert out["episodes"] == 41 and out["abstained"] == 1
    assert out["true_positive_rate"] == pytest.approx(16 / 20)
    assert out["true_negative_rate"] == pytest.approx(10 / 20)
    assert out["balanced_accuracy"] == pytest.approx(0.65)
    assert out["judged_successful"] == pytest.approx(26 / 40)
    # abstentions are paid for too: 41 calls of 2,000 input and 200 output tokens at $4 and $20 per million
    assert out["dollars"] == pytest.approx(41 * (2000 * 4 + 200 * 20) / 1e6)
    assert out["dollars_per_video_hour"] == pytest.approx(out["dollars"] / (41 * 36 / 3600))


def test_balanced_accuracy_equals_accuracy_on_a_balanced_set():
    rng = np.random.default_rng(0)
    labels = np.array([True] * 50 + [False] * 50)
    p = np.clip(np.where(labels, 0.7, 0.4) + rng.normal(0, 0.2, 100), 0, 1)
    out = report(verdicts(p, labels))
    assert out["balanced_accuracy"] == pytest.approx(out["accuracy"])
