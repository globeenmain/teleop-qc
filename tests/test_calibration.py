import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from teleop_qc import calibration as cal


def calibrated_sample(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    return p, (rng.uniform(0, 1, n) < p).astype(int)


def test_calibrated_judge_has_small_ece():
    p, y = calibrated_sample()
    assert cal.ece(p, y) < 0.02
    assert cal.ece(p, y, strategy="quantile") < 0.02


def test_overconfident_judge_has_large_ece_that_recalibration_removes():
    p, y = calibrated_sample()
    overconfident = np.where(p > 0.5, 0.99, 0.01)  # right ranking, absurd confidence
    assert cal.ece(overconfident, y) > 0.2
    fixed = cal.cross_fit(overconfident, y, method="isotonic")
    assert cal.ece(fixed, y) < 0.03


def test_platt_scaling_fixes_a_tempered_judge():
    p, y = calibrated_sample()
    logit = np.log(p / (1 - p)) * 3  # too sharp
    sharp = 1 / (1 + np.exp(-logit))
    fixed = cal.cross_fit(sharp, y, method="platt")
    assert cal.ece(fixed, y) < cal.ece(sharp, y) / 3


def test_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(1)
    p = np.round(rng.uniform(0, 1, 500), 1)  # many ties
    y = (rng.uniform(0, 1, 500) < p).astype(int)
    assert cal.auroc(p, y) == pytest.approx(roc_auc_score(y, p))


def test_auroc_is_nan_with_one_class():
    assert np.isnan(cal.auroc([0.2, 0.8], [1, 1]))


def test_kappa_matches_sklearn():
    a = [1, 1, 0, 0, 1, 0, 1, 1]
    b = [1, 0, 0, 0, 1, 1, 1, 1]
    assert cal.cohen_kappa(np.array(a), np.array(b)) == pytest.approx(cohen_kappa_score(a, b))


def test_isotonic_matches_sklearn_on_training_points():
    rng = np.random.default_rng(2)
    p = np.round(rng.uniform(0, 1, 300), 2)
    y = (rng.uniform(0, 1, 300) < p).astype(int)
    ours = cal.fit_isotonic(p, y)(p)
    ref = IsotonicRegression(out_of_bounds="clip").fit(p, y).predict(p)
    assert np.allclose(ours, ref)
    order = np.argsort(p)
    assert np.all(np.diff(ours[order]) >= -1e-12)


def test_selective_threshold_reaches_the_target_and_maximizes_coverage():
    p = np.array([0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
    y = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 1])
    t = cal.selective_threshold(p, y, target_precision=0.8)
    assert t is not None and t.precision >= 0.8
    assert t.value == 0.6 and t.count == 6  # 5 successes out of the 6 episodes with p >= 0.6
    reject = cal.selective_threshold(p, y, target_precision=0.75, accept=False)
    assert reject.value == 0.4 and reject.count == 4  # 3 failures out of the 4 episodes with p <= 0.4
    assert cal.selective_threshold(p, y, target_precision=1.01) is None


def test_bootstrap_interval_contains_the_point_estimate():
    p, y = calibrated_sample(n=2000)
    low, high = cal.bootstrap_ci(cal.auroc, p, y, n=200)
    assert low < cal.auroc(p, y) < high


def test_inputs_are_validated():
    with pytest.raises(ValueError):
        cal.brier([0.5, 1.2], [0, 1])
    with pytest.raises(ValueError):
        cal.brier([0.5], [0, 1])
