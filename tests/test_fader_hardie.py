import argparse
import pytest
import pandas as pd
from axia import SBGSurvival
from axia.dataset import fader_hardie


def _percent_diff(val1, val2):
    return abs(val1 - val2) / val2


def test_basic_model():
    data = fader_hardie(n_samples=5000, noise_features=0)

    sbs = SBGSurvival(
        age="age", alive="alive", features=["is_high_end"], gamma=1e-6, verbose=True
    )
    sbs.fit(data)
    pred = pd.concat([data, sbs.predict_params(data)], axis=1)
    means = pred.groupby("is_high_end").mean()

    print(means)
    assert _percent_diff(means.loc[0, "alpha"], 0.704072) < 5e-2
    assert _percent_diff(means.loc[1, "alpha"], 0.668621) < 5e-2

    assert _percent_diff(means.loc[0, "beta"], 1.182035) < 5e-2
    assert _percent_diff(means.loc[1, "beta"], 3.810440) < 5e-2

    assert _percent_diff(means.loc[0, "alive"], 0.241) < 5e-2
    assert _percent_diff(means.loc[1, "alive"], 0.491) < 5e-2


def test_noisy_model_1():
    data = fader_hardie(n_samples=5000, noise_features=1)

    sbs = SBGSurvival(
        age="age",
        alive="alive",
        features=[c for c in data.columns if c not in ["age", "alive"]],
        gamma=1e-6,
        verbose=True,
    )
    sbs.fit(data)
    pred = pd.concat([data, sbs.predict_params(data)], axis=1)
    means = pred.groupby("is_high_end").mean()

    print(means)
    assert _percent_diff(means.loc[0, "alpha"], 0.704072) < 5e-2
    assert _percent_diff(means.loc[1, "alpha"], 0.668621) < 5e-2

    assert _percent_diff(means.loc[0, "beta"], 1.182035) < 5e-2
    assert _percent_diff(means.loc[1, "beta"], 3.810440) < 5e-2

    assert _percent_diff(means.loc[0, "alive"], 0.241) < 5e-2
    assert _percent_diff(means.loc[1, "alive"], 0.491) < 5e-2


def test_noisy_model_10():
    data = fader_hardie(n_samples=5000, noise_features=10)

    sbs = SBGSurvival(
        age="age",
        alive="alive",
        features=[c for c in data.columns if c not in ["age", "alive"]],
        gamma=1e-6,
        verbose=True,
    )
    sbs.fit(data)
    pred = pd.concat([data, sbs.predict_params(data)], axis=1)
    means = pred.groupby("is_high_end").mean()

    print(means)
    assert _percent_diff(means.loc[0, "alpha"], 0.704072) < 5e-2
    assert _percent_diff(means.loc[1, "alpha"], 0.668621) < 5e-2

    assert _percent_diff(means.loc[0, "beta"], 1.182035) < 5e-2
    assert _percent_diff(means.loc[1, "beta"], 3.810440) < 5e-2

    assert _percent_diff(means.loc[0, "alive"], 0.241) < 5e-2
    assert _percent_diff(means.loc[1, "alive"], 0.491) < 5e-2


if __name__ == "__main__":
    pytest.main([__file__])
