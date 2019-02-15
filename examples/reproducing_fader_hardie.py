import argparse
import pandas as pd
from axia import SBGSurvival
from axia.dataset import fader_hardie


def basic_model(n_samples=5000):
    data = fader_hardie(n_samples=n_samples, noise_features=0)

    sbs = SBGSurvival(
        age='age',
        alive='alive',
        features=['is_high_end'],
        gamma=1e-6,
        verbose=True
    )
    sbs.fit(data)
    print(sbs.summary())

    pred = pd.concat([data, sbs.predict_params(data)], axis=1)
    print(pred.head())
    print(pred.groupby('is_high_end').mean())


def noisy_model(n_samples=5000, noise_features=1):
    data = fader_hardie(n_samples=n_samples, noise_features=noise_features)

    sbs = SBGSurvival(
        age='age',
        alive='alive',
        features=[c for c in data.columns if c not in ["age", "alive"]],
        gamma=1e-1,
        verbose=True
    )
    sbs.fit(data)
    print(sbs.summary())

    pred = pd.concat([data, sbs.predict_params(data)], axis=1)
    print(pred.head())
    print(pred.groupby('is_high_end').mean())


def main():
    basic_model()
    noisy_model(n_samples=10000, noise_features=10)
    noisy_model(n_samples=100000, noise_features=100)


if __name__ == '__main__':
    main()
