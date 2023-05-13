import numpy as np
import pandas as pd


def _user_level_data(fraction_lost, n_samples=1000):
    data = np.zeros((n_samples, 2))
    data[:, 0] = len(fraction_lost)

    for fraction in reversed(fraction_lost):
        age = int(n_samples * fraction)
        data[:age, 0] -= 1

    data[:, 1][data[:, 0] == len(fraction_lost)] = 1

    np.random.shuffle(data)
    return data


def fader_hardie(n_samples, noise_features=0):
    """
    A function to generate the customer level data used in [1]

    :return: pandas DataFrame
    """

    # List of number of customers lost per year per quality level
    highend_lost = np.asarray([0, .131, .257, .347, .407, .449, .483, .509])
    regular_lost = np.asarray([0, .369, .532, .618, .674, .711, .738, .759])

    data_he = _user_level_data(highend_lost, n_samples=n_samples)
    data_re = _user_level_data(regular_lost, n_samples=n_samples)

    data = pd.DataFrame(
        data=np.concatenate((data_he, data_re), axis=0),
        columns=["age", "alive"]
    )

    data["is_high_end"] = \
        np.concatenate((np.ones(n_samples), np.zeros(n_samples)))

    for i in range(noise_features):
        data["noise_feature_{}".format(i)] = np.random.randn(2 * n_samples)

    return data.sample(frac=1).reset_index(drop=True)


# Define parameters of the model!
_params = dict(
    alpha=dict(
        bias=0.2,
        categ={'cat_a': 0.09736, 'cat_b': -0.368, 'cat_c': -1e-2},
        count=0.065,
        numer=0.292
    ),
    beta=dict(
        bias=0.87,
        categ={'cat_a': -0.12, 'cat_b': -0.4425, 'cat_c': 0.69},
        count=-0.148,
        numer=0.021
    )
)


def _simulate_sample_life(alpha, beta, max_age=10):
    """
    A function to simulate the life of a sample given its alpha and beta
    parameters
    """
    age = 1
    alive = 1

    pchurn = np.random.beta(alpha, beta)

    while age < max_age:
        if np.random.random() <= pchurn:
            alive = 0
            break

        age += 1

    return age, alive


def _compute_alpha(row):
    # Get alpha params
    pdict = _params['alpha']

    # Start with bias
    alpha = np.exp(pdict['bias'])

    # add categorical contribution
    alpha *= np.exp(pdict['categ'][row['category']])

    # Add count and numerical contributions
    alpha *= np.exp(pdict['count'] * row['counts'] +
                    pdict['numer'] * row['numerical'])
    # add noise
    alpha *= np.exp(2e-2 * np.random.randn())
    return alpha


def _compute_beta(row):
    # Get beta params
    pdict = _params['beta']

    # Start with bias
    beta = np.exp(pdict['bias'])

    # add categorical contribution
    beta *= np.exp(pdict['categ'][row['category']])

    # Add count and numerical contributions
    beta *= np.exp(pdict['count'] * row['counts'] +
                   pdict['numer'] * row['numerical'])
    # add noise
    beta *= np.exp(1e-2 * np.random.randn())

    return beta


def _compute_age(row):
    max_age = np.random.choice([8, 9, 10, 11], 1, p=[0.3, 0.3, 0.2, 0.2])[0]
    age, alive = _simulate_sample_life(
        alpha=row['alpha_true'],
        beta=row['beta_true'],
        max_age=max_age
    )
    return age, alive


def synthetic_data(n_samples=10000):
    data = pd.DataFrame()
    data['id'] = np.arange(n_samples)

    # add categories
    data['category'] = np.random.choice(['cat_a', 'cat_b', 'cat_c'],
                                        n_samples,
                                        p=[0.47, 0.36, 0.17])

    # transform cotegory type
    data['category'] = data['category'].astype('category')

    # Add counts feature
    data['counts'] = np.random.poisson(lam=0.25, size=n_samples)

    # add numerical gaussian feature
    data['numerical'] = 0.5 * np.random.randn(n_samples) + 1

    # Add true alpha and beta params
    data['alpha_true'] = data.apply(_compute_alpha, axis=1)
    data['beta_true'] = data.apply(_compute_beta, axis=1)

    # Simulate age
    sim = data.apply(_compute_age, axis=1)

    # Update age values
    data['age'] = [s[0] for s in sim]

    # for simplicity we assume all come from same cohort, so it is easy to set
    # alive value
    data['alive'] = [s[1] for s in sim]

    # split in half
    tr = data.iloc[:n_samples//2]
    te = data.iloc[n_samples//2:].reset_index(drop=True)

    return {'train': tr, 'test': te, 'params': _params}
