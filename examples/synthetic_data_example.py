import pandas as pd
from axia import SBGSurvival
from axia.dataset import synthetic_data


def main():
    # simulate data
    data = synthetic_data(50000)
    train, test, _ = data['train'], data['test'], data['params']

    # sneak peak
    print("This is what the data looks like.")
    print(train.head())
    print()

    # Look at obvious correlations
    print("This is how the true parameters differ among categories")
    print(train.groupby('category').mean().drop('id', axis=1))
    print()

    print("This is how the true parameters differ among counts")
    print(train.groupby('counts').mean().drop('id', axis=1))
    print()

    print('There is some correlation with the numerical variable too.')
    print(train[['numerical', 'alpha_true', 'beta_true', 'age', 'alive']].corr())
    print()

    # START MODELING
    # Create the sbs object using all features. Lets keep gamma small and let
    # the model "overfit" if necessary. We have enough data.
    sbs = SBGSurvival(
        age='age',
        alive='alive',
        features=['category', 'counts', 'numerical'],
        gamma=1e-6,
        verbose=True
    )

    # Train model
    sbs.fit(train)

    # Summary of results
    print(sbs.summary())
    print()

    # Make some predictions
    pred = pd.concat([test,
                      sbs.predict_params(test)], axis=1)

    print(
        "Mean Percentage Error for Alpha: " +
        "{}".format(
            (
                (pred['alpha_true'] - pred['alpha']).abs() /
                pred['alpha_true']
            ).mean()
        )
    )

    print(
        "Mean Percentage Error for Beta: " +
        "{}".format(
            (
                (pred['beta_true'] - pred['beta']).abs() /
                pred['beta_true']
            ).mean()
        )
    )
    print()

    # correlation between true and predicitons
    print("Predictons better be correlated with true values.")
    print(pred[['alpha_true', 'alpha']].corr())
    print(pred[['beta_true', 'beta']].corr())

    # Done
    print("Not bad.")


if __name__ == '__main__':
    main()
