import pandas as pd
from generate_data import make_raw_article_data
from axia import SBGSurvival


def basic_model():
    data = make_raw_article_data()

    print(data.head())

    # START MODELING
    # Create the sbs object using all features. Lets keep gamma small and let
    # the model "overfit" if necessary. We have enough data.
    sbs = SBGSurvival(age='age',
                              alive='alive',
                              features=['is_high_end'],
                              gamma=1e-6,
                              verbose=True)

    sbs.fit(data)

    print(sbs.summary())

    pred = pd.concat([data, sbs.predict_params(data)], axis=1)

    print(pred.head())
    print(pred.groupby('is_high_end').mean().drop('id', axis=1))


def add_random():

    data = make_raw_article_data()
    # Create the sbs object using all features. Lets keep gamma small and let
    # the model "overfit" if necessary. We have enough data.
    sbs = SBGSurvival(age='age',
                              alive='alive',
                              features=['is_high_end', 'random'],
                              gamma=1e-1,
                              verbose=True)

    sbs.fit(data)

    print(sbs.summary())

    pred = pd.concat([data, sbs.predict_params(data)], axis=1)

    print(pred.head())
    print(pred.groupby('is_high_end').mean().drop('id', axis=1))


def main():
    basic_model()
    add_random()


if __name__ == '__main__':
    main()
