import numpy as np
import pandas as pd
from datetime import date, datetime
from axia.util import SubscriptionData


test_data = pd.DataFrame(
    data=[
        [date(2018, 4, 1), np.nan, 10, 10, "b"], # id = 3
        [date(2018, 7, 1), np.nan, 20, 10, "c"], # id = 5
        [date(2018, 11, 1), np.nan, 10, 20, "a"], # id = 21
        [date(2018, 3, 1), np.nan, 10, 20, "b"], # id = 82
        [date(2018, 12, 1), np.nan, 10, 30, "c"], # id = 12
        [date(2018, 9, 1), np.nan, 10, 10, "b"], # id = 973
        [date(2018, 5, 1), date(2018, 11, 23), 30, 30, "b"], # id = 218
        [date(2018, 3, 12), date(2018, 10, 2), 10, 10, "a"], # id = 312
        [date(2018, 9, 12), date(2018, 12, 2), 10, 30, "c"], # id = 188
        [date(2018, 6, 12), date(2018, 12, 2), 20, 30, "c"], # id = 62
        [date(2018, 6, 12), date(2018, 9, 2), 20, 20, "b"], # id = 89
        [date(2018, 5, 12), date(2019, 1, 2), 10, 10, "a"], # id = 11
        [date(2018, 5, 17), date(2018, 5, 22), 20, 20, "c"], # id = 782
        [date(2019, 2, 17), date(2019, 2, 22), 20, 20, "c"], # id = 1001
        [date(2019, 2, 27), np.nan, 30, 30, "b"], # id = 1002
        [date(2018, 7, 7), np.nan, 0, 10, "a"], # id = 1003
        [date(2018, 11, 13), date(2018, 11, 13), 20, 20, "a"], # id = 1004
        [date(2018, 10, 3), date(2018, 10, 3), 20, 30, "a"], # id = 1005
    ],
    columns=[
        "start_date",
        "end_date",
        "subscription_initial",
        "subscription_current",
        "category",
    ],
    index=pd.Series(
        [
            3, 5, 21, 82, 12, 973, 218, 312, 188, 62, 89, 11, 782,
            1001, 1002, 1003, 1004, 1005,
        ],
        name="account_id"
    ),
)
test_data["start_date"] = pd.to_datetime(test_data["start_date"])
test_data["end_date"] = pd.to_datetime(test_data["end_date"])


def _calculate_age(end, start):
    now = datetime.now()
    end_dates = end.fillna(now)

    return np.maximum(12 * (end_dates.dt.year - start.dt.year) + (end_dates.dt.month - start.dt.month), 1)


def test_tabular_data():
    data = SubscriptionData(
        data=test_data,
        subscription_initial="subscription_initial",
        subscription_current="subscription_current",
        additional_cols=[
            "category",
            pd.Series(test_data["category"] + "_", name="category_"),
        ],
    )

    assert all(_calculate_age(data.df["end_date"], data.df["start_date"]) == data.df["age"])
    assert all(data.df["alive"] * data.df["subscription_current"] == data.df["subscription_current"])
    assert all((data.df["alive"] == 0) == (data.df["subscription_current"] == 0))
    assert all(data.df["category"] == test_data["category"])
    assert all(data.df["category_"] == test_data["category"] + "_")

def test_cross_data():
    data = SubscriptionData(
        data=test_data,
        subscription_initial="subscription_initial",
        subscription_current="subscription_current",
    )

    assert data.cdf.loc[3].index.min() >= data.cdf.loc[3]["start_date"].min()
    assert all(
        data.cdf.reset_index("end_of_month").apply(
            lambda row: int(pd.isnull(row["end_date"]) or (row["end_date"] > row["end_of_month"]))
             == row["alive"], axis=1)
    )

    assert data.cdf.loc[(3, "2018-11-30")]["months_alive"] == 8
    assert data.cdf.loc[(3, "2019-02-28")]["months_alive"] == 11
    assert data.cdf.loc[(312, "2018-11-30")]["months_alive"] == 7
    assert data.cdf.loc[(312, "2019-01-31")]["months_alive"] == 7
    assert data.cdf.loc[(11, "2019-01-31")]["is_cancelation_month"] == 1
    assert data.cdf.loc[(782, "2018-05-31")]["is_starting_month"] == 1
    assert data.cdf.loc[(782, "2018-05-31")]["is_cancelation_month"] == 1

    for value_type in ["base", "linear"]:
        assert (
            data.cdf.loc[(12, "2019-01-31")]["value_to_date_{}".format(value_type)] <
            data.cdf.loc[(12, "2019-02-28")]["value_to_date_{}".format(value_type)]
        )
        assert (
            data.cdf.loc[(89, "2019-01-31")]["value_to_date_{}".format(value_type)] ==
            data.cdf.loc[(89, "2019-02-28")]["value_to_date_{}".format(value_type)]
        )

    assert data.cdf.loc[(973, "2018-09-30")]["value_to_date_linear"] == 10
    assert data.cdf.loc[(973, "2018-10-31")]["value_to_date_linear"] == 20
    assert (
        data.cdf.loc[(188, "2018-09-30")]["subscription_value_linear"] ==
        test_data.loc[188, "subscription_initial"]
    )
    assert (
        data.cdf.loc[(188, "2018-11-30")]["subscription_value_linear"] ==
        test_data.loc[188, "subscription_current"]
    )

    # print(data.cdf.columns)
    # print(data.cdf.loc[3].iloc[:, :4])
    # print(data.cdf.loc[3].iloc[:, 4:])
    # print(data.cdf.loc[11].iloc[:, :4])
    # print(data.cdf.loc[11].iloc[:, 4:])
    # print(data.cdf.loc[188].iloc[:, :4])
    # print(data.cdf.loc[188].iloc[:, 4:7])
    # print(data.cdf.loc[188].iloc[:, 7:])
    # assert False

def test_tabular_data_trva_date_split():
    data = SubscriptionData(
        data=test_data,
        subscription_initial="subscription_initial",
        subscription_current="subscription_current",
        split_at="2018-10-21",
    )

    print(data.dtr.iloc[:, :4])
    print(data.dva.iloc[:, :4])
    print(data.dtr.iloc[:, 4:])
    print(data.dva.iloc[:, 4:])
    assert True

def test_crossdata_trva_date_split():
    data = SubscriptionData(
        data=test_data,
        subscription_initial="subscription_initial",
        subscription_current="subscription_current",
        split_at="2018-10-21",
    )

    assert data.cdtr.index.get_level_values("end_of_month").max() < datetime(2018, 10, 21)
    assert data.cdva.index.get_level_values("end_of_month").min() >= datetime(2018, 10, 21)

    assert data.cdtr.loc[5]["alive"].iloc[0] == 1
    assert data.cdtr.loc[782]["alive"].iloc[0] == 0

    assert data.cdtr.loc[3]["alive"].min() == 1
    assert data.cdva.loc[3]["alive"].min() == 1

    assert data.cdtr.loc[188]["alive"].min() == 1
    assert data.cdva.loc[188]["alive"].min() == 0

    assert data.cdtr.loc[89]["alive"].min() == 0
    assert data.cdva.loc[89]["alive"].max() == 0

    assert (data.cdtr.loc[62]["age"] * data.cdtr.loc[62]["alive"]).max() == 3
    assert (data.cdva.loc[62]["age"] * data.cdva.loc[62]["alive"]).max() == 5


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_bayesian_optimization.py
    """
    pytest.main([__file__])
